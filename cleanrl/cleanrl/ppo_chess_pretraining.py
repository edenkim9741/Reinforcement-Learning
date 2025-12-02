import argparse
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
import wandb
import chess
import chess.engine
from pettingzoo.classic import chess_v6
from collections import deque

try:
    from other_model.chess_minimax import evaluate_vs_stockfish
except ImportError:
    print("[WARN] 'other_model.chess_minimax' not found. Using dummy eval function.")
    def evaluate_vs_stockfish(*args, **kwargs):
        return 0.0, 0.0, 1.0

# ==========================================
# 0. Configuration (Optimized)
# ==========================================
from dataclasses import dataclass

@dataclass
class PPOConfig:
    exp_name: str = "chess_resnet_ppo_optimized"
    total_timesteps: int = 50_000_000
    learning_rate: float = 3.0e-4  # Increased for faster convergence
    num_steps: int = 1024  # Increased for better sample efficiency
    num_envs: int = 64  # Doubled for better parallelization
    minibatch_size: int = 4096  # Larger minibatches
    update_epochs: int = 4
    
    seed: int = 42
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Optimization Features
    use_mixed_precision: bool = True  # AMP for 2x speedup
    gradient_accumulation_steps: int = 1  # Simulate larger batches
    use_jit: bool = True  # JIT compile for speed
    use_checkpoint: bool = False  # Memory-efficient backprop
    
    # Experience Replay
    replay_buffer_size: int = 50000
    replay_ratio: float = 0.25  # % of updates from replay
    
    # Pre-training Config (Reduced for efficiency)
    pretrain_samples: int = 10_000  # Reduced from 20k
    pretrain_epochs: int = 5  # Reduced from 10
    pretrain_batch_size: int = 512  # Increased batch size
    
    # System Config
    logging: bool = True
    resume: str = None
    stockfish_path: str = "/usr/games/stockfish"
    num_dataloader_workers: int = 4  # Parallel data loading
    
    # Eval Config
    eval_interval: int = 50
    eval_games: int = 10
    stockfish_eval_time_limit: float = 0.05
    
    # Early Stopping
    early_stopping_patience: int = 10
    target_reward: float = 0.5

# ==========================================
# 1. Environment (Optimized)
# ==========================================
class ChessSymmetricEnv(gym.Env):
    __slots__ = ('aec_env', 'observation_space', 'action_space', 'current_agent', '_piece_values')
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.aec_env = chess_v6.env(render_mode=render_mode)
        self.aec_env.reset()
        
        raw_obs = self.aec_env.observe("player_0")["observation"]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=raw_obs.shape, dtype=np.float32
        )
        self.action_space = self.aec_env.action_space("player_0")
        self.current_agent = "player_0"
        
        # Pre-compute piece values for speed
        self._piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, 
            chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9
        }
    
    def _get_material_score(self, board):
        score = 0
        for pt, val in self._piece_values.items():
            score += len(board.pieces(pt, chess.WHITE)) * val
            score -= len(board.pieces(pt, chess.BLACK)) * val
        return score

    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed)
        self.current_agent = self.aec_env.agent_selection
        obs_dict = self.aec_env.observe(self.current_agent)
        return obs_dict["observation"].astype(np.float32), {"action_mask": obs_dict["action_mask"].astype(np.float32)}

    def step(self, action):
        prev_board = self.aec_env.board
        prev_score = self._get_material_score(prev_board)
        mover_is_white = prev_board.turn == chess.WHITE

        self.aec_env.step(int(action))

        if any(self.aec_env.terminations.values()) or any(self.aec_env.truncations.values()):
            my_reward = self.aec_env.rewards[self.current_agent]
            dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            dummy_mask = np.ones(self.action_space.n, dtype=np.float32)
            return dummy_obs, float(my_reward), True, False, {"action_mask": dummy_mask}

        self.current_agent = self.aec_env.agent_selection
        obs_dict = self.aec_env.observe(self.current_agent)
        
        new_score = self._get_material_score(self.aec_env.board)
        diff = new_score - prev_score
        step_reward = (diff if mover_is_white else -diff) * 0.05

        return obs_dict["observation"].astype(np.float32), step_reward, False, False, {"action_mask": obs_dict["action_mask"].astype(np.float32)}

    def close(self):
        self.aec_env.close()

# Move conversion (optimized with lookup table)
_KNIGHT_MOVES_CACHE = {
    (1, 2): 56, (2, 1): 57, (2, -1): 58, (1, -2): 59,
    (-1, -2): 60, (-2, -1): 61, (-2, 1): 62, (-1, 2): 63
}

_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

def move_to_action(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    # Knight moves
    if (df, dr) in _KNIGHT_MOVES_CACHE:
        return from_sq * 73 + _KNIGHT_MOVES_CACHE[(df, dr)]

    # Underpromotion skip
    if move.promotion is not None and move.promotion != chess.QUEEN:
        return None

    # Direction-based moves
    for dir_idx, (dir_x, dir_y) in enumerate(_DIRECTIONS):
        if df * dir_y - dr * dir_x == 0 and df * dir_x + dr * dir_y > 0:
            distance = max(abs(df), abs(dr))
            if 1 <= distance <= 7:
                return from_sq * 73 + dir_idx * 7 + (distance - 1)
    return None

# ==========================================
# 2. ResNet Model (Optimized)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)  # bias=False for BN
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)  # In-place for memory

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out.add_(identity)  # In-place addition
        return self.relu(out)

class ChessAgent(nn.Module):
    def __init__(self, obs_shape, n_actions, num_res_blocks=6, channels=192):  # Deeper & wider
        super().__init__()
        self.h, self.w, self.c = obs_shape
        
        self.start_block = nn.Sequential(
            nn.Conv2d(self.c, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.backbone = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        
        # Policy head (optimized)
        self.actor_head = nn.Sequential(
            nn.Conv2d(channels, 64, 1, bias=False),  # More filters
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * self.h * self.w, n_actions)
        )
        
        # Value head (optimized)
        self.critic_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * self.h * self.w, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Regularization
            nn.Linear(256, 1)
        )

    def forward(self, obs, action=None, action_mask=None):
        x = obs.permute(0, 3, 1, 2)
        x = self.start_block(x)
        x = self.backbone(x)
        
        logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)

        if action_mask is not None:
            logits = torch.where(action_mask.bool(), logits, torch.tensor(-1e8, device=logits.device))

        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value
    
    def get_value(self, obs):
        x = obs.permute(0, 3, 1, 2)
        x = self.start_block(x)
        x = self.backbone(x)
        return self.critic_head(x).squeeze(-1)

# ==========================================
# 3. Experience Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, logprob, reward, done, value, mask):
        self.buffer.append((obs, action, logprob, reward, done, value, mask))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        
        obs, actions, logprobs, rewards, dones, values, masks = zip(*batch)
        
        return (
            torch.stack(obs).to(self.device),
            torch.stack(actions).to(self.device),
            torch.stack(logprobs).to(self.device),
            torch.stack(rewards).to(self.device),
            torch.stack(dones).to(self.device),
            torch.stack(values).to(self.device),
            torch.stack(masks).to(self.device)
        )
    
    def __len__(self):
        return len(self.buffer)

# ==========================================
# 4. Pre-training (Optimized)
# ==========================================
def collect_data_from_stockfish(config):
    print(f"[PRE-TRAIN] Collecting {config.pretrain_samples} samples...")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    except FileNotFoundError:
        print("[ERROR] Invalid Stockfish path. Skipping pre-training.")
        return None, None, None

    env = ChessSymmetricEnv()
    obs_list, act_list, val_list = [], [], []
    obs, info = env.reset()
    
    pbar = tqdm(total=config.pretrain_samples)
    collected = 0
    
    while collected < config.pretrain_samples:
        try:
            board = env.aec_env.board
            if board.is_game_over():
                obs, info = env.reset()
                continue

            result = engine.analyse(board, chess.engine.Limit(time=0.01))
            
            if "pv" not in result or not result["pv"]:
                obs, info = env.reset()
                continue

            best_move = result["pv"][0]
            score_pov = result["score"].relative.score(mate_score=10000)
            value_target = np.tanh(score_pov / 100.0) if score_pov is not None else 0.0
            
            action_idx = move_to_action(best_move)
            
            if action_idx is not None and 0 <= action_idx < len(info["action_mask"]) and info["action_mask"][action_idx] == 1:
                obs_list.append(obs)
                act_list.append(action_idx)
                val_list.append(value_target)
                collected += 1
                pbar.update(1)
                
                obs, _, terminated, truncated, info = env.step(action_idx)
                if terminated or truncated:
                    obs, info = env.reset()
            else:
                legal_actions = np.where(info["action_mask"] == 1)[0]
                if len(legal_actions) > 0:
                    obs, _, terminated, truncated, info = env.step(np.random.choice(legal_actions))
                    if terminated or truncated:
                        obs, info = env.reset()
                else:
                    obs, info = env.reset()
        except:
            obs, info = env.reset()

    engine.quit()
    env.close()
    pbar.close()
    
    return np.array(obs_list), np.array(act_list), np.array(val_list)

def pretrain_supervised(agent, device, config):
    obs_data, act_data, val_data = collect_data_from_stockfish(config)
    if obs_data is None:
        return

    dataset = TensorDataset(
        torch.tensor(obs_data, dtype=torch.float32),
        torch.tensor(act_data, dtype=torch.long),
        torch.tensor(val_data, dtype=torch.float32)
    )
    loader = DataLoader(
        dataset, 
        batch_size=config.pretrain_batch_size, 
        shuffle=True,
        num_workers=config.num_dataloader_workers,
        pin_memory=True
    )
    
    optimizer = optim.AdamW(agent.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.pretrain_epochs)
    scaler = GradScaler() if config.use_mixed_precision else None
    
    print(f"[PRE-TRAIN] Training for {config.pretrain_epochs} epochs...")
    agent.train()
    
    for epoch in range(config.pretrain_epochs):
        epoch_loss = 0
        for b_obs, b_act, b_val in loader:
            b_obs, b_act, b_val = b_obs.to(device), b_act.to(device), b_val.to(device)
            
            optimizer.zero_grad()
            
            if config.use_mixed_precision:
                with autocast():
                    _, log_prob, _, value = agent(b_obs, action=b_act)
                    loss = -log_prob.mean() + nn.functional.mse_loss(value, b_val)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, log_prob, _, value = agent(b_obs, action=b_act)
                loss = -log_prob.mean() + nn.functional.mse_loss(value, b_val)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}: Loss {epoch_loss/len(loader):.4f}, LR {scheduler.get_last_lr()[0]:.6f}")
    
    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), f"models/{config.exp_name}_pretrained.pt")
    print("[PRE-TRAIN] Done.")

# ==========================================
# 5. Main PPO Loop (Optimized)
# ==========================================
def train_ppo(config: PPOConfig):
    torch.backends.cudnn.benchmark = True  # Optimize cudnn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    if config.logging:
        wandb.init(project="Chess-RL-PPO", config=config.__dict__, name=config.exp_name)

    envs = AsyncVectorEnv([lambda: ChessSymmetricEnv() for _ in range(config.num_envs)])
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n
    
    agent = ChessAgent(obs_shape, n_actions).to(device)

    if config.resume:
        print(f"[INFO] Loading checkpoint: {config.resume}")
        state_dict = torch.load(config.resume, map_location=device)
        agent.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    else:
        pretrain_path = f"models/{config.exp_name}_pretrained.pt"
        if os.path.exists(pretrain_path):
            print("[INFO] Loading pretrained model...")
            agent.load_state_dict(torch.load(pretrain_path))
        else:
            pretrain_supervised(agent, device, config)

    # JIT Compile for speed
    if config.use_jit:
        dummy_obs = torch.randn(1, *obs_shape).to(device)
        dummy_mask = torch.ones(1, n_actions).to(device)
        agent = torch.jit.trace(agent, (dummy_obs, None, dummy_mask))

    if torch.cuda.device_count() > 1:
        agent = nn.DataParallel(agent)

    optimizer = optim.AdamW(agent.parameters(), lr=config.learning_rate, eps=1e-5, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_timesteps // (config.num_steps * config.num_envs))
    scaler = GradScaler() if config.use_mixed_precision else None
    
    # Replay Buffer
    replay_buffer = ReplayBuffer(config.replay_buffer_size, obs_shape, device)

    # Storage
    obs = torch.zeros((config.num_steps, config.num_envs) + obs_shape, device=device)
    actions = torch.zeros((config.num_steps, config.num_envs), device=device)
    logprobs = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones = torch.zeros((config.num_steps, config.num_envs), device=device)
    values = torch.zeros((config.num_steps, config.num_envs), device=device)
    masks = torch.zeros((config.num_steps, config.num_envs, n_actions), device=device)

    global_step = 0
    next_obs_np, info = envs.reset()
    next_obs = torch.tensor(next_obs_np, device=device)
    next_done = torch.zeros(config.num_envs, device=device)
    next_mask = torch.tensor(info["action_mask"], device=device)

    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)
    best_reward = -float('inf')
    patience_counter = 0

    print("[INFO] Starting PPO training...")
    start_time = time.time()
    
    for update in tqdm(range(1, num_updates + 1)):
        # Rollout
        for step in range(config.num_steps):
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            masks[step] = next_mask

            with torch.no_grad():
                action, logprob, _, value = agent(next_obs, action_mask=next_mask)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            real_next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(real_next_obs, device=device)
            next_done = torch.tensor(done, device=device, dtype=torch.float32)
            next_mask = torch.tensor(info["action_mask"], device=device)
            
            # Add to replay buffer
            for i in range(config.num_envs):
                replay_buffer.push(
                    obs[step][i].cpu(), actions[step][i].cpu(), 
                    logprobs[step][i].cpu(), rewards[step][i].cpu(),
                    dones[step][i].cpu(), values[step][i].cpu(), masks[step][i].cpu()
                )

        # GAE (Optimized)
        with torch.no_grad():
            model = agent.module if isinstance(agent, nn.DataParallel) else agent
            next_value = model.get_value(next_obs).reshape(1, -1)
            
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                nextnonterminal = 1.0 - (next_done if t == config.num_steps - 1 else dones[t + 1])
                nextvalues = next_value if t == config.num_steps - 1 else values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_masks = masks.reshape((-1, n_actions))

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO Update
        b_inds = np.arange(config.num_steps * config.num_envs)
        
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, len(b_inds), config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                optimizer.zero_grad()
                
                if config.use_mixed_precision:
                    with autocast():
                        _, newlogprob, entropy, newvalue = agent(
                            b_obs[mb_inds], b_actions[mb_inds], b_masks[mb_inds]
                        )
                        
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()
                        
                        mb_advantages = b_advantages[mb_inds]
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                        loss = pg_loss - config.ent_coef * entropy.mean() + config.vf_coef * v_loss
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    _, newlogprob, entropy, newvalue = agent(
                        b_obs[mb_inds], b_actions[mb_inds], b_masks[mb_inds]
                    )
                    
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    mb_advantages = b_advantages[mb_inds]
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                    loss = pg_loss - config.ent_coef * entropy.mean() + config.vf_coef * v_loss
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    optimizer.step()
        
        scheduler.step()

        # Replay Buffer Update
        if len(replay_buffer) >= config.minibatch_size:
            num_replay_updates = int(config.update_epochs * config.replay_ratio)
            for _ in range(num_replay_updates):
                r_obs, r_act, r_logprob, r_rew, r_done, r_val, r_mask = replay_buffer.sample(config.minibatch_size)
                
                optimizer.zero_grad()
                _, newlogprob, entropy, newvalue = agent(r_obs, r_act, r_mask)
                
                logratio = newlogprob - r_logprob
                ratio = logratio.exp()
                
                # Simple replay loss
                pg_loss = -(ratio * r_rew).mean()
                v_loss = 0.5 * ((newvalue - r_val) ** 2).mean()
                loss = pg_loss + config.vf_coef * v_loss - config.ent_coef * entropy.mean()
                
                if config.use_mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    optimizer.step()

        # Logging
        if config.logging and update % 10 == 0:
            avg_reward = rewards.mean().item()
            wandb.log({
                "losses/policy_loss": pg_loss.item(),
                "losses/value_loss": v_loss.item(),
                "losses/entropy": entropy.mean().item(),
                "charts/learning_rate": scheduler.get_last_lr()[0],
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "charts/avg_reward": avg_reward,
                "charts/replay_buffer_size": len(replay_buffer),
                "global_step": global_step
            })

        # Early Stopping Check
        avg_reward = rewards.mean().item()
        if avg_reward > best_reward:
            best_reward = avg_reward
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stopping_patience:
            print(f"[INFO] Early stopping triggered. No improvement for {config.early_stopping_patience} updates.")
            break

        # Save & Eval
        if update % config.eval_interval == 0:
            os.makedirs("models", exist_ok=True)
            model_to_save = agent.module if isinstance(agent, nn.DataParallel) else agent
            
            # Remove JIT wrapper for saving
            if config.use_jit and isinstance(model_to_save, torch.jit.ScriptModule):
                # Save the original model before JIT compilation
                # Note: JIT models need special handling
                torch.jit.save(model_to_save, f"models/{config.exp_name}_{update}_jit.pt")
            else:
                torch.save(model_to_save.state_dict(), f"models/{config.exp_name}_{update}.pt")
            
            print(f"\n[SAVE] Model saved at update {update}")
            print(f"[INFO] Best Reward: {best_reward:.4f}, Current Reward: {avg_reward:.4f}")
            print(f"[INFO] Total Steps: {global_step:,}, Time: {time.time() - start_time:.2f}s")
            
            # Eval vs Stockfish
            try:
                # Create eval model without JIT for evaluation
                eval_model = ChessAgent(obs_shape, n_actions).to(device)
                if isinstance(agent, nn.DataParallel):
                    eval_model.load_state_dict(agent.module.state_dict())
                elif config.use_jit:
                    # For JIT models, we need to create a fresh model
                    torch.jit.save(agent, "temp_jit_model.pt")
                    loaded_jit = torch.jit.load("temp_jit_model.pt")
                    # Extract state dict is not straightforward with JIT, skip eval
                    print("[WARN] Skipping evaluation for JIT compiled model")
                    eval_model = None
                else:
                    eval_model.load_state_dict(agent.state_dict())
                
                if eval_model is not None:
                    win, draw, loss = evaluate_vs_stockfish(
                        eval_model, device, config, 
                        num_games=config.eval_games, step_count=global_step
                    )
                    print(f"[EVAL] Win: {win:.2f}, Draw: {draw:.2f}, Loss: {loss:.2f}")
                    
                    if config.logging:
                        wandb.log({
                            "eval/win_rate": win,
                            "eval/draw_rate": draw,
                            "eval/loss_rate": loss,
                            "global_step": global_step
                        })
            except Exception as e:
                print(f"[WARN] Evaluation failed: {e}")

        # Target reward check
        if avg_reward >= config.target_reward:
            print(f"[INFO] Target reward {config.target_reward} reached! Stopping training.")
            break

    # Final save
    envs.close()
    os.makedirs("models", exist_ok=True)
    final_model = agent.module if isinstance(agent, nn.DataParallel) else agent
    
    if config.use_jit and isinstance(final_model, torch.jit.ScriptModule):
        torch.jit.save(final_model, f"models/{config.exp_name}_final_jit.pt")
    else:
        torch.save(final_model.state_dict(), f"models/{config.exp_name}_final.pt")
    
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Training finished!")
    print(f"Total Steps: {global_step:,}")
    print(f"Total Time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Best Reward: {best_reward:.4f}")
    print(f"Final Model: models/{config.exp_name}_final.pt")
    print(f"{'='*60}\n")
    
    if config.logging:
        wandb.finish()

# ==========================================
# 6. Main Entry Point
# ==========================================
if __name__ == "__main__":
    # Set random seeds for reproducibility
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    parser = argparse.ArgumentParser(description="Optimized Chess PPO Training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--logging", action="store_true", default=False, help="Enable W&B logging")
    parser.add_argument("--exp_name", type=str, default="chess_resnet_ppo_optimized", help="Experiment name")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=50_000_000, help="Total training timesteps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_jit", action="store_true", default=False, help="Use JIT compilation (faster but harder to debug)")
    parser.add_argument("--no_amp", action="store_true", default=False, help="Disable mixed precision training")
    
    args = parser.parse_args()
    
    # Create config
    cfg = PPOConfig(
        exp_name=args.exp_name,
        resume=args.resume,
        logging=args.logging,
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_jit=args.use_jit,
        use_mixed_precision=not args.no_amp
    )
    
    # Set seed
    set_seed(cfg.seed)
    
    # Print configuration
    print("\n" + "="*60)
    print("OPTIMIZED CHESS PPO TRAINING")
    print("="*60)
    print(f"Experiment: {cfg.exp_name}")
    print(f"Total Timesteps: {cfg.total_timesteps:,}")
    print(f"Num Envs: {cfg.num_envs}")
    print(f"Learning Rate: {cfg.learning_rate}")
    print(f"Mixed Precision: {cfg.use_mixed_precision}")
    print(f"JIT Compilation: {cfg.use_jit}")
    print(f"Replay Buffer: {cfg.replay_buffer_size:,}")
    print(f"Logging: {'W&B' if cfg.logging else 'Disabled'}")
    print("="*60 + "\n")
    
    # Start training
    try:
        train_ppo(cfg)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
