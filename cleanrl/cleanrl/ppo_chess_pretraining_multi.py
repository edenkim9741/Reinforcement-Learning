import argparse
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
import wandb
import chess
import chess.engine
from pettingzoo.classic import chess_v6

# Try importing custom module, else use dummy
try:
    from other_model.chess_minimax import evaluate_vs_stockfish
except ImportError:
    def evaluate_vs_stockfish(*args, **kwargs):
        return 0.0, 0.0, 1.0

import multiprocessing as mp

# ==========================================
# 0. Configuration
# ==========================================
from dataclasses import dataclass

@dataclass
class PPOConfig:
    exp_name: str = "chess_resnet_ppo_optimized"
    total_timesteps: int = 50_000_000
    learning_rate: float = 2.5e-4
    
    # Environment & Rollout
    num_steps: int = 512       # Agent steps per update per env
    num_envs: int = 32         # Number of parallel environments
    
    # Optimization
    minibatch_size: int = 1024
    update_epochs: int = 4

    # Pre-training
    pretrain_samples: int = 10_000
    pretrain_epochs: int = 10
    pretrain_batch_size: int = 256
    
    # Dataset Saving/Loading (새로 추가됨)
    dataset_path: str = "data/stockfish_pretrain_data.npz"  # 저장할 파일 경로
    
    # Coefficients
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # System
    seed: int = 42
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Experience Replay (Optional for PPO)
    use_replay_buffer: bool = True
    replay_buffer_size: int = 10_000
    replay_ratio: float = 0.2
    
    # Paths & Logging
    logging: bool = False
    resume: str = None
    stockfish_path: str = "/usr/games/stockfish"  # Update this path!
    stockfish_eval_time_limit: float = 0.05  # Time limit per move during evaluation
    stockfish_eval_skill: int = 0  # Skill level during evaluation (0-20)
    
    # Evaluation
    eval_interval: int = 50
    eval_games: int = 32
    
    # Early Stopping
    early_stopping_patience: int = 20
    target_reward: float = 0.8

# ==========================================
# 1. Environment & Utilities
# ==========================================

# Move conversion caching
_KNIGHT_MOVES_CACHE = {
    (1, 2): 56, (2, 1): 57, (2, -1): 58, (1, -2): 59,
    (-1, -2): 60, (-2, -1): 61, (-2, 1): 62, (-1, 2): 63
}
_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

def move_to_action(move: chess.Move) -> int:
    """Converts a chess.Move object to an integer action index (0-4671)."""
    if move is None: return 0
    from_sq = move.from_square
    to_sq = move.to_square
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    if (df, dr) in _KNIGHT_MOVES_CACHE:
        return from_sq * 73 + _KNIGHT_MOVES_CACHE[(df, dr)]

    if move.promotion is not None and move.promotion != chess.QUEEN:
        return None # Only allow Queen promotion for simplicity in this encoding

    for dir_idx, (dir_x, dir_y) in enumerate(_DIRECTIONS):
        if df * dir_y - dr * dir_x == 0 and df * dir_x + dr * dir_y > 0:
            distance = max(abs(df), abs(dr))
            if 1 <= distance <= 7:
                return from_sq * 73 + dir_idx * 7 + (distance - 1)
    return None

class ChessSymmetricEnv(gym.Env):
    """
    Wraps PettingZoo chess_v6 (AEC) into a standard Gym API.
    Handles self-play by always presenting the board from the current player's perspective.
    """
    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.aec_env = chess_v6.env(render_mode=render_mode)
        
        # Standard observation shape for Chess (8x8x111 typically in PettingZoo)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 111), dtype=np.float32)
        # 8*8*73 = 4672 actions
        self.action_space = spaces.Discrete(4672)
        
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
        # 수정됨: obs는 딕셔너리입니다. {'observation': ..., 'action_mask': ...}
        obs_dict, _, _, _, info = self.aec_env.last()
        
        # 딕셔너리에서 실제 관측값과 마스크를 분리합니다.
        obs = obs_dict["observation"]
        mask = obs_dict["action_mask"]
        
        return obs.astype(np.float32), {"action_mask": mask.astype(np.float32)}

    def step(self, action):
        # 1. Execute action for current agent
        self.aec_env.step(action)
        
        # 2. Check game state after move
        obs_dict, reward, terminated, truncated, info = self.aec_env.last()
        
        # 만약 게임이 끝났다면 obs_dict가 None일 수 있는 구버전 이슈 방지 및 데이터 추출
        if obs_dict is None:
             # 종료 시 더미 데이터 반환 (PettingZoo 버전에 따라 다를 수 있음)
             obs = np.zeros(self.observation_space.shape, dtype=np.float32)
             mask = np.ones(self.action_space.n, dtype=np.float32)
        else:
             obs = obs_dict["observation"]
             mask = obs_dict["action_mask"]

        # If the game ended immediately after my move
        if terminated or truncated:
            board = self.aec_env.unwrapped.board
            
            if board.is_repetition(3):
                reward = -0.5
            elif board.is_fifty_moves():
                reward = -0.5
            elif board.is_stalemate() or board.is_insufficient_material():
                reward = -0.1

            return obs.astype(np.float32), float(reward), True, False, {"action_mask": mask.astype(np.float32)}

        # 3. Reward Shaping (Optional)
        board = self.aec_env.unwrapped.board
        # material_score = self._get_material_score(board) # 필요한 경우 사용
        
        current_reward = 0.0 # 체스는 일반적으로 종료 시에만 보상을 줌
        
        return obs.astype(np.float32), current_reward, False, False, {"action_mask": mask.astype(np.float32)}

    def close(self):
        self.aec_env.close()

# ==========================================
# 2. Model (Optimized)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out.add_(identity)
        return self.relu(out)

class ChessAgent(nn.Module):
    def __init__(self, obs_shape, n_actions, num_res_blocks=4, channels=128):
        super().__init__()
        self.h, self.w, self.c = obs_shape
        
        self.start_block = nn.Sequential(
            nn.Conv2d(self.c, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.backbone = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        
        self.actor_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * self.h * self.w, n_actions)
        )
        
        self.critic_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * self.h * self.w, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def get_value(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.start_block(x)
        x = self.backbone(x)
        return self.critic_head(x)

    def forward(self, obs, action=None, action_mask=None):
        x = obs.permute(0, 3, 1, 2)
        x = self.start_block(x)
        x = self.backbone(x)
        
        logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)

        if action_mask is not None:
            # === [Safety Fix] ===
            # 어떤 행(sample)의 action_mask가 모두 0이라면(둘 수 있는 수가 없는 경우),
            # 모델이 죽지 않도록 임의로 첫 번째 행동(0번 인덱스)을 유효하게 만듭니다.
            # (이 데이터는 어차피 done=True라서 학습 가치 계산 시 무시되거나 끊깁니다)
            is_all_invalid = action_mask.sum(dim=1) == 0
            if is_all_invalid.any():
                # 원본 마스크를 건드리지 않기 위해 복사
                action_mask = action_mask.clone()
                action_mask[is_all_invalid, 0] = 1.0

            # 유효하지 않은 행동의 확률을 -1e9로 낮춤 (마스킹)
            logits = torch.where(action_mask.bool(), logits, torch.tensor(-1e9, device=logits.device))

        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value

# ==========================================
# 3. Simple Local Replay Buffer
# ==========================================
class LocalReplayBuffer:
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.logprobs = np.zeros((capacity,), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.masks = np.zeros((capacity, 4672), dtype=np.float32) # Standard chess action space
        self.idx = 0
        self.full = False
    
    def push(self, obs, action, logprob, reward, done, value, mask):
        # Expecting batches of data
        batch_size = len(obs)
        end_idx = self.idx + batch_size
        
        if end_idx > self.capacity:
            # Wrap around or just fill up (simplification: simple FIFO for PPO extra epochs)
            # For simplicity, we just fill until full then stop or overwrite from start
            # Let's do simple overwrite
            end_idx = self.capacity
            batch_size = end_idx - self.idx
        
        self.obs[self.idx:end_idx] = obs[:batch_size]
        self.actions[self.idx:end_idx] = action[:batch_size]
        self.logprobs[self.idx:end_idx] = logprob[:batch_size]
        self.rewards[self.idx:end_idx] = reward[:batch_size]
        self.dones[self.idx:end_idx] = done[:batch_size]
        self.values[self.idx:end_idx] = value[:batch_size]
        self.masks[self.idx:end_idx] = mask[:batch_size]
        
        self.idx = (self.idx + batch_size) % self.capacity
        if self.idx == 0 and batch_size > 0:
            self.full = True

    def sample(self, batch_size):
        limit = self.capacity if self.full else self.idx
        if limit < batch_size: return None
        
        indices = np.random.randint(0, limit, size=batch_size)
        
        return (
            torch.tensor(self.obs[indices], device=self.device),
            torch.tensor(self.actions[indices], device=self.device),
            torch.tensor(self.logprobs[indices], device=self.device),
            torch.tensor(self.rewards[indices], device=self.device),
            torch.tensor(self.dones[indices], device=self.device),
            torch.tensor(self.values[indices], device=self.device),
            torch.tensor(self.masks[indices], device=self.device)
        )
    
    def __len__(self):
        return self.capacity if self.full else self.idx

# ==========================================
# 4. Pre-training (Simplified)
# ==========================================
def collect_worker(rank, sample_count, config, seed):
    """
    개별 프로세스에서 실행되는 워커 함수입니다.
    자신만의 환경과 Stockfish 엔진을 생성하여 데이터를 수집합니다.
    """
    # 각 프로세스마다 다른 시드 설정 (중복 데이터 방지)
    process_seed = seed + rank
    np.random.seed(process_seed)
    random.seed(process_seed)
    
    # 환경 및 엔진 초기화
    env = ChessSymmetricEnv()
    try:
        engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        # config에 stockfish_eval_skill이 없다면 기본값 사용 (안전장치)
        skill = getattr(config, 'stockfish_eval_skill', 10)
        engine.configure({"Skill Level": skill})
    except Exception as e:
        print(f"[Worker {rank}] Error starting Stockfish: {e}")
        return [], [], []

    obs_list, act_list, val_list = [], [], []
    obs, info = env.reset(seed=process_seed)
    
    # 설정된 개수만큼 수집
    while len(obs_list) < sample_count:
        board = env.aec_env.unwrapped.board
        
        if board.is_game_over():
            obs, info = env.reset()
            continue
            
        try:
            time_limit = getattr(config, 'stockfish_eval_time_limit', 0.01)
            result = engine.analyse(board, chess.engine.Limit(time=time_limit))
            
            if "pv" not in result or not result["pv"]:
                obs, info = env.reset()
                continue
                
            best_move = result["pv"][0]
            score = result["score"].relative.score(mate_score=10000)
            value_target = np.tanh(score / 100.0) if score is not None else 0.0
            
            action_idx = move_to_action(best_move)
            
            if action_idx is not None and info["action_mask"][action_idx] == 1:
                obs_list.append(obs)
                act_list.append(action_idx)
                val_list.append(value_target)
                
                obs, _, terminated, truncated, info = env.step(action_idx)
                if terminated or truncated:
                    obs, info = env.reset()
            else:
                # Fallback: Random move
                valid_actions = np.where(info["action_mask"])[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                    obs, _, term, trunc, info = env.step(action)
                    if term or trunc: obs, info = env.reset()
                else:
                    obs, info = env.reset()
        except Exception:
            obs, info = env.reset()

    # 리소스 정리
    engine.quit()
    env.close()
    
    return obs_list, act_list, val_list

def collect_stockfish_data(config):
    """
    Main function to manage multi-processing data collection.
    """
    # 1. 저장된 데이터 확인 및 로드
    if os.path.exists(config.dataset_path):
        print(f"[PRE] Found saved dataset at '{config.dataset_path}'. Loading...")
        try:
            data = np.load(config.dataset_path)
            print(f"[PRE] Loaded {len(data['obs'])} samples successfully.")
            return data['obs'], data['acts'], data['vals']
        except Exception as e:
            print(f"[WARN] Failed to load dataset: {e}. Starting fresh collection.")

    # 2. 멀티프로세싱 설정
    # CPU 코어 수 확인 (너무 많이 쓰면 시스템이 느려질 수 있으므로 적절히 조절)
    num_workers = min(mp.cpu_count(), 8)  # 최대 8개 혹은 CPU 코어 수
    samples_per_worker = config.pretrain_samples // num_workers
    remainder = config.pretrain_samples % num_workers
    
    print(f"[PRE] Collecting {config.pretrain_samples} samples using {num_workers} workers...")
    
    # 각 워커에 전달할 인자 리스트 생성
    worker_args = []
    for i in range(num_workers):
        # 마지막 워커는 남은 자투리 샘플까지 처리
        count = samples_per_worker + (remainder if i == num_workers - 1 else 0)
        worker_args.append((i, count, config, config.seed))

    # 3. 병렬 실행
    t_start = time.time()
    with mp.Pool(processes=num_workers) as pool:
        # starmap은 인자 리스트를 언패킹해서 함수에 전달함
        results = pool.starmap(collect_worker, worker_args)
    
    # 4. 결과 병합 (Aggregation)
    all_obs, all_acts, all_vals = [], [], []
    total_collected = 0
    
    for r_obs, r_acts, r_vals in results:
        all_obs.extend(r_obs)
        all_acts.extend(r_acts)
        all_vals.extend(r_vals)
        total_collected += len(r_obs)
    
    print(f"[PRE] Collection finished in {time.time() - t_start:.2f}s. Total samples: {total_collected}")

    # numpy array로 변환
    obs_arr = np.array(all_obs, dtype=np.float32)
    acts_arr = np.array(all_acts, dtype=np.int64)
    vals_arr = np.array(all_vals, dtype=np.float32)

    # 5. 저장
    if config.dataset_path:
        os.makedirs(os.path.dirname(config.dataset_path), exist_ok=True)
        print(f"[PRE] Saving dataset to '{config.dataset_path}'...")
        np.savez_compressed(config.dataset_path, obs=obs_arr, acts=acts_arr, vals=vals_arr)
        print("[PRE] Save complete.")
    
    return obs_arr, acts_arr, vals_arr

def pretrain_supervised(agent, device, config):
    obs, acts, vals = collect_stockfish_data(config)
    if obs is None: return

    dataset = TensorDataset(
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor(acts, dtype=torch.long),
        torch.tensor(vals, dtype=torch.float32)
    )
    # Using small num_workers to avoid overhead
    loader = DataLoader(dataset, batch_size=config.pretrain_batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.AdamW(agent.parameters(), lr=1e-3)
    scaler = GradScaler('cuda') if config.use_mixed_precision else None
    
    print(f"[PRE] Training supervised for {config.pretrain_epochs} epochs...")
    agent.train()
    
    for epoch in range(config.pretrain_epochs):
        total_loss = 0
        for b_obs, b_act, b_val in tqdm(loader, leave=False):
            b_obs, b_act, b_val = b_obs.to(device), b_act.to(device), b_val.to(device)
            
            optimizer.zero_grad()
            
            if config.use_mixed_precision:
                with autocast('cuda'):
                    # Pass None for mask in pretraining
                    _, log_prob, _, value = agent(b_obs, action=b_act, action_mask=None)
                    loss = -log_prob.mean() + nn.functional.mse_loss(value, b_val)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, log_prob, _, value = agent(b_obs, action=b_act, action_mask=None)
                loss = -log_prob.mean() + nn.functional.mse_loss(value, b_val)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
    
    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), f"models/{config.exp_name}_pretrained.pt")

# ==========================================
# 5. Main Training Loop
# ==========================================
def make_env():
    """Helper for AsyncVectorEnv"""
    return ChessSymmetricEnv()

def train(config: PPOConfig):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")
    
    # Initialize WandB
    if config.logging:
        wandb.init(project="Chess-PPO-Optimized", config=config.__dict__, name=config.exp_name)

    # 1. Environment Setup (AsyncVectorEnv handles MP efficiently)
    print(f"[INFO] Creating {config.num_envs} environments...")
    envs = AsyncVectorEnv(
        [make_env for _ in range(config.num_envs)],
        daemon=True  # Important for cleanup
    )

    # 2. Agent Setup
    agent = ChessAgent(
        obs_shape=(8, 8, 111), 
        n_actions=4672
    ).to(device)
    
    # Load/Pretrain
    if config.resume:
        agent.load_state_dict(torch.load(config.resume, map_location=device, weights_only=True))
        print("[INFO] Model loaded.")
    else:
        pretrain_path = f"models/{config.exp_name}_pretrained.pt"
        if os.path.exists(pretrain_path):
            agent.load_state_dict(torch.load(pretrain_path, map_location=device, weights_only=True))
            print("[INFO] Pretrained model loaded.")
        else:
            pretrain_supervised(agent, device, config)

    optimizer = optim.AdamW(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    scaler = GradScaler() if config.use_mixed_precision else None
    
    # Replay Buffer
    replay_buffer = None
    if config.use_replay_buffer:
        replay_buffer = LocalReplayBuffer(config.replay_buffer_size, (8,8,111), device)

    # Storage for Rollouts (On GPU to save transfers)
    obs_buffer = torch.zeros((config.num_steps, config.num_envs, 8, 8, 111), device=device)
    actions_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    values_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    masks_buffer = torch.zeros((config.num_steps, config.num_envs, 4672), dtype=torch.bool, device=device)

    # Start Envs
    next_obs_np, next_info = envs.reset(seed=config.seed)
    next_obs = torch.tensor(next_obs_np, device=device)
    next_done = torch.zeros(config.num_envs, device=device)
    next_mask = torch.tensor(next_info["action_mask"], device=device)
    
    global_step = 0
    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)
    start_time = time.time()

    print("[INFO] Starting training loop...")

    for update in range(1, num_updates + 1):
        # --- 1. Rollout ---
        for step in range(config.num_steps):
            global_step += config.num_envs
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done
            masks_buffer[step] = next_mask.bool()

            with torch.no_grad():
                action, logprob, _, value = agent(next_obs, action_mask=next_mask)
                values_buffer[step] = value.flatten()
            
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            # Step Envs
            # Note: AsyncVectorEnv accepts numpy actions
            real_next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards_buffer[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(real_next_obs, device=device)
            next_done = torch.tensor(done, device=device, dtype=torch.float32)
            next_mask = torch.tensor(info["action_mask"], device=device)
            
            # Push to Replay Buffer
            if config.use_replay_buffer:
                replay_buffer.push(
                    obs_buffer[step].cpu().numpy(),
                    actions_buffer[step].cpu().numpy(),
                    logprobs_buffer[step].cpu().numpy(),
                    rewards_buffer[step].cpu().numpy(),
                    dones_buffer[step].cpu().numpy(),
                    values_buffer[step].cpu().numpy(),
                    masks_buffer[step].cpu().numpy()
                )

        # --- 2. GAE Calculation ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buffer)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                delta = rewards_buffer[t] + config.gamma * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buffer

        # Flatten the batch
        b_obs = obs_buffer.reshape((-1, 8, 8, 111))
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)
        b_masks = masks_buffer.reshape((-1, 4672))

        # --- 3. PPO Update ---
        b_inds = np.arange(config.num_steps * config.num_envs)
        clipfracs = []
        
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                # Mixed Precision Context
                with autocast('cuda', enabled=config.use_mixed_precision):
                    _, newlogprob, entropy, newvalue = agent(b_obs[mb_inds], b_actions[mb_inds], b_masks[mb_inds])
                    
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                        # old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    # Normalize advantages
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy Loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # --- 4. Optional: Auxiliary Replay Buffer Update ---
        if config.use_replay_buffer and len(replay_buffer) >= config.minibatch_size:
            # Train on some past data
            replay_epochs = int(config.update_epochs * config.replay_ratio)
            for _ in range(replay_epochs):
                batch = replay_buffer.sample(config.minibatch_size)
                if batch is None: break
                r_obs, r_act, r_lp, r_rew, r_done, r_val, r_mask = batch
                
                with autocast(enabled=config.use_mixed_precision):
                    _, newlogprob, entropy, newvalue = agent(r_obs, r_act, r_mask)
                    logratio = newlogprob - r_lp
                    ratio = logratio.exp()
                    pg_loss = -(ratio * r_rew).mean() # Simplified loss for replay
                    v_loss = 0.5 * ((newvalue - r_val) ** 2).mean()
                    loss = pg_loss + config.vf_coef * v_loss - config.ent_coef * entropy.mean()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # --- 5. Logging & Saving ---
        if update % 10 == 0:
            print(f"Update {update}/{num_updates} | Step {global_step} | Reward: {rewards_buffer.mean().item():.4f}")
            if config.logging:
                wandb.log({
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                    "charts/reward": rewards_buffer.mean().item()
                })

        if update % config.eval_interval == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(agent.state_dict(), f"models/{config.exp_name}_{update}.pt")
            eval_agent = agent.module if isinstance(agent, nn.DataParallel) else agent
            
            try:
                win, draw, loss = evaluate_vs_stockfish(
                    eval_agent,
                    device, 
                    config, 
                    num_games=config.eval_games,
                    update=update
                )
                
                print(f"[EVAL] Result - Win: {win*100:.1f}%, Draw: {draw*100:.1f}%, Loss: {loss*100:.1f}%")
                
                if config.logging:
                    wandb.log({
                        "eval/win_rate": win, 
                        "eval/draw_rate": draw, 
                        "eval/loss_rate": loss, 
                        "eval/stockfish_time_limit": config.stockfish_eval_time_limit,
                        "global_step": global_step,
                        "stockfish_skill": config.stockfish_eval_skill
                    })
                
                if win + draw == 1.0:
                    print("[EVAL] Perfect performance against Stockfish achieved! stockfish_eval_time_limit increased.")
                    # config.stockfish_eval_skill = min(config.stockfish_eval_skill + 1, 20)
                    config.stockfish_eval_time_limit = min(config.stockfish_eval_time_limit + 0.01, 0.1)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Eval failed: {e}")

    envs.close()
    if config.logging:
        wandb.finish()

if __name__ == "__main__":
    # Mac/Linux compatibility
    mp.set_start_method("spawn", force=True)
    

    
    # You can parse args here to override config
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish_path", type=str, default="/usr/games/stockfish")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=50)
    args = parser.parse_args()
    
    config = PPOConfig(
        stockfish_path = args.stockfish_path,
        logging = args.logging,
        eval_interval = args.eval_interval
    )

    try:
        train(config)

    except KeyboardInterrupt:
        print("Training interrupted.")