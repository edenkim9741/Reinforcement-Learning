import argparse
import time
from dataclasses import dataclass
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from gymnasium.vector import AsyncVectorEnv
import wandb
import chess
from pettingzoo.classic import chess_v6

# 기존 모듈 임포트 유지
from other_model.chess_minimax import PIECE_VALUES, evaluate_vs_stockfish

# ==========================================
# 1. Model Definition (Actor-Critic)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.h, self.w, self.c = obs_shape
        self.network = nn.Sequential(
            nn.Conv2d(self.c, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * self.h * self.w, 1024),
            nn.ReLU(),
        )
        self.actor = nn.Linear(1024, n_actions)
        self.critic = nn.Linear(1024, 1)

    # 기존 get_action_and_value의 로직을 forward로 이동합니다.
    def forward(self, obs, action=None, action_mask=None):
        # obs: (B, H, W, C) -> (B, C, H, W)
        obs = obs.permute(0, 3, 1, 2)
        hidden = self.network(obs)
        logits = self.actor(hidden)

        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
            
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(hidden).squeeze(-1)
        
        return action, log_prob, entropy, value

    # 기존 코드와의 호환성 및 단일 추론을 위해 남겨둠
    def get_action_and_value(self, obs, action=None, action_mask=None):
        return self.forward(obs, action, action_mask)

    def get_value(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        hidden = self.network(obs)
        return self.critic(hidden).squeeze(-1)

# ==========================================
# 2. Symmetric Self-Play Environment
# ==========================================
class ChessSymmetricEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "name": "ChessSymmetricEnv"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.aec_env = chess_v6.env(render_mode=render_mode)
        self.aec_env.reset()
        
        raw_obs_space = self.aec_env.observation_space("player_0")["observation"]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=raw_obs_space.shape, dtype=np.float32
        )
        self.action_space = self.aec_env.action_space("player_0")
        
        self.current_agent = "player_0"
        self.step_count = 0
        
    def _get_material_value(self, board):
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }

        white_score = 0
        black_score = 0
        for piece_type, v in piece_values.items():
            white_score += len(board.pieces(piece_type, chess.WHITE)) * v
            black_score += len(board.pieces(piece_type, chess.BLACK)) * v

        # 항상 "화이트 - 블랙" 형태로만 반환
        return white_score - black_score


    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed)
        self.current_agent = self.aec_env.agent_selection
        self.step_count = 0
        
        obs_dict = self.aec_env.observe(self.current_agent)
        obs = obs_dict["observation"].astype(np.float32)
        mask = obs_dict["action_mask"].astype(np.float32)
        return obs, {"action_mask": mask}

    def step(self, action):
        # 현재 수를 둘 플레이어의 색을 미리 저장
        board_before = self.aec_env.board
        mover_is_white = board_before.turn == chess.WHITE

        prev_score = self._get_material_value(board_before)

        # 실제 수 두기
        self.aec_env.step(int(action))
        self.step_count += 1

        terminations = self.aec_env.terminations
        truncations = self.aec_env.truncations

        # 게임 종료 처리
        if any(terminations.values()) or any(truncations.values()):
            rewards = self.aec_env.rewards
            my_reward = rewards[self.current_agent]  # 이 시점에서 current_agent는 여전히 수를 둔 쪽

            # 승/패 보상 스케일 (필요시 조절)
            final_reward = float(my_reward) * 10.0

            dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            dummy_mask = np.ones(self.action_space.n, dtype=np.float32)

            return dummy_obs, final_reward, True, False, {
                "action_mask": dummy_mask,
                "winner": self.current_agent,
            }

        # 비종료 상태: 다음 플레이어 관점 관찰
        self.current_agent = self.aec_env.agent_selection
        obs_dict = self.aec_env.observe(self.current_agent)
        next_obs = obs_dict["observation"].astype(np.float32)
        next_mask = obs_dict["action_mask"].astype(np.float32)

        # 새 점수 (항상 화이트-블랙)
        new_score = self._get_material_value(self.aec_env.board)

        # 이번 수를 둔 쪽 관점에서의 점수 변화량
        if mover_is_white:
            material_diff = new_score - prev_score   # 화이트 입장
        else:
            material_diff = prev_score - new_score   # 블랙 입장(부호 반전)

        # 스케일링 (0.01 ~ 0.1 정도 시도)
        material_reward = material_diff * 0.1

        return next_obs, material_reward, False, False, {"action_mask": next_mask}


    def render(self):
        return self.aec_env.render()

    def close(self):
        self.aec_env.close()

# ==========================================
# 3. PPO Configuration
# ==========================================
@dataclass
class PPOConfig:
    exp_name: str = "ppo_chess_selfplay_gpu_cnn"
    total_timesteps: int = 300_000_000
    learning_rate: float = 2.5e-4
    num_steps: int = 512
    num_envs: int = 32
    minibatch_size: int = 2048
    update_epochs: int = 4
    
    seed: int = 42
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    logging: bool = True
    resume: str = None

    # Stockfish 경로 확인 필요 (User 환경에 맞게)
    stockfish_path: str = "/usr/games/stockfish"
    stockfish_eval_skill: int = 0
    stockfish_eval_time_limit: float = 0
    eval_interval: int = 50
    eval_games: int = 16

# ==========================================
# 4. Helper Functions
# ==========================================
def make_single_env(seed_offset=0):
    def _init():
        env = ChessSymmetricEnv(render_mode=None)
        env = gym.wrappers.RecordEpisodeStatistics(env) 
        env.reset(seed=seed_offset)
        return env
    return _init

def make_vector_env(num_envs):
    return AsyncVectorEnv([make_single_env(i) for i in range(num_envs)])

# ==========================================
# 5. Main Training Loop
# ==========================================
def train(config: PPOConfig):
    # [Multi-GPU 수정] GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Device: {device}")
    
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs for training!")
    
    if config.logging:
        wandb.init(project="Reinforcement-Learning", config=config.__dict__, name=config.exp_name)

    env = make_vector_env(config.num_envs)
    
    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n
    
    agent = ActorCritic(obs_shape, n_actions).to(device)
    
    # [Multi-GPU 수정] 모델 로딩 시 분기 처리
    if config.resume:
        print(f"[INFO] Loading model from {config.resume}")
        checkpoint = torch.load(config.resume, map_location=device)
        
        # DataParallel로 저장된 모델(module. prefix)을 일반 모델에 로드할 때 처리
        # 만약 저장된 키가 'module.'로 시작하면 제거하고 로드
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        agent.load_state_dict(new_state_dict)

    # [Multi-GPU 수정] DataParallel 래핑
    if torch.cuda.device_count() > 1:
        agent = nn.DataParallel(agent)

    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    obs_buf = torch.zeros((config.num_steps, config.num_envs) + obs_shape, dtype=torch.float32, device=device)
    masks_buf = torch.zeros((config.num_steps, config.num_envs, n_actions), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    
    global_step = 0
    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)

    next_obs_np, info = env.reset()
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device)
    next_done = torch.zeros(config.num_envs, dtype=torch.float32, device=device)

    print(f"[INFO] Start Training for {num_updates} updates...")

    for update in tqdm(range(1, num_updates + 1)):
        # --- 1. Rollout Collection ---
        for step in range(config.num_steps):
            global_step += config.num_envs
            obs_buf[step] = next_obs
            masks_buf[step] = next_mask
            dones_buf[step] = next_done

            with torch.no_grad():
                # [Multi-GPU 수정] agent(x) 호출 -> forward() 실행 -> 자동 병렬화
                action, logprob, _, value = agent(next_obs, action_mask=next_mask)
                values_buf[step] = value
            
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            real_next_obs, rewards, terminated, truncated, infos = env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards_buf[step] = torch.tensor(rewards, dtype=torch.float32, device=device)
            next_obs = torch.tensor(real_next_obs, dtype=torch.float32, device=device)
            next_mask = torch.tensor(infos["action_mask"], dtype=torch.float32, device=device)
            next_done = torch.tensor(done, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for i, info_item in enumerate(infos["final_info"]):
                    if info_item and "episode" in info_item:
                        if config.logging:
                            wandb.log({
                                "charts/episodic_return": info_item["episode"]["r"],
                                "charts/episodic_length": info_item["episode"]["l"],
                                "global_step": global_step
                            })

        # --- 2. GAE ---
        with torch.no_grad():
            # [Multi-GPU Note] get_value는 DataParallel로 자동 분산되지 않으므로(forward가 아님)
            # 메인 GPU에서 실행되거나, agent.module.get_value를 호출해야 합니다.
            # 하지만 여기서는 배치가 작으므로(num_envs) 그냥 실행해도 무방합니다.
            # DataParallel 객체는 forward 외의 메서드를 바로 호출할 수 없으므로 module 접근 필요.
            if isinstance(agent, nn.DataParallel):
                next_value = agent.module.get_value(next_obs)
            else:
                next_value = agent.get_value(next_obs)
                
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                
                delta = rewards_buf[t] + config.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values_buf

        # --- 3. PPO Update (Multi-GPU Benefit Here) ---
        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_masks = masks_buf.reshape((-1, n_actions))

        b_inds = np.arange(config.num_steps * config.num_envs)
        
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                # [Multi-GPU 수정] forward 호출 (자동 병렬화)
                _, newlogprob, entropy, newvalue = agent(
                    b_obs[mb_inds], 
                    action=b_actions[mb_inds], 
                    action_mask=b_masks[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss - config.ent_coef * entropy.mean() + config.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

        if config.logging:
            wandb.log({
                "losses/policy_loss": pg_loss.item(),
                "losses/value_loss": v_loss.item(),
                "losses/entropy": entropy.mean().item(),
                "global_step": global_step
            })

        # --- 4. Save & Eval ---
        if update % 100 == 0:
            save_path = f"models/{config.exp_name}_{update}.pt"
            os.makedirs("models", exist_ok=True)
            # [Multi-GPU 수정] 저장 시 module의 state_dict 저장 (Load 호환성 위함)
            model_to_save = agent.module if isinstance(agent, nn.DataParallel) else agent
            torch.save(model_to_save.state_dict(), save_path)
            print(f"[INFO] Saved model to {save_path}")

            
        
        if update % config.eval_interval == 0:
            eval_agent = agent.module if isinstance(agent, nn.DataParallel) else agent
            
            try:
                win, draw, loss = evaluate_vs_stockfish(
                    eval_agent,
                    device, 
                    config, 
                    num_games=config.eval_games,
                    step_count=global_step 
                )
                
                print(f"[EVAL] Result - Win: {win*100:.1f}%, Draw: {draw*100:.1f}%, Loss: {loss*100:.1f}%")
                
                if config.logging:
                    wandb.log({
                        "eval/win_rate": win, 
                        "eval/draw_rate": draw, 
                        "eval/loss_rate": loss, 
                        "eval/stockfish_time_limit": config.stockfish_eval_time_limit,
                        "global_step": global_step,
                        # "stockfish_skill": config.stockfish_eval_skill
                    })
                
                if win + draw == 1.0:
                    print("[EVAL] Perfect performance against Stockfish achieved! stockfish_eval_time_limit increased.")
                    # config.stockfish_eval_skill = min(config.stockfish_eval_skill + 1, 20)
                    config.stockfish_eval_time_limit = min(config.stockfish_eval_time_limit + 0.01, 0.1)

                    
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Eval failed: {e}")

    env.close()
    if config.logging: wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logging", action="store_true", default=False)
    args = parser.parse_args()
    
    cfg = PPOConfig(resume=args.resume, logging=args.logging)
    train(cfg)