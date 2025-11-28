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

# [중요] 기존에 사용하시던 모듈이 있다고 가정합니다.
# 만약 없다면, 아래 PIECE_VALUES와 유사한 로직이 필요합니다.
from other_model.chess_minimax import PIECE_VALUES

# ==========================================
# 1. Model Definition (Actor-Critic)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        # PettingZoo Chess v6는 (8, 8, 111) 형태의 관측 공간을 가짐
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

    def get_value(self, obs):
        obs = obs.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        hidden = self.network(obs)
        return self.critic(hidden).squeeze(-1)

    def get_action_and_value(self, obs, action=None, action_mask=None):
        obs = obs.permute(0, 3, 1, 2)
        hidden = self.network(obs)
        logits = self.actor(hidden)

        if action_mask is not None:
            # 불가능한 수는 매우 작은 값으로 마스킹
            logits = logits.masked_fill(action_mask == 0, -1e9)
            
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(hidden).squeeze(-1)
        return action, log_prob, entropy, value

# ==========================================
# 2. Symmetric Self-Play Environment
# ==========================================
class ChessSymmetricEnv(gym.Env):
    """
    이 환경은 '나 자신'과의 대결을 위해 설계되었습니다.
    step()을 호출하면 턴이 변경됩니다 (White -> Black -> White ...).
    에이전트는 매 스텝마다 현재 턴 플레이어의 관점에서 최적의 수를 둡니다.
    """
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "name": "ChessSymmetricEnv"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.aec_env = chess_v6.env(render_mode=render_mode)
        self.aec_env.reset()
        
        # Observation & Action Space 정의
        # Player_0(White) 기준으로 Space를 가져옴
        raw_obs_space = self.aec_env.observation_space("player_0")["observation"]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=raw_obs_space.shape, dtype=np.float32
        )
        self.action_space = self.aec_env.action_space("player_0")
        
        self.current_agent = "player_0"
        self.step_count = 0

    def _get_board_value(self, agent_id):
        """보상 shaping을 위한 간단한 기물 점수 계산"""
        board = self.aec_env.unwrapped.board
        value = 0.0
        # PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, ...}
        for piece in board.piece_map().values():
            score = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                value += score
            else:
                value -= score
        
        # agent_id 관점에서 점수 반환
        if agent_id == "player_0": return value  # White
        else: return -value                      # Black

    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed)
        self.current_agent = self.aec_env.agent_selection
        self.step_count = 0
        
        obs_dict = self.aec_env.observe(self.current_agent)
        obs = obs_dict["observation"].astype(np.float32)
        mask = obs_dict["action_mask"].astype(np.float32)
        
        return obs, {"action_mask": mask}

    def step(self, action):
        # 1. 현재 플레이어의 Action 실행
        self.aec_env.step(int(action))
        self.step_count += 1
        
        # 2. 종료 조건 확인
        terminations = self.aec_env.terminations
        truncations = self.aec_env.truncations
        
        # 누군가 이겼거나 비겼다면
        if any(terminations.values()) or any(truncations.values()):
            # 승패 판정
            # 방금 수를 둔 플레이어(self.current_agent)가 이겼는지 확인
            rewards = self.aec_env.rewards
            my_reward = rewards[self.current_agent]
            
            # 종료 시에는 다음 관측값이 의미가 없으므로 0으로 채움
            dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            dummy_mask = np.ones(self.action_space.n, dtype=np.float32)
            
            return dummy_obs, float(my_reward), True, False, {"action_mask": dummy_mask, "winner": self.current_agent}

        # 3. 게임이 안 끝났으면 -> 다음 플레이어로 턴 넘어감
        self.current_agent = self.aec_env.agent_selection
        
        obs_dict = self.aec_env.observe(self.current_agent)
        next_obs = obs_dict["observation"].astype(np.float32)
        next_mask = obs_dict["action_mask"].astype(np.float32)
        
        # [중요] Self-Play 보상 설계
        # 게임이 진행 중일 때의 보상은 0이 기본이지만, 
        # 학습 가속화를 위해 약간의 Material Advantage(기물 점수)를 줄 수 있음.
        # 단, Zero-Sum 게임이므로 신중해야 함. 여기서는 0으로 둠.
        step_reward = 0.0
        
        return next_obs, step_reward, False, False, {"action_mask": next_mask}

    def render(self):
        return self.aec_env.render()

    def close(self):
        self.aec_env.close()

# ==========================================
# 3. PPO Configuration
# ==========================================
@dataclass
class PPOConfig:
    exp_name: str = "ppo_chess_selfplay_gpu"
    total_timesteps: int = 50_000_000
    learning_rate: float = 2.5e-4
    num_steps: int = 1024       # 롤아웃 길이 증가
    num_envs: int = 64          # GPU 배치 효율을 위해 환경 개수 증가 (CPU 코어 감안하여 조절)
    minibatch_size: int = 2048  # 배치 사이즈 증가
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

# ==========================================
# 4. Helper Functions (Env Setup)
# ==========================================
def make_single_env(seed_offset=0):
    def _init():
        env = ChessSymmetricEnv(render_mode=None)
        # Gym Wrapper로 감싸서 호환성 확보
        env = gym.wrappers.RecordEpisodeStatistics(env) 
        env.reset(seed=seed_offset)
        return env
    return _init

def make_vector_env(num_envs):
    # AsyncVectorEnv를 사용하여 멀티프로세싱 병렬화
    return AsyncVectorEnv([make_single_env(i) for i in range(num_envs)])

# ==========================================
# 5. Main Training Loop
# ==========================================
def train(config: PPOConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Device: {device}")
    
    if config.logging:
        wandb.init(project="Chess-RL-SelfPlay", config=config.__dict__, name=config.exp_name)

    # Vector Environment 생성
    env = make_vector_env(config.num_envs)
    
    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n
    
    agent = ActorCritic(obs_shape, n_actions).to(device)
    if config.resume:
        print(f"[INFO] Loading model from {config.resume}")
        agent.load_state_dict(torch.load(config.resume, map_location=device))
        
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # Buffer Initialization
    obs_buf = torch.zeros((config.num_steps, config.num_envs) + obs_shape, dtype=torch.float32, device=device)
    masks_buf = torch.zeros((config.num_steps, config.num_envs, n_actions), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    
    global_step = 0
    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)

    # Initial Observation
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
                action, logprob, _, value = agent.get_action_and_value(next_obs, action_mask=next_mask)
                values_buf[step] = value
            
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # Step Environment (CPU 병렬 처리)
            # 여기서는 Stockfish를 기다리지 않으므로 매우 빠름
            real_next_obs, rewards, terminated, truncated, infos = env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards_buf[step] = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            # 다음 상태 업데이트
            next_obs = torch.tensor(real_next_obs, dtype=torch.float32, device=device)
            next_mask = torch.tensor(infos["action_mask"], dtype=torch.float32, device=device)
            next_done = torch.tensor(done, dtype=torch.float32, device=device)

            # Logging (에피소드 끝난 경우)
            if "final_info" in infos:
                for i, info_item in enumerate(infos["final_info"]):
                    if info_item and "episode" in info_item:
                        if config.logging:
                            wandb.log({
                                "charts/episodic_return": info_item["episode"]["r"],
                                "charts/episodic_length": info_item["episode"]["l"],
                                "global_step": global_step
                            })

        # --- 2. GAE (Generalized Advantage Estimation) ---
        with torch.no_grad():
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

        # --- 3. PPO Update ---
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    action=b_actions[mb_inds], 
                    action_mask=b_masks[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss - config.ent_coef * entropy.mean() + config.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

        if config.logging and update % 10 == 0:
            wandb.log({
                "losses/policy_loss": pg_loss.item(),
                "losses/value_loss": v_loss.item(),
                "losses/entropy": entropy.mean().item(),
                "global_step": global_step
            })

        # --- 4. Video Recording (Optional) ---
        if update % 50 == 0:
            save_path = f"models/{config.exp_name}_{update}.pt"
            os.makedirs("models", exist_ok=True)
            torch.save(agent.state_dict(), save_path)
            # 비디오 녹화는 별도 함수로 호출 (위에서 정의한 ChessSelfPlayEnv 사용)

    env.close()
    if config.logging: wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logging", action="store_true", default=False)
    args = parser.parse_args()
    
    cfg = PPOConfig(resume=args.resume, logging=args.logging)
    train(cfg)