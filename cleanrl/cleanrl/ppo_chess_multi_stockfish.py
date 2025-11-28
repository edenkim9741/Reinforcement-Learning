import argparse
import time
from dataclasses import dataclass
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from gymnasium.vector import AsyncVectorEnv
import wandb
import chess
from pettingzoo.classic import chess_v6
import chess.engine

# 공통 모듈 임포트
from other_model.chess_minimax import encode_move, PIECE_VALUES

STOCKFISH_PATH = "/usr/games/stockfish"  # [수정 필요] 본인 경로 입력

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.h, self.w, self.c = obs_shape
        self.network = nn.Sequential(
            nn.Conv2d(self.c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * self.h * self.w, 512),
            nn.ReLU(),
        )
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def get_value(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        hidden = self.network(obs)
        return self.critic(hidden).squeeze(-1)

    def get_action_and_value(self, obs, action=None, action_mask=None):
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

# ==============================
#  Environment (With Dynamic Skill Setter)
# ==============================
class ChessSelfPlayEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "name": "ChessSelfPlayEnv"}

    def __init__(self, render_mode=None, opponent_skill=-1):
        super().__init__()
        self.render_mode = render_mode
        self.aec_env = chess_v6.env(render_mode=render_mode)
        self.aec_env.reset()
        
        self.opponent_skill = opponent_skill
        
        self.agents = self.aec_env.agents
        self.agent_id = "player_0"
        self.opponent_id = "player_1"
        raw_obs_space = self.aec_env.observation_space("player_0")["observation"]
        self.observation_space = spaces.Box(low=raw_obs_space.low, high=raw_obs_space.high, shape=raw_obs_space.shape, dtype=np.float32)
        self.action_space = self.aec_env.action_space("player_0")
        self.last_board_value = 0.0

        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        # Skill Level 설정을 위한 UCI 옵션 설정 (Stockfish 버전에 따라 다를 수 있음)
        self.engine.configure({"Skill Level": max(0, opponent_skill)})


    def set_skill_level(self, level):
        self.opponent_skill = level
        # 엔진에도 즉시 반영
        self.engine.configure({"Skill Level": max(0, level)})
        return self.opponent_skill

    def _opponent_step(self):
        board = self.aec_env.unwrapped.board
        
        # [최적화] 이미 켜져 있는 엔진에 요청만 보냄 (오버헤드 대폭 감소)
        # 시간 제한을 0.01초로 매우 짧게 주어 CPU 대기 시간을 줄임
        limit = chess.engine.Limit(time=0.01)
        try:
            result = self.engine.play(board, limit)
            best_move = result.move
        except:
            best_move = None
        
        if best_move is None:
            legal_moves = list(board.legal_moves)
            best_move = random.choice(legal_moves)

        if self.opponent_id == "player_1":
            return encode_move(best_move, should_mirror=(board.turn == chess.BLACK))

        return encode_move(best_move, should_mirror=(board.turn == chess.BLACK))

    def _get_board_value(self):
        board = self.aec_env.unwrapped.board
        value = 0.0
        for piece in board.piece_map().values():
            score = PIECE_VALUES.get(piece.piece_type, 0)
            if self.agent_id == "player_0":
                if piece.color == chess.WHITE: value += score
                else: value -= score
            else:
                if piece.color == chess.BLACK: value += score
                else: value -= score
        return value


    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed)
        self.last_board_value = 0.0
        if random.random() < 0.5:
            self.agent_id, self.opponent_id = "player_0", "player_1"
        else:
            self.agent_id, self.opponent_id = "player_1", "player_0"

        while self.aec_env.agent_selection != self.agent_id:
            if all(self.aec_env.terminations.values()) or all(self.aec_env.truncations.values()): break
            action_opp = self._opponent_step()
            self.aec_env.step(int(action_opp))

        self.last_board_value = self._get_board_value()
        if not (all(self.aec_env.terminations.values()) or all(self.aec_env.truncations.values())):
            raw_obs = self.aec_env.observe(self.agent_id)
            obs = raw_obs["observation"].copy().astype(np.float32)
            action_mask = raw_obs["action_mask"].copy().astype(np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            action_mask = np.ones(self.action_space.n, dtype=np.float32)
        return obs, {"action_mask": action_mask}

    def step(self, action):
        self.aec_env.step(int(action))
        terminated = all(self.aec_env.terminations.values())
        truncated = all(self.aec_env.truncations.values())
        done = terminated or truncated

        if not done:
            action_opp = self._opponent_step()
            self.aec_env.step(int(action_opp))
            terminated = all(self.aec_env.terminations.values())
            truncated = all(self.aec_env.truncations.values())
            done = terminated or truncated

        original_reward = float(self.aec_env.rewards[self.agent_id])
        current_board_value = self._get_board_value()
        shaping_reward = current_board_value - self.last_board_value
        shaping_coeff = 0.02
        total_reward = original_reward + (shaping_reward * shaping_coeff)
        self.last_board_value = current_board_value

        if not done:
            raw_obs = self.aec_env.observe(self.agent_id)
            obs = raw_obs["observation"].copy().astype(np.float32)
            action_mask = raw_obs["action_mask"].copy().astype(np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            action_mask = np.ones(self.action_space.n, dtype=np.float32)
        return obs, total_reward, terminated, truncated, {"action_mask": action_mask}
    
    def render(self): return self.aec_env.render()
    def close(self): 
        self.aec_env.close()
        if hasattr(self, 'engine'):
            self.engine.quit()

def record_chess_video(agent, device, video_dir, update_idx, max_steps=300):
    os.makedirs(video_dir, exist_ok=True)
    eval_env = ChessSelfPlayEnv(render_mode="rgb_array", opponent_skill=0) # 비디오는 약한 상대로
    eval_env = RecordVideo(eval_env, video_folder=video_dir, episode_trigger=lambda ep_id: True, name_prefix=f"update_{update_idx}")
    agent.eval()
    obs, info = eval_env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
    done = False; step = 0
    while not done and step < max_steps:
        with torch.no_grad(): action, _, _, _ = agent.get_action_and_value(obs_t, action_mask=mask_t)
        obs, _, terminated, truncated, info = eval_env.step(action.item())
        done = terminated or truncated
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        step += 1
    eval_env.close()
    agent.train()

@dataclass
class PPOConfig:
    exp_name: str = "ppo_chess_auto_curriculum"
    total_timesteps: int = 40_000_000
    learning_rate: float = 2.5e-4
    num_steps: int = 512
    num_envs: int = 64
    update_epochs: int = 5
    minibatch_size: int = 2048
    logging: bool = True 
    opponent_skill_start: int = -1
    # 자동 조정 관련 설정
    curriculum_threshold: float = 0.6  # 승률 점수가 0.6 이상이면 난이도 UP
    curriculum_check_interval: int = 50 # 50 update 마다 확인
    
    seed: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    resume: str = None

# ... (make_single_env, make_vector_env 동일) ...
def make_single_env(seed_offset=0, opponent_skill=0):
    def _init():
        env = ChessSelfPlayEnv(render_mode=None, opponent_skill=opponent_skill)
        env.reset(seed=seed_offset)
        return env
    return _init

def make_vector_env(num_envs, opponent_skill=0):
    return AsyncVectorEnv([make_single_env(seed_offset=i, opponent_skill=opponent_skill) for i in range(num_envs)])

def train(config: PPOConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    if config.logging:
        run = wandb.init(project="Reinforcement-Learning", config=config.__dict__)

    # 현재 Skill Level 추적 변수
    current_skill = config.opponent_skill_start

    # Env 생성
    env = make_vector_env(config.num_envs, opponent_skill=current_skill)
    
    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n
    agent = ActorCritic(obs_shape, n_actions).to(device)
    if config.resume: agent.load_state_dict(torch.load(config.resume, map_location=device))
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # ... Buffer Init ...
    # (버퍼 초기화 코드는 이전과 동일하여 생략, 실제로는 그대로 넣으세요)
    obs_buf = torch.zeros((config.num_steps, config.num_envs) + obs_shape, dtype=torch.float32, device=device)
    masks_buf = torch.zeros((config.num_steps, config.num_envs, n_actions), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
    advantages = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)

    episode_returns = []
    current_ep_return = np.zeros(config.num_envs, dtype=np.float32)

    next_obs_np, info = env.reset()
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device)
    next_done = torch.zeros((config.num_envs,), dtype=torch.float32, device=device)

    global_step = 0
    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)
    start_time = time.time()

    bar = tqdm(range(1, num_updates + 1), desc="PPO Training")
    for update in bar:
        # 1. Rollout Collection (이전과 동일)
        for step in range(config.num_steps):
            global_step += config.num_envs
            obs_buf[step], masks_buf[step], dones_buf[step] = next_obs, next_mask, next_done
            
            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(next_obs, action_mask=next_mask)
            
            actions_buf[step], logprobs_buf[step], values_buf[step] = actions, logprobs, values
            
            actions_np = actions.cpu().numpy()
            obs_step, rewards, terminated, truncated, infos = env.step(actions_np)
            done = np.logical_or(terminated, truncated)
            
            rewards_buf[step] = torch.tensor(rewards, device=device, dtype=torch.float32)
            current_ep_return += rewards
            
            if np.any(done):
                for i in range(config.num_envs):
                    if done[i]:
                        episode_returns.append(current_ep_return[i])
                        current_ep_return[i] = 0.0
            
            next_obs = torch.tensor(obs_step, dtype=torch.float32, device=device)
            next_mask = torch.tensor(infos["action_mask"], dtype=torch.float32, device=device)
            next_done = torch.tensor(done, dtype=torch.float32, device=device)

        # 2. GAE Calculation (이전과 동일)
        with torch.no_grad(): next_value = agent.get_value(next_obs)
        lastgaelam = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
        for t in reversed(range(config.num_steps)):
            if t == config.num_steps - 1: nextnonterminal = 1.0 - next_done; nextvalues = next_value
            else: nextnonterminal = 1.0 - dones_buf[t + 1]; nextvalues = values_buf[t + 1]
            delta = rewards_buf[t] + config.gamma * nextvalues * nextnonterminal - values_buf[t]
            lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values_buf

        # 3. Optimization (이전과 동일)
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
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], action_mask=b_masks[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss = torch.max(-mb_advantages * ratio, -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)).mean()
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * torch.max((newvalue - b_returns[mb_inds]) ** 2, (b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -config.clip_coef, config.clip_coef) - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - config.ent_coef * entropy.mean() + config.vf_coef * v_loss
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm); optimizer.step()

        # ==========================================
        # 4. [AUTO CURRICULUM] Skill Level Adjustment
        # ==========================================
        mean_ep_ret = np.mean(episode_returns[-50:]) if len(episode_returns) > 0 else 0.0
        
        # 주기적으로 체크
        if update % config.curriculum_check_interval == 0:
            # 승률(점수)이 기준치(0.6) 이상이고, 아직 최고 레벨(20)이 아니면
            if mean_ep_ret > config.curriculum_threshold and current_skill < 20:
                current_skill += 1
                
                # [핵심] 모든 벡터 환경 프로세스에 Skill 변경 명령 전송
                env.call("set_skill_level", current_skill)
                
                print(f"\n[CURRICULUM] Level UP! New Opponent Skill: {current_skill} (Avg Return: {mean_ep_ret:.2f})")
                
                # 리턴 기록 초기화 (새로운 레벨에서의 성능을 다시 측정하기 위해)
                episode_returns = [] 

        # WandB Logging
        if config.logging:
            run.log({
                "train/mean_ep_ret": mean_ep_ret,
                "train/opponent_skill": current_skill,
                "global_step": global_step
            })
            
        bar.set_postfix({"skill": current_skill, "ret": f"{mean_ep_ret:.2f}"})

        if update % 50 == 0:
            record_chess_video(agent, device, f"videos/{config.exp_name}", update)

    env.close()
    torch.save(agent.state_dict(), f"{config.exp_name}_final.pt")
    print(f"[INFO] Training completed. Model saved to {config.exp_name}_final.pt")
    if config.logging: run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logging", action="store_true", default=False)
    args = parser.parse_args()
    cfg = PPOConfig(
        resume=args.resume,
        logging=args.logging
    )
    train(cfg)