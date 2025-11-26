import argparse
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from gymnasium.vector import AsyncVectorEnv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from pettingzoo.classic import tictactoe_v3

import os

def record_chess_video(agent, device, video_dir, update_idx, max_steps=300):
    """
    현재 agent policy로 체스 한 판(or max_steps) 두면서 video를 저장.
    video_dir/update_{update_idx}.* 로 파일 생성됨.
    """
    os.makedirs(video_dir, exist_ok=True)

    # 비디오 파일 prefix (파일명이 update_20-episode-0 이런 식으로 저장됨)
    name_prefix = f"update_{update_idx}"

    # rgb_array 모드로 단일 env 생성 + RecordVideo 래퍼
    eval_env = ChessSelfPlayEnv(render_mode="rgb_array")
    eval_env = RecordVideo(
        eval_env,
        video_folder=video_dir,
        episode_trigger=lambda ep_id: True,  # 첫 에피소드만 저장
        name_prefix=name_prefix,
    )

    agent.eval()  # eval 모드

    obs, info = eval_env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)

    done = False
    step = 0

    while not done and step < max_steps:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(
                obs_t, action_mask=mask_t
            )
        action_id = action.item()

        obs, reward, terminated, truncated, info = eval_env.step(action_id)
        done = terminated or truncated

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        step += 1

    eval_env.close()
    agent.train()  # 다시 train 모드



# ==============================
#  ChessSelfPlayEnv (PettingZoo -> Gymnasium 단일 env)
# ==============================
class ChessSelfPlayEnv(gym.Env):
    """
    PettingZoo classic.chess_v6 AEC 환경을
    '한 개의 PPO 정책이 양쪽 플레이어(백/흑)를 모두 담당하는'
    self-play 단일 에이전트 Gymnasium 환경으로 감싼 래퍼입니다.

    obs:   raw_obs["observation"]
    info:  {"action_mask": raw_obs["action_mask"]}
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "ChessSelfPlayEnv"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        # self.aec_env = chess_v6.env(render_mode=render_mode)
        self.aec_env = tictactoe_v3.env(render_mode=render_mode)  # 임시로 틱택토로 테스트
        self.aec_env.reset()

        first_agent = self.aec_env.agent_selection
        raw_obs_space = self.aec_env.observation_space(first_agent)["observation"]

        self.observation_space = spaces.Box(
            low=raw_obs_space.low,
            high=raw_obs_space.high,
            shape=raw_obs_space.shape,
            dtype=np.float32,
        )
        self.action_space = self.aec_env.action_space(first_agent)

    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed)
        self.current_agent = self.aec_env.agent_selection

        raw_obs = self.aec_env.observe(self.current_agent)
        obs = raw_obs["observation"].astype(np.float32)
        action_mask = raw_obs["action_mask"].astype(np.float32)

        info = {"action_mask": action_mask}
        return obs, info

    def step(self, action):
        acting_agent = self.aec_env.agent_selection

        # 한 수 진행
        self.aec_env.step(int(action))

        reward = float(self.aec_env.rewards[acting_agent])
        terminated = all(self.aec_env.terminations.values())
        truncated = all(self.aec_env.truncations.values())
        done = terminated or truncated

        if not done:
            # 다음 agent 차례
            self.current_agent = self.aec_env.agent_selection
            raw_obs = self.aec_env.observe(self.current_agent)
            obs = raw_obs["observation"].astype(np.float32)
            action_mask = raw_obs["action_mask"].astype(np.float32)
        else:
            # 게임 끝: dummy obs & mask (vector env에서 바로 reset해 줄 거라 크게 상관 없음)
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            action_mask = np.ones(self.action_space.n, dtype=np.float32)

        info = {"action_mask": action_mask}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.aec_env.render()

    def close(self):
        self.aec_env.close()


# ==============================
#  PPO 네트워크 & 하이퍼파라미터
# ==============================

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))
        hidden_dim = 256

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def get_value(self, obs):
        x = self.encoder(obs)
        return self.value_head(x).squeeze(-1)  # (B,)

    def get_action_and_value(self, obs, action=None, action_mask=None):
        """
        obs: (B, *obs_shape)
        action_mask: (B, n_actions) - 1.0 for legal, 0.0 for illegal
        """
        x = self.encoder(obs)
        logits = self.policy_head(x)  # (B, n_actions)

        if action_mask is not None:
            # illegal move의 logit을 매우 작은 값으로 마스킹 → 선택 불가
            logits = logits.masked_fill(action_mask == 0, -1e9)

        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()  # (B,)

        log_prob = dist.log_prob(action)  # (B,)
        entropy = dist.entropy()          # (B,)
        value = self.value_head(x).squeeze(-1)  # (B,)

        return action, log_prob, entropy, value


@dataclass
class PPOConfig:
    exp_name: str = "ppo_pettingzoo_chess_vector"
    seed: int = 1
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    num_steps: int = 256        # rollout horizon
    num_envs: int = 8           # 병렬 환경 수
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    minibatch_size: int = 1024  # (num_steps * num_envs) 기준


# ==============================
#  벡터 환경 생성 (SyncVectorEnv 사용)
# ==============================

def make_single_env(seed_offset=0):
    def _init():
        env = ChessSelfPlayEnv(render_mode=None)
        env.reset(seed=seed_offset)
        return env
    return _init


def make_vector_env(num_envs):
    return AsyncVectorEnv(
        [make_single_env(seed_offset=i) for i in range(num_envs)]
    )


# ==============================
#  PPO 학습 루프
# ==============================

def train(config: PPOConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    env = make_vector_env(config.num_envs)
    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n
    num_envs = config.num_envs

    agent = ActorCritic(obs_shape, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # Rollout buffers: (T, E, ...)
    obs_buf = torch.zeros((config.num_steps, num_envs) + obs_shape,
                          dtype=torch.float32, device=device)
    masks_buf = torch.zeros((config.num_steps, num_envs, n_actions),
                            dtype=torch.float32, device=device)
    actions_buf = torch.zeros((config.num_steps, num_envs),
                              dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((config.num_steps, num_envs),
                               dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((config.num_steps, num_envs),
                              dtype=torch.float32, device=device)
    dones_buf = torch.zeros((config.num_steps, num_envs),
                            dtype=torch.float32, device=device)
    values_buf = torch.zeros((config.num_steps, num_envs),
                             dtype=torch.float32, device=device)
    advantages = torch.zeros((config.num_steps, num_envs),
                             dtype=torch.float32, device=device)

    # 에피소드 리턴 통계
    episode_returns = []
    current_ep_return = np.zeros(num_envs, dtype=np.float32)

    # Reset vector env
    next_obs_np, info = env.reset()
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)  # (E, *obs)
    next_mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device)  # (E, A)
    next_done = torch.zeros((num_envs,), dtype=torch.float32, device=device)

    global_step = 0
    # 한 update당 수집되는 transition 수: num_steps * num_envs
    num_updates = config.total_timesteps // (config.num_steps * num_envs)

    start_time = time.time()

    bar = tqdm(range(1, num_updates + 1), desc="PPO Training")
    for update in bar:
        # ==========================
        # Rollout 수집
        # ==========================
        for step in range(config.num_steps):
            global_step += num_envs  # env가 E개이므로

            obs_buf[step] = next_obs
            masks_buf[step] = next_mask
            dones_buf[step] = next_done

            with torch.no_grad():
                actions, logprobs, entropy, values = agent.get_action_and_value(
                    next_obs, action_mask=next_mask
                )
                # actions, logprobs, values: (E,)

            actions_buf[step] = actions
            logprobs_buf[step] = logprobs
            values_buf[step] = values

            # 벡터 env step
            actions_np = actions.cpu().numpy()
            obs_step, rewards, terminated, truncated, infos = env.step(actions_np)
            done = np.logical_or(terminated, truncated)  # (E,)

            rewards_buf[step] = torch.tensor(rewards, device=device, dtype=torch.float32)
            current_ep_return += rewards.astype(np.float32)

            # done인 env들 리턴만 기록 (reset은 AsyncVectorEnv가 자동으로 처리)
            if np.any(done):
                for i in range(num_envs):
                    if done[i]:
                        episode_returns.append(current_ep_return[i])
                        current_ep_return[i] = 0.0


            # 다음 상태 준비
            next_obs = torch.tensor(obs_step, dtype=torch.float32, device=device)
            next_mask = torch.tensor(infos["action_mask"], dtype=torch.float32, device=device)
            next_done = torch.tensor(done, dtype=torch.float32, device=device)

        # ==========================
        # GAE advantage 계산
        # ==========================
        with torch.no_grad():
            next_value = agent.get_value(next_obs)  # (E,)

        lastgaelam = torch.zeros(num_envs, dtype=torch.float32, device=device)
        for t in reversed(range(config.num_steps)):
            if t == config.num_steps - 1:
                next_nonterminal = 1.0 - next_done  # (E,)
                next_values = next_value            # (E,)
            else:
                next_nonterminal = 1.0 - dones_buf[t + 1]  # (E,)
                next_values = values_buf[t + 1]            # (E,)

            delta = rewards_buf[t] + config.gamma * next_values * next_nonterminal - values_buf[t]
            lastgaelam = delta + config.gamma * config.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values_buf  # (T, E)

        # ==========================
        # PPO 업데이트
        # ==========================
        # (T, E, ...) -> (T*E, ...)
        T, E = config.num_steps, num_envs
        batch_size = T * E

        b_obs = obs_buf.reshape(batch_size, -1)
        b_actions = actions_buf.reshape(batch_size)
        b_logprobs = logprobs_buf.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_values = values_buf.reshape(batch_size)
        b_masks = masks_buf.reshape(batch_size, n_actions)

        inds = np.arange(batch_size)

        # 손실 통계용 누적 변수
        total_loss = 0.0
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_minibatches = 0

        for epoch in range(config.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = inds[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_logprobs_old = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values_old = b_values[mb_inds]
                mb_masks = b_masks[mb_inds]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    mb_obs, mb_actions, action_mask=mb_masks
                )
                ratio = (newlogprob - mb_logprobs_old).exp()

                # Normalize advantage
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Policy loss (clipped surrogate)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_loss_unclipped = (value - mb_returns) ** 2
                v_clipped = mb_values_old + torch.clamp(
                    value - mb_values_old, -config.clip_coef, config.clip_coef
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

                # 통계 누적
                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy_loss.item()
                n_minibatches += 1

        # 한 update 기준 평균 손실
        mean_loss = total_loss / max(1, n_minibatches)
        mean_pg_loss = total_pg_loss / max(1, n_minibatches)
        mean_v_loss = total_v_loss / max(1, n_minibatches)
        mean_entropy = total_entropy / max(1, n_minibatches)

        # 최근 에피소드 리턴 (예: 마지막 20개 평균)
        if len(episode_returns) > 0:
            mean_ep_ret = float(np.mean(episode_returns[-20:]))
        else:
            mean_ep_ret = 0.0

        fps = int(global_step / (time.time() - start_time + 1e-8))

        # 🔹 tqdm bar에 정보 표시 (이전 스타일 유지)
        bar.set_postfix(
            {
                "ep_ret": f"{mean_ep_ret:.2f}",
                "loss": f"{mean_loss:.3f}",
                "pg": f"{mean_pg_loss:.3f}",
                "v": f"{mean_v_loss:.3f}",
                "ent": f"{mean_entropy:.3f}",
                "fps": fps,
            }
        )

        if update % 10 == 0:
            record_chess_video(
                agent=agent,
                device=device,
                video_dir="videos/chess",  # 원하는 폴더 이름
                update_idx=update,
                max_steps=300,             # 한 에피소드 최대 수 (원하면 늘리기)
            )


    env.close()
    # torch.save(agent.state_dict(), f"{config.exp_name}_final.pt")
    torch.save(agent.state_dict(), f"tictactoe_final.pt")
    print(f"[INFO] Training finished. Model saved to {config.exp_name}_final.pt")


# ==============================
#  main
# ==============================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train(cfg)
