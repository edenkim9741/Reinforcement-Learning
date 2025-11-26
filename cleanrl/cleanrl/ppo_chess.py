import argparse
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pettingzoo.classic import chess_v6


# ==============================
#  ChessSelfPlayEnv (PettingZoo -> Gymnasium)
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
        self.aec_env = chess_v6.env(render_mode=render_mode)
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
            # 게임 끝: dummy obs & mask (어차피 곧 reset될 것)
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
        return self.value_head(x).squeeze(-1)

    def get_action_and_value(self, obs, action=None, action_mask=None):
        """
        obs: (B, *obs_shape)
        action_mask: (B, n_actions) - 1.0 for legal, 0.0 for illegal
        """
        x = self.encoder(obs)
        logits = self.policy_head(x)

        if action_mask is not None:
            # illegal move(logit) 을 -1e9로 마스킹 → 확률 ~0
            logits = logits.masked_fill(action_mask == 0, -1e9)

        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(x).squeeze(-1)

        return action, log_prob, entropy, value


@dataclass
class PPOConfig:
    exp_name: str = "ppo_pettingzoo_chess_masked"
    seed: int = 1
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    num_steps: int = 1024  # rollout horizon
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    minibatch_size: int = 256


# ==============================
#  PPO 학습 루프
# ==============================

def make_env():
    return ChessSelfPlayEnv(render_mode=None)


def train(config: PPOConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    env = make_env()
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = ActorCritic(obs_shape, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # Rollout buffers
    obs_buf = torch.zeros((config.num_steps,) + obs_shape, dtype=torch.float32, device=device)
    actions_buf = torch.zeros((config.num_steps,), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((config.num_steps,), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((config.num_steps,), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((config.num_steps,), dtype=torch.float32, device=device)
    values_buf = torch.zeros((config.num_steps,), dtype=torch.float32, device=device)
    masks_buf = torch.zeros((config.num_steps, n_actions), dtype=torch.float32, device=device)

    advantages = torch.zeros((config.num_steps,), dtype=torch.float32, device=device)

    # Reset env
    next_obs_np, info = env.reset()
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device)
    next_done = torch.zeros((), dtype=torch.float32, device=device)

    global_step = 0
    num_updates = config.total_timesteps // config.num_steps
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # ==========================
        # Rollout 수집
        # ==========================
        for step in range(config.num_steps):
            global_step += 1

            obs_buf[step] = next_obs
            masks_buf[step] = next_mask
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(
                    next_obs.unsqueeze(0),
                    action_mask=next_mask.unsqueeze(0),
                )
                action = action.squeeze(0)
                logprob = logprob.squeeze(0)
                value = value.squeeze(0)

            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value

            # Env step
            next_obs_np, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            rewards_buf[step] = reward

            # 다음 상태 준비
            if done:
                # episode 종료 시 바로 reset
                next_obs_np, info = env.reset()
                next_done = torch.zeros((), dtype=torch.float32, device=device)
            else:
                next_done = torch.tensor(float(done), dtype=torch.float32, device=device)

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device)

        # ==========================
        # GAE advantage 계산
        # ==========================
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).squeeze(0)

        lastgaelam = 0
        for t in reversed(range(config.num_steps)):
            if t == config.num_steps - 1:
                next_nonterminal = 1.0 - next_done
                next_values = next_value
            else:
                next_nonterminal = 1.0 - dones_buf[t + 1]
                next_values = values_buf[t + 1]

            delta = rewards_buf[t] + config.gamma * next_values * next_nonterminal - values_buf[t]
            lastgaelam = delta + config.gamma * config.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values_buf

        # ==========================
        # PPO 업데이트
        # ==========================
        b_obs = obs_buf  # (T, *obs_shape)
        b_actions = actions_buf
        b_logprobs = logprobs_buf
        b_advantages = advantages
        b_returns = returns
        b_values = values_buf
        b_masks = masks_buf  # (T, n_actions)

        batch_size = config.num_steps
        inds = np.arange(batch_size)

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

        fps = int(global_step / (time.time() - start_time + 1e-8))
        print(
            f"Update {update}/{num_updates} | "
            f"global_step={global_step} | "
            f"fps={fps}"
        )

    env.close()
    torch.save(agent.state_dict(), f"{config.exp_name}_final.pt")
    print(f"[INFO] Training finished. Model saved to {config.exp_name}_final.pt")


# ==============================
#  main
# ==============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train(cfg)
