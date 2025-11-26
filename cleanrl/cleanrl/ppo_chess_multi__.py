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
from pettingzoo.classic import chess_v6

import os
import wandb
import time


def record_chess_video(agent, device, video_dir, update_idx, max_steps=300):
    """
    현재 agent policy로 한 판(or max_steps) 두면서 video를 저장.
    video_dir/update_{update_idx}.* 로 파일 생성됨.
    """
    os.makedirs(video_dir, exist_ok=True)

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
    mask_t = torch.tensor(
        info["action_mask"], dtype=torch.float32, device=device
    ).unsqueeze(0)

    done = False
    step = 0

    while not done and step < max_steps:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t, action_mask=mask_t)
        action_id = action.item()

        obs, reward, terminated, truncated, info = eval_env.step(action_id)
        done = terminated or truncated

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(
            info["action_mask"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        step += 1

    eval_env.close()
    agent.train()  # 다시 train 모드


# ==============================
#  ChessSelfPlayEnv (PettingZoo -> Gymnasium 단일 env)
#  한 쪽은 PPO, 다른 쪽은 항상 랜덤으로 두는 버전
# ==============================
import chess


class ChessSelfPlayEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "name": "ChessSelfPlayEnv",
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.aec_env = chess_v6.env(render_mode=render_mode)
        self.aec_env.reset()

        agents = self.aec_env.agents
        self.agent_id = agents[0]  # PPO (White라고 가정)
        self.opponent_id = agents[1]  # Random (Black)

        # Observation / Action Space 설정 (기존과 동일)
        raw_obs_space = self.aec_env.observation_space(self.agent_id)["observation"]
        self.observation_space = spaces.Box(
            low=raw_obs_space.low,
            high=raw_obs_space.high,
            shape=raw_obs_space.shape,
            dtype=np.float32,
        )
        self.action_space = self.aec_env.action_space(self.agent_id)

        # [Reward Shaping] 이전 턴의 기물 점수 차이를 저장할 변수
        self.last_board_value = 0.0

    def _get_board_value(self):
        """
        현재 보드 상태에서 (내 기물 점수 - 상대 기물 점수)를 계산하여 반환.
        PettingZoo의 내부 chess.Board 객체에 접근해야 합니다.
        """
        # PettingZoo chess_v6는 내부적으로 python-chess의 board 객체를 가지고 있습니다.
        # 구조: aec_env -> env -> env -> board (접근 경로가 버전에 따라 다를 수 있어 unwrapped 사용)
        board = self.aec_env.unwrapped.board

        # 기물 점수 매핑 (일반적인 체스 점수)
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,  # 킹은 잡히지 않으므로 0 (혹은 체크메이트 보상으로 대체됨)
        }

        value = 0.0
        # board.piece_map()은 {square_index: Piece객체} 딕셔너리 반환
        for piece in board.piece_map().values():
            score = piece_values.get(piece.piece_type, 0)

            # 내 기물이면 +, 상대 기물이면 -
            # self.agent_id가 "player_0" (White)라고 가정
            if self.agent_id == "player_0":
                if piece.color == chess.WHITE:
                    value += score
                else:
                    value -= score
            else:  # 내가 Black인 경우
                if piece.color == chess.BLACK:
                    value += score
                else:
                    value -= score

        return value

    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed)

        # [Reward Shaping] 초기 보드 점수 (보통 0)
        self.last_board_value = 0.0

        # 상대 턴이면 랜덤으로 진행 (기존 코드와 동일)
        while self.aec_env.agent_selection != self.agent_id:
            curr = self.aec_env.agent_selection
            obs_opp = self.aec_env.observe(curr)
            mask_opp = obs_opp["action_mask"]
            action_opp = self.aec_env.action_space(curr).sample(mask_opp)
            self.aec_env.step(int(action_opp))
            if all(self.aec_env.terminations.values()) or all(
                self.aec_env.truncations.values()
            ):
                break

        # Reset 직후 내 기물 점수 계산 (초기화)
        self.last_board_value = self._get_board_value()

        # 관측 반환 (기존 코드와 동일)
        if not (
            all(self.aec_env.terminations.values())
            or all(self.aec_env.truncations.values())
        ):
            raw_obs = self.aec_env.observe(self.agent_id)
            obs = raw_obs["observation"].astype(np.float32)
            action_mask = raw_obs["action_mask"].astype(np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            action_mask = np.ones(self.action_space.n, dtype=np.float32)

        return obs, {"action_mask": action_mask}

    def step(self, action):
        # 1. 우리 에이전트 수 두기
        self.aec_env.step(int(action))

        terminated = all(self.aec_env.terminations.values())
        truncated = all(self.aec_env.truncations.values())
        done = terminated or truncated

        # 2. 상대(랜덤) 수 두기 (게임 안 끝났으면)
        if not done:
            obs_opp = self.aec_env.observe(self.opponent_id)
            mask_opp = obs_opp["action_mask"]
            action_opp = self.aec_env.action_space(self.opponent_id).sample(mask_opp)
            self.aec_env.step(int(action_opp))

            terminated = all(self.aec_env.terminations.values())
            truncated = all(self.aec_env.truncations.values())
            done = terminated or truncated

        # 3. 보상 계산 (기존 승패 보상 + Shaping 보상)
        original_reward = float(self.aec_env.rewards[self.agent_id])

        # [Reward Shaping] 현재 보드 점수 계산
        current_board_value = self._get_board_value()

        # 점수 변화량 (내 점수가 늘거나, 상대 점수가 줄면 이득)
        # 예: 내가 상대 폰을 잡음 -> (내점수 - (상대점수-1)) - (내점수 - 상대점수) = +1
        shaping_reward = current_board_value - self.last_board_value

        # [중요] Shaping 계수 조절 (Coefficient)
        # 기물 점수 1점이 승리(1.0)보다 크면 안 되므로, 적절히 줄여줍니다.
        # 예: 폰 하나 잡는 것 = 0.02점 (폰 50개 잡아야 승리 1점과 맞먹음 -> 승리가 더 중요함을 유지)
        shaping_coeff = 0.02

        total_reward = original_reward + (shaping_reward * shaping_coeff)

        # 상태 업데이트
        self.last_board_value = current_board_value

        # 4. 다음 관측 (기존 코드와 동일)
        if not done:
            raw_obs = self.aec_env.observe(self.agent_id)
            obs = raw_obs["observation"].astype(np.float32)
            action_mask = raw_obs["action_mask"].astype(np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            action_mask = np.ones(self.action_space.n, dtype=np.float32)

        return obs, total_reward, terminated, truncated, {"action_mask": action_mask}

    def render(self):
        # PettingZoo 환경의 render를 그대로 호출
        return self.aec_env.render()

    def close(self):
        self.aec_env.close()


# ==============================
#  PPO 네트워크 & 하이퍼파라미터
# ==============================


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
        obs = obs.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        hidden = self.network(obs)
        return self.critic(hidden).squeeze(-1)  # (B,)

    def get_action_and_value(self, obs, action=None, action_mask=None):
        obs = obs.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        hidden = self.network(obs)
        logits = self.actor(hidden)  # (B, n_actions)

        if action_mask is not None:
            # illegal move의 logit을 매우 작은 값으로 마스킹 → 선택 불가
            logits = logits.masked_fill(action_mask == 0, -1e9)

        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()  # (B,)

        log_prob = dist.log_prob(action)  # (B,)
        entropy = dist.entropy()  # (B,)
        value = self.critic(hidden).squeeze(-1)  # (B,)

        return action, log_prob, entropy, value


@dataclass
class PPOConfig:
    exp_name: str = "ppo_pettingzoo_chess_vector"
    seed: int = 1
    total_timesteps: int = 5_000_000
    learning_rate: float = 2.5e-4
    num_steps: int = 512  # rollout horizon
    num_envs: int = 32  # 병렬 환경 수
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    minibatch_size: int = 2048  # (num_steps * num_envs) 기준
    logging: bool = True


# ==============================
#  벡터 환경 생성 (AsyncVectorEnv 사용)
# ==============================


def make_single_env(seed_offset=0):
    def _init():
        env = ChessSelfPlayEnv(render_mode=None)
        env.reset(seed=seed_offset)
        return env

    return _init


def make_vector_env(num_envs):
    return AsyncVectorEnv([make_single_env(seed_offset=i) for i in range(num_envs)])


# ==============================
#  PPO 학습 루프
# ==============================


def train(config: PPOConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if config.logging:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="edenkim9741-chonnam-national-university",
            # Set the wandb project where this run will be logged.
            project="Reinforcement-Learning",
            name=time.strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_chess_selfplay",
            # Track hyperparameters and run metadata.
            config={
                "architecture": "CNN",
                "dataset": "Chess",
            },
        )

    env = make_vector_env(config.num_envs)
    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n
    num_envs = config.num_envs

    agent = ActorCritic(obs_shape, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # Rollout buffers: (T, E, ...)
    obs_buf = torch.zeros(
        (config.num_steps, num_envs) + obs_shape, dtype=torch.float32, device=device
    )
    masks_buf = torch.zeros(
        (config.num_steps, num_envs, n_actions), dtype=torch.float32, device=device
    )
    actions_buf = torch.zeros(
        (config.num_steps, num_envs), dtype=torch.long, device=device
    )
    logprobs_buf = torch.zeros(
        (config.num_steps, num_envs), dtype=torch.float32, device=device
    )
    rewards_buf = torch.zeros(
        (config.num_steps, num_envs), dtype=torch.float32, device=device
    )
    dones_buf = torch.zeros(
        (config.num_steps, num_envs), dtype=torch.float32, device=device
    )
    values_buf = torch.zeros(
        (config.num_steps, num_envs), dtype=torch.float32, device=device
    )
    advantages = torch.zeros(
        (config.num_steps, num_envs), dtype=torch.float32, device=device
    )

    # 에피소드 리턴 통계
    episode_returns = []
    current_ep_return = np.zeros(num_envs, dtype=np.float32)

    # Reset vector env
    next_obs_np, info = env.reset()
    next_obs = torch.tensor(
        next_obs_np, dtype=torch.float32, device=device
    )  # (E, *obs)
    next_mask = torch.tensor(
        info["action_mask"], dtype=torch.float32, device=device
    )  # (E, A)
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

            rewards_buf[step] = torch.tensor(
                rewards, device=device, dtype=torch.float32
            )
            current_ep_return += rewards.astype(np.float32)
            if config.logging:
                run.log({"train/step_reward": np.mean(rewards), "global_step": global_step})

            # done인 env들 리턴만 기록 (AsyncVectorEnv는 auto-reset로 새 episode 시작)
            if np.any(done):
                for i in range(num_envs):
                    if done[i]:
                        episode_returns.append(current_ep_return[i])
                        current_ep_return[i] = 0.0

            # 다음 상태 준비
            next_obs = torch.tensor(obs_step, dtype=torch.float32, device=device)
            next_mask = torch.tensor(
                infos["action_mask"], dtype=torch.float32, device=device
            )
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
                next_values = next_value  # (E,)
            else:
                next_nonterminal = 1.0 - dones_buf[t + 1]  # (E,)
                next_values = values_buf[t + 1]  # (E,)

            delta = (
                rewards_buf[t]
                + config.gamma * next_values * next_nonterminal
                - values_buf[t]
            )
            lastgaelam = (
                delta + config.gamma * config.gae_lambda * next_nonterminal * lastgaelam
            )
            advantages[t] = lastgaelam

        returns = advantages + values_buf  # (T, E)

        # ==========================
        # PPO 업데이트
        # ==========================
        # (T, E, ...) -> (T*E, ...)
        T, E = config.num_steps, num_envs
        batch_size = T * E

        b_obs = obs_buf.reshape((-1,) + obs_shape)
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
                loss = (
                    pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
                )

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

        # 🔹 10 update마다 비디오 저장 (우리 vs 랜덤)
        if update % 100 == 0:
            record_chess_video(
                agent=agent,
                device=device,
                video_dir="videos/chess",
                update_idx=update,
                max_steps=300,
            )

    env.close()
    torch.save(agent.state_dict(), f"{config.exp_name}_final.pt")
    print(f"[INFO] Training finished. Model saved to {config.exp_name}_final.pt")


# ==============================
#  main
# ==============================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=40_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--logging", type=bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        logging=args.logging,)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train(cfg)
