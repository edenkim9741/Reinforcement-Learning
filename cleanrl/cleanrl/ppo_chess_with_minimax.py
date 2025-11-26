import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from pettingzoo.classic import chess_v6


# ==============================
#  ActorCritic (학습 때 쓰던 거랑 동일 구조)
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
        entropy = dist.entropy()          # (B,)
        value = self.critic(hidden).squeeze(-1)  # (B,)

        return action, log_prob, entropy, value


# ==============================
#  유틸 함수들
# ==============================

def make_env(render_mode=None):
    env = chess_v6.env(render_mode=render_mode)
    env.reset()
    return env


def build_model(device, checkpoint_path: Optional[str]) -> Optional[ActorCritic]:
    """
    checkpoint_path가 주어지면 ActorCritic을 만들어서 weight 로드,
    안 주어지면 None 반환 (→ 랜덤 정책으로 사용)
    """
    if checkpoint_path is None:
        return None

    # env 하나 열어서 observation / action space 파악
    tmp_env = make_env(render_mode=None)
    obs_space = tmp_env.observation_space("player_0")["observation"]
    act_space = tmp_env.action_space("player_0")
    obs_shape = obs_space.shape
    n_actions = act_space.n
    tmp_env.close()

    model = ActorCritic(obs_shape, n_actions).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def select_action_model(model: ActorCritic, obs_dict, device):
    """
    ActorCritic 모델로부터 행동 하나 샘플
    obs_dict: {"observation": ..., "action_mask": ...}
    """
    obs_np = obs_dict["observation"].astype(np.float32)
    mask_np = obs_dict["action_mask"].astype(np.float32)

    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0)

    action_t, _, _, _ = model.get_action_and_value(obs_t, action_mask=mask_t)
    return int(action_t.item())


def select_action_random(env, agent_name: str, obs_dict):
    """
    PettingZoo action_space의 mask 샘플 기능 사용 (합법 수 중 랜덤).
    """
    mask = obs_dict["action_mask"]
    return env.action_space(agent_name).sample(mask)


def play_one_game(env, device, model_p0: Optional[ActorCritic],
                  model_p1: Optional[ActorCritic],
                  render: bool = False):
    """
    env: chess_v6 AEC env
    model_p0: player_0에 사용할 모델 (None이면 랜덤)
    model_p1: player_1에 사용할 모델 (None이면 랜덤)

    return: winner_name: "player_0" / "player_1" / None (무승부)
    """
    env.reset()
    rewards = {agent: 0.0 for agent in env.agents}

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        rewards[agent] += reward

        if termination or truncation:
            action = None
        else:
            if agent == "player_0":
                if model_p0 is None:
                    action = select_action_random(env, agent, obs)
                else:
                    action = select_action_model(model_p0, obs, device)
            else:  # player_1
                if model_p1 is None:
                    action = select_action_random(env, agent, obs)
                else:
                    action = select_action_model(model_p1, obs, device)

        env.step(action)

        if render:
            env.render()

    # 결과 판정: chess_v6는 set_game_result에서
    # player_0 reward = result_val * 1, player_1 reward = result_val * -1
    r0 = rewards["player_0"]
    if r0 > 0:
        return "player_0"
    elif r0 < 0:
        return "player_1"
    else:
        return None  # draw


# ==============================
#  메인: PPO vs 상대 모델 비교
# ==============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-checkpoint",
        type=str,
        default="ppo_pettingzoo_chess_vector_final.pt",
        help="우리 PPO 에이전트 checkpoint 경로",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=str,
        default=None,
        help="비교할 opponent checkpoint 경로 (없으면 랜덤 정책)",
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=20,
        help="대국 횟수 (절반은 우리가 white, 절반은 black)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="대국 진행 상황을 렌더링(창 띄우기)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 모델 로드
    agent_model = build_model(device, args.agent_checkpoint)
    if agent_model is None:
        raise ValueError("agent-checkpoint가 필요합니다.")

    opponent_model = build_model(device, args.opponent_checkpoint)
    if opponent_model is None:
        print("[INFO] Opponent: RANDOM policy (legal moves only).")
    else:
        print(f"[INFO] Opponent model loaded from: {args.opponent_checkpoint}")

    # 평가용 env (렌더 여부는 arg에 따라)
    render_mode = "human" if args.render else None
    env = make_env(render_mode=render_mode)

    a_wins = 0
    b_wins = 0
    draws = 0

    # n_games 번 플레이, 짝수/홀수 게임마다 색 바꿔줌
    for game_idx in range(args.n_games):
        # 짝수 게임: agent = player_0(white), opponent = player_1(black)
        # 홀수 게임: agent = player_1(black), opponent = player_0(white)
        if game_idx % 2 == 0:
            model_p0 = agent_model
            model_p1 = opponent_model
            agent_as_p0 = True
            color_str = "agent=WHITE, opponent=BLACK"
        else:
            model_p0 = opponent_model
            model_p1 = agent_model
            agent_as_p0 = False
            color_str = "agent=BLACK, opponent=WHITE"

        print(f"\n[GAME {game_idx+1}/{args.n_games}] {color_str}")

        winner = play_one_game(
            env,
            device=device,
            model_p0=model_p0,
            model_p1=model_p1,
            render=args.render,
        )

        if winner is None:
            print("  -> Draw")
            draws += 1
        else:
            print(f"  -> Winner: {winner}")
            if winner == "player_0":
                if agent_as_p0:
                    a_wins += 1
                    print("     (agent win)")
                else:
                    b_wins += 1
                    print("     (opponent win)")
            else:  # player_1
                if agent_as_p0:
                    b_wins += 1
                    print("     (opponent win)")
                else:
                    a_wins += 1
                    print("     (agent win)")

    env.close()

    print("\n================ Evaluation Result ================")
    print(f"Agent wins    : {a_wins}")
    print(f"Opponent wins : {b_wins}")
    print(f"Draws         : {draws}")
    total = a_wins + b_wins + draws
    if total > 0:
        print(f"Agent win rate vs opponent: {a_wins / total:.3f}")
    print("==================================================")


if __name__ == "__main__":
    main()
