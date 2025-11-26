import argparse
import time
from dataclasses import dataclass
import os
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

# [NEW] 공통 모듈 임포트
from minimax.chess_minimax import get_best_move_minimax, encode_move, PIECE_VALUES

# ==============================
#  ChessSelfPlayEnv
# ==============================

class ChessSelfPlayEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "name": "ChessSelfPlayEnv",
    }

    def __init__(self, render_mode=None, opponent_depth=1):
        super().__init__()
        self.render_mode = render_mode
        self.aec_env = chess_v6.env(render_mode=render_mode)
        self.aec_env.reset()
        
        self.opponent_depth = opponent_depth
        agents = self.aec_env.agents
        self.agent_id = agents[0]  # PPO (White)
        self.opponent_id = agents[1]  # Minimax (Black)

        raw_obs_space = self.aec_env.observation_space(self.agent_id)["observation"]
        self.observation_space = spaces.Box(
            low=raw_obs_space.low,
            high=raw_obs_space.high,
            shape=raw_obs_space.shape,
            dtype=np.float32,
        )
        self.action_space = self.aec_env.action_space(self.agent_id)
        self.last_board_value = 0.0

    def _get_board_value(self):
        board = self.aec_env.unwrapped.board
        value = 0.0
        for piece in board.piece_map().values():
            score = PIECE_VALUES.get(piece.piece_type, 0)
            if self.agent_id == "player_0": # White
                if piece.color == chess.WHITE: value += score
                else: value -= score
            else:
                if piece.color == chess.BLACK: value += score
                else: value -= score
        return value

    def _opponent_step(self):
        raw_env = self.aec_env.unwrapped
        if not hasattr(raw_env, 'board') and hasattr(self.aec_env, 'env'):
             raw_env = self.aec_env.env
        board = raw_env.board
        
        # [NEW] 모듈 함수 사용
        best_move = get_best_move_minimax(board, depth=self.opponent_depth)
        
        if best_move is None:
            curr = self.aec_env.agent_selection
            mask = self.aec_env.observe(curr)["action_mask"]
            return self.aec_env.action_space(curr).sample(mask)

        try:
            # [NEW] 모듈 함수 사용 (상대는 Black이므로 should_mirror=True)
            action_opp = encode_move(best_move, should_mirror=True)
        except:
            curr = self.aec_env.agent_selection
            mask = self.aec_env.observe(curr)["action_mask"]
            action_opp = self.aec_env.action_space(curr).sample(mask)
            
        return action_opp

    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed)
        self.last_board_value = 0.0

        while self.aec_env.agent_selection != self.agent_id:
            action_opp = self._opponent_step()
            self.aec_env.step(int(action_opp))
            if all(self.aec_env.terminations.values()) or all(self.aec_env.truncations.values()):
                break

        self.last_board_value = self._get_board_value()

        if not (all(self.aec_env.terminations.values()) or all(self.aec_env.truncations.values())):
            raw_obs = self.aec_env.observe(self.agent_id)
            # .copy() 필수
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
            # .copy() 필수
            obs = raw_obs["observation"].copy().astype(np.float32)
            action_mask = raw_obs["action_mask"].copy().astype(np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            action_mask = np.ones(self.action_space.n, dtype=np.float32)

        return obs, total_reward, terminated, truncated, {"action_mask": action_mask}

    def render(self):
        return self.aec_env.render()
    
    def close(self):
        self.aec_env.close()

def record_chess_video(agent, device, video_dir, update_idx, max_steps=300):
    os.makedirs(video_dir, exist_ok=True)
    name_prefix = f"update_{update_idx}"

    eval_env = ChessSelfPlayEnv(render_mode="rgb_array", opponent_depth=1)
    eval_env = RecordVideo(
        eval_env,
        video_folder=video_dir,
        episode_trigger=lambda ep_id: True,
        name_prefix=name_prefix,
    )

    agent.eval()
    obs, info = eval_env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)

    done = False
    step = 0
    while not done and step < max_steps:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t, action_mask=mask_t)
        action_id = action.item()
        
        obs, reward, terminated, truncated, info = eval_env.step(action_id)
        done = terminated or truncated

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        step += 1

    eval_env.close()
    agent.train()

# ==============================
#  PPO Network & Train
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

@dataclass
class PPOConfig:
    exp_name: str = "ppo_chess_vs_minimax"
    seed: int = 1
    total_timesteps: int = 5_000_000
    learning_rate: float = 2.5e-4
    num_steps: int = 512
    num_envs: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    minibatch_size: int = 2048
    logging: bool = True
    opponent_depth: int = 1
    resume: str = None

def make_single_env(seed_offset=0, opponent_depth=1):
    def _init():
        env = ChessSelfPlayEnv(render_mode=None, opponent_depth=opponent_depth)
        env.reset(seed=seed_offset)
        return env
    return _init

def make_vector_env(num_envs, opponent_depth=1):
    return AsyncVectorEnv([
        make_single_env(seed_offset=i, opponent_depth=opponent_depth) 
        for i in range(num_envs)
    ])

def train(config: PPOConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    if config.logging:
        run = wandb.init(
            entity="edenkim9741-chonnam-national-university",
            project="Reinforcement-Learning",
            name=time.strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_chess_vs_minimax",
            config=config.__dict__,
        )

    env = make_vector_env(config.num_envs, opponent_depth=config.opponent_depth)
    
    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n
    num_envs = config.num_envs

    agent = ActorCritic(obs_shape, n_actions).to(device)
    if config.resume is not None:
        agent.load_state_dict(torch.load(config.resume, map_location=device))
        print(f"[INFO] Loaded model from {config.resume}")

    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    obs_buf = torch.zeros((config.num_steps, num_envs) + obs_shape, dtype=torch.float32, device=device)
    masks_buf = torch.zeros((config.num_steps, num_envs, n_actions), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((config.num_steps, num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((config.num_steps, num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((config.num_steps, num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((config.num_steps, num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((config.num_steps, num_envs), dtype=torch.float32, device=device)
    advantages = torch.zeros((config.num_steps, num_envs), dtype=torch.float32, device=device)

    episode_returns = []
    current_ep_return = np.zeros(num_envs, dtype=np.float32)

    next_obs_np, info = env.reset()
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device)
    next_done = torch.zeros((num_envs,), dtype=torch.float32, device=device)

    global_step = 0
    num_updates = config.total_timesteps // (config.num_steps * num_envs)
    start_time = time.time()

    bar = tqdm(range(1, num_updates + 1), desc="PPO Training")
    for update in bar:
        for step in range(config.num_steps):
            global_step += num_envs

            obs_buf[step] = next_obs
            masks_buf[step] = next_mask
            dones_buf[step] = next_done

            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(next_obs, action_mask=next_mask)

            actions_buf[step] = actions
            logprobs_buf[step] = logprobs
            values_buf[step] = values

            actions_np = actions.cpu().numpy()
            obs_step, rewards, terminated, truncated, infos = env.step(actions_np)
            done = np.logical_or(terminated, truncated)

            rewards_buf[step] = torch.tensor(rewards, device=device, dtype=torch.float32)
            current_ep_return += rewards
            if config.logging:
                run.log({"train/step_reward": np.mean(rewards), "global_step": global_step})

            if np.any(done):
                for i in range(num_envs):
                    if done[i]:
                        episode_returns.append(current_ep_return[i])
                        current_ep_return[i] = 0.0

            next_obs = torch.tensor(obs_step, dtype=torch.float32, device=device)
            next_mask = torch.tensor(infos["action_mask"], dtype=torch.float32, device=device)
            next_done = torch.tensor(done, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            
        lastgaelam = torch.zeros(num_envs, dtype=torch.float32, device=device)
        for t in reversed(range(config.num_steps)):
            if t == config.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]
            delta = rewards_buf[t] + config.gamma * nextvalues * nextnonterminal - values_buf[t]
            lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_masks = masks_buf.reshape((-1, n_actions))

        b_inds = np.arange(config.num_steps * num_envs)
        
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds], action_mask=b_masks[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -config.clip_coef, config.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

        mean_ep_ret = np.mean(episode_returns[-20:]) if len(episode_returns) > 0 else 0.0
        fps = int(global_step / (time.time() - start_time + 1e-8))
        bar.set_postfix({"ep_ret": f"{mean_ep_ret:.2f}", "fps": fps})

        if update % 100 == 0:
            record_chess_video(agent, device, "videos/chess_minimax", update, max_steps=300)

    env.close()
    torch.save(agent.state_dict(), f"{config.exp_name}_final.pt")
    if config.logging: run.finish()
    print("[INFO] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--opponent-depth", type=int, default=2)
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        opponent_depth=args.opponent_depth,
        logging=args.logging,
        resume=args.resume,
    )
    
    train(cfg)