import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import chess
from pettingzoo.classic import chess_v6

# [NEW] 공통 모듈 임포트
from other_model.chess_minimax import get_best_move_minimax, encode_move, get_best_move_stockfish

# ==============================
#  Helpers & Model
# ==============================

def select_action_minimax(env, obs_dict, use_stockfish=False):
    raw_env = env.unwrapped 
    if not hasattr(raw_env, 'board'):
        if hasattr(env, 'env'): raw_env = env.env
    board = raw_env.board
    
    # Eval할 때는 보통 조금 더 똑똑한 상대를 원할 수 있으므로 depth=2
    if use_stockfish:
        best_move = get_best_move_stockfish(board, time_limit=0.01)
    else:
        best_move = get_best_move_minimax(board, depth=2)
    
    if best_move is None:
        mask = obs_dict["action_mask"]
        return env.action_space(env.agent_selection).sample(mask)

    try:
        # 흑번이면 mirror=True
        action = encode_move(best_move, should_mirror=(board.turn == chess.BLACK))
    except:
        mask = obs_dict["action_mask"]
        return env.action_space(env.agent_selection).sample(mask)
    return action

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
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden).squeeze(-1)

def make_env():
    env = chess_v6.env(render_mode=None)
    env.reset()
    return env

def build_model(device, checkpoint_path: str):
    if checkpoint_path is None or checkpoint_path.lower() == "minimax" or checkpoint_path.lower() == "stockfish":
        return None
    
    tmp_env = make_env()
    obs_space = tmp_env.observation_space("player_0")["observation"]
    act_space = tmp_env.action_space("player_0")
    model = ActorCritic(obs_space.shape, act_space.n).to(device)
    tmp_env.close()

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

@torch.no_grad()
def select_action_model(model, obs_dict, device):
    # .copy() 필수
    obs_np = obs_dict["observation"].copy()
    mask_np = obs_dict["action_mask"].copy()
    
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0)
    
    action, _, _, _ = model.get_action_and_value(obs, action_mask=mask)
    return int(action.item())

def select_action_random(env, agent_name, obs_dict):
    return env.action_space(agent_name).sample(obs_dict["action_mask"])

# ==============================
#  Main (DDP)
# ==============================

def play_game(game_idx, agent_model, opponent_model, opponent_type, device):
    env = make_env()
    
    if game_idx % 2 == 0:
        model_p0, type_p0 = agent_model, "model"
        model_p1, type_p1 = opponent_model, opponent_type
        agent_as_p0 = True
    else:
        model_p0, type_p0 = opponent_model, opponent_type
        model_p1, type_p1 = agent_model, "model"
        agent_as_p0 = False

        
    rewards = {agent: 0.0 for agent in env.agents}
    for agent in env.agent_iter():
        obs, reward, termination, truncation, _ = env.last()
        rewards[agent] += reward
        if termination or truncation:
            action = None
        else:
            if agent == "player_0":
                cur_model, cur_type = model_p0, type_p0
            else:
                cur_model, cur_type = model_p1, type_p1
                
            if cur_type == "minimax" :
                action = select_action_minimax(env, obs)
            elif cur_type == "stockfish":
                action = select_action_minimax(env, obs, use_stockfish=True)
            elif cur_type == "model" and cur_model:
                action = select_action_model(cur_model, obs, device)
            else:
                action = select_action_random(env, agent, obs)
        env.step(action)

    r0 = rewards["player_0"]
    env.close()

    if r0 > 0: return 0 if agent_as_p0 else 1
    elif r0 < 0: return 1 if agent_as_p0 else 0
    else: return 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-checkpoint", type=str, default="ppo_pettingzoo_chess_vector_final.pt")
    parser.add_argument("--opponent-checkpoint", type=str, default="minimax")
    parser.add_argument("--n-games", type=int, default=20)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        device_id = local_rank % num_gpus
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        print(f"[INFO] Running with torchrun. World size: {world_size}")
        print(f"[INFO] Rank 0 is using device: {device}")

    agent_model = build_model(device, args.agent_checkpoint)
    
    opponent_types = ("minimax", "stockfish")
    opponent_model = None
    opponent_type = None
    if args.opponent_checkpoint not in opponent_types and args.opponent_checkpoint is not None:
        opponent_type = "model"
        opponent_model = build_model(device, args.opponent_checkpoint)
    else:
        opponent_type = args.opponent_checkpoint

    my_games = [i for i in range(args.n_games) if i % world_size == rank]
    local_stats = torch.zeros(3, dtype=torch.int, device=device) 

    for idx in my_games:
        res = play_game(idx, agent_model, opponent_model, opponent_type, device)
        local_stats[res] += 1

    dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

    if rank == 0:
        a_wins = local_stats[0].item()
        b_wins = local_stats[1].item()
        draws = local_stats[2].item()
        total = a_wins + b_wins + draws
        
        print("\n================ Evaluation Result ================")
        print(f"Total Games   : {total}")
        print(f"Agent wins    : {a_wins}")
        print(f"Opponent wins : {b_wins}")
        print(f"Draws         : {draws}")
        if total > 0:
            print(f"Agent Win Rate: {a_wins/total:.3f}")
        print("==================================================")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()