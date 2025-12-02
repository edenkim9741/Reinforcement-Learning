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
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
import wandb
import chess
import chess.engine
from pettingzoo.classic import chess_v6
import cv2

# ==========================================
# 0. Configuration (개선됨)
# ==========================================
from dataclasses import dataclass

# =============================================================================
#  [Helper Class] Stockfish Environment
# =============================================================================
class StockfishEvalEnv(gym.Env):
    def __init__(self, stockfish_path, skill_level=0, time_limit=0.05, render_mode=None):
        super().__init__()
        self.aec_env = chess_v6.env(render_mode=render_mode)
        self.time_limit = time_limit
        self.aec_env.reset()
        
        # Stockfish 엔진 인스턴스 생성
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.engine.configure({"Skill Level": skill_level})
        except FileNotFoundError:
            raise FileNotFoundError(f"Stockfish path incorrect: {stockfish_path}")

        raw_obs_space = self.aec_env.observation_space("player_0")["observation"]
        self.observation_space = spaces.Box(low=0, high=1, shape=raw_obs_space.shape, dtype=np.float32)
        self.action_space = self.aec_env.action_space("player_0")

    def render(self):
        return self.aec_env.render()

    def play_match(self, agent, device, play_as_white=True, record_video=False):
        self.aec_env.reset()
        agent_player = "player_0" if play_as_white else "player_1"
        frames = [] 

        for agent_selection in self.aec_env.agent_iter():
            observation, reward, termination, truncation, info = self.aec_env.last()
            
            if record_video:
                frame = self.render()
                if frame is not None: frames.append(frame)

            if termination or truncation:
                rewards = self.aec_env.rewards
                agent_score = rewards[agent_player]
                action = None
                self.aec_env.step(action)
                
                result = 0
                if agent_score > 0: result = 1    # Win
                elif agent_score < 0: result = -1 # Loss
                
                return result, frames

            if agent_selection == agent_player:
                # === Agent Turn ===
                obs_data = observation["observation"].copy() 
                mask_data = observation["action_mask"].copy()
                
                obs_tensor = torch.tensor(obs_data, dtype=torch.float32, device=device).unsqueeze(0)
                mask_tensor = torch.tensor(mask_data, dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    # 병렬 워커에서는 DDP나 module이 없을 수 있으므로 체크
                    if hasattr(agent, 'module'):
                        action_idx, _, _, _ = agent.module.forward(obs_tensor, action_mask=mask_tensor)
                    else:
                        action_idx, _, _, _ = agent.forward(obs_tensor, action_mask=mask_tensor)
                
                action = action_idx.item()
            else:
                # === Stockfish Turn ===
                board = self.aec_env.unwrapped.board
                limit = chess.engine.Limit(time=self.time_limit)
                result_engine = self.engine.play(board, limit)
                
                if result_engine.move is None:
                    action = self.aec_env.action_space(agent_selection).sample(observation["action_mask"])
                else:
                    is_black = (agent_selection == "player_1")
                    try:
                        # encode_move 함수가 scope 내에 있어야 함
                        from other_model.chess_minimax import encode_move
                        action = encode_move(result_engine.move, should_mirror=is_black)
                    except:
                        action = self.aec_env.action_space(agent_selection).sample(observation["action_mask"])

            self.aec_env.step(action)

        return 0, frames

    def close(self):
        if hasattr(self, 'engine'):
            self.engine.quit()
        self.aec_env.close()

# =============================================================================
#  [Parallel Worker Function]
#  이 함수는 각 프로세스(CPU 코어)에서 독립적으로 실행됩니다.
# =============================================================================
def run_eval_worker(game_idx, config, model_state_dict, obs_shape, n_actions):
    """
    단일 게임을 수행하는 워커 함수
    """
    # 1. 각 프로세스마다 별도의 CPU Device 사용 (CUDA Context 충돌 방지)
    device = torch.device("cpu")
    
    # 2. 모델 재생성 및 가중치 로드
    # (ActorCritic 클래스가 이 파일 내에 있거나 import 가능해야 함)
    local_agent = ChessAgent(obs_shape, n_actions).to(device)
    local_agent.load_state_dict(model_state_dict)
    local_agent.eval()

    # 3. 환경 생성
    try:
        env = StockfishEvalEnv(config.stockfish_path, skill_level=config.stockfish_eval_skill, time_limit=config.stockfish_eval_time_limit)
    except Exception as e:
        print(f"[Worker Error] Failed to init Env: {e}")
        return 0 # 에러 시 무승부 처리 또는 예외 발생

    # 4. 게임 진행
    play_as_white = (game_idx % 2 == 0)
    try:
        # 비디오 녹화는 워커에서 하지 않음 (False)
        result, _ = env.play_match(local_agent, device, play_as_white=play_as_white, record_video=False)
    except Exception as e:
        print(f"[Worker Error] Game {game_idx} failed: {e}")
        result = 0
    finally:
        env.close()

    return result

# =============================================================================
#  [Main Evaluation Function]
# =============================================================================
def evaluate_vs_stockfish(agent, device, config, num_games=10, update=0):
    
    # -----------------------------------------------------
    # 1. 첫 번째 게임: 비디오 녹화를 위해 메인 프로세스에서 실행
    # -----------------------------------------------------
    print(f"[EVAL] Playing video match (Game 1/{num_games})...")
    
    # 영상 저장을 위한 Eval Env
    video_env = StockfishEvalEnv(
        config.stockfish_path, 
        skill_level=config.stockfish_eval_skill, 
        time_limit=config.stockfish_eval_time_limit,
        render_mode="rgb_array"
    )
    
    # DDP 등으로 감싸진 모델의 원본을 가져오기 (state_dict 추출용)
    raw_model = agent.module if hasattr(agent, "module") else agent
    raw_model.eval()

    # 첫 번째 판 실행
    video_result, frames = video_env.play_match(
        raw_model, device, play_as_white=True, record_video=True
    )
    video_env.close()
    
    results = [video_result] # 결과 리스트 시작

    # 영상 저장 (OpenCV)
    video_dir = f"videos/{config.exp_name}"
    os.makedirs(video_dir, exist_ok=True)
    
    if frames:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        outcome = "Win" if video_result == 1 else "Loss" if video_result == -1 else "Draw"
        filename = f"{video_dir}/step_{update}_White_{outcome}.mp4"
        
        try:
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(filename, fourcc, 2.0, (width, height))
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"[VIDEO] Saved to {filename}")
        except Exception as e:
            print(f"[ERROR] Video save failed: {e}")

    # -----------------------------------------------------
    # 2. 나머지 게임: 멀티프로세싱으로 병렬 실행
    # -----------------------------------------------------
    remaining_games = num_games - 1
    if remaining_games > 0:
        print(f"[EVAL] Playing {remaining_games} games in parallel...")
        
        # 모델의 가중치를 CPU로 이동 (프로세스 간 공유를 위해)
        cpu_state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
        
        # 모델 구조 정보 추출 (ActorCritic 재생성용)
        # obs_shape=(8,8,111), n_actions=4672 등
        obs_shape = (video_env.aec_env.observation_space("player_0")["observation"].shape[0],
                     video_env.aec_env.observation_space("player_0")["observation"].shape[1],
                     video_env.aec_env.observation_space("player_0")["observation"].shape[2])
        n_actions = video_env.aec_env.action_space("player_0").n

        # 워커 함수에 전달할 고정 인자들을 묶음 (partial)
        worker_fn = functools.partial(
            run_eval_worker, 
            config=config, 
            model_state_dict=cpu_state_dict,
            obs_shape=obs_shape,
            n_actions=n_actions
        )

        # 게임 인덱스 리스트 (1부터 시작)
        game_indices = range(1, num_games)

        # CPU 코어 수만큼 프로세스 풀 생성 (너무 많으면 오버헤드 발생, 보통 cpu_count 사용)
        num_workers = min(multiprocessing.cpu_count(), 4)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 병렬 실행 및 결과 수집
            parallel_results = list(executor.map(worker_fn, game_indices))
        
        results.extend(parallel_results)

    # -----------------------------------------------------
    # 3. 결과 집계
    # -----------------------------------------------------
    agent.train() # 다시 학습 모드로 전환
    
    win_rate = results.count(1) / num_games
    draw_rate = results.count(0) / num_games
    loss_rate = results.count(-1) / num_games
    
    return win_rate, draw_rate, loss_rate

@dataclass
class PPOConfig:
    exp_name: str = "chess_resnet_ppo_fixed"
    total_timesteps: int = 50_000_000
    learning_rate: float = 1e-4  # 사전학습 후 낮춤
    
    # Environment & Rollout (조정됨)
    num_steps: int = 256        # 512->256 (메모리 절약)
    num_envs: int = 16          # 32->16 (안정성)
    
    # Optimization (개선됨)
    minibatch_size: int = 512   # 1024->512 (더 많은 업데이트)
    update_epochs: int = 8      # 4->8 (충분한 학습)

    # Pre-training (대폭 증가)
    pretrain_samples: int = 50_000   # 10k->50k
    pretrain_epochs: int = 20         # 10->20
    pretrain_batch_size: int = 256
    
    dataset_path: str = "data/stockfish_pretrain_data.npz"
    
    # Coefficients (조정됨)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.02        # 0.01->0.02 (탐험 증가)
    vf_coef: float = 1.0          # 0.5->1.0 (가치 학습 강화)
    max_grad_norm: float = 0.5
    
    # System
    seed: int = 42
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 2  # 실제로 사용
    
    # Experience Replay 제거 (PPO는 on-policy)
    use_replay_buffer: bool = False
    
    # Curriculum Learning 추가
    use_curriculum: bool = True
    curriculum_stages: list = None
    
    # Paths & Logging
    logging: bool = False
    resume: str = None
    stockfish_path: str = "/usr/games/stockfish"
    stockfish_eval_time_limit: float = 0.05
    stockfish_eval_skill: int = 0
    
    # Evaluation
    eval_interval: int = 50
    eval_games: int = 32
    
    # Early Stopping
    early_stopping_patience: int = 20
    target_reward: float = 0.8
    
    def __post_init__(self):
        if self.curriculum_stages is None:
            self.curriculum_stages = [
                {"time": 0.01, "skill": 0},   # 매우 약함
                {"time": 0.02, "skill": 1},   # 약함
                {"time": 0.05, "skill": 2},   # 중간
            ]



# ==========================================
# 1. Environment (보상 구조 개선)
# ==========================================

# _KNIGHT_MOVES_CACHE = {
#     (1, 2): 56, (2, 1): 57, (2, -1): 58, (1, -2): 59,
#     (-1, -2): 60, (-2, -1): 61, (-2, 1): 62, (-1, 2): 63
# }
# _DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

# def move_to_action(move: chess.Move) -> int:
#     if move is None: return 0
#     from_sq = move.from_square
#     to_sq = move.to_square
#     df = chess.square_file(to_sq) - chess.square_file(from_sq)
#     dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

#     if (df, dr) in _KNIGHT_MOVES_CACHE:
#         return from_sq * 73 + _KNIGHT_MOVES_CACHE[(df, dr)]

#     if move.promotion is not None and move.promotion != chess.QUEEN:
#         return None

#     for dir_idx, (dir_x, dir_y) in enumerate(_DIRECTIONS):
#         if df * dir_y - dr * dir_x == 0 and df * dir_x + dr * dir_y > 0:
#             distance = max(abs(df), abs(dr))
#             if 1 <= distance <= 7:
#                 return from_sq * 73 + dir_idx * 7 + (distance - 1)
#     return None

def encode_move(move: chess.Move, should_mirror: bool = False) -> int:
    """
    chess.Move 객체를 PettingZoo Action ID(int)로 변환합니다.
    should_mirror: 현재 턴이 Black이라면 True (PettingZoo는 흑번일 때 보드를 뒤집어서 계산함)
    """
    if should_mirror:
        move = mirror_move(move)

    TOTAL = 73
    source = move.from_square
    coord = square_to_coord(source)
    panel = get_move_plane(move)
    
    # (row * 8 + col) * 73 + panel
    # 주의: square_to_coord의 리턴은 (col, row) 형태
    cur_action = (coord[0] * 8 + coord[1]) * TOTAL + panel
    return cur_action

class ChessSymmetricEnv(gym.Env):
    """개선된 보상 구조"""
    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.aec_env = chess_v6.env(render_mode=render_mode)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 111), dtype=np.float32)
        self.action_space = spaces.Discrete(4672)
        
        self._piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, 
            chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9
        }
        self.prev_material = 0
        self.move_count = 0

    def _get_material_score(self, board):
        score = 0
        for pt, val in self._piece_values.items():
            score += len(board.pieces(pt, chess.WHITE)) * val
            score -= len(board.pieces(pt, chess.BLACK)) * val
        return score

    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed)
        obs_dict, _, _, _, info = self.aec_env.last()
        obs = obs_dict["observation"]
        mask = obs_dict["action_mask"]
        
        board = self.aec_env.unwrapped.board
        self.prev_material = self._get_material_score(board)
        self.move_count = 0
        
        return obs.astype(np.float32), {"action_mask": mask.astype(np.float32)}

    def step(self, action):
        mover = self.aec_env.agent_selection
        self.aec_env.step(action)
        
        obs_dict, _, terminated, truncated, info = self.aec_env.last()
        
        if obs_dict is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            mask = np.ones(self.action_space.n, dtype=np.float32)
        else:
            obs = obs_dict["observation"]
            mask = obs_dict["action_mask"]

        board = self.aec_env.unwrapped.board
        reward = 0.0
        self.move_count += 1

        # === 개선된 보상 구조 ===
        if terminated or truncated:
            # 1. 게임 종료 보상
            if board.is_checkmate():
                # 승리자 확인
                winner = self.aec_env.unwrapped.board.turn  # False=Black won, True=White won
                is_white = (mover == "player_0")
                
                # 현재 플레이어가 이겼는지 확인
                if (is_white and not winner) or (not is_white and winner):
                    reward = 1.0  # 승리
                else:
                    reward = -1.0  # 패배
            else:
                # 무승부는 약간 부정적
                reward = -0.3
                
            # 2. 빠른 승리 보너스
            if reward > 0:
                reward += max(0, (100 - self.move_count) / 100)
                
        else:
            # 3. 중간 보상 (매우 중요!)
            current_material = self._get_material_score(board)
            
            # 기물 가치 변화
            is_white = (mover == "player_0")
            material_change = current_material - self.prev_material
            
            if is_white:
                reward += material_change * 0.01  # 백: 재료 증가 시 보상
            else:
                reward -= material_change * 0.01  # 흑: 재료 감소 시 보상
            
            self.prev_material = current_material
            
            # 4. 체크 보너스
            if board.is_check():
                reward += 0.05
            
            # 5. 이동 다양성 장려 (너무 긴 게임 방지)
            if self.move_count > 100:
                reward -= 0.001 * (self.move_count - 100)

        return obs.astype(np.float32), float(reward), terminated or truncated, False, {"action_mask": mask.astype(np.float32)}

    def close(self):
        self.aec_env.close()

# ==========================================
# 2. Model (동일)
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

    def forward(self, obs, action=None, action_mask=None, eval_mode=False):
        x = obs.permute(0, 3, 1, 2)
        x = self.start_block(x)
        x = self.backbone(x)
        
        logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)

        if action_mask is not None:
            is_all_invalid = action_mask.sum(dim=1) == 0
            if is_all_invalid.any():
                action_mask = action_mask.clone()
                action_mask[is_all_invalid, 0] = 1.0
            logits = torch.where(action_mask.bool(), logits, torch.tensor(-1e9, device=logits.device))

        if eval_mode:
            action = torch.argmax(logits, dim=1)

        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value

# ==========================================
# 3. Pre-training (동일하게 유지)
# ==========================================
def collect_worker(rank, sample_count, config, seed):
    process_seed = seed + rank
    np.random.seed(process_seed)
    random.seed(process_seed)
    
    env = ChessSymmetricEnv()
    try:
        engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        skill = getattr(config, 'stockfish_eval_skill', 0)
        engine.configure({"Skill Level": skill})
    except Exception as e:
        print(f"[Worker {rank}] Error starting Stockfish: {e}")
        return [], [], []

    obs_list, act_list, val_list = [], [], []
    obs, info = env.reset(seed=process_seed)
    
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
            
            action_idx = encode_move(best_move, should_mirror=(board.turn == chess.BLACK))
            
            if action_idx is not None and info["action_mask"][action_idx] == 1:
                obs_list.append(obs)
                act_list.append(action_idx)
                val_list.append(value_target)
                
                obs, _, terminated, truncated, info = env.step(action_idx)
                if terminated or truncated:
                    obs, info = env.reset()
            else:
                valid_actions = np.where(info["action_mask"])[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                    obs, _, term, trunc, info = env.step(action)
                    if term or trunc: obs, info = env.reset()
                else:
                    obs, info = env.reset()
        except Exception:
            obs, info = env.reset()

    engine.quit()
    env.close()
    
    return obs_list, act_list, val_list

def collect_stockfish_data(config):
    if os.path.exists(config.dataset_path):
        print(f"[PRE] Found saved dataset at '{config.dataset_path}'. Loading...")
        try:
            data = np.load(config.dataset_path)
            print(f"[PRE] Loaded {len(data['obs'])} samples successfully.")
            return data['obs'], data['acts'], data['vals']
        except Exception as e:
            print(f"[WARN] Failed to load dataset: {e}. Starting fresh collection.")

    num_workers = min(mp.cpu_count(), 12)
    samples_per_worker = config.pretrain_samples // num_workers
    remainder = config.pretrain_samples % num_workers
    
    print(f"[PRE] Collecting {config.pretrain_samples} samples using {num_workers} workers...")
    
    worker_args = []
    for i in range(num_workers):
        count = samples_per_worker + (remainder if i == num_workers - 1 else 0)
        worker_args.append((i, count, config, config.seed))

    t_start = time.time()
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(collect_worker, worker_args)
    
    all_obs, all_acts, all_vals = [], [], []
    total_collected = 0
    
    for r_obs, r_acts, r_vals in results:
        all_obs.extend(r_obs)
        all_acts.extend(r_acts)
        all_vals.extend(r_vals)
        total_collected += len(r_obs)
    
    print(f"[PRE] Collection finished in {time.time() - t_start:.2f}s. Total samples: {total_collected}")

    obs_arr = np.array(all_obs, dtype=np.float32)
    acts_arr = np.array(all_acts, dtype=np.int64)
    vals_arr = np.array(all_vals, dtype=np.float32)

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
# 4. Main Training Loop (수정됨)
# ==========================================
def make_env():
    return ChessSymmetricEnv()

def train(config: PPOConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")
    
    if config.logging:
        wandb.init(project="Chess-PPO-Fixed", config=config.__dict__, name=config.exp_name)

    print(f"[INFO] Creating {config.num_envs} environments...")
    envs = AsyncVectorEnv([make_env for _ in range(config.num_envs)], daemon=True)

    agent = ChessAgent(obs_shape=(8, 8, 111), n_actions=4672).to(device)
    
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
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_timesteps // (config.num_steps * config.num_envs))
    scaler = GradScaler('cuda') if config.use_mixed_precision else None

    # Storage
    obs_buffer = torch.zeros((config.num_steps, config.num_envs, 8, 8, 111), device=device)
    actions_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    values_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    masks_buffer = torch.zeros((config.num_steps, config.num_envs, 4672), dtype=torch.bool, device=device)

    next_obs_np, next_info = envs.reset(seed=config.seed)
    next_obs = torch.tensor(next_obs_np, device=device)
    next_done = torch.zeros(config.num_envs, device=device)
    next_mask = torch.tensor(next_info["action_mask"], device=device)
    
    global_step = 0
    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)
    start_time = time.time()
    
    # Curriculum Learning
    curriculum_idx = 0

    print("[INFO] Starting training loop...")

    for update in range(1, num_updates + 1):
        # Curriculum 업데이트
        if config.use_curriculum and curriculum_idx < len(config.curriculum_stages) - 1:
            progress = global_step / config.total_timesteps
            if progress > (curriculum_idx + 1) / len(config.curriculum_stages):
                curriculum_idx += 1
                stage = config.curriculum_stages[curriculum_idx]
                print(f"[CURRICULUM] Stage {curriculum_idx}: time={stage['time']}, skill={stage['skill']}")

        # --- Rollout ---
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

            real_next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards_buffer[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(real_next_obs, device=device)
            next_done = torch.tensor(done, device=device, dtype=torch.float32)
            next_mask = torch.tensor(info["action_mask"], device=device)

        # --- GAE 계산 (수정됨) ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)  # (num_envs,)
            advantages = torch.zeros_like(rewards_buffer)
            lastgaelam = 0
            
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done  # (num_envs,)
                    nextvalues = next_value            # (num_envs,)
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]  # (num_envs,)
                    nextvalues = values_buffer[t + 1]            # (num_envs,)
                
                delta = rewards_buffer[t] + config.gamma * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values_buffer

        # Flatten
        b_obs = obs_buffer.reshape((-1, 8, 8, 111))
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)
        b_masks = masks_buffer.reshape((-1, 4672))
# --- PPO Update (Gradient Accumulation 추가) ---
        b_inds = np.arange(config.num_steps * config.num_envs)
        clipfracs = []
        
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, len(b_inds), config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                # Gradient Accumulation 구현
                mini_batch_size = len(mb_inds) // config.gradient_accumulation_steps
                
                for acc_step in range(config.gradient_accumulation_steps):
                    acc_start = acc_step * mini_batch_size
                    acc_end = acc_start + mini_batch_size
                    acc_inds = mb_inds[acc_start:acc_end]
                    
                    with autocast('cuda', enabled=config.use_mixed_precision):
                        _, newlogprob, entropy, newvalue = agent(
                            b_obs[acc_inds], 
                            b_actions[acc_inds], 
                            b_masks[acc_inds]
                        )
                        
                        logratio = newlogprob - b_logprobs[acc_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[acc_inds]
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy Loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value Loss (Clipped)
                        newvalue = newvalue.view(-1)
                        v_loss_unclipped = (newvalue - b_returns[acc_inds]) ** 2
                        v_clipped = b_values[acc_inds] + torch.clamp(
                            newvalue - b_values[acc_inds],
                            -config.clip_coef,
                            config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[acc_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
                        
                        # Gradient Accumulation을 위한 정규화
                        loss = loss / config.gradient_accumulation_steps

                    if config.use_mixed_precision:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                # Gradient Accumulation 후 업데이트
                if config.use_mixed_precision:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
        
        scheduler.step()

        # --- Logging ---
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if update % 10 == 0:
            sps = int(global_step / (time.time() - start_time))
            print(f"Update {update}/{num_updates} | Step {global_step:,} | "
                  f"Reward: {rewards_buffer.mean().item():.4f} | "
                  f"SPS: {sps} | "
                  f"ExplainedVar: {explained_var:.3f}")
            
            if config.logging:
                wandb.log({
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/SPS": sps,
                    "charts/global_step": global_step,
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "rewards/mean": rewards_buffer.mean().item(),
                    "rewards/std": rewards_buffer.std().item(),
                    "rewards/max": rewards_buffer.max().item(),
                    "rewards/min": rewards_buffer.min().item(),
                })

        # --- Evaluation & Saving ---
        if update % config.eval_interval == 0:
            os.makedirs("models", exist_ok=True)
            save_path = f"models/{config.exp_name}_update{update}.pt"
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'update': update,
            }, save_path)
            print(f"[SAVE] Model saved to {save_path}")
            
            eval_agent = agent.module if isinstance(agent, nn.DataParallel) else agent
            
            try:
                # Curriculum 단계에 맞는 Stockfish 설정
                if config.use_curriculum:
                    stage = config.curriculum_stages[curriculum_idx]
                    eval_time = stage['time']
                    eval_skill = stage['skill']
                    config.stockfish_eval_time_limit = eval_time
                    config.stockfish_eval_skill = eval_skill
                else:
                    eval_time = config.stockfish_eval_time_limit
                    eval_skill = config.stockfish_eval_skill
                
                win, draw, loss = evaluate_vs_stockfish(
                    eval_agent,
                    device, 
                    config, 
                    num_games=config.eval_games,
                    update=update
                )
                
                print(f"[EVAL] vs Stockfish(skill={eval_skill}, time={eval_time:.3f}s) - "
                      f"Win: {win*100:.1f}%, Draw: {draw*100:.1f}%, Loss: {loss*100:.1f}%")
                
                if config.logging:
                    wandb.log({
                        "eval/win_rate": win, 
                        "eval/draw_rate": draw, 
                        "eval/loss_rate": loss,
                        "eval/score": win + 0.5 * draw,  # Chess scoring
                        "eval/stockfish_time_limit": eval_time,
                        "eval/stockfish_skill": eval_skill,
                        "eval/curriculum_stage": curriculum_idx,
                        "global_step": global_step
                    })
                
                # Curriculum 자동 진행 (선택적)
                if config.use_curriculum and (win + draw) > 0.7:
                    if curriculum_idx < len(config.curriculum_stages) - 1:
                        curriculum_idx += 1
                        print(f"[CURRICULUM] Advanced to stage {curriculum_idx}!")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Eval failed: {e}")

    envs.close()
    
    # Final save
    final_path = f"models/{config.exp_name}_final.pt"
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
    }, final_path)
    print(f"[SAVE] Final model saved to {final_path}")
    
    if config.logging:
        wandb.finish()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish_path", type=str, default="/usr/games/stockfish")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    config = PPOConfig(
        stockfish_path=args.stockfish_path,
        logging=args.logging,
        eval_interval=args.eval_interval,
        resume=args.resume
    )

    try:
        train(config)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] Training failed: {e}")