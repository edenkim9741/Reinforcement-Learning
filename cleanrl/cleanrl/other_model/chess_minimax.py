import chess
import chess.engine
import random
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn # ActorCritic 재생성을 위해 필요
from pettingzoo.classic import chess_v6
import os
import time
import cv2
import functools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# [주의] 이 코드가 실행되는 파일 내에 ActorCritic 클래스가 정의되어 있거나 import 되어 있어야 합니다.
# from your_model_file import ActorCritic 

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

    # [Multi-GPU 수정] DataParallel은 forward 메서드를 자동으로 병렬화합니다.
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

# =============================================================================
#  [Helper Class] Stockfish Environment
# =============================================================================
class StockfishEvalEnv(gym.Env):
    def __init__(self, stockfish_path, skill_level=0, render_mode=None):
        super().__init__()
        self.aec_env = chess_v6.env(render_mode=render_mode)
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
                limit = chess.engine.Limit(time=0.05)
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
    local_agent = ActorCritic(obs_shape, n_actions).to(device)
    local_agent.load_state_dict(model_state_dict)
    local_agent.eval()

    # 3. 환경 생성
    try:
        env = StockfishEvalEnv(config.stockfish_path, skill_level=config.stockfish_eval_skill)
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
def evaluate_vs_stockfish(agent, device, config, num_games=10, step_count=0):
    
    # -----------------------------------------------------
    # 1. 첫 번째 게임: 비디오 녹화를 위해 메인 프로세스에서 실행
    # -----------------------------------------------------
    print(f"[EVAL] Playing video match (Game 1/{num_games})...")
    
    # 영상 저장을 위한 Eval Env
    video_env = StockfishEvalEnv(
        config.stockfish_path, 
        skill_level=config.stockfish_eval_skill, 
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
        filename = f"{video_dir}/step_{step_count}_White_{outcome}.mp4"
        
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
        num_workers = min(multiprocessing.cpu_count(), remaining_games)
        
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

def get_best_move_stockfish_skill(board: chess.Board, time_limit=0.01, skill_level=0):
    if skill_level <0 :
        return select_action_random(board)

    global engine
    if engine is None:
        try:
            # SimpleEngine으로 프로세스 실행
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            engine.configure({"Skill Level": skill_level})
        except FileNotFoundError:
            print(f"[ERROR] Stockfish not found at {STOCKFISH_PATH}. Please install it.")
            return None

    # time_limit(초) 만큼만 생각하고 수를 둠 (0.01초면 매우 빠름)
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    return result.move


def select_action_random(board):
    legal_moves = list(board.legal_moves)

    return random.choice(legal_moves)


def square_to_coord(s):
    col = s % 8
    row = s // 8
    return (col, row)

def diff(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    return (x2 - x1, y2 - y1)

def sign(v):
    return -1 if v < 0 else (1 if v > 0 else 0)

def mirror_move(move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )

def get_queen_dir(diff):
    dx, dy = diff
    assert dx == 0 or dy == 0 or abs(dx) == abs(dy)
    magnitude = max(abs(dx), abs(dy)) - 1
    counter = 0
    for x in range(-1, 1 + 1):
        for y in range(-1, 1 + 1):
            if x == 0 and y == 0: continue
            if x == sign(dx) and y == sign(dy): return magnitude, counter
            counter += 1
    return 0, 0

def get_queen_plane(diff):
    NUM_COUNTERS = 8
    mag, counter = get_queen_dir(diff)
    return mag * NUM_COUNTERS + counter

def get_knight_dir(diff):
    dx, dy = diff
    counter = 0
    for x in range(-2, 2 + 1):
        for y in range(-2, 2 + 1):
            if abs(x) + abs(y) == 3:
                if dx == x and dy == y: return counter
                counter += 1
    return 0

def is_knight_move(diff):
    dx, dy = diff
    return abs(dx) + abs(dy) == 3 and 1 <= abs(dx) <= 2

def get_pawn_promotion_move(diff):
    dx, dy = diff
    return dx + 1

def get_pawn_promotion_num(promotion):
    return 0 if promotion == chess.KNIGHT else (1 if promotion == chess.BISHOP else 2)

def get_move_plane(move):
    source = move.from_square
    dest = move.to_square
    difference = diff(square_to_coord(source), square_to_coord(dest))

    QUEEN_MOVES = 56
    KNIGHT_MOVES = 8
    QUEEN_OFFSET = 0
    KNIGHT_OFFSET = QUEEN_MOVES
    UNDER_OFFSET = KNIGHT_OFFSET + KNIGHT_MOVES

    if is_knight_move(difference):
        return KNIGHT_OFFSET + get_knight_dir(difference)
    else:
        if move.promotion is not None and move.promotion != chess.QUEEN:
            return (
                UNDER_OFFSET
                + 3 * get_pawn_promotion_move(difference)
                + get_pawn_promotion_num(move.promotion)
            )
        else:
            return QUEEN_OFFSET + get_queen_plane(difference)

def encode_move(move: chess.Move, should_mirror: bool) -> int:
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


# =============================================================================
#  [PART 2] Minimax Algorithm
# =============================================================================

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def evaluate_board(board: chess.Board):
    if board.is_checkmate():
        return -9999 if board.turn == chess.WHITE else 9999
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # 기존 for loop 제거 -> Bitboard 연산으로 대체
    # (White 점수 - Black 점수)
    score = 0
    
    # Pawn (1점)
    score += 1 * (len(board.pieces(chess.PAWN, chess.WHITE)) - len(board.pieces(chess.PAWN, chess.BLACK)))
    # Knight (3점)
    score += 3 * (len(board.pieces(chess.KNIGHT, chess.WHITE)) - len(board.pieces(chess.KNIGHT, chess.BLACK)))
    # Bishop (3점)
    score += 3 * (len(board.pieces(chess.BISHOP, chess.WHITE)) - len(board.pieces(chess.BISHOP, chess.BLACK)))
    # Rook (5점)
    score += 5 * (len(board.pieces(chess.ROOK, chess.WHITE)) - len(board.pieces(chess.ROOK, chess.BLACK)))
    # Queen (9점)
    score += 9 * (len(board.pieces(chess.QUEEN, chess.WHITE)) - len(board.pieces(chess.QUEEN, chess.BLACK)))
    
    return score

def order_moves(board, moves):
    """
    좋은 수(잡는 수, 승진 등)를 리스트 앞쪽으로 배치
    """
    def score_move(move):
        score = 0
        # 1. 기물을 잡는 수 (높은 점수)
        if board.is_capture(move):
            # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)의 단순화 버전
            # 잡히는 기물의 가치가 높을수록 우선순위 둠
            if board.is_en_passant(move):
                score += 10 # 폰 잡음
            else:
                victim = board.piece_at(move.to_square)
                if victim:
                    score += PIECE_VALUES[victim.piece_type] * 10
        
        # 2. 승진하는 수
        if move.promotion:
            score += 90 # 퀸 승진 등은 매우 중요
            
        return score

    # 점수가 높은 순으로 정렬 (내림차순)
    return sorted(moves, key=score_move, reverse=True)

def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    # [수정] 정렬된 Move 리스트 사용
    legal_moves = order_moves(board, list(board.legal_moves))

    if maximizing_player:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def get_best_move_minimax(board: chess.Board, depth=2):
    """
    Minimax를 통해 최적의 chess.Move를 반환합니다.
    """
    best_move = None
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return None
    
    # White 차례면 Maximize, Black 차례면 Minimize
    maximizing_player = (board.turn == chess.WHITE)
    
    if maximizing_player:
        best_value = -float('inf')
        for move in legal_moves:
            board.push(move)
            val = minimax(board, depth - 1, -float('inf'), float('inf'), False)
            board.pop()
            if val > best_value:
                best_value = val
                best_move = move
    else:
        best_value = float('inf')
        for move in legal_moves:
            board.push(move)
            val = minimax(board, depth - 1, -float('inf'), float('inf'), True)
            board.pop()
            if val < best_value:
                best_value = val
                best_move = move
                
    return best_move