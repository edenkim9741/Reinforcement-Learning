#!/usr/bin/env python3
"""
PPO Agent for Chess vs Stockfish (Curriculum Learning)

목표: Stockfish Level 0 (time_limit=0.01s)에 대해 50% 이상 승률 달성

학습 파이프라인:
    Phase 1: Stockfish 데이터셋 기반 지도학습 (Imitation Learning)
    Phase 2: Random 상대와 강화학습 (기초 전략 학습)
    Phase 3: Stockfish 상대와 강화학습 (고급 전략 학습)

구성:
    - ChessEnvWrapper: Gymnasium 호환 체스 환경 (Stockfish/Random 상대)
    - ChessAgent: ResNet 기반 Actor-Critic 네트워크 (Action Masking)
    - PPOTrainer: Curriculum Learning을 지원하는 PPO 학습기
    - ThreadedVecEnv: 병렬 환경 실행

사용법:
    python ppo_chess_vs_stockfish_clean.py --mode curriculum --use-wandb
    python ppo_chess_vs_stockfish_clean.py --mode eval --model-path models/ppo_chess_final.pt
"""

import os
import time
import random
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import wandb

import gymnasium as gym
from gymnasium import spaces
import chess
import chess.engine
from pettingzoo.classic import chess_v6
from tqdm import tqdm


# =============================================================================
# 설정 (Configuration)
# =============================================================================
@dataclass
class Config:
    """학습 및 환경 설정"""
    # Stockfish 설정
    stockfish_path: str = "/usr/games/stockfish"
    stockfish_skill: int = 0
    stockfish_time_limit: float = 0.01
    
    # 학습 하이퍼파라미터
    seed: int = 42
    total_timesteps: int = 16_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: Optional[float] = None

    skip_phases: Tuple[int, ...] = ()
    
    # 모델 아키텍처
    num_res_blocks: int = 4
    channels: int = 128
    
    # 평가 설정
    eval_episodes: int = 20
    eval_interval: int = 10
    
    # 경로
    model_dir: str = "models"
    
    # Pretraining
    pretrain_lr: float = 1e-3
    
    # Curriculum Learning 비율
    phase2_ratio: float = 0.3  # Random 상대 학습 비율
    phase3_ratio: float = 0.7  # Stockfish 상대 학습 비율


# =============================================================================
# 무브 인코딩/디코딩 유틸리티
# =============================================================================
def _square_to_coord(s: int) -> Tuple[int, int]:
    """체스판 칸 인덱스를 (col, row) 좌표로 변환"""
    return (s % 8, s // 8)


def _coord_diff(c1: Tuple[int, int], c2: Tuple[int, int]) -> Tuple[int, int]:
    """두 좌표의 차이 계산"""
    return (c2[0] - c1[0], c2[1] - c1[1])


def _sign(v: int) -> int:
    """부호 반환"""
    return -1 if v < 0 else (1 if v > 0 else 0)


def _mirror_move(move: chess.Move) -> chess.Move:
    """흑색 관점에서의 무브 미러링"""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def _get_queen_plane(diff: Tuple[int, int]) -> int:
    """퀸 이동에 대한 평면 인덱스 계산"""
    dx, dy = diff
    magnitude = max(abs(dx), abs(dy)) - 1
    counter = 0
    for x in range(-1, 2):
        for y in range(-1, 2):
            if x == 0 and y == 0:
                continue
            if x == _sign(dx) and y == _sign(dy):
                return magnitude * 8 + counter
            counter += 1
    return 0


def _get_knight_dir(diff: Tuple[int, int]) -> int:
    """나이트 이동 방향 인덱스 계산"""
    dx, dy = diff
    counter = 0
    for x in range(-2, 3):
        for y in range(-2, 3):
            if abs(x) + abs(y) == 3:
                if dx == x and dy == y:
                    return counter
                counter += 1
    return 0


def _is_knight_move(diff: Tuple[int, int]) -> bool:
    """나이트 이동 여부 확인"""
    dx, dy = diff
    return abs(dx) + abs(dy) == 3 and 1 <= abs(dx) <= 2


def _get_move_plane(move: chess.Move) -> int:
    """무브의 평면 인덱스 계산 (PettingZoo 액션 인코딩용)"""
    source = move.from_square
    dest = move.to_square
    diff = _coord_diff(_square_to_coord(source), _square_to_coord(dest))
    
    QUEEN_OFFSET, KNIGHT_OFFSET, UNDER_OFFSET = 0, 56, 64
    
    if _is_knight_move(diff):
        return KNIGHT_OFFSET + _get_knight_dir(diff)
    elif move.promotion is not None and move.promotion != chess.QUEEN:
        promo_dir = diff[0] + 1
        promo_type = {chess.KNIGHT: 0, chess.BISHOP: 1}.get(move.promotion, 2)
        return UNDER_OFFSET + 3 * promo_dir + promo_type
    else:
        return QUEEN_OFFSET + _get_queen_plane(diff)


def encode_move(move: chess.Move, should_mirror: bool = False) -> int:
    """chess.Move를 PettingZoo 액션 인덱스로 변환"""
    if should_mirror:
        move = _mirror_move(move)
    coord = _square_to_coord(move.from_square)
    plane = _get_move_plane(move)
    return (coord[0] * 8 + coord[1]) * 73 + plane


# =============================================================================
# 보상 형성 유틸리티
# =============================================================================
PIECE_VALUES = {
    chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
    chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0
}


def compute_intermediate_reward(board_before: chess.Board, 
                                 board_after: chess.Board, 
                                 our_color: chess.Color) -> float:
    """
    중간 보상 계산: 기물 가치 변화 + 체크 보너스
    
    Args:
        board_before: 이동 전 보드 상태
        board_after: 이동 후 보드 상태
        our_color: 에이전트의 색상
    
    Returns:
        중간 보상 값
    """
    def count_material(board: chess.Board, color: chess.Color) -> float:
        return sum(len(board.pieces(pt, color)) * val for pt, val in PIECE_VALUES.items())
    
    def get_balance(board: chess.Board) -> float:
        our = count_material(board, our_color)
        opp = count_material(board, not our_color)
        return (our - opp) / 39.0  # 정규화
    
    reward = (get_balance(board_after) - get_balance(board_before)) * 0.1
    
    # 체크 보너스
    if board_after.is_check() and board_after.turn != our_color:
        reward += 0.02
    
    return reward


# =============================================================================
# 체스 환경 래퍼
# =============================================================================
class ChessEnvWrapper(gym.Env):
    """
    Gymnasium 호환 체스 환경 래퍼
    
    Stockfish 또는 Random 상대와 대전
    
    Attributes:
        opponent_type: 상대 유형 ("stockfish" 또는 "random")
        play_as_white: 에이전트가 백색인지 여부
    """
    
    def __init__(self, 
                 opponent_type: str = "stockfish",
                 stockfish_path: str = "/usr/games/stockfish",
                 stockfish_skill: int = 0,
                 stockfish_time_limit: float = 0.01,
                 reward_shaping: bool = True):
        super().__init__()
        
        self.opponent_type = opponent_type
        self.stockfish_path = stockfish_path
        self.stockfish_skill = stockfish_skill
        self.stockfish_time_limit = stockfish_time_limit
        self.reward_shaping = reward_shaping
        
        # 플레이어 설정 (초기화)
        self.play_as_white = random.choice([True, False])
        self.agent_player = "player_0" if self.play_as_white else "player_1"
        self.opponent_player = "player_1" if self.play_as_white else "player_0"
        
        # PettingZoo 환경 생성
        self.aec_env = chess_v6.env()
        self.aec_env.reset()
        
        # 관찰/행동 공간 설정
        raw_obs_space = self.aec_env.observation_space("player_0")["observation"]
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0, high=1, shape=raw_obs_space.shape, dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(4672,), dtype=np.int8)
        })
        self.action_space = spaces.Discrete(4672)
        
        # Stockfish 엔진 초기화
        self.engine = None
        if self.opponent_type == "stockfish":
            self._init_stockfish()
    
    def _init_stockfish(self):
        """Stockfish 엔진 초기화"""
        try:
            if self.engine is not None:
                self.engine.quit()
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self.engine.configure({"Skill Level": self.stockfish_skill})
        except FileNotFoundError:
            raise FileNotFoundError(f"Stockfish not found at {self.stockfish_path}")
    
    def _get_opponent_action(self, observation: dict) -> int:
        """상대방의 액션 결정"""
        action_mask = observation["action_mask"]
        valid_actions = np.where(action_mask)[0]
        
        if self.opponent_type == "random" or len(valid_actions) == 0:
            return np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
        
        # Stockfish 무브 획득
        board = self.aec_env.unwrapped.board
        try:
            result = self.engine.play(board, chess.engine.Limit(time=self.stockfish_time_limit))
            if result.move:
                is_black = (self.opponent_player == "player_1")
                action = encode_move(result.move, should_mirror=is_black)
                if action_mask[action] == 1:
                    return action
        except Exception:
            pass
        
        return np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
    
    def _opponent_turn(self):
        """상대방 턴 실행"""
        if self.aec_env.agent_selection != self.opponent_player:
            return
        
        observation, _, termination, truncation, _ = self.aec_env.last()
        if termination or truncation:
            self.aec_env.step(None)
            return
        
        action = self._get_opponent_action(observation)
        self.aec_env.step(action)
    
    def _get_obs_and_info(self) -> Tuple[dict, dict]:
        """현재 관찰과 정보 반환"""
        observation, _, termination, truncation, _ = self.aec_env.last()
        obs = {
            "observation": observation["observation"].astype(np.float32),
            "action_mask": observation["action_mask"].astype(np.int8)
        }
        info = {"termination": termination, "truncation": truncation}
        return obs, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 랜덤하게 색상 선택
        self.play_as_white = options.get("play_as_white", random.choice([True, False])) if options else random.choice([True, False])
        self.agent_player = "player_0" if self.play_as_white else "player_1"
        self.opponent_player = "player_1" if self.play_as_white else "player_0"
        
        self.aec_env.reset(seed=seed)
        
        # 상대가 먼저 시작하면 턴 진행
        if not self.play_as_white:
            self._opponent_turn()
        
        return self._get_obs_and_info()
    
    def step(self, action: int):
        board_before = self.aec_env.unwrapped.board.copy() if self.reward_shaping else None
        our_color = chess.WHITE if self.play_as_white else chess.BLACK
        
        observation, _, termination, truncation, _ = self.aec_env.last()
        if termination or truncation:
            obs, info = self._get_obs_and_info()
            return obs, 0.0, True, False, info
        
        # 에이전트 액션 실행
        self.aec_env.step(action)
        
        # 중간 보상 계산
        intermediate_reward = 0.0
        if self.reward_shaping and board_before is not None:
            intermediate_reward = compute_intermediate_reward(
                board_before, self.aec_env.unwrapped.board, our_color
            )
        
        # 게임 종료 확인
        observation, _, termination, truncation, _ = self.aec_env.last()
        if termination or truncation:
            final_reward = self.aec_env.rewards.get(self.agent_player, 0.0)
            obs, info = self._get_obs_and_info()
            return obs, final_reward, True, False, info
        
        # 상대방 턴
        self._opponent_turn()
        
        # 상대 턴 후 게임 종료 확인
        observation, _, termination, truncation, _ = self.aec_env.last()
        if termination or truncation:
            final_reward = self.aec_env.rewards.get(self.agent_player, 0.0)
            obs, info = self._get_obs_and_info()
            return obs, final_reward, True, False, info
        
        obs, info = self._get_obs_and_info()
        return obs, intermediate_reward, False, False, info
    
    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except:
                pass
        self.aec_env.close()


# =============================================================================
# 병렬 환경 실행
# =============================================================================
class SimpleVecEnv:
    """
    순차 실행 Vectorized Environment
    
    Random 상대처럼 I/O가 없는 경우 스레딩 오버헤드 없이 빠르게 실행
    """
    
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def step(self, actions):
        obs_list, rewards, terminateds, truncateds, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        return obs_list, np.array(rewards), np.array(terminateds), np.array(truncateds), infos
    
    def reset(self, seed=None, options=None):
        obs_list, info_list = [], []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed, options=options)
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list
    
    def close(self):
        for env in self.envs:
            env.close()


class ThreadedVecEnv:
    """
    ThreadPoolExecutor 기반 병렬 환경
    
    Stockfish I/O 작업에 적합한 멀티스레딩 구현
    """
    
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self.executor = ThreadPoolExecutor(max_workers=self.num_envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def _step_env(self, args):
        env, action = args
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        return obs, reward, terminated, truncated, info
    
    def step(self, actions):
        results = list(self.executor.map(self._step_env, zip(self.envs, actions)))
        obs_list, rewards, terminateds, truncateds, infos = zip(*results)
        return list(obs_list), np.array(rewards), np.array(terminateds), np.array(truncateds), list(infos)
    
    def reset(self, seed=None, options=None):
        obs_list, info_list = [], []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed, options=options)
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list
    
    def close(self):
        for env in self.envs:
            env.close()
        self.executor.shutdown(wait=True)


# =============================================================================
# 신경망 모델
# =============================================================================
class ResBlock(nn.Module):
    """Residual Block"""
    
    def __init__(self, channels: int):
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
        return self.relu(out + identity)


class ChessAgent(nn.Module):
    """
    ResNet 기반 PPO 에이전트
    
    Architecture:
        - Input: 8x8x111 (PettingZoo chess 관찰 형식)
        - Backbone: Conv + N개의 Residual Block
        - Actor Head: 정책 네트워크 (Action Masking 지원)
        - Critic Head: 가치 네트워크
    """
    
    def __init__(self, 
                 obs_shape: Tuple[int, int, int] = (8, 8, 111), 
                 n_actions: int = 4672,
                 num_res_blocks: int = 4,
                 channels: int = 128):
        super().__init__()
        
        self.h, self.w, self.c = obs_shape
        self.n_actions = n_actions
        
        # 초기 컨볼루션
        self.start_block = nn.Sequential(
            nn.Conv2d(self.c, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual 백본
        self.backbone = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        
        # Actor Head (정책)
        self.actor_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * self.h * self.w, n_actions)
        )
        
        # Critic Head (가치)
        self.critic_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * self.h * self.w, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """관찰에서 특징 추출 (batch, h, w, c) -> (batch, c, h, w)"""
        x = obs.permute(0, 3, 1, 2)
        return self.backbone(self.start_block(x))
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """상태 가치 반환"""
        return self.critic_head(self._extract_features(obs))
    
    def get_action_and_value(self, 
                             obs: torch.Tensor, 
                             action: Optional[torch.Tensor] = None,
                             action_mask: Optional[torch.Tensor] = None):
        """
        액션, 로그 확률, 엔트로피, 가치 반환
        
        Args:
            obs: 관찰 텐서 (batch, h, w, c)
            action: 평가할 액션 (선택적)
            action_mask: 유효한 액션 마스크 (batch, n_actions)
        
        Returns:
            action, log_prob, entropy, value
        """
        features = self._extract_features(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        
        # Action Masking 적용
        if action_mask is not None:
            invalid_mask = action_mask.sum(dim=1) == 0
            if invalid_mask.any():
                action_mask = action_mask.clone()
                action_mask[invalid_mask, 0] = 1.0
            
            logits = torch.where(
                action_mask.bool(), logits,
                torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
            )
        
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value


# =============================================================================
# 데이터셋 생성 (병렬)
# =============================================================================
def _collect_single_game(args):
    """
    단일 게임 데이터 수집 (워커 함수)
    
    Args:
        args: (game_idx, stockfish_path, stockfish_skill, stockfish_time_limit)
    
    Returns:
        (observations, actions, masks) 리스트
    """
    game_idx, stockfish_path, stockfish_skill, stockfish_time_limit = args
    
    observations = []
    actions = []
    masks = []
    
    # 각 워커에서 Stockfish 엔진 생성
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Skill Level": stockfish_skill})
    except Exception:
        return [], [], []
    
    # PettingZoo 환경 생성
    env = chess_v6.env()
    env.reset()
    
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            env.step(None)
            continue
        
        board = env.unwrapped.board
        action_mask = observation["action_mask"]
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            env.step(0)
            continue
        
        # Stockfish에게 최선의 수 요청
        try:
            result = engine.play(board, chess.engine.Limit(time=stockfish_time_limit))
            best_move = result.move
        except Exception:
            action = np.random.choice(valid_actions)
            env.step(action)
            continue
        
        if best_move is None:
            action = np.random.choice(valid_actions)
            env.step(action)
            continue
        
        # 무브 인코딩
        is_black = (agent_name == "player_1")
        target_action = encode_move(best_move, should_mirror=is_black)
        
        # 유효한 액션인지 확인
        if action_mask[target_action] != 1:
            action = np.random.choice(valid_actions)
            env.step(action)
            continue
        
        # 데이터 수집
        observations.append(observation["observation"].astype(np.float32))
        actions.append(target_action)
        masks.append(action_mask.astype(np.float32))
        
        env.step(target_action)
    
    engine.quit()
    env.close()
    
    return observations, actions, masks


def generate_stockfish_dataset(
    output_path: str = "data/stockfish_pretrain_data.npz",
    num_games: int = 1000,
    num_workers: int = 8,
    stockfish_path: str = "/usr/games/stockfish",
    stockfish_skill: int = 10,
    stockfish_time_limit: float = 0.1,
    verbose: bool = True
):
    """
    Stockfish 자가 대전을 통한 학습 데이터셋 생성 (병렬 처리)
    
    Stockfish가 양측을 플레이하며 각 수에서의 관찰, 액션, 마스크를 수집합니다.
    이 데이터는 지도학습(Imitation Learning) 프리트레이닝에 사용됩니다.
    
    Args:
        output_path: 저장할 .npz 파일 경로
        num_games: 생성할 게임 수
        num_workers: 병렬 워커 수 (기본: 8)
        stockfish_path: Stockfish 실행 파일 경로
        stockfish_skill: Stockfish 스킬 레벨 (0-20, 높을수록 강함)
        stockfish_time_limit: 수당 시간 제한 (초)
        verbose: 진행 상황 출력 여부
    
    Returns:
        생성된 샘플 수
    """
    print(f"\n{'='*60}")
    print("Generating Stockfish Dataset (Parallel)")
    print(f"{'='*60}")
    print(f"Games: {num_games}, Workers: {num_workers}")
    print(f"Skill: {stockfish_skill}, Time: {stockfish_time_limit}s")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # 워커 인자 준비
    worker_args = [
        (i, stockfish_path, stockfish_skill, stockfish_time_limit)
        for i in range(num_games)
    ]
    
    # 데이터 저장소
    all_observations = []
    all_actions = []
    all_masks = []
    
    # 병렬 실행
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        if verbose:
            results = list(tqdm(
                executor.map(_collect_single_game, worker_args),
                total=num_games,
                desc="Generating games"
            ))
        else:
            results = list(executor.map(_collect_single_game, worker_args))
    
    # 결과 병합
    games_with_data = 0
    for obs_list, act_list, mask_list in results:
        if len(obs_list) > 0:
            all_observations.extend(obs_list)
            all_actions.extend(act_list)
            all_masks.extend(mask_list)
            games_with_data += 1
    
    elapsed_time = time.time() - start_time
    
    # NumPy 배열로 변환
    obs_array = np.array(all_observations)
    acts_array = np.array(all_actions, dtype=np.int64)
    masks_array = np.array(all_masks)
    
    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        obs=obs_array,
        acts=acts_array,
        masks=masks_array
    )
    
    print(f"\n{'='*60}")
    print("Dataset Generation Complete")
    print(f"{'='*60}")
    print(f"Games completed: {games_with_data}/{num_games}")
    print(f"Total samples: {len(all_observations)}")
    print(f"Average moves per game: {len(all_observations)/max(games_with_data, 1):.1f}")
    print(f"Time elapsed: {elapsed_time:.1f}s ({num_games/elapsed_time:.1f} games/s)")
    print(f"Saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"{'='*60}")
    
    return len(all_observations)


# =============================================================================
# PPO 학습기
# =============================================================================
class PPOTrainer:
    """
    Curriculum Learning을 지원하는 PPO 학습기
    
    Phase 1: 데이터셋 기반 지도학습 (Pretraining)
    Phase 2: Random 상대와 강화학습
    Phase 3: Stockfish 상대와 강화학습
    """
    
    def __init__(self, config: Config, device: torch.device, use_wandb: bool = False):
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        self.obs_shape = (8, 8, 111)
        self.n_actions = 4672
        
        # 에이전트 생성
        self.agent = ChessAgent(
            obs_shape=self.obs_shape,
            n_actions=self.n_actions,
            num_res_blocks=config.num_res_blocks,
            channels=config.channels
        ).to(device)
        
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)
        
        # 환경 및 저장소는 학습 시 초기화
        self.vec_env = None
        
        # 통계
        self.global_step = 0
        self.start_time = None
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
    
    def _create_vec_env(self, opponent_type: str, num_envs: int):
        """
        환경 생성: 상대 유형에 따라 적절한 VecEnv 선택
        
        - random: SimpleVecEnv (순차 실행, 오버헤드 없음)
        - stockfish: ThreadedVecEnv (I/O 병렬화)
        """
        def make_env(opp_type):
            def _init():
                return ChessEnvWrapper(
                    opponent_type=opp_type,
                    stockfish_path=self.config.stockfish_path,
                    stockfish_skill=self.config.stockfish_skill,
                    stockfish_time_limit=self.config.stockfish_time_limit
                )
            return _init
        
        env_fns = [make_env(opponent_type) for _ in range(num_envs)]
        
        # Random 상대는 I/O가 없으므로 순차 실행이 더 빠름
        if opponent_type == "random":
            return SimpleVecEnv(env_fns)
        else:
            return ThreadedVecEnv(env_fns)
    
    def _init_storage(self, num_envs: int):
        """롤아웃 저장소 초기화"""
        cfg = self.config
        self.obs = torch.zeros((cfg.num_steps, num_envs) + self.obs_shape).to(self.device)
        self.actions = torch.zeros((cfg.num_steps, num_envs), dtype=torch.long).to(self.device)
        self.logprobs = torch.zeros((cfg.num_steps, num_envs)).to(self.device)
        self.rewards = torch.zeros((cfg.num_steps, num_envs)).to(self.device)
        self.dones = torch.zeros((cfg.num_steps, num_envs)).to(self.device)
        self.values = torch.zeros((cfg.num_steps, num_envs)).to(self.device)
        self.action_masks = torch.zeros((cfg.num_steps, num_envs, self.n_actions)).to(self.device)
        
        self.next_obs = torch.zeros((num_envs,) + self.obs_shape).to(self.device)
        self.next_done = torch.zeros(num_envs).to(self.device)
        self.next_action_mask = torch.zeros((num_envs, self.n_actions)).to(self.device)
    
    def _reset_stats(self):
        """통계 초기화"""
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
    
    def collect_rollout(self, num_envs: int):
        """롤아웃 데이터 수집"""
        for step in range(self.config.num_steps):
            self.global_step += num_envs
            
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done
            self.action_masks[step] = self.next_action_mask
            
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(
                    self.next_obs, action_mask=self.next_action_mask
                )
                self.values[step] = value
            
            self.actions[step] = action
            self.logprobs[step] = logprob
            
            # 병렬 환경 스텝
            actions_np = action.cpu().numpy()
            obs_list, rewards, terminateds, truncateds, _ = self.vec_env.step(actions_np)
            
            for i in range(num_envs):
                self.rewards[step, i] = rewards[i]
                done = terminateds[i] or truncateds[i]
                
                if done:
                    if rewards[i] > 0:
                        self.win_count += 1
                    elif rewards[i] < 0:
                        self.loss_count += 1
                    else:
                        self.draw_count += 1
                
                self.next_obs[i] = torch.from_numpy(obs_list[i]["observation"]).to(self.device)
                self.next_action_mask[i] = torch.from_numpy(
                    obs_list[i]["action_mask"].astype(np.float32)
                ).to(self.device)
                self.next_done[i] = float(done)
    
    def compute_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE를 사용한 어드밴티지 계산"""
        cfg = self.config
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                
                delta = self.rewards[t] + cfg.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + self.values
        
        return advantages, returns
    
    def update(self, advantages: torch.Tensor, returns: torch.Tensor, num_envs: int):
        """정책 및 가치 네트워크 업데이트"""
        cfg = self.config
        batch_size = num_envs * cfg.num_steps
        minibatch_size = batch_size // cfg.num_minibatches
        
        # 배치 평탄화
        b_obs = self.obs.reshape((-1,) + self.obs_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        b_action_masks = self.action_masks.reshape(-1, self.n_actions)
        
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]
                
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds], b_action_masks[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                
                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value Loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
            
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break
        
        return pg_loss.item(), v_loss.item(), entropy_loss.item(), np.mean(clipfracs)
    
    def train_phase(self, opponent_type: str, total_timesteps: int, phase_name: str) -> float:
        """
        단일 학습 페이즈 실행
        
        Args:
            opponent_type: "random" 또는 "stockfish"
            total_timesteps: 총 타임스텝
            phase_name: 로깅용 페이즈 이름
        
        Returns:
            최종 승률
        """
        cfg = self.config
        num_envs = cfg.num_envs
        if opponent_type == "stockfish":
            num_envs = cfg.num_envs * 2
        batch_size = num_envs * cfg.num_steps
        num_iterations = total_timesteps // batch_size
        
        print(f"\n{'='*60}")
        print(f"{phase_name}: Training vs {opponent_type.upper()}")
        print(f"Iterations: {num_iterations}, Timesteps: {total_timesteps}, Envs: {num_envs}")
        print(f"{'='*60}")
        
        # 환경 생성
        self.vec_env = self._create_vec_env(opponent_type, num_envs)
        self._init_storage(num_envs)
        
        # 초기 상태 설정
        obs_list, _ = self.vec_env.reset(seed=cfg.seed)
        for i, obs in enumerate(obs_list):
            self.next_obs[i] = torch.from_numpy(obs["observation"]).to(self.device)
            self.next_action_mask[i] = torch.from_numpy(obs["action_mask"].astype(np.float32)).to(self.device)
        
        self.global_step = 0
        self.start_time = time.time()
        self._reset_stats()
        
        pbar = tqdm(range(1, num_iterations + 1), desc=phase_name)
        
        for iteration in pbar:
            # Learning Rate Annealing
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                self.optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate
            
            self.collect_rollout(num_envs)
            advantages, returns = self.compute_advantages()
            pg_loss, v_loss, entropy_loss, clipfrac = self.update(advantages, returns, num_envs)
            
            # 로깅
            if iteration % 10 == 0:
                total_games = self.win_count + self.loss_count + self.draw_count
                win_rate = self.win_count / max(total_games, 1)
                lose_rate = self.loss_count / max(total_games, 1)
                sps = int(self.global_step / (time.time() - self.start_time))
                
                pbar.set_postfix({
                    'W': self.win_count, 'L': self.loss_count, 'D': self.draw_count,
                    'WR': f'{win_rate:.1%}', 'LR': f'{lose_rate:.1%}', 'SPS': sps
                })
                
                if self.use_wandb:
                    wandb.log({
                        f"{phase_name}/iteration": iteration,
                        f"{phase_name}/global_step": self.global_step,
                        f"{phase_name}/win_rate": win_rate,
                        f"{phase_name}/lose_rate": lose_rate,
                        f"{phase_name}/win_count": self.win_count,
                        f"{phase_name}/loss_count": self.loss_count,
                        f"{phase_name}/draw_count": self.draw_count,
                        f"{phase_name}/policy_loss": pg_loss,
                        f"{phase_name}/value_loss": v_loss,
                        f"{phase_name}/entropy": entropy_loss,
                        f"{phase_name}/sps": sps,
                    })
                
                # 높은 승률 달성 시 조기 종료 (Stockfish 대전 시)
                if opponent_type == "stockfish" and win_rate >= 0.7:
                    print(f"\n✓ High win rate achieved: {win_rate:.1%}")
                    self.save_model(f"{cfg.model_dir}/ppo_chess_{phase_name}_wr{int(win_rate*100)}.pt")
                    break
                
                self._reset_stats()
            
            # 평가
            if opponent_type == "stockfish" and iteration % cfg.eval_interval == 0:
                eval_wr = self.evaluate()
                print(f"  [EVAL] Win rate vs Stockfish: {eval_wr:.1%}")
                if self.use_wandb:
                    wandb.log({f"{phase_name}/eval_win_rate": eval_wr})
                if eval_wr >= 0.5:
                    self.save_model(f"{cfg.model_dir}/ppo_chess_best.pt")
        
        self.vec_env.close()
        
        # 최종 평가
        final_win_rate = self.evaluate()
        print(f"{phase_name} Complete. Final Win Rate: {final_win_rate:.1%}")
        
        return final_win_rate
    
    def pretrain_from_dataset(self, data_path: str):
        """
        데이터셋 기반 지도학습 (Phase 1)
        
        Args:
            data_path: Stockfish 데이터셋 경로 (.npz)
        """
        print(f"\n{'='*60}")
        print("Phase 1: Supervised Pretraining from Dataset")
        print(f"{'='*60}")
        
        if not os.path.exists(data_path):
            print(f"Dataset not found at {data_path}, skipping pretraining")
            return
        
        data = np.load(data_path)
        obs_data, acts_data, masks_data = data['obs'], data['acts'], data['masks']
        print(f"Loaded {len(obs_data)} samples")
        
        if self.use_wandb:
            wandb.log({"phase1/dataset_size": len(obs_data)})
        
        optimizer = optim.Adam(self.agent.parameters(), lr=self.config.pretrain_lr)
        criterion = nn.CrossEntropyLoss()
        
        batch_size, num_epochs = 128, 20
        indices = np.arange(len(obs_data))
        
        for epoch in range(num_epochs):
            np.random.shuffle(indices)
            total_loss, correct, total = 0, 0, 0
            
            for start in range(0, len(obs_data), batch_size):
                batch_idx = indices[start:start + batch_size]
                
                obs_t = torch.from_numpy(obs_data[batch_idx]).to(self.device)
                mask_t = torch.from_numpy(masks_data[batch_idx]).to(self.device)
                target_t = torch.from_numpy(acts_data[batch_idx]).to(self.device)
                
                self.agent.train()
                features = self.agent._extract_features(obs_t)
                logits = self.agent.actor_head(features)
                logits = torch.where(mask_t.bool(), logits, torch.tensor(-1e9, device=self.device))
                
                loss = criterion(logits, target_t)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == target_t).sum().item()
                total += len(target_t)
            
            accuracy = correct / total
            avg_loss = total_loss / (len(obs_data) // batch_size)
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1%}")
            
            if self.use_wandb:
                wandb.log({"phase1/epoch": epoch+1, "phase1/loss": avg_loss, "phase1/accuracy": accuracy})
        
        self.save_model(f"{self.config.model_dir}/ppo_chess_pretrained.pt")
        print("Pretraining complete!")
    
    def curriculum_train(self, data_path: str) -> float:
        """
        Curriculum Learning 전체 파이프라인 실행
        
        Args:
            data_path: 프리트레이닝 데이터셋 경로
        
        Returns:
            최종 승률
        """
        cfg = self.config

        if 1 not in cfg.skip_phases:
            # Phase 1: 데이터셋 기반 프리트레이닝
            pretrain_path = f"{cfg.model_dir}/ppo_chess_phase1.pt"
            if os.path.exists(pretrain_path):
                print(f"Loading pretrained model from {pretrain_path}")
                self.load_model(pretrain_path)
            else:
                self.pretrain_from_dataset(data_path)
                self.save_model(pretrain_path)
        
        if 2 not in cfg.skip_phases:
            # Phase 2: Random 상대와 학습
            phase2_path = f"{cfg.model_dir}/ppo_chess_phase2.pt"
            if os.path.exists(phase2_path):
                print(f"Loading Phase 2 model from {phase2_path}")
                self.load_model(phase2_path)
            else:
                phase2_timesteps = int(cfg.total_timesteps * cfg.phase2_ratio)
                self.train_phase("random", phase2_timesteps, "Phase2")
                self.save_model(phase2_path)
        
        # Phase 3: Stockfish 상대와 학습
        phase3_timesteps = int(cfg.total_timesteps * cfg.phase3_ratio)
        final_win_rate = self.train_phase("stockfish", phase3_timesteps, "Phase3")
    
        self.save_model(f"{cfg.model_dir}/ppo_chess_phase3.pt")
        
        return final_win_rate
    
    def evaluate(self, num_games: int = None) -> float:
        """Stockfish 상대로 평가"""
        num_games = num_games or self.config.eval_episodes
        
        self.agent.eval()
        eval_env = ChessEnvWrapper(
            opponent_type="stockfish",
            stockfish_path=self.config.stockfish_path,
            stockfish_skill=self.config.stockfish_skill,
            stockfish_time_limit=self.config.stockfish_time_limit
        )
        
        wins, draws, losses = 0, 0, 0
        
        for game in range(num_games):
            obs, _ = eval_env.reset(options={"play_as_white": game % 2 == 0})
            done = False
            
            while not done:
                obs_t = torch.from_numpy(obs["observation"]).unsqueeze(0).to(self.device)
                mask_t = torch.from_numpy(obs["action_mask"].astype(np.float32)).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(obs_t, action_mask=mask_t)
                
                obs, reward, terminated, truncated, _ = eval_env.step(action.item())
                done = terminated or truncated
                
                if done:
                    if reward > 0:
                        wins += 1
                    elif reward < 0:
                        losses += 1
                    else:
                        draws += 1
        
        eval_env.close()
        self.agent.train()
        
        win_rate = wins / num_games
        lose_rate = losses / num_games
        print(f"  Eval: W:{wins} D:{draws} L:{losses} ({win_rate:.1%}) ({lose_rate:.1%})")
        return win_rate
    
    def save_model(self, path: str):
        """모델 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'global_step': self.global_step,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        print(f"Model loaded from {path}")


# =============================================================================
# 메인 함수
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="PPO Chess Agent vs Stockfish")
    parser.add_argument("--mode", type=str, default="curriculum", 
                        choices=["curriculum", "eval", "generate-data"],
                        help="Mode: curriculum (train), eval, or generate-data")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Model path for evaluation")
    parser.add_argument("--stockfish-path", type=str, default="/usr/games/stockfish")
    parser.add_argument("--stockfish-skill", type=int, default=0)
    parser.add_argument("--stockfish-time", type=float, default=0.01)
    parser.add_argument("--total-timesteps", type=int, default=16_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--data-path", type=str, default="data/stockfish_pretrain_data_v4.npz")
    parser.add_argument("--use-wandb", action="store_true")
    # 데이터 생성 관련 인자
    parser.add_argument("--num-games", type=int, default=1000,
                        help="Number of games for dataset generation")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel workers for dataset generation")
    parser.add_argument("--data-skill", type=int, default=0,
                        help="Stockfish skill level for dataset generation (0-20)")
    parser.add_argument("--data-time", type=float, default=0.01,
                        help="Time limit per move for dataset generation")
    parser.add_argument("--skip-phases", type=int, nargs='*', default=[],
                        help="Phases to skip during training (1, 2, 3)")
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 설정 생성
    config = Config(
        stockfish_path=args.stockfish_path,
        stockfish_skill=args.stockfish_skill,
        stockfish_time_limit=args.stockfish_time,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        skip_phases=tuple(args.skip_phases)
    )
    
    if args.mode == "generate-data":
        # 데이터셋 생성 모드
        generate_stockfish_dataset(
            output_path=args.data_path,
            num_games=args.num_games,
            num_workers=args.num_workers,
            stockfish_path=args.stockfish_path,
            stockfish_skill=args.data_skill,
            stockfish_time_limit=args.data_time,
            verbose=True
        )
        return
    
    if args.mode == "curriculum":
        # WandB 초기화
        if args.use_wandb:
            run_name = datetime.now().strftime("%y-%m-%d-%H-%M")
            wandb.init(
                project="Reinforcement-Learning",
                name=run_name,
                config=asdict(config),
                save_code=True,
            )
            print(f"WandB run: {run_name}")
        
        # 학습
        trainer = PPOTrainer(config, device, use_wandb=args.use_wandb)
        final_win_rate = trainer.curriculum_train(args.data_path)
        
        # 결과 출력
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Final Win Rate vs Stockfish: {final_win_rate:.1%}")
        
        if final_win_rate >= 0.5:
            print("✓ SUCCESS: >50% win rate achieved!")
        else:
            print("✗ Target not reached. Consider longer training.")
        
        if args.use_wandb:
            wandb.log({"final/win_rate": final_win_rate, "final/success": final_win_rate >= 0.5})
            wandb.finish()
    
    elif args.mode == "eval":
        # 평가 모드
        model_path = args.model_path
        if model_path is None:
            for p in [f"{config.model_dir}/ppo_chess_best.pt", f"{config.model_dir}/ppo_chess_final.pt"]:
                if os.path.exists(p):
                    model_path = p
                    break
        
        if model_path is None or not os.path.exists(model_path):
            print("No model found. Please specify --model-path or train first.")
            return
        
        trainer = PPOTrainer(config, device)
        trainer.load_model(model_path)
        
        print(f"\nEvaluating: {model_path}")
        print(f"Against Stockfish (skill={config.stockfish_skill}, time={config.stockfish_time_limit}s)")
        
        win_rate = trainer.evaluate(num_games=args.eval_episodes)
        
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULT: {win_rate:.1%} win rate")
        print("✓ SUCCESS!" if win_rate >= 0.5 else "✗ Below 50% target")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
