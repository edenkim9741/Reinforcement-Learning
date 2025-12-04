#!/usr/bin/env python3
"""
PPO Agent for Chess (PettingZoo chess_v6) vs Stockfish

목표: Stockfish Level 0 (time_limit=0.01s)에 대해 50% 이상 승률 달성

구성:
1. 환경 래퍼 (Gymnasium 호환)
2. ResNet 기반 PPO 에이전트 (Action Masking 지원)
3. Self-play 및 Stockfish 대전 훈련
4. 평가 기능

사용법:
    python ppo_chess_vs_stockfish.py --mode train
    python ppo_chess_vs_stockfish.py --mode eval --model-path models/ppo_chess_final.pt
"""

import os
import sys
import time
import random
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple, List
from multiprocessing import Process, Pipe
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import wandb

# TensorBoard - optional
HAS_TENSORBOARD = False
SummaryWriter = None
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except Exception:
    pass

import gymnasium as gym
from gymnasium import spaces
import chess
import chess.engine
from pettingzoo.classic import chess_v6
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    # Environment
    stockfish_path: str = "/usr/games/stockfish"
    stockfish_skill: int = 0
    stockfish_time_limit: float = 0.01
    
    # Training
    seed: int = 42
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    num_envs: int = 4
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
    
    # Model
    num_res_blocks: int = 4
    channels: int = 128
    
    # Eval
    eval_episodes: int = 20
    eval_interval: int = 10  # iterations
    
    # Paths
    model_dir: str = "models"
    log_dir: str = "runs"
    
    # Pretraining
    pretrain_episodes: int = 1000
    pretrain_lr: float = 1e-3
    
    # Mixed training
    self_play_ratio: float = 0.5  # Self-play vs Stockfish ratio
    use_wandb: bool = False


# =============================================================================
# Move Encoding/Decoding Utilities
# =============================================================================
def square_to_coord(s: int) -> Tuple[int, int]:
    """Convert square index to (col, row)"""
    return (s % 8, s // 8)


def coord_diff(c1: Tuple[int, int], c2: Tuple[int, int]) -> Tuple[int, int]:
    return (c2[0] - c1[0], c2[1] - c1[1])


def sign(v: int) -> int:
    return -1 if v < 0 else (1 if v > 0 else 0)


def mirror_move(move: chess.Move) -> chess.Move:
    """Mirror a move for black's perspective"""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def get_queen_dir(diff: Tuple[int, int]) -> Tuple[int, int]:
    dx, dy = diff
    magnitude = max(abs(dx), abs(dy)) - 1
    counter = 0
    for x in range(-1, 2):
        for y in range(-1, 2):
            if x == 0 and y == 0:
                continue
            if x == sign(dx) and y == sign(dy):
                return magnitude, counter
            counter += 1
    return 0, 0


def get_queen_plane(diff: Tuple[int, int]) -> int:
    mag, counter = get_queen_dir(diff)
    return mag * 8 + counter


def get_knight_dir(diff: Tuple[int, int]) -> int:
    dx, dy = diff
    counter = 0
    for x in range(-2, 3):
        for y in range(-2, 3):
            if abs(x) + abs(y) == 3:
                if dx == x and dy == y:
                    return counter
                counter += 1
    return 0


def is_knight_move(diff: Tuple[int, int]) -> bool:
    dx, dy = diff
    return abs(dx) + abs(dy) == 3 and 1 <= abs(dx) <= 2


def get_pawn_promotion_move(diff: Tuple[int, int]) -> int:
    return diff[0] + 1


def get_pawn_promotion_num(promotion: int) -> int:
    if promotion == chess.KNIGHT:
        return 0
    elif promotion == chess.BISHOP:
        return 1
    else:
        return 2


def get_move_plane(move: chess.Move) -> int:
    source = move.from_square
    dest = move.to_square
    diff = coord_diff(square_to_coord(source), square_to_coord(dest))
    
    QUEEN_MOVES = 56
    KNIGHT_MOVES = 8
    QUEEN_OFFSET = 0
    KNIGHT_OFFSET = QUEEN_MOVES
    UNDER_OFFSET = KNIGHT_OFFSET + KNIGHT_MOVES
    
    if is_knight_move(diff):
        return KNIGHT_OFFSET + get_knight_dir(diff)
    elif move.promotion is not None and move.promotion != chess.QUEEN:
        return (UNDER_OFFSET + 
                3 * get_pawn_promotion_move(diff) + 
                get_pawn_promotion_num(move.promotion))
    else:
        return QUEEN_OFFSET + get_queen_plane(diff)


def encode_move(move: chess.Move, should_mirror: bool = False) -> int:
    """Convert chess.Move to PettingZoo action index"""
    if should_mirror:
        move = mirror_move(move)
    
    coord = square_to_coord(move.from_square)
    plane = get_move_plane(move)
    return (coord[0] * 8 + coord[1]) * 73 + plane


def decode_move(action: int, board: chess.Board, is_black: bool = False) -> Optional[chess.Move]:
    """Convert PettingZoo action index to chess.Move (for debugging)"""
    plane = action % 73
    pos = action // 73
    col, row = pos // 8, pos % 8
    
    # This is a simplified decoder - mainly for debugging
    # The actual legal moves are checked via action mask
    return None


# =============================================================================
# Reward Shaping Utilities
# =============================================================================
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0
}


def count_material(board: chess.Board, color: chess.Color) -> float:
    """Count material for a given color"""
    total = 0.0
    for piece_type, value in PIECE_VALUES.items():
        total += len(board.pieces(piece_type, color)) * value
    return total


def get_material_balance(board: chess.Board, our_color: chess.Color) -> float:
    """Get material balance from our perspective"""
    our_material = count_material(board, our_color)
    opp_material = count_material(board, not our_color)
    return (our_material - opp_material) / 39.0  # Normalize by max material difference


def compute_intermediate_reward(board_before: chess.Board, 
                                board_after: chess.Board, 
                                our_color: chess.Color) -> float:
    """
    Compute intermediate reward based on:
    - Material gain/loss
    - Checkmate threats
    - Position control
    """
    reward = 0.0
    
    # Material change
    balance_before = get_material_balance(board_before, our_color)
    balance_after = get_material_balance(board_after, our_color)
    material_change = balance_after - balance_before
    reward += material_change * 0.1  # Small reward for material gain
    
    # Check bonus
    if board_after.is_check():
        if board_after.turn != our_color:  # We gave check
            reward += 0.02
    
    return reward


# =============================================================================
# Chess Environment Wrapper
# =============================================================================
class ChessEnvWrapper(gym.Env):
    """
    Gymnasium-compatible wrapper for PettingZoo chess_v6.
    Plays against Stockfish as the opponent.
    """
    
    def __init__(self, 
                 stockfish_path: str = "/usr/games/stockfish",
                 stockfish_skill: int = 0,
                 stockfish_time_limit: float = 0.01,
                 render_mode: Optional[str] = None,
                 reward_shaping: bool = True):
        super().__init__()
        
        play_as_white = random.choice([True, False])
        self.play_as_white = play_as_white
        self.agent_player = "player_0" if play_as_white else "player_1"
        self.stockfish_player = "player_1" if play_as_white else "player_0"
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping
        
        # Create PettingZoo environment
        self.aec_env = chess_v6.env(render_mode=render_mode)
        self.aec_env.reset()
        
        # Setup observation and action spaces
        raw_obs_space = self.aec_env.observation_space("player_0")["observation"]
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=0, high=1, shape=raw_obs_space.shape, dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0, high=1, shape=(4672,), dtype=np.int8
            )
        })
        self.action_space = spaces.Discrete(4672)
        
        # Initialize Stockfish
        self.stockfish_path = stockfish_path
        self.stockfish_skill = stockfish_skill
        self.stockfish_time_limit = stockfish_time_limit
        self.engine = None
        self._init_stockfish()
    
    def _init_stockfish(self):
        """Initialize Stockfish engine"""
        try:
            if self.engine is not None:
                self.engine.quit()
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self.engine.configure({"Skill Level": self.stockfish_skill})
        except FileNotFoundError:
            raise FileNotFoundError(f"Stockfish not found at {self.stockfish_path}")
    
    def _get_stockfish_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get Stockfish's move"""
        try:
            result = self.engine.play(
                board, 
                chess.engine.Limit(time=self.stockfish_time_limit)
            )
            return result.move
        except Exception as e:
            print(f"Stockfish error: {e}")
            return None
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Randomly choose side
        if options and "play_as_white" in options:
            self.play_as_white = options["play_as_white"]
        else:
            self.play_as_white = random.choice([True, False])
        
        self.agent_player = "player_0" if self.play_as_white else "player_1"
        self.stockfish_player = "player_1" if self.play_as_white else "player_0"
        
        self.aec_env.reset(seed=seed)
        
        # If Stockfish moves first (we're black), let it move
        if not self.play_as_white:
            self._stockfish_turn()
        
        obs, info = self._get_obs_and_info()
        return obs, info
    
    def _stockfish_turn(self):
        """Execute Stockfish's turn"""
        if self.aec_env.agent_selection != self.stockfish_player:
            return
        
        observation, reward, termination, truncation, info = self.aec_env.last()
        
        if termination or truncation:
            self.aec_env.step(None)
            return
        
        board = self.aec_env.unwrapped.board
        stockfish_move = self._get_stockfish_move(board)
        
        if stockfish_move is None:
            # Random fallback
            action_mask = observation["action_mask"]
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                action = 0
        else:
            is_black = (self.stockfish_player == "player_1")
            action = encode_move(stockfish_move, should_mirror=is_black)
            
            # Verify action is valid
            action_mask = observation["action_mask"]
            if action_mask[action] != 1:
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = 0
        
        self.aec_env.step(action)
    
    def _get_obs_and_info(self):
        """Get observation for our agent"""
        current_agent = self.aec_env.agent_selection
        observation, reward, termination, truncation, info = self.aec_env.last()
        
        obs = {
            "observation": observation["observation"].astype(np.float32),
            "action_mask": observation["action_mask"].astype(np.int8)
        }
        
        info_dict = {
            "termination": termination,
            "truncation": truncation,
            "action_mask": observation["action_mask"]
        }
        
        return obs, info_dict
    
    def step(self, action: int):
        # Save board state before move for reward shaping
        board_before = self.aec_env.unwrapped.board.copy() if self.reward_shaping else None
        our_color = chess.WHITE if self.play_as_white else chess.BLACK
        
        # Execute our action
        observation, reward, termination, truncation, info = self.aec_env.last()
        
        if termination or truncation:
            obs, info_dict = self._get_obs_and_info()
            return obs, 0.0, True, False, info_dict
        
        self.aec_env.step(action)
        
        # Compute intermediate reward
        intermediate_reward = 0.0
        if self.reward_shaping and board_before is not None:
            board_after = self.aec_env.unwrapped.board
            intermediate_reward = compute_intermediate_reward(board_before, board_after, our_color)
        
        # Check if game ended after our move
        observation, reward, termination, truncation, info = self.aec_env.last()
        
        if termination or truncation:
            # Get final reward
            rewards = self.aec_env.rewards
            final_reward = rewards.get(self.agent_player, 0.0)
            obs, info_dict = self._get_obs_and_info()
            return obs, final_reward, True, False, info_dict
        
        # Stockfish's turn
        self._stockfish_turn()
        
        # Check if game ended after Stockfish's move
        observation, reward, termination, truncation, info = self.aec_env.last()
        
        if termination or truncation:
            rewards = self.aec_env.rewards
            final_reward = rewards.get(self.agent_player, 0.0)
            obs, info_dict = self._get_obs_and_info()
            return obs, final_reward, True, False, info_dict
        
        obs, info_dict = self._get_obs_and_info()
        return obs, intermediate_reward, False, False, info_dict
    
    def render(self):
        if self.render_mode:
            return self.aec_env.render()
        return None
    
    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except:
                pass
        self.aec_env.close()


class RandomOpponentEnvWrapper(gym.Env):
    """
    Environment where the agent plays against a random opponent.
    This is useful for initial training before facing Stockfish.
    """
    
    def __init__(self, render_mode: Optional[str] = None, reward_shaping: bool = True):
        super().__init__()
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping
        self.aec_env = chess_v6.env(render_mode=render_mode)
        self.aec_env.reset()
        
        raw_obs_space = self.aec_env.observation_space("player_0")["observation"]
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=0, high=1, shape=raw_obs_space.shape, dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0, high=1, shape=(4672,), dtype=np.int8
            )
        })
        self.action_space = spaces.Discrete(4672)
        
        self.play_as_white = True
        self.agent_player = "player_0"
        self.random_player = "player_1"
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Randomly choose side
        self.play_as_white = random.choice([True, False])
        self.agent_player = "player_0" if self.play_as_white else "player_1"
        self.random_player = "player_1" if self.play_as_white else "player_0"
        
        self.aec_env.reset(seed=seed)
        
        # If random opponent moves first, let it move
        if not self.play_as_white:
            self._random_turn()
        
        obs, info = self._get_obs_and_info()
        return obs, info
    
    def _random_turn(self):
        """Execute random opponent's turn"""
        if self.aec_env.agent_selection != self.random_player:
            return
        
        observation, reward, termination, truncation, info = self.aec_env.last()
        
        if termination or truncation:
            self.aec_env.step(None)
            return
        
        action_mask = observation["action_mask"]
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            action = 0
        
        self.aec_env.step(action)
    
    def _get_obs_and_info(self):
        observation, reward, termination, truncation, info = self.aec_env.last()
        
        obs = {
            "observation": observation["observation"].astype(np.float32),
            "action_mask": observation["action_mask"].astype(np.int8)
        }
        
        info_dict = {
            "termination": termination,
            "truncation": truncation,
            "action_mask": observation["action_mask"]
        }
        
        return obs, info_dict
    
    def step(self, action: int):
        board_before = self.aec_env.unwrapped.board.copy() if self.reward_shaping else None
        our_color = chess.WHITE if self.play_as_white else chess.BLACK
        
        observation, reward, termination, truncation, info = self.aec_env.last()
        
        if termination or truncation:
            obs, info_dict = self._get_obs_and_info()
            return obs, 0.0, True, False, info_dict
        
        self.aec_env.step(action)
        
        intermediate_reward = 0.0
        if self.reward_shaping and board_before is not None:
            board_after = self.aec_env.unwrapped.board
            intermediate_reward = compute_intermediate_reward(board_before, board_after, our_color)
        
        observation, reward, termination, truncation, info = self.aec_env.last()
        
        if termination or truncation:
            rewards = self.aec_env.rewards
            final_reward = rewards.get(self.agent_player, 0.0)
            obs, info_dict = self._get_obs_and_info()
            return obs, final_reward, True, False, info_dict
        
        # Random opponent's turn
        self._random_turn()
        
        observation, reward, termination, truncation, info = self.aec_env.last()
        
        if termination or truncation:
            rewards = self.aec_env.rewards
            final_reward = rewards.get(self.agent_player, 0.0)
            obs, info_dict = self._get_obs_and_info()
            return obs, final_reward, True, False, info_dict
        
        obs, info_dict = self._get_obs_and_info()
        return obs, intermediate_reward, False, False, info_dict
    
    def render(self):
        if self.render_mode:
            return self.aec_env.render()
        return None
    
    def close(self):
        self.aec_env.close()




class ThreadedVecEnv:
    """
    Vectorized environment using ThreadPoolExecutor.
    Faster startup than SubprocVecEnv but limited by GIL for CPU-bound tasks.
    Good for I/O-bound tasks like Stockfish communication.
    """
    
    def __init__(self, env_fns, max_workers=None):
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self.max_workers = max_workers or self.num_envs
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Get spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def _step_env(self, args):
        """Step a single environment"""
        env, action = args
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        return obs, reward, terminated, truncated, info
    
    def step(self, actions):
        """Step all environments with given actions in parallel"""
        futures = list(self.executor.map(self._step_env, zip(self.envs, actions)))
        obs_list, rewards, terminateds, truncateds, infos = zip(*futures)
        return list(obs_list), np.array(rewards), np.array(terminateds), np.array(truncateds), list(infos)
    
    def reset(self, seed=None, options=None):
        """Reset all environments"""
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed, options=options)
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()
        self.executor.shutdown(wait=True)


# =============================================================================
# Neural Network Model
# =============================================================================
class ResBlock(nn.Module):
    """Residual block for the chess agent"""
    
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
        out = out + identity
        return self.relu(out)


class ChessAgent(nn.Module):
    """
    ResNet-based PPO Agent for Chess with action masking.
    
    Architecture:
    - Input: 8x8x111 observation (PettingZoo chess format)
    - Backbone: Initial conv + N residual blocks
    - Actor head: Policy network with action masking
    - Critic head: Value network
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int] = (8, 8, 111), 
                 n_actions: int = 4672,
                 num_res_blocks: int = 4,
                 channels: int = 128):
        super().__init__()
        
        self.h, self.w, self.c = obs_shape
        self.n_actions = n_actions
        
        # Initial convolution
        self.start_block = nn.Sequential(
            nn.Conv2d(self.c, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual backbone
        self.backbone = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * self.h * self.w, n_actions)
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * self.h * self.w, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
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
        """Extract features from observation"""
        # obs: (batch, h, w, c) -> (batch, c, h, w)
        x = obs.permute(0, 3, 1, 2)
        x = self.start_block(x)
        x = self.backbone(x)
        return x
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value"""
        features = self._extract_features(obs)
        return self.critic_head(features)
    
    def get_action_and_value(self, 
                             obs: torch.Tensor, 
                             action: Optional[torch.Tensor] = None,
                             action_mask: Optional[torch.Tensor] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Args:
            obs: Observation tensor (batch, h, w, c)
            action: Optional action to evaluate
            action_mask: Boolean mask for valid actions (batch, n_actions)
        
        Returns:
            action, log_prob, entropy, value
        """
        features = self._extract_features(obs)
        
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        
        # Apply action masking
        if action_mask is not None:
            # Handle all-invalid masks (shouldn't happen in normal play)
            is_all_invalid = action_mask.sum(dim=1) == 0
            if is_all_invalid.any():
                action_mask = action_mask.clone()
                action_mask[is_all_invalid, 0] = 1.0
            
            # Mask invalid actions with very negative logits
            logits = torch.where(
                action_mask.bool(),
                logits,
                torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
            )
        
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value


# =============================================================================
# PPO Training
# =============================================================================
class PPOTrainer:
    """PPO Trainer for Chess Agent with Parallel Environment Support"""
    
    def __init__(self, config: Config, device: torch.device, use_parallel: bool = True, use_wandb: bool = False):
        self.config = config
        self.device = device
        self.use_parallel = use_parallel
        self.use_wandb = use_wandb
        
        # Create environments
        if use_parallel:
            # Use ThreadedVecEnv for parallel Stockfish calls
            env_fns = [
                lambda i=i: ChessEnvWrapper(
                    stockfish_path=config.stockfish_path,
                    stockfish_skill=config.stockfish_skill,
                    stockfish_time_limit=config.stockfish_time_limit
                )
                for i in range(config.num_envs)
            ]
            self.vec_env = ThreadedVecEnv(env_fns)
            self.envs = None  # Not used in parallel mode
            
            # Get observation shape
            sample_obs_list, _ = self.vec_env.reset(seed=config.seed)
            sample_obs = sample_obs_list[0]
        else:
            # Use sequential environments
            self.vec_env = None
            self.envs = [
                ChessEnvWrapper(
                    stockfish_path=config.stockfish_path,
                    stockfish_skill=config.stockfish_skill,
                    stockfish_time_limit=config.stockfish_time_limit
                )
                for _ in range(config.num_envs)
            ]
            sample_obs, _ = self.envs[0].reset()
        
        self.obs_shape = sample_obs["observation"].shape
        self.n_actions = 4672
        
        # Create agent
        self.agent = ChessAgent(
            obs_shape=self.obs_shape,
            n_actions=self.n_actions,
            num_res_blocks=config.num_res_blocks,
            channels=config.channels
        ).to(device)
        
        self.optimizer = optim.Adam(
            self.agent.parameters(), 
            lr=config.learning_rate, 
            eps=1e-5
        )
        
        # Compute derived values
        self.batch_size = config.num_envs * config.num_steps
        self.minibatch_size = self.batch_size // config.num_minibatches
        self.num_iterations = config.total_timesteps // self.batch_size
        
        
        # Storage for rollout
        self.obs = torch.zeros(
            (config.num_steps, config.num_envs) + self.obs_shape
        ).to(device)
        self.actions = torch.zeros(
            (config.num_steps, config.num_envs), dtype=torch.long
        ).to(device)
        self.logprobs = torch.zeros(
            (config.num_steps, config.num_envs)
        ).to(device)
        self.rewards = torch.zeros(
            (config.num_steps, config.num_envs)
        ).to(device)
        self.dones = torch.zeros(
            (config.num_steps, config.num_envs)
        ).to(device)
        self.values = torch.zeros(
            (config.num_steps, config.num_envs)
        ).to(device)
        self.action_masks = torch.zeros(
            (config.num_steps, config.num_envs, self.n_actions)
        ).to(device)
        
        # Initialize environment states
        self.next_obs = torch.zeros((config.num_envs,) + self.obs_shape).to(device)
        self.next_done = torch.zeros(config.num_envs).to(device)
        self.next_action_mask = torch.zeros((config.num_envs, self.n_actions)).to(device)
        
        if use_parallel:
            obs_list, _ = self.vec_env.reset(seed=config.seed)
            for i, obs in enumerate(obs_list):
                self.next_obs[i] = torch.from_numpy(obs["observation"])
                self.next_action_mask[i] = torch.from_numpy(obs["action_mask"].astype(np.float32))
        else:
            for i, env in enumerate(self.envs):
                obs, info = env.reset(seed=config.seed + i)
                self.next_obs[i] = torch.from_numpy(obs["observation"])
                self.next_action_mask[i] = torch.from_numpy(obs["action_mask"].astype(np.float32))
        
        # Statistics
        self.global_step = 0
        self.start_time = time.time()
        self.episode_returns = []
        self.episode_lengths = []
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
    
    def collect_rollout(self):
        """Collect rollout data from environments (parallel or sequential)"""
        for step in range(self.config.num_steps):
            self.global_step += self.config.num_envs
            
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done
            self.action_masks[step] = self.next_action_mask
            
            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(
                    self.next_obs,
                    action_mask=self.next_action_mask
                )
                self.values[step] = value
            
            self.actions[step] = action
            self.logprobs[step] = logprob
            
            if self.use_parallel:
                # Parallel execution using ThreadedVecEnv
                actions_np = action.cpu().numpy()
                obs_list, rewards, terminateds, truncateds, infos = self.vec_env.step(actions_np)
                
                for i in range(self.config.num_envs):
                    obs = obs_list[i]
                    reward = rewards[i]
                    done = terminateds[i] or truncateds[i]
                    
                    self.rewards[step, i] = reward
                    
                    if done:
                        if reward > 0:
                            self.win_count += 1
                        elif reward < 0:
                            self.loss_count += 1
                        else:
                            self.draw_count += 1
                    
                    self.next_obs[i] = torch.from_numpy(obs["observation"]).to(self.device)
                    self.next_action_mask[i] = torch.from_numpy(
                        obs["action_mask"].astype(np.float32)
                    ).to(self.device)
                    self.next_done[i] = float(done)
            else:
                # Sequential execution
                for i, env in enumerate(self.envs):
                    obs, reward, terminated, truncated, info = env.step(action[i].item())
                    done = terminated or truncated
                    
                    self.rewards[step, i] = reward
                    
                    if done:
                        if reward > 0:
                            self.win_count += 1
                        elif reward < 0:
                            self.loss_count += 1
                        else:
                            self.draw_count += 1
                        obs, info = env.reset()
                    
                    self.next_obs[i] = torch.from_numpy(obs["observation"]).to(self.device)
                    self.next_action_mask[i] = torch.from_numpy(
                        obs["action_mask"].astype(np.float32)
                    ).to(self.device)
                    self.next_done[i] = float(done)
    
    def compute_advantages(self):
        """Compute advantages using GAE"""
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            
            for t in reversed(range(self.config.num_steps)):
                if t == self.config.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                
                delta = (self.rewards[t] + 
                        self.config.gamma * nextvalues * nextnonterminal - 
                        self.values[t])
                advantages[t] = lastgaelam = (
                    delta + 
                    self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
                )
            
            returns = advantages + self.values
        
        return advantages, returns
    
    def update(self, advantages: torch.Tensor, returns: torch.Tensor):
        """Update policy and value networks"""
        # Flatten batch
        b_obs = self.obs.reshape((-1,) + self.obs_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        b_action_masks = self.action_masks.reshape(-1, self.n_actions)
        
        # Training loop
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_action_masks[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()
                    )
                
                mb_advantages = b_advantages[mb_inds]
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                break
        
        return pg_loss.item(), v_loss.item(), entropy_loss.item(), np.mean(clipfracs)
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.num_iterations} iterations...")
        print(f"Batch size: {self.batch_size}, Minibatch size: {self.minibatch_size}")
        
        bar = tqdm(range(1, self.num_iterations + 1), desc="Training", unit="it")
        for iteration in bar:
            # Learning rate annealing
            if self.config.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            
            # Collect rollout
            self.collect_rollout()
            
            # Compute advantages
            advantages, returns = self.compute_advantages()
            
            # Update networks
            pg_loss, v_loss, entropy_loss, clipfrac = self.update(advantages, returns)
            
            # Logging
            sps = int(self.global_step / (time.time() - self.start_time))
            
            total_games = self.win_count + self.loss_count + self.draw_count
            
            if iteration % 10 == 0:
                total_games = self.win_count + self.loss_count + self.draw_count
                win_rate = self.win_count / max(total_games, 1)
                print(f"Iter {iteration}/{self.num_iterations} | "
                      f"Step {self.global_step} | "
                      f"Win: {self.win_count}, Loss: {self.loss_count}, Draw: {self.draw_count} | "
                      f"WinRate: {win_rate:.2%} | "
                      f"SPS: {sps}")
                
                # WandB logging
                if self.use_wandb:
                    wandb.log({
                        "phase3/iteration": iteration,
                        "phase3/global_step": self.global_step,
                        "phase3/win_count": self.win_count,
                        "phase3/loss_count": self.loss_count,
                        "phase3/draw_count": self.draw_count,
                        "phase3/total_games": total_games,
                        "phase3/win_rate": win_rate,
                        "phase3/policy_loss": pg_loss,
                        "phase3/value_loss": v_loss,
                        "phase3/entropy": entropy_loss,
                        "phase3/clipfrac": clipfrac,
                        "phase3/sps": sps,
                        "phase3/learning_rate": self.optimizer.param_groups[0]["lr"],
                    })

                if win_rate >= 0.7:
                    print(f"  [SAVE] Saving model at iteration {iteration} with win rate {win_rate:.2%}")
                    self.save_model(f"{self.config.model_dir}/ppo_chess_iter{iteration}_winrate{int(win_rate*100)}.pt")

                    return win_rate

                




                
                # Reset counts
                self.win_count = 0
                self.loss_count = 0
                self.draw_count = 0
            
            # Evaluation
            if iteration % self.config.eval_interval == 0:
                eval_win_rate = self.evaluate()
                if self.use_wandb:
                    wandb.log({
                        "phase3/eval_win_rate": eval_win_rate,
                        "phase3/eval_iteration": iteration,
                    })
                print(f"  [EVAL] Win rate vs Stockfish: {eval_win_rate:.2%}")
                
                # Save if good performance
                if eval_win_rate >= 0.5:
                    self.save_model(f"{self.config.model_dir}/ppo_chess_best.pt")
                    print(f"  [SAVE] Model saved with {eval_win_rate:.2%} win rate!")
        
        # Save final model
        self.save_model(f"{self.config.model_dir}/ppo_chess_final.pt")
        print("Training complete!")
        
        # Final evaluation
        final_win_rate = self.evaluate(num_games=50)
        print(f"Final evaluation: {final_win_rate:.2%} win rate vs Stockfish")
        
        self.close()
        return final_win_rate
    
    def evaluate(self, num_games: int = None) -> float:
        """Evaluate agent against Stockfish"""
        if num_games is None:
            num_games = self.config.eval_episodes
        
        self.agent.eval()
        
        eval_env = ChessEnvWrapper(
            stockfish_path=self.config.stockfish_path,
            stockfish_skill=self.config.stockfish_skill,
            stockfish_time_limit=self.config.stockfish_time_limit
        )
        
        wins = 0
        draws = 0
        losses = 0
        
        for game in range(num_games):
            obs, info = eval_env.reset(options={"play_as_white": game % 2 == 0})
            done = False
            
            while not done:
                obs_tensor = torch.from_numpy(obs["observation"]).unsqueeze(0).to(self.device)
                mask_tensor = torch.from_numpy(
                    obs["action_mask"].astype(np.float32)
                ).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(
                        obs_tensor, action_mask=mask_tensor
                    )
                
                obs, reward, terminated, truncated, info = eval_env.step(action.item())
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
        
        total = wins + draws + losses
        win_rate = wins / total if total > 0 else 0
        
        print(f"  Eval results: W:{wins} D:{draws} L:{losses} ({win_rate:.2%})")
        return win_rate
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'draw_count': self.draw_count,
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        print(f"Loaded model from {path}")
    
    def close(self):
        """Clean up resources"""
        if self.use_parallel and self.vec_env is not None:
            self.vec_env.close()
        elif self.envs is not None:
            for env in self.envs:
                env.close()


# =============================================================================
# Dataset-based Pretraining
# =============================================================================
def pretrain_from_dataset(config: Config, device: torch.device, data_path: str = "data/stockfish_pretrain_data_v4.npz", use_wandb: bool = False):
    """
    Pretrain the agent using pre-collected Stockfish data.
    This is more efficient than online data collection.
    """
    print("=" * 60)
    print("Pretraining from dataset")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return None
    
    # Load dataset
    data = np.load(data_path)
    obs_data = data['obs']
    acts_data = data['acts']
    masks_data = data['masks']
    
    print(f"Loaded {len(obs_data)} samples from {data_path}")
    
    if use_wandb:
        wandb.log({"phase1/dataset_size": len(obs_data)})
    
    # Get observation shape
    obs_shape = obs_data.shape[1:]
    
    # Create agent
    agent = ChessAgent(
        obs_shape=obs_shape,
        n_actions=4672,
        num_res_blocks=config.num_res_blocks,
        channels=config.channels
    ).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=config.pretrain_lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    batch_size = 128
    num_epochs = 20
    num_samples = len(obs_data)
    
    # Create indices for shuffling
    indices = np.arange(num_samples)
    
    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            obs_tensor = torch.from_numpy(obs_data[batch_indices]).to(device)
            mask_tensor = torch.from_numpy(masks_data[batch_indices]).to(device)
            target_tensor = torch.from_numpy(acts_data[batch_indices]).to(device)
            
            # Forward pass
            agent.train()
            features = agent._extract_features(obs_tensor)
            logits = agent.actor_head(features)
            
            # Apply mask
            logits = torch.where(
                mask_tensor.bool(),
                logits,
                torch.tensor(-1e9, device=device)
            )
            
            loss = criterion(logits, target_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Compute accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == target_tensor).sum().item()
            total += len(target_tensor)
        
        avg_loss = total_loss / num_batches
        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
        
        if use_wandb:
            wandb.log({
                "phase1/epoch": epoch + 1,
                "phase1/loss": avg_loss,
                "phase1/accuracy": accuracy,
            })
    
    # Save pretrained model
    os.makedirs(config.model_dir, exist_ok=True)
    pretrain_path = f"{config.model_dir}/ppo_chess_pretrained.pt"
    torch.save({
        'model_state_dict': agent.state_dict(),
        'config': config,
    }, pretrain_path)
    print(f"Pretrained model saved to {pretrain_path}")
    
    return agent



# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="PPO Chess Agent vs Stockfish")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "eval", "pretrain", "train-random", "curriculum"],
                        help="Mode: train, eval, pretrain, train-random, or curriculum")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model for evaluation or continued training")
    parser.add_argument("--stockfish-path", type=str, default="/usr/games/stockfish",
                        help="Path to Stockfish executable")
    parser.add_argument("--stockfish-skill", type=int, default=0,
                        help="Stockfish skill level (0-20)")
    parser.add_argument("--stockfish-time", type=float, default=0.01,
                        help="Stockfish time limit per move")
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="Use CUDA if available")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--pretrain-episodes", type=int, default=500,
                        help="Number of pretraining episodes")
    parser.add_argument("--use-dataset", action="store_true",
                        help="Use pre-collected dataset for pretraining")
    parser.add_argument("--data-path", type=str, default="data/stockfish_pretrain_data_v4.npz",
                        help="Path to pretraining dataset")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create config
    config = Config(
        stockfish_path=args.stockfish_path,
        stockfish_skill=args.stockfish_skill,
        stockfish_time_limit=args.stockfish_time,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        pretrain_episodes=args.pretrain_episodes,
        use_wandb=args.use_wandb
    )
    
    if args.mode == "pretrain":
        # Just pretrain
        if args.use_dataset:
            agent = pretrain_from_dataset(config, device, args.data_path)
        else:
            agent = pretrain_with_stockfish(config, device)
        if agent is not None:
            print("Pretraining complete!")
    
    elif args.mode == "train":
        # Create trainer
        trainer = PPOTrainer(config, device)
        
        # Load pretrained model if exists
        pretrain_path = f"{config.model_dir}/ppo_chess_pretrained.pt"
        if args.model_path:
            trainer.load_model(args.model_path)
        elif os.path.exists(pretrain_path):
            print(f"Loading pretrained model from {pretrain_path}")
            trainer.load_model(pretrain_path)
        else:
            # Pretrain first
            print("No pretrained model found. Running pretraining first...")
            if args.use_dataset or os.path.exists(args.data_path):
                agent = pretrain_from_dataset(config, device, args.data_path)
            else:
                agent = pretrain_with_stockfish(config, device)
            if agent is not None:
                trainer.agent.load_state_dict(agent.state_dict())
        
        # Train
        final_win_rate = trainer.train()
        
        while final_win_rate >= 0.7:
            print("increse the difficulty level for stockfish")
            config.stockfish_skill = min(20, config.stockfish_skill + 1)
            new_trainer = PPOTrainer(config, device)
            new_trainer.agent.load_state_dict(trainer.agent.state_dict())
            final_win_rate = new_trainer.train()
            trainer = new_trainer


        
        # Report
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Final win rate vs Stockfish (skill={config.stockfish_skill}, "
              f"time={config.stockfish_time_limit}s): {final_win_rate:.2%}")
        
        if final_win_rate >= 0.5:
            print("✓ SUCCESS: Achieved >50% win rate!")
        else:
            print("✗ Target not reached. Consider training longer or tuning hyperparameters.")
    
    elif args.mode == "eval":
        if args.model_path is None:
            # Try to find best model
            model_path = f"{config.model_dir}/ppo_chess_best.pt"
            if not os.path.exists(model_path):
                model_path = f"{config.model_dir}/ppo_chess_final.pt"
            if not os.path.exists(model_path):
                print("No model found. Please specify --model-path or train first.")
                return
        else:
            model_path = args.model_path
        
        # Create trainer and load model
        trainer = PPOTrainer(config, device)
        trainer.load_model(model_path)
        
        # Evaluate
        print(f"\nEvaluating model: {model_path}")
        print(f"Against Stockfish (skill={config.stockfish_skill}, "
              f"time={config.stockfish_time_limit}s)")
        
        win_rate = trainer.evaluate(num_games=args.eval_episodes)
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Win rate: {win_rate:.2%}")
        
        
        if win_rate >= 0.5:
            print("✓ SUCCESS: >50% win rate achieved!")
        else:
            print("✗ Below 50% win rate target.")
        
        trainer.close()
    
    elif args.mode == "curriculum":
        # Curriculum learning: Random -> Stockfish
        print("=" * 60)
        print("CURRICULUM LEARNING")
        print("=" * 60)
        print("Phase 1: Dataset pretraining")
        print("Phase 2: Training vs Random opponent (60% of timesteps)")
        print("Phase 3: Fine-tuning vs Stockfish (40% of timesteps)")
        print("=" * 60)
        
        # Initialize wandb
        if args.use_wandb:
            run_name = datetime.now().strftime("%y-%m-%d-%H-%M")
            wandb.init(
                project="Reinforcement-Learning",
                name=run_name,
                config={
                    **asdict(config),
                    "mode": "curriculum",
                    "phase1_epochs": 20,
                    "phase2_ratio": 0.6,
                    "phase3_ratio": 0.4,
                },
                save_code=True,
            )
            print(f"WandB run initialized: {run_name}")
        
        obs_shape = (8, 8, 111)
        n_actions = 4672
        
        # Phase 1: Pretrain from dataset
        print("\n" + "=" * 60)
        print("Phase 1: Dataset Pretraining")
        print("=" * 60)
        if args.use_wandb:
            wandb.log({"current_phase": 1})
        
        data_path = args.data_path
        if os.path.exists(data_path):
            agent = pretrain_from_dataset(config, device, data_path, use_wandb=args.use_wandb)
        else:
            print("No dataset found, training from scratch")
            agent = ChessAgent(
                obs_shape=obs_shape,
                n_actions=n_actions,
                num_res_blocks=config.num_res_blocks,
                channels=config.channels
            ).to(device)
        
        # Phase 2: Train vs Random (60% of timesteps)
        print("\n" + "=" * 60)
        print("Phase 2: Training vs Random opponent")
        print("=" * 60)
        if args.use_wandb:
            wandb.log({"current_phase": 2})
        
        phase2_timesteps = int(config.total_timesteps * 0.6)
        
        # Create random opponent environments
        random_envs = [RandomOpponentEnvWrapper() for _ in range(config.num_envs)]
        
        optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
        
        batch_size = config.num_envs * config.num_steps
        num_iterations = phase2_timesteps // batch_size
        
        # Initialize states
        next_obs_list = []
        next_mask_list = []
        
        for i, env in enumerate(random_envs):
            obs, info = env.reset(seed=config.seed + i)
            next_obs_list.append(torch.from_numpy(obs["observation"]).to(device))
            next_mask_list.append(torch.from_numpy(obs["action_mask"].astype(np.float32)).to(device))
        
        next_obs = torch.stack(next_obs_list)
        next_mask = torch.stack(next_mask_list)
        next_done = torch.zeros(config.num_envs).to(device)
        
        # Storage
        obs_storage = torch.zeros((config.num_steps, config.num_envs) + obs_shape).to(device)
        actions_storage = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long).to(device)
        logprobs_storage = torch.zeros((config.num_steps, config.num_envs)).to(device)
        rewards_storage = torch.zeros((config.num_steps, config.num_envs)).to(device)
        dones_storage = torch.zeros((config.num_steps, config.num_envs)).to(device)
        values_storage = torch.zeros((config.num_steps, config.num_envs)).to(device)
        masks_storage = torch.zeros((config.num_steps, config.num_envs, n_actions)).to(device)
        
        global_step = 0
        start_time = time.time()
        win_count = 0
        loss_count = 0
        draw_count = 0
        
        phase2_path = f"{config.model_dir}/ppo_chess_phase2_random.pt"
        
        if os.path.exists(phase2_path):
            checkpoint = torch.load(phase2_path, map_location=device)
            agent.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded Phase 2 model from {phase2_path}")

        else:
            print(f"{phase2_path} not found, starting Phase 2 training from scratch.")
            print(f"Phase 2: Training for {num_iterations} iterations ({phase2_timesteps} timesteps)")
            
            for iteration in range(1, num_iterations + 1):
                # LR annealing
                if config.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / num_iterations
                    optimizer.param_groups[0]["lr"] = frac * config.learning_rate
                
                # Collect rollout
                for step in range(config.num_steps):
                    global_step += config.num_envs
                    obs_storage[step] = next_obs
                    dones_storage[step] = next_done
                    masks_storage[step] = next_mask
                    
                    with torch.no_grad():
                        action, logprob, _, value = agent.get_action_and_value(
                            next_obs, action_mask=next_mask
                        )
                        values_storage[step] = value
                    
                    actions_storage[step] = action
                    logprobs_storage[step] = logprob
                    
                    next_obs_list = []
                    next_mask_list = []
                    
                    for i, env in enumerate(random_envs):
                        obs, reward, terminated, truncated, info = env.step(action[i].item())
                        done = terminated or truncated
                        
                        rewards_storage[step, i] = reward
                        next_done[i] = float(done)
                        
                        if done:
                            if reward > 0:
                                win_count += 1
                            elif reward < 0:
                                loss_count += 1
                            else:
                                draw_count += 1
                            obs, info = env.reset()
                        
                        next_obs_list.append(torch.from_numpy(obs["observation"]).to(device))
                        next_mask_list.append(torch.from_numpy(obs["action_mask"].astype(np.float32)).to(device))
                    
                    next_obs = torch.stack(next_obs_list)
                    next_mask = torch.stack(next_mask_list)
                
                # Compute advantages (GAE)
                with torch.no_grad():
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards_storage).to(device)
                    lastgaelam = 0
                    for t in reversed(range(config.num_steps)):
                        if t == config.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones_storage[t + 1]
                            nextvalues = values_storage[t + 1]
                        delta = rewards_storage[t] + config.gamma * nextvalues * nextnonterminal - values_storage[t]
                        advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values_storage
                
                # Flatten
                b_obs = obs_storage.reshape((-1,) + obs_shape)
                b_logprobs = logprobs_storage.reshape(-1)
                b_actions = actions_storage.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values_storage.reshape(-1)
                b_masks = masks_storage.reshape(-1, n_actions)
                
                # Update
                b_inds = np.arange(batch_size)
                minibatch_size = batch_size // config.num_minibatches
                
                for epoch in range(config.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                            b_obs[mb_inds], b_actions[mb_inds], b_masks[mb_inds]
                        )
                        
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()
                        
                        mb_advantages = b_advantages[mb_inds]
                        if config.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        
                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        # Value loss
                        newvalue = newvalue.view(-1)
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                        
                        # Entropy loss
                        entropy_loss = entropy.mean()
                        
                        loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
                        
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                        optimizer.step()
                
                if iteration % 10 == 0:
                    total_games = win_count + loss_count + draw_count
                    win_rate = win_count / max(total_games, 1)
                    sps = int(global_step / (time.time() - start_time))
                    print(f"Phase 2 | Iter {iteration}/{num_iterations} | Step {global_step} | "
                        f"W:{win_count} D:{draw_count} L:{loss_count} | WinRate: {win_rate:.2%} | SPS: {sps}")
                    
                    # WandB logging for Phase 2
                    if args.use_wandb:
                        wandb.log({
                            "phase2/iteration": iteration,
                            "phase2/global_step": global_step,
                            "phase2/win_count": win_count,
                            "phase2/loss_count": loss_count,
                            "phase2/draw_count": draw_count,
                            "phase2/total_games": total_games,
                            "phase2/win_rate": win_rate,
                            "phase2/sps": sps,
                            "phase2/policy_loss": pg_loss.item(),
                            "phase2/value_loss": v_loss.item(),
                            "phase2/entropy": entropy_loss.item(),
                            "phase2/learning_rate": optimizer.param_groups[0]["lr"],
                        })
            
            # Close random environments
            for env in random_envs:
                env.close()
            
            # Save Phase 2 model
            os.makedirs(config.model_dir, exist_ok=True)
            torch.save({'model_state_dict': agent.state_dict()}, phase2_path)
            print(f"\nPhase 2 complete! Model saved to {phase2_path}")
            print(f"Phase 2 stats: W:{win_count} D:{draw_count} L:{loss_count}")
            
            # Log Phase 2 final stats
            if args.use_wandb:
                final_win_rate_phase2 = win_count / max(win_count + loss_count + draw_count, 1)
                wandb.log({
                    "phase2/final_win_count": win_count,
                    "phase2/final_loss_count": loss_count,
                    "phase2/final_draw_count": draw_count,
                    "phase2/final_win_rate": final_win_rate_phase2,
                })
        
        # Phase 3: Fine-tune vs Stockfish (40% of timesteps) with PARALLEL environments
        print("\n" + "=" * 60)
        print("Phase 3: Fine-tuning vs Stockfish (PARALLEL MODE)")
        print("=" * 60)
        if args.use_wandb:
            wandb.log({"current_phase": 3})
        
        phase3_timesteps = int(config.total_timesteps * 0.4)
        
        # Use more environments for parallel execution
        num_parallel_envs = config.num_envs * 2  # Double the environments for parallel
        
        stockfish_config = Config(
            stockfish_path=config.stockfish_path,
            stockfish_skill=config.stockfish_skill,
            stockfish_time_limit=config.stockfish_time_limit,
            total_timesteps=phase3_timesteps,
            num_envs=num_parallel_envs,
            seed=config.seed,
            eval_episodes=config.eval_episodes,
            eval_interval=config.eval_interval,
        )
        
        # Create trainer with parallel environments and wandb
        trainer = PPOTrainer(stockfish_config, device, use_parallel=True, use_wandb=args.use_wandb)
        trainer.agent.load_state_dict(agent.state_dict())
        
        print(f"Phase 3: Training for {trainer.num_iterations} iterations ({phase3_timesteps} timesteps)")
        print(f"Using {num_parallel_envs} parallel environments")
        
        final_win_rate = trainer.train()
        
        print("\n" + "=" * 60)
        print("CURRICULUM LEARNING COMPLETE")
        print("=" * 60)
        print(f"Final win rate vs Stockfish: {final_win_rate:.2%}")
        
        # Log final results to wandb
        if args.use_wandb:
            wandb.log({
                "final/win_rate": final_win_rate,
                "final/success": final_win_rate >= 0.5,
            })
            
            # Save final model and log as artifact
            final_model_path = f"{config.model_dir}/ppo_chess_curriculum_final.pt"
            torch.save({
                'model_state_dict': trainer.agent.state_dict(),
                'config': asdict(config),
                'final_win_rate': final_win_rate,
            }, final_model_path)
            
            # Log model as wandb artifact
            artifact = wandb.Artifact(
                name="chess-ppo-model",
                type="model",
                description=f"PPO Chess model with {final_win_rate:.2%} win rate vs Stockfish"
            )
            artifact.add_file(final_model_path)
            wandb.log_artifact(artifact)
            
            wandb.finish()
        
        if final_win_rate >= 0.5:
            print("✓ SUCCESS: >50% win rate achieved!")
        else:
            print("✗ Target not reached. Consider longer training or hyperparameter tuning.")
        
        trainer.close()


if __name__ == "__main__":
    main()
