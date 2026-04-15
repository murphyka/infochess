"""Train a dual-head GRU model for king location prediction and king evasion.

Generates games with mixed opponents (random + vismax), trains:
  - First-order head: predict opponent king location (cross-entropy loss)
  - Second-order head: predict which squares are visible to the opponent (binary cross-entropy)

The trained model can be loaded by BeliefVisMaxAgent, HidingVisMaxAgent,
and HidingBeliefVisMaxAgent.

Usage:
  python train_guesser.py --board_config configs/8x8_1R_1B_2P.json
  python train_guesser.py --board_config configs/8x8_1R_1B_2P.json --num_games 2000 --output models/my_model.pth
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment.game import Game, NUMBER_PIECE_TYPES
from agents.model import DualHeadGRU, save_model
from agents.random_agent import RandomAgent
from agents.visibility_maximizing_agent import VisibilityMaximizingAgent

parser = argparse.ArgumentParser()
parser.add_argument('--board_config', required=True, help='Path to board config JSON')
parser.add_argument('--num_games', type=int, default=500)
parser.add_argument('--num_turns', type=int, default=50,
                    help='Max turns per game (total, both sides)')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=10, help='Games per training batch')
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--output', type=str, default='models/guesser.pth')
args = parser.parse_args()

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

# ── Phase 1: Generate game data ──────────────────────────────────────────────

print(f"Phase 1: Generating {args.num_games} games...")

# Infer board dimensions from a test game
_probe = Game(max_turns=2, board_config_path=args.board_config)
rows, cols = _probe.board.rows, _probe.board.cols
del _probe

turns_per_side = args.num_turns // 2
dim_per_square = 2 + NUMBER_PIECE_TYPES
input_dim = rows * cols * dim_per_square
output_dim = rows * cols

white_states, black_states = [], []
white_kings, black_kings = [], []
white_fog_targets, black_fog_targets = [], []

def make_agent(team):
    return random.choice([RandomAgent(team), VisibilityMaximizingAgent(team)])

for i in range(args.num_games):
    if (i + 1) % 100 == 0:
        print(f"  Game {i + 1}/{args.num_games}")
    game = Game(max_turns=args.num_turns, board_config_path=args.board_config,
                probabilistic_guesses=False)
    white_agent = make_agent('white')
    black_agent = make_agent('black')
    while not game.is_over():
        agent = white_agent if game.turn == 'white' else black_agent
        game.step(agent.get_action(game.board))
        game.step_king(agent.get_king_action(game.board))

        board = game.board
        if agent.team == 'white':
            white_states.append(board.encode_state('white'))
            white_kings.append(board.get_opponent_king_position('white'))
            fog_mask = np.zeros(output_dim, dtype=np.float32)
            for r, c in board.fogs['black'].visible:
                fog_mask[r * cols + c] = 1.0
            white_fog_targets.append(fog_mask)
        else:
            state = board.encode_state('black')[::-1, ::-1, :]
            black_states.append(state)
            r, c = board.get_opponent_king_position('black')
            black_kings.append((rows - 1 - r, cols - 1 - c))
            fog_mask = np.zeros(output_dim, dtype=np.float32)
            for fr, fc in board.fogs['white'].visible:
                fog_mask[(rows - 1 - fr) * cols + (cols - 1 - fc)] = 1.0
            black_fog_targets.append(fog_mask)

        game.make_guess(agent.team, agent.get_guess(board))

n = args.num_games
white_states = np.array(white_states).reshape(n, turns_per_side, rows, cols, dim_per_square)
black_states = np.array(black_states).reshape(n, turns_per_side, rows, cols, dim_per_square)
white_kings = np.array(white_kings).reshape(n, turns_per_side, 2)
black_kings = np.array(black_kings).reshape(n, turns_per_side, 2)
white_fog = np.array(white_fog_targets).reshape(n, turns_per_side, output_dim)
black_fog = np.array(black_fog_targets).reshape(n, turns_per_side, output_dim)

print(f"  Collected {n} games × {turns_per_side} turns/side")

# ── Phase 2: Train ───────────────────────────────────────────────────────────

print(f"\nPhase 2: Training dual-head GRU ({args.hidden_dim}d × {args.num_layers}L)...")

model = DualHeadGRU(input_dim, output_dim, args.hidden_dim, args.num_layers)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

w_states_t = torch.tensor(white_states.reshape(n, turns_per_side, -1), dtype=torch.float32)
b_states_t = torch.tensor(black_states.reshape(n, turns_per_side, -1), dtype=torch.float32)
w_king_idx = torch.tensor(white_kings[:, :, 0] * cols + white_kings[:, :, 1], dtype=torch.long)
b_king_idx = torch.tensor(black_kings[:, :, 0] * cols + black_kings[:, :, 1], dtype=torch.long)
w_fog_t = torch.tensor(white_fog, dtype=torch.float32)
b_fog_t = torch.tensor(black_fog, dtype=torch.float32)

B, T = args.batch_size, turns_per_side

batches_per_epoch = max(n // B, 1)
replace = B > n  # allow replacement if fewer games than batch size

for epoch in range(args.num_epochs):
    for _ in range(batches_per_epoch):
        idx = np.random.choice(n, size=B, replace=replace)

        w_fo, w_so, _ = model(w_states_t[idx])
        b_fo, b_so, _ = model(b_states_t[idx])

        fo_loss = (
            nn.CrossEntropyLoss()(w_fo.reshape(B * T, output_dim), w_king_idx[idx].reshape(B * T))
            + nn.CrossEntropyLoss()(b_fo.reshape(B * T, output_dim), b_king_idx[idx].reshape(B * T))
        )
        so_loss = (
            F.binary_cross_entropy_with_logits(w_so, w_fog_t[idx])
            + F.binary_cross_entropy_with_logits(b_so, b_fog_t[idx])
        )
        loss = fo_loss + so_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1}/{args.num_epochs}  "
              f"loss={loss.item():.4f}  fo={fo_loss.item():.4f}  so={so_loss.item():.4f}")

save_model(model, args.output, input_dim, output_dim, args.hidden_dim, args.num_layers)
print(f"\nModel saved to {args.output}")
