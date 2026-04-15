"""Shared base class for GRU-based agents."""

import random
import numpy as np
import torch

from .base_agent import BaseAgent, fog_filtered_copy
from .model import load_model


class NeuralAgent(BaseAgent):
    """Base for agents that use a trained dual-head GRU model.

    Subclasses override get_action, get_king_action, and get_guess to
    combine model beliefs with different movement and evasion strategies.

    Call order each turn: get_action → get_king_action → get_guess
    get_guess always advances the GRU hidden state so that subsequent
    turns benefit from the full game history.
    """

    def __init__(self, team, model_path):
        super().__init__(team)
        self.model, meta = load_model(model_path)
        self.model.eval()
        self.rows = int(meta['output_dim'] ** 0.5)
        self.cols = self.rows
        self._hidden = None

    # ── Model utilities ───────────────────────────────────────────────────

    def _encode(self, board):
        state = board.encode_state(self.team)
        if self.team == 'black':
            state = state[::-1, ::-1, :]
        return torch.tensor(state.reshape(1, 1, -1), dtype=torch.float32)

    def _to_dists(self, fo_logits, so_logits):
        """Convert raw logits to position→probability dicts."""
        fo_probs = torch.softmax(fo_logits[0, 0], dim=-1).detach().numpy()
        so_probs = torch.sigmoid(so_logits[0, 0]).detach().numpy()
        fo_dist, so_dist = {}, {}
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                pos = (self.rows - 1 - r, self.cols - 1 - c) if self.team == 'black' else (r, c)
                fo_dist[pos] = float(fo_probs[idx])
                so_dist[pos] = float(so_probs[idx])
        return fo_dist, so_dist

    def _run(self, board):
        """Advance GRU hidden state, return (fo_dist, so_dist)."""
        with torch.no_grad():
            fo, so, self._hidden = self.model(self._encode(board), self._hidden)
        return self._to_dists(fo, so)

    def _peek(self, board):
        """Run model without advancing hidden state, return (fo_dist, so_dist)."""
        with torch.no_grad():
            fo, so, _ = self.model(self._encode(board), self._hidden)
        return self._to_dists(fo, so)

    # ── Movement utilities ────────────────────────────────────────────────

    _EPS = 1e-12

    def _vismax_move(self, board, belief=None):
        """Pick the non-king move that minimises expected posterior entropy of belief.

        If belief is None, falls back to maximising raw visible square count.
        """
        best_moves, best_score = [], float('-inf')

        for move in board.get_legal_non_king_moves(self.team):
            board_copy = fog_filtered_copy(board, self.team)
            copied_piece = board_copy.get_piece_at(move['piece'].position)
            board_copy.move_piece(copied_piece, move['new_pos'])

            visible = set()
            for p in board_copy.get_team_pieces(self.team):
                visible.update(p.vision(board_copy))

            if belief is not None:
                fogged = [v for sq, v in belief.items() if sq not in visible]
                p_fog = sum(fogged)
                if p_fog < self._EPS:
                    score = 0.0
                else:
                    h = -sum((p / p_fog) * np.log(p / p_fog) for p in fogged if p > self._EPS)
                    score = -(p_fog * h)
            else:
                score = len(visible)

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves) if best_moves else None

    def _so_king_action(self, board):
        """Move king to the adjacent square least visible to the opponent.

        Uses the second-order head (sigmoid per square = opponent visibility probability).
        Picks the king destination with the lowest predicted opponent visibility.
        """
        moves = board.get_legal_king_moves(self.team)
        if not moves:
            return None

        _, so_belief = self._peek(board)

        best_moves, best_vis = [], float('inf')
        for move in moves:
            vis = so_belief.get(move['new_pos'], 1.0)
            if vis < best_vis:
                best_vis = vis
                best_moves = [move]
            elif vis == best_vis:
                best_moves.append(move)
        return random.choice(best_moves)

    def reset(self):
        self._hidden = None
