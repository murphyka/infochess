import random
from .base_agent import BaseAgent, fog_filtered_copy, guess_king_if_visible

class VisibilityMaximizingAgent(BaseAgent):
    def __init__(self, team):
        super().__init__(team)

    def get_action(self, board):
        best_moves = []
        best_score = -1

        for move in board.get_legal_non_king_moves(self.team):
            board_copy = fog_filtered_copy(board, self.team)
            copied_piece = board_copy.get_piece_at(move['piece'].position)
            board_copy.move_piece(copied_piece, move['new_pos'])

            # Recompute visibility after move
            visible = set()
            for p in board_copy.get_team_pieces(self.team):
                visible.update(p.vision(board_copy))

            score = len(visible)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        if best_moves:
            return random.choice(best_moves)
        else:
            return None

    def get_guess(self, board):
        return {guess_king_if_visible(board, self.team): 1}
