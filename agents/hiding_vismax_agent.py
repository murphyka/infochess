"""HidingVisMaxAgent: plain vismax movement + SO-belief king evasion + uniform guessing."""

from .neural_agent import NeuralAgent


class HidingVisMaxAgent(NeuralAgent):
    """Maximises raw visibility for piece movement (no belief weighting),
    evades the opponent using second-order belief for king moves, and
    guesses model-free (point mass if king visible, else uniform over fogged squares).

    Requires a trained dual-head GRU model (only the second-order head is used for king moves).
    The GRU state is still advanced each turn so the SO head has a useful game history.
    """

    def get_action(self, board):
        return self._vismax_move(board)

    def get_king_action(self, board):
        return self._so_king_action(board)

    def get_guess(self, board):
        # Advance GRU state so king evasion has useful history next turn.
        self._run(board)

        opponent_king = board.get_opponent_king_position(self.team)
        if opponent_king in board.fogs[self.team].visible:
            return {opponent_king: 1.0}
        fogged = [pos for pos in board.get_legal_guesses(self.team)
                  if pos not in board.fogs[self.team].visible]
        if not fogged:
            return {opponent_king: 1.0}
        p = 1.0 / len(fogged)
        return {pos: p for pos in fogged}
