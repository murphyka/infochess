"""HidingBeliefVisMaxAgent: belief-weighted vismax + SO-belief king evasion + GRU guessing."""

from .neural_agent import NeuralAgent


class HidingBeliefVisMaxAgent(NeuralAgent):
    """Combines all three model-driven behaviours:
      - Piece movement: maximise information gain about opponent king (FO belief)
      - King movement: evade opponent using second-order belief (SO head)
      - Guessing: return FO belief distribution

    Requires a trained dual-head GRU model.
    """

    def get_action(self, board):
        opponent_king = board.get_opponent_king_position(self.team)
        if opponent_king in board.fogs[self.team].visible:
            return self._vismax_move(board)
        fo, _ = self._peek(board)
        return self._vismax_move(board, fo)

    def get_king_action(self, board):
        return self._so_king_action(board)

    def get_guess(self, board):
        fo, _ = self._run(board)
        opponent_king = board.get_opponent_king_position(self.team)
        if opponent_king in board.fogs[self.team].visible:
            return {opponent_king: 1.0}
        return fo
