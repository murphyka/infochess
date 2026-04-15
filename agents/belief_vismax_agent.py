"""BeliefVisMaxAgent: belief-weighted visibility movement + GRU guessing."""

from .neural_agent import NeuralAgent


class BeliefVisMaxAgent(NeuralAgent):
    """Moves to maximise information gain about the opponent king (info-gain movement),
    guesses using the GRU first-order belief distribution, and moves the king randomly.

    Requires a trained dual-head GRU model (only the first-order head is used).
    """

    def get_action(self, board):
        opponent_king = board.get_opponent_king_position(self.team)
        if opponent_king in board.fogs[self.team].visible:
            return self._vismax_move(board)
        fo, _ = self._peek(board)
        return self._vismax_move(board, fo)

    def get_guess(self, board):
        fo, _ = self._run(board)
        opponent_king = board.get_opponent_king_position(self.team)
        if opponent_king in board.fogs[self.team].visible:
            return {opponent_king: 1.0}
        return fo
