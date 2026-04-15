import random
import copy

def fog_filtered_copy(board, team):
    """Deep copy a board with hidden enemy pieces removed.

    The copy contains all friendly pieces and only the enemy pieces currently
    visible to `team`.  This prevents move-simulation from leaking information
    about fog-hidden pieces (e.g. pawns that would block line-of-sight).
    """
    board_copy = copy.deepcopy(board)
    visible = board.fogs[team].visible
    opponent = 'black' if team == 'white' else 'white'
    for piece in list(board_copy.pieces):
        if piece.team == opponent and piece.position not in visible:
            board_copy.grid[piece.position[0]][piece.position[1]] = None
            board_copy.pieces.remove(piece)
    return board_copy


class BaseAgent:
    def __init__(self, team):
        self.team = team

    def get_action(self, board):
        pass

    def get_king_action(self, board):
        """Return a king move, or None if no legal king moves."""
        moves = board.get_legal_king_moves(self.team)
        if not moves:
            return None
        return random.choice(moves)

    def get_guess(self, board):
        king_pos = board.get_opponent_king_position(self.team)
        if king_pos in board.fogs[self.team].visible:
            return {king_pos: 1.0}
        fogged = [pos for pos in board.get_legal_guesses(self.team)
                  if pos not in board.fogs[self.team].visible]
        if not fogged:
            return {king_pos: 1.0}
        p = 1.0 / len(fogged)
        return {pos: p for pos in fogged}

    def get_team(self):
        return self.team

def guess_king_if_visible(board, team):
    # If the king is visible, guess the king's position
    if board.get_opponent_king_position(team) in board.fogs[team].visible:
        return board.get_opponent_king_position(team)
    # Otherwise, guess a random hidden position
    not_visible = set(board.get_legal_guesses(team)) - set(board.fogs[team].visible)
    return random.choice(list(not_visible))
