''' Team-specific fog of war. '''

class FogOfWar:
    def __init__(self, board, team: str):
        from .board import Board  # delay to avoid circular import
        assert isinstance(board, Board)
        self.board = board
        self.team = team
        self.visible = set()  # visible squares
        self.update()

    def update(self):
        """Recompute the visible squares based on all team pieces."""
        self.visible.clear()
        team_pieces = self.board.get_team_pieces(self.team)
        for piece in team_pieces:
            self.visible.update(piece.vision(self.board))

    def is_visible(self, pos: tuple) -> bool:
        return pos in self.visible
