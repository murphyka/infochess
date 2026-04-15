
class Piece:
    def __init__(self, team: str, position: tuple):
        self.team = team  # 'white' or 'black'
        self.position = position  # (row, col)
        self.type = type(self).__name__.lower()
        self.move_directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        self.long_view_directions = []
        self.type_to_id = {
            'king': 0,
            'pawn': 1,
            'rook': 2,
            'bishop': 3
        }
        self.type_id = self.type_to_id[self.type]


    def possible_moves(self, board) -> list:
        """Default movement: 1 square in any direction (like a king)."""
        from .board import Board  # delay to avoid circular import
        assert isinstance(board, Board)
        moves = set()
        r0, c0 = self.position
        for dr, dc in self.move_directions:
            r, c = r0 + dr, c0 + dc
            if board.is_within_bounds((r, c)):
                if board.get_piece_at((r, c)) is None: ## make sure the square is empty
                    moves.add((r, c))
        return moves

    def vision(self, board) -> set:
        """Returns the set of squares this piece can 'see' based on its vision rules."""
        return self.near_vision(board) | self.long_vision(board)
    
    def near_vision(self, board) -> set:
        """Same near-vision for all pieces --> immediate vicinity in all directions."""
        from .board import Board  # delay to avoid circular import
        assert isinstance(board, Board)
        visible = set([self.position])
        for dr, dc in self.move_directions:
            r, c = self.position
            r += dr
            c += dc
            if not board.is_within_bounds((r, c)):
                continue
            visible.add((r, c))
        return visible

    def long_vision(self, board) -> set:
        from .board import Board  # delay to avoid circular import
        assert isinstance(board, Board)
        visible = set([self.position])
        for dr, dc in self.long_view_directions:
            r, c = self.position
            while True:
                r += dr
                c += dc
                if not board.is_within_bounds((r, c)):
                    break
                visible.add((r, c))
                occupant = board.get_piece_at((r, c))
                if occupant is not None and occupant.blocks_vision():
                    break
        return visible
    def blocks_vision(self) -> bool:
        """Whether this piece blocks vision (e.g. pawn)."""
        return False

    def move(self, new_position: tuple):
        """Update the piece’s internal state to reflect a move."""
        self.position = new_position


class Pawn(Piece):
    def blocks_vision(self): return True
        
class Rook(Piece):
    def __init__(self, team: str, position: tuple):
        super().__init__(team, position)
        self.long_view_directions = [(0,1), (0,-1), (1,0), (-1,0)]

class Bishop(Piece):
    def __init__(self, team: str, position: tuple):
        super().__init__(team, position)
        self.long_view_directions = [(1,1), (1,-1), (-1,1), (-1,-1)]

class King(Piece):
    pass