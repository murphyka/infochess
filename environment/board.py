from .pieces import Piece, King, Pawn, Rook, Bishop
from .fog import FogOfWar
import json
import random
import numpy as np


class Board:
    def __init__(self, config_path):
        with open(config_path) as f:
            config = json.load(f)

        self.rows = config.get('rows', 8)
        self.cols = config.get('cols', 8)
        self.grid = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.pieces = []  # list of all active pieces

        piece_classes = {'king': King, 'pawn': Pawn, 'rook': Rook, 'bishop': Bishop}

        # If multiple kings are listed for a team, pick one at random.
        kings_by_team = {}
        non_king_configs = []
        for piece_config in config.get('pieces', []):
            if piece_config['type'].lower() == 'king':
                kings_by_team.setdefault(piece_config['team'], []).append(piece_config)
            else:
                non_king_configs.append(piece_config)

        selected = list(non_king_configs)
        for team, king_configs in kings_by_team.items():
            selected.append(random.choice(king_configs))

        for piece_config in selected:
            piece_type = piece_config['type'].lower()
            if piece_type not in piece_classes:
                raise ValueError(f"Unknown piece type: {piece_config['type']}")
            self.place_piece(piece_classes[piece_type](
                piece_config['team'], tuple(piece_config['position'])
            ))

        self.fogs = {
            'white': FogOfWar(self, 'white'),
            'black': FogOfWar(self, 'black')
        }

    def place_piece(self, piece: Piece):
        """Add a piece to the board and to the grid."""
        self.grid[piece.position[0]][piece.position[1]] = piece
        self.pieces.append(piece)

    def move_piece(self, piece: Piece, new_pos: tuple):
        """Update the board and piece location."""
        old_pos = piece.position
        self.grid[old_pos[0]][old_pos[1]] = None
        piece.move(new_pos)
        self.grid[new_pos[0]][new_pos[1]] = piece
        self.fogs[piece.team].update()

    def get_piece_at(self, pos: tuple) -> Piece:
        return self.grid[pos[0]][pos[1]]

    def is_within_bounds(self, pos: tuple) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_team_pieces(self, team: str) -> list:
        return [p for p in self.pieces if p.team == team]

    def get_opponent_king_position(self, team: str) -> tuple:
        for p in self.pieces:
            if isinstance(p, King) and p.team != team:
                return p.position
        return None

    def get_legal_moves(self, team: str) -> list:
        # Each move is a dict of {'piece': piece, 'new_pos': new_pos}
        return [{'piece': p, 'new_pos': m} for p in self.get_team_pieces(team) for m in p.possible_moves(self)]

    def get_legal_king_moves(self, team: str) -> list:
        return [{'piece': p, 'new_pos': m} for p in self.get_team_pieces(team)
                if isinstance(p, King) for m in p.possible_moves(self)]

    def get_legal_non_king_moves(self, team: str) -> list:
        return [{'piece': p, 'new_pos': m} for p in self.get_team_pieces(team)
                if not isinstance(p, King) for m in p.possible_moves(self)]
    
    def get_legal_guesses(self, team: str) -> list:
        # return all positions in the grid
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]
    
    def encode_state(self, team: str) -> np.ndarray:
        """
        Encode the board state for the guesser.
        The first dimension of each square is the team (-1, 0, 1),
        the second dimension is the fog (0, 1),
        the third dimension is the piece (0, 1, 2, 3 = king, pawn, rook, bishop)
        """
        state = np.zeros((self.rows, self.cols, 6))
        for r in range(self.rows):
            for c in range(self.cols):
                if self.fogs[team].is_visible((r, c)):
                    state[r, c, 1] = 1
                    piece = self.get_piece_at((r, c))
                    if piece is not None:
                        state[r, c, 0] = 1 if piece.team == team else -1
                        state[r, c, 2] = 1 if piece.type == 'king' else 0
                        state[r, c, 3] = 1 if piece.type == 'pawn' else 0
                        state[r, c, 4] = 1 if piece.type == 'rook' else 0
                        state[r, c, 5] = 1 if piece.type == 'bishop' else 0
        return state
