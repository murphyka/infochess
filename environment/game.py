from .board import Board
from .fog import FogOfWar

NUMBER_PIECE_TYPES = 4

class Game:
    def __init__(self, max_turns: int = 10, probabilistic_guesses: bool = False, board_config_path: str = 'configs/6x6_1R_1B_2P.json'):
        self.board = Board(board_config_path)
        self.fogs = {
            'white': FogOfWar(self.board, 'white'),
            'black': FogOfWar(self.board, 'black')
        }
        self.turn = 'white'
        self.turn_count = 0
        self.max_turns = max_turns
        self.probabilistic_guesses = probabilistic_guesses
        self.gt_king_pos = {
            'white': [],
            'black': []
        }
        self.king_loc_guesses = {
            'white': [],
            'black': []
        }

    def step(self, action):
        """Take a non-king piece action for the current player."""
        if action['piece'].team != self.turn:
            print(f"Position {action['piece']} is not owned by {self.turn}")
            raise ValueError("Piece is not owned by current player")
        if action['new_pos'] not in action['piece'].possible_moves(self.board):
            print(f"Invalid move for {action['piece']}")
            print("Valid moves: ", action['piece'].possible_moves(self.board))
            raise ValueError("Invalid move")
        self.board.move_piece(action['piece'], action['new_pos'])
        self.fogs[self.turn].update()

    def step_king(self, action):
        """Take a king move action for the current player. Pass None to skip."""
        if action is None:
            return
        if action['piece'].team != self.turn:
            raise ValueError("King is not owned by current player")
        if action['new_pos'] not in action['piece'].possible_moves(self.board):
            raise ValueError("Invalid king move")
        self.board.move_piece(action['piece'], action['new_pos'])
        self.fogs[self.turn].update()


    def make_guess(self, team: str, probabilities: dict):
        # A guess is a dict of positions (tuples), each with a probability mass
        # The sum of the probabilities should be 1
        if not self.probabilistic_guesses:
            assert len(probabilities) == 1
        else:
            if abs(sum(probabilities.values()) - 1.0) > 1e-6:
                print(f"Probabilities for {team} must sum to 1 (got {sum(probabilities.values())})")
                raise ValueError("Probabilities must sum to 1")
        self.king_loc_guesses[team].append(probabilities)
        self.gt_king_pos[team].append(self.board.get_opponent_king_position(team))
        self.turn = 'black' if self.turn == 'white' else 'white'
        self.turn_count += 1
        self.fogs[self.turn].update()

    def is_over(self) -> bool:
        return self.turn_count >= self.max_turns

    def get_winner(self):
        scores = self.get_score()
        # print(scores)
        if scores['white'] > scores['black']:
            return 'white', scores['white']
        elif scores['black'] > scores['white']:
            return 'black', scores['black']
        else:
            return 'draw', scores['white']

    def get_score(self):
        # Compare guesses to opponent king's position
        if not self.is_over():
            return None, None

        scores = {}
        for team in ['white', 'black']:
            king_pos_series = self.gt_king_pos[team]
            guess_probs_series = self.king_loc_guesses[team]
            scores[team] = 0
            for guess_probs, gt_king_pos in zip(guess_probs_series, king_pos_series):
                scores[team] += guess_probs.get(gt_king_pos, 0)
        return scores