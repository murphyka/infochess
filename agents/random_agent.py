# random agent
import random
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, team):
        super().__init__(team)
    
    def get_action(self, board):
        return random.choice(list(board.get_legal_non_king_moves(self.team)))
