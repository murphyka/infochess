# Render the board and pieces
import numpy as np
import pygame

from environment.board import Board

# Constants
TILE_SIZE = 60
MARGIN = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
FOG = (50, 50, 50)
HIGHLIGHT_PIECE = (0, 255, 0)
HIGHLIGHT_SQUARE = (255, 0, 0)
HIGHLIGHT_GUESS = (0, 0, 255)

# Piece glyphs (simplified text for now)
PIECE_SYMBOLS = {
    "pawn": "P",
    "rook": "R",
    "bishop": "B",
    "king": "K"
}

pygame.font.init()
FONT = pygame.font.SysFont("arial", 32)
LABEL_FONT = pygame.font.SysFont("arial", 22)
HEATMAP_FONT = pygame.font.SysFont("arial", 16)


class DualBoardRenderer:
    def __init__(self, board, fogs, current_turn='white'):
        self.board = board
        self.board_side_length = board.rows
        self.board_height = self.board_side_length * TILE_SIZE
        self.display_width = 2 * (self.board_side_length * TILE_SIZE) + 3 * MARGIN
        self.display_height = 2 * self.board_height + 3 * MARGIN
        self.fogs = fogs  # {'white': FogOfWar, 'black': FogOfWar}
        self.current_turn = current_turn
        self.turn_count = 0
        self.selected_piece = None
        self.selected_square = None
        self.selected_guess = None
        self.belief_maps = {'white': None, 'black': None}
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("King Inference Game")

    def draw_board(self, top_left, fog):
        for row in range(self.board_side_length):
            for col in range(self.board_side_length):
                x = top_left[0] + col * TILE_SIZE
                y = top_left[1] + row * TILE_SIZE
                color = LIGHT if (row + col) % 2 == 0 else DARK
                rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                if fog and (row, col) not in fog.visible:
                    pygame.draw.rect(self.screen, FOG, rect)
                if self.selected_piece == (row, col):
                    pygame.draw.rect(self.screen, HIGHLIGHT_PIECE, rect, 4)
                if self.selected_square == (row, col):
                    pygame.draw.rect(self.screen, HIGHLIGHT_SQUARE, rect, 4)
                if self.selected_guess == (row, col):
                    pygame.draw.rect(self.screen, HIGHLIGHT_GUESS, rect, 4)
    def draw_pieces(self, top_left, fog):
        for piece in self.board.pieces:
            r, c = piece.position
            if fog and (r, c) not in fog.visible:
                continue
            label = PIECE_SYMBOLS[type(piece).__name__.lower()]
            text_color = BLACK if piece.team == 'black' else WHITE
            text = FONT.render(label, True, text_color)
            x = top_left[0] + c * TILE_SIZE + TILE_SIZE // 2
            y = top_left[1] + r * TILE_SIZE + TILE_SIZE // 2
            rect = text.get_rect(center=(x, y))
            self.screen.blit(text, rect)

    def draw_turn_indicator(self):
        label = f"{self.current_turn.upper()}, turn {self.turn_count}"
        right_side_flag = self.current_turn == 'white'
        color = BLACK if self.current_turn == 'white' else WHITE
        bg_color = WHITE if self.current_turn == 'white' else BLACK
        text = LABEL_FONT.render(label, True, color, bg_color)
        rect = text.get_rect(center=(self.display_width // 4 if right_side_flag else (self.display_width * 3) // 4, MARGIN // 2))
        self.screen.blit(text, rect)

    def get_clicked_square(self, mouse_pos):
        x, y = mouse_pos
        if not (MARGIN <= y <= self.display_height - MARGIN):
            return None, None
        for side, offset_x in [('white', MARGIN), ('black', self.display_width // 2 + MARGIN // 2)]:
            board_x0 = offset_x
            board_x1 = offset_x + self.board_side_length * TILE_SIZE
            if board_x0 <= x < board_x1:
                col = (x - board_x0) // TILE_SIZE
                row = (y - MARGIN) // TILE_SIZE
                return (row, col), side
        return None, None

    def draw_heatmap(self, top_left, belief_map, fog, king_pos, label_text="Belief Map"):
        COLD = (200, 220, 255)
        HOT = (255, 0, 0)

        max_prob = max(belief_map.values()) if belief_map else 0
        norm = max_prob if max_prob > 0 else 1.0
        shrink = int(TILE_SIZE * 0.25)  # 25% inset on each side = 50% inner square

        # Label
        label = LABEL_FONT.render(label_text, True, WHITE)
        label_rect = label.get_rect(centerx=top_left[0] + self.board_side_length * TILE_SIZE // 2,
                                    bottom=top_left[1] - 2)
        self.screen.blit(label, label_rect)

        for row in range(self.board_side_length):
            for col in range(self.board_side_length):
                x = top_left[0] + col * TILE_SIZE
                y = top_left[1] + row * TILE_SIZE
                prob = belief_map.get((row, col), 0.0)
                t = prob / norm
                color = tuple(int(COLD[i] + (HOT[i] - COLD[i]) * t) for i in range(3))

                visible = fog and (row, col) in fog.visible
                if visible:
                    # Draw dark background for the full tile, then shrunk colored square
                    full_rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
                    pygame.draw.rect(self.screen, FOG, full_rect)
                    pygame.draw.rect(self.screen, BLACK, full_rect, 1)
                    inner_rect = pygame.Rect(x + shrink, y + shrink,
                                             TILE_SIZE - 2 * shrink, TILE_SIZE - 2 * shrink)
                    pygame.draw.rect(self.screen, color, inner_rect)
                else:
                    rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, BLACK, rect, 1)

                if prob > 0.01:
                    text = HEATMAP_FONT.render(f"{int(prob * 100)}", True, BLACK)
                    text_rect = text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
                    self.screen.blit(text, text_rect)

                if king_pos == (row, col):
                    star = HEATMAP_FONT.render("*", True, WHITE)
                    self.screen.blit(star, (x + 1, y + 1))

    def update(self):
        self.screen.fill((100, 100, 100))  # gray background

        white_offset = (MARGIN, MARGIN)
        black_offset = (self.display_width // 2 + MARGIN // 2, MARGIN)

        self.draw_board(white_offset, self.fogs['white'])
        self.draw_pieces(white_offset, self.fogs['white'])

        self.draw_board(black_offset, self.fogs['black'])
        self.draw_pieces(black_offset, self.fogs['black'])

        heatmap_y = MARGIN + self.board_height + MARGIN
        if self.belief_maps['white'] is not None:
            black_king = self.board.get_opponent_king_position('white')
            self.draw_heatmap((MARGIN, heatmap_y), self.belief_maps['white'],
                              self.fogs['white'], black_king, "White Belief")
        if self.belief_maps['black'] is not None:
            white_king = self.board.get_opponent_king_position('black')
            self.draw_heatmap((self.display_width // 2 + MARGIN // 2, heatmap_y), self.belief_maps['black'],
                              self.fogs['black'], white_king, "Black Belief")

        self.draw_turn_indicator()

        pygame.display.flip()

    def save_frame(self, path):
        pygame.image.save(self.screen, path)

    def close(self):
        pygame.quit()