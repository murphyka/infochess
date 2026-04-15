"""
King Inference Game — Playable Prototype

A fog-of-war chess variant where two players move pieces and guess
the opponent's king location each turn.
Score = sum of probabilities assigned to the correct king position.

Controls (click on YOUR side of the board):
  1. Click a non-king piece to select it
  2. Click a destination square to move it
  3. Click your king, then click where to move it (or press S to skip)
  4. Click a square to guess where the opponent's king is

Press R to reset current selection. Escape to quit.

Modes:
  --mode pvp       Two humans play on the same screen (default)
  --mode pve       You (black, right board) vs AI (white, left board)

Run:
  python play_manual.py
  python play_manual.py --mode pve
  python play_manual.py --board_config configs/6x6_1R_1B_2P.json
"""

import argparse
import pygame
from environment.game import Game
from environment.pieces import King
from visual.render import DualBoardRenderer
from agents.random_agent import RandomAgent
from agents.visibility_maximizing_agent import VisibilityMaximizingAgent

parser = argparse.ArgumentParser(description='King Inference Game')
parser.add_argument('--mode', type=str, default='pvp', choices=['pvp', 'pve'],
                    help='pvp = two humans, pve = human (black) vs AI (white)')
parser.add_argument('--ai', type=str, default='vismax', choices=['random', 'vismax'],
                    help='AI agent type for pve mode')
parser.add_argument('--board_config', type=str, default='configs/8x8_1R_1B_2P.json')
parser.add_argument('--max_turns', type=int, default=20)
args = parser.parse_args()

game = Game(max_turns=args.max_turns, board_config_path=args.board_config)
renderer = DualBoardRenderer(game.board, game.fogs, current_turn='white')

# AI opponent (only used in pve mode)
if args.mode == 'pve':
    if args.ai == 'vismax':
        ai_agent = VisibilityMaximizingAgent('white')
    else:
        ai_agent = RandomAgent('white')

clock = pygame.time.Clock()

# Phases: select_piece -> select_dest -> select_king_dest -> guess
phase = 'select_piece'
selected_piece = None  # the non-king piece being moved

PHASE_HINTS = {
    'select_piece': 'Select a piece to move',
    'select_dest': 'Click destination (R to reselect)',
    'select_king_dest': 'Click to move your king (S to skip)',
    'guess': "Click where you think the opponent's king is",
}

print("=" * 50)
print("King Inference Game")
print("=" * 50)
if args.mode == 'pvp':
    print("Two-player mode: click on your side of the board.")
    print("White plays on the LEFT, Black plays on the RIGHT.")
else:
    print(f"You are BLACK (right board). AI ({args.ai}) is white.")
print()
print("Each turn:")
print("  1) Select a non-king piece  2) Move it")
print("  3) Move your king (or S to skip)  4) Guess opponent's king")
print("Press R to reset selection, Escape to quit.")
print("=" * 50)

running = True
while running and not game.is_over():
    turn_team = game.turn

    # --- AI turn (pve mode, white) ---
    if args.mode == 'pve' and turn_team == 'white':
        renderer.current_turn = 'white'
        renderer.selected_piece = None
        renderer.selected_square = None
        renderer.selected_guess = None
        renderer.update()

        action_ai = ai_agent.get_action(game.board)
        renderer.selected_piece = action_ai['piece'].position
        renderer.selected_square = action_ai['new_pos']
        game.step(action_ai)

        king_action = ai_agent.get_king_action(game.board)
        game.step_king(king_action)
        renderer.update()
        pygame.time.delay(400)

        guess = ai_agent.get_guess(game.board)
        if isinstance(guess, dict):
            renderer.selected_guess = max(guess, key=guess.get)
        else:
            renderer.selected_guess = guess
            guess = {guess: 1}
        game.make_guess('white', guess)
        renderer.update()
        pygame.time.delay(400)

        renderer.selected_piece = None
        renderer.selected_square = None
        renderer.selected_guess = None
        continue

    # --- Human turn ---
    renderer.current_turn = turn_team
    renderer.turn_count = game.turn_count // 2

    # Update highlights
    if phase == 'select_piece':
        renderer.selected_piece = None
        renderer.selected_square = None
        renderer.selected_guess = None
    elif phase == 'select_dest':
        renderer.selected_piece = selected_piece.position if selected_piece else None
        renderer.selected_square = None
        renderer.selected_guess = None
    elif phase == 'select_king_dest':
        # Show king highlighted
        king_pieces = [p for p in game.board.get_team_pieces(turn_team) if isinstance(p, King)]
        renderer.selected_piece = king_pieces[0].position if king_pieces else None
        renderer.selected_square = None
        renderer.selected_guess = None
    elif phase == 'guess':
        renderer.selected_piece = None
        renderer.selected_square = None
        renderer.selected_guess = None

    pygame.display.set_caption(f"King Inference — {turn_team.upper()} — {PHASE_HINTS[phase]}")
    renderer.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break
            if event.key == pygame.K_r and phase in ('select_dest', 'select_piece'):
                selected_piece = None
                phase = 'select_piece'
            if event.key == pygame.K_s and phase == 'select_king_dest':
                # Skip king move
                game.step_king(None)
                phase = 'guess'

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            square, side = renderer.get_clicked_square(pos)
            if square is None or side is None:
                continue
            if side != turn_team:
                continue

            if phase == 'select_piece':
                piece = game.board.get_piece_at(square)
                if piece is None or piece.team != turn_team:
                    continue
                if isinstance(piece, King):
                    continue  # king is moved in a separate phase
                selected_piece = piece
                phase = 'select_dest'
                renderer.selected_piece = piece.position
                renderer.update()

            elif phase == 'select_dest':
                if square not in selected_piece.possible_moves(game.board):
                    # Invalid — reset piece selection
                    selected_piece = None
                    phase = 'select_piece'
                    continue
                renderer.selected_square = square
                game.step({'piece': selected_piece, 'new_pos': square})
                renderer.update()

                # Check if king has any legal moves
                king_moves = game.board.get_legal_king_moves(turn_team)
                if king_moves:
                    phase = 'select_king_dest'
                else:
                    game.step_king(None)
                    phase = 'guess'

            elif phase == 'select_king_dest':
                piece = game.board.get_piece_at(square)
                # Allow clicking the king's current position (to see it) or
                # a valid destination
                king_moves = game.board.get_legal_king_moves(turn_team)
                matching = [m for m in king_moves if m['new_pos'] == square]
                if not matching:
                    continue
                game.step_king(matching[0])
                renderer.selected_square = square
                renderer.update()
                phase = 'guess'

            elif phase == 'guess':
                renderer.selected_guess = square
                game.make_guess(turn_team, {square: 1})
                renderer.update()
                pygame.time.delay(300)

                # Reset for next turn
                selected_piece = None
                phase = 'select_piece'
                renderer.selected_piece = None
                renderer.selected_square = None
                renderer.selected_guess = None
                break

    clock.tick(30)

# Game over
if game.is_over():
    scores = game.get_score()
    winner, _ = game.get_winner()
    print()
    print("=" * 50)
    print(f"GAME OVER — White: {scores['white']:.2f}  Black: {scores['black']:.2f}")
    print(f"Winner: {winner.upper()}")
    print("=" * 50)

    pygame.display.set_caption(f"GAME OVER — {winner.upper()} wins!")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                waiting = False

renderer.close()
