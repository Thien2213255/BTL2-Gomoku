import sys
import os
import random
import time
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.constants import *
from src.game_logic import Board
from src.minimax_agent import MinimaxAgent
from src.ml_agent import MLAgent

# Config
NUM_GAMES = 100
MINIMAX_DEPTH = 2

# Random move hợp lệ dựa trên game logic
class RandomAgent:
    def __init__(self, player_color):
        self.player = player_color
    
    def get_move(self, board):
        moves = board.get_valid_moves()
        if not moves: return None
        return random.choice(moves)

def run():
    print("\n" + "="*60)
    print("Minimax vs Random")
    print("="*60)
    
    board = Board()
    random_agent = RandomAgent(WHITE) # Random luôn trắng
    
    # Minimax vs Random
    print(f"\n[TEST 1] Minimax Agent (Depth {MINIMAX_DEPTH}) vs Random Agent")
    
    minimax_agent = MinimaxAgent(BLACK, depth=MINIMAX_DEPTH)
    minimax_wins = 0
    
    for _ in tqdm(range(NUM_GAMES), desc="Minimax vs Random"):
        board.reset()
        # Minimax (Black) đi trước
        minimax_agent.player = BLACK
        random_agent.player = WHITE
        
        game_over = False
        while not game_over:
            # Minimax turn
            move = minimax_agent.get_move(board)
            if move:
                board.make_move(move[0], move[1], BLACK)
                if board.check_win(BLACK):
                    minimax_wins += 1
                    game_over = True
                    continue
                if board.is_full(): game_over = True; continue
            
            # Random turn
            move = random_agent.get_move(board)
            if move:
                board.make_move(move[0], move[1], WHITE)
                if board.check_win(WHITE):
                    game_over = True
                if board.is_full(): game_over = True

    rate1 = (minimax_wins / NUM_GAMES) * 100
    print(f"-> Kết quả: Minimax thắng {minimax_wins}/{NUM_GAMES} ({rate1}%)")

    # ML vs Random
    print(f"\n[TEST 2] Machine Learning Agent vs Random Agent")
    
    # Load model
    ml_agent = MLAgent(BLACK, model_path="models/gomoku_model.pth")
    ml_wins = 0
    
    for _ in tqdm(range(NUM_GAMES), desc="ML vs Random"):
        board.reset()
        # ML (Black) đi trước
        ml_agent.player = BLACK
        random_agent.player = WHITE
        
        game_over = False
        while not game_over:
            # ML turn
            move = ml_agent.get_move(board)
            if move:
                board.make_move(move[0], move[1], BLACK)
                if board.check_win(BLACK):
                    ml_wins += 1
                    game_over = True
                    continue
                if board.is_full(): game_over = True; continue
            
            # Random turn
            move = random_agent.get_move(board)
            if move:
                board.make_move(move[0], move[1], WHITE)
                if board.check_win(WHITE):
                    game_over = True
                if board.is_full(): game_over = True

    rate2 = (ml_wins / NUM_GAMES) * 100
    print(f"-> Kết quả: ML Agent thắng {ml_wins}/{NUM_GAMES} ({rate2}%)")

if __name__ == "__main__":
    run()