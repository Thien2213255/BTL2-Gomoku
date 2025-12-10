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
from src.mcts_agent import MCTSAgent

# Config
NUM_GAMES = 100
MCTS_SIM_TIME = 2.0
MINIMAX_DEPTH = 2

def run():
    print("\n" + "="*70)
    print("AGENTS VS MCTS")
    print("="*70)
    
    board = Board()
    
    mcts_opponent = MCTSAgent(WHITE, simulation_time=MCTS_SIM_TIME)

    # Minimax vs MCTS
    print(f"\n[MATCH 1] Minimax (Depth {MINIMAX_DEPTH}) vs MCTS ({MCTS_SIM_TIME}s)")
    
    minimax_agent = MinimaxAgent(BLACK, depth=MINIMAX_DEPTH)
    minimax_wins = 0
    draws = 0
    
    # Half game black, half game white 
    for i in tqdm(range(NUM_GAMES), desc="Minimax vs MCTS"):
        board.reset()
        
        # Swap sides every game
        if i % 2 == 0:
            minimax_agent.player = BLACK
            mcts_opponent.player = WHITE
            minimax_agent.opponent = WHITE
        else:
            minimax_agent.player = WHITE
            mcts_opponent.player = BLACK
            minimax_agent.opponent = BLACK
            
        game_over = False
        winner = None
        
        while not game_over:
            if board.turn == minimax_agent.player:
                move = minimax_agent.get_move(board)
            else:
                move = mcts_opponent.get_move(board)
            
            if move:
                board.make_move(move[0], move[1], board.turn)
                
                if board.check_win(board.turn):
                    winner = board.turn
                    game_over = True
                elif board.is_full():
                    winner = 0
                    game_over = True
                else:
                    board.switch_turn()
            else:
                break
        
        if winner == minimax_agent.player:
            minimax_wins += 1
        elif winner == 0:
            draws += 1

    rate1 = (minimax_wins / NUM_GAMES) * 100
    print(f"-> Result: Minimax Wins: {minimax_wins}, MCTS Wins: {NUM_GAMES - minimax_wins - draws}, Draws: {draws}")
    print(f"-> Win Rate: {rate1}%")

    # ML vs MCTS
    print(f"\n[MATCH 2] ML Agent vs MCTS ({MCTS_SIM_TIME}s)")
    
    if os.path.exists("models/gomoku_model.pth"):
        ml_agent = MLAgent(BLACK, model_path="models/gomoku_model.pth")
    else:
        print("Error: Model file not found. Please train first.")
        return

    ml_wins = 0
    draws = 0
    
    for i in tqdm(range(NUM_GAMES), desc="ML vs MCTS"):
        board.reset()
        
        # Swap sides
        if i % 2 == 0:
            ml_agent.player = BLACK
            mcts_opponent.player = WHITE
        else:
            ml_agent.player = WHITE
            mcts_opponent.player = BLACK
            
        game_over = False
        winner = None
        
        while not game_over:
            if board.turn == ml_agent.player:
                move = ml_agent.get_move(board)
            else:
                move = mcts_opponent.get_move(board)
            
            if move:
                board.make_move(move[0], move[1], board.turn)
                
                if board.check_win(board.turn):
                    winner = board.turn
                    game_over = True
                elif board.is_full():
                    winner = 0
                    game_over = True
                else:
                    board.switch_turn()
            else:
                break

        if winner == ml_agent.player:
            ml_wins += 1
        elif winner == 0:
            draws += 1

    rate2 = (ml_wins / NUM_GAMES) * 100
    print(f"-> Result: ML Agent Wins: {ml_wins}, MCTS Wins: {NUM_GAMES - ml_wins - draws}, Draws: {draws}")
    print(f"-> Win Rate: {rate2}%")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    run()