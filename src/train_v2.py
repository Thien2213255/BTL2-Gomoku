import torch
import numpy as np
import random
import os
import pickle
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from src.constants import *
from src.game_logic import Board
from src.minimax_agent import MinimaxAgent
from src.ml_agent import MLAgent

# Config 
DATA_FILE = "models/training_data_v2.pkl"
NUM_GAMES = 1000
TEACHER_DEPTH = 3
BATCH_SIZE = 128
EPOCHS = 50

def generate_smart_data():
    print(f"--- ĐANG SINH DỮ LIỆU TỪ ({NUM_GAMES} VÁN) ---")
    data = []
    board = Board()
    
    agent1 = MinimaxAgent(BLACK, depth=TEACHER_DEPTH)
    agent2 = MinimaxAgent(WHITE, depth=TEACHER_DEPTH)
    
    # Biến đếm thống kê
    black_wins = 0
    white_wins = 0
    
    for i in tqdm(range(NUM_GAMES)):
        board.reset()
        game_moves = [] # Lưu tạm nước đi của ván này: (state, action_idx, player_color)
        game_over = False
        
        start_random_moves = random.randint(1, 3)
        move_count = 0
        
        while not game_over:
            current_player = board.turn
            
            if move_count < start_random_moves:
                valid_moves = board.get_valid_moves()
                if not valid_moves: break
                move = random.choice(valid_moves)
            else:
                if current_player == BLACK:
                    move = agent1.get_move(board)
                else:
                    move = agent2.get_move(board)
            
            if move:
                # Lưu lại trạng thái TRƯỚC khi đi
                board_np = board.board
                my_pieces = (board_np == current_player).astype(np.float32)
                opp_pieces = (board_np == -current_player).astype(np.float32)
                state_input = np.stack([my_pieces, opp_pieces]) # (2, 15, 15)
                
                action_idx = move[0] * BOARD_SIZE + move[1]
                
                # Lưu vào bộ nhớ tạm
                game_moves.append((state_input, action_idx, current_player))
                
                # Thực hiện nước đi
                board.make_move(move[0], move[1], current_player)
                move_count += 1
                
                # Kiểm tra kết quả
                if board.check_win(current_player):
                    # Chỉ lấy dữ liệu của người thắng!
                    winner = current_player
                    if winner == BLACK: black_wins += 1
                    else: white_wins += 1
                    
                    for state, action, p_color in game_moves:
                        if p_color == winner:
                            data.append((state, action))
                    game_over = True
                    
                elif board.is_full():
                    # Không học ván hòa 
                    game_over = True
            else:
                break
                
    print(f"-> Kết quả: Đen thắng {black_wins}, Trắng thắng {white_wins}")
    print(f"-> Thu thập được {len(data)} nước đi CHUẨN (Winner only).")
    return data

def augment_data(data):
    print("--- ĐANG TĂNG CƯỜNG DỮ LIỆU (AUGMENTATION) ---")
    augmented_data = []
    
    for state_input, action_idx in data:
        # state_input shape: (2, 15, 15)
        my_p = state_input[0]
        opp_p = state_input[1]
        
        r, c = action_idx // BOARD_SIZE, action_idx % BOARD_SIZE
        move_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE))
        move_matrix[r, c] = 1
        
        for k in range(4):
            # Xoay
            rot_my = np.rot90(my_p, k)
            rot_opp = np.rot90(opp_p, k)
            rot_move = np.rot90(move_matrix, k)
            
            rot_state = np.stack([rot_my, rot_opp])
            mr, mc = np.argwhere(rot_move == 1)[0]
            augmented_data.append((rot_state.copy(), mr * BOARD_SIZE + mc))
            
            # Lật gương
            flip_my = np.fliplr(rot_my)
            flip_opp = np.fliplr(rot_opp)
            flip_move = np.fliplr(rot_move)
            
            flip_state = np.stack([flip_my, flip_opp])
            mr, mc = np.argwhere(flip_move == 1)[0]
            augmented_data.append((flip_state.copy(), mr * BOARD_SIZE + mc))
            
    print(f"-> Tổng dữ liệu sau khi nhân bản: {len(augmented_data)} mẫu.")
    return augmented_data

def main():
    if os.path.exists(DATA_FILE):
        print("Tìm thấy data cũ, sinh lại data mới ...")
    
    raw_data = generate_smart_data()
    
    if len(raw_data) == 0:
        print("Lỗi: Không có ván thắng nào được ghi nhận.")
        return

    dataset = augment_data(raw_data)
    
    # Lưu backup
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(dataset, f)

    # Train
    ml_agent = MLAgent(BLACK)
    
    # Reset weights để train mới
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
    ml_agent.model.apply(weights_init)
    
    print(f"Bắt đầu Training {EPOCHS} Epochs...")
    ml_agent.train_on_data(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()