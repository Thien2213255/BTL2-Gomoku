import torch
import numpy as np
import random
import os
import pickle
from tqdm import tqdm

from src.constants import *
from src.game_logic import Board
from src.minimax_agent import MinimaxAgent
from src.ml_agent import MLAgent

# Cấu hình
DATA_FILE = "models/training_data.pkl"
NUM_GAMES = 200     # 200 ván là đủ vì có Augmentation x8 = 1600 mẫu
MINIMAX_DEPTH = 2   # Depth 2 để dạy căn bản, tránh lỗi timeout

def generate_data():
    print(f"--- ĐANG SINH DỮ LIỆU TỪ {NUM_GAMES} VÁN CỜ ---")
    data = []
    board = Board()
    
    # Thầy giáo Minimax
    teacher = MinimaxAgent(BLACK, depth=MINIMAX_DEPTH)
    
    for i in tqdm(range(NUM_GAMES)):
        board.reset()
        game_over = False
        first_moves = 0
        
        while not game_over:
            # 2 nước đầu đi random để tạo thế cờ đa dạng
            if first_moves < 2:
                valid_moves = board.get_valid_moves()
                if not valid_moves: break
                move = random.choice(valid_moves)
                first_moves += 1
            else:
                teacher.player = board.turn
                teacher.opponent = -board.turn
                move = teacher.get_move(board)
            
            if move:
                # --- CHUẨN BỊ INPUT 2 KÊNH (QUAN TRỌNG) ---
                # Channel 0: Quân của người chơi hiện tại
                # Channel 1: Quân của đối thủ
                board_np = board.board
                current_player = board.turn
                
                my_pieces = (board_np == current_player).astype(np.float32)
                opp_pieces = (board_np == -current_player).astype(np.float32)
                
                # Input shape: (2, 15, 15)
                state_input = np.stack([my_pieces, opp_pieces])
                
                # Output: Index nước đi
                action_idx = move[0] * BOARD_SIZE + move[1]
                
                data.append((state_input, action_idx))
                
                # Đi quân
                board.make_move(move[0], move[1], board.turn)
                
                if board.check_win(board.turn) or board.is_full():
                    game_over = True
            else:
                break
                
    print(f"-> Thu thập được {len(data)} mẫu dữ liệu gốc.")
    return data

def augment_data(data):
    print("--- ĐANG TĂNG CƯỜNG DỮ LIỆU (AUGMENTATION) ---")
    augmented_data = []
    
    for state_input, action_idx in tqdm(data):
        # state_input shape: (2, 15, 15)
        # Cần tách ra để xoay
        my_p = state_input[0]
        opp_p = state_input[1]
        
        r, c = action_idx // BOARD_SIZE, action_idx % BOARD_SIZE
        move_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE))
        move_matrix[r, c] = 1
        
        for k in range(4):
            # Xoay bàn cờ (cả 2 kênh)
            rot_my = np.rot90(my_p, k)
            rot_opp = np.rot90(opp_p, k)
            rot_move = np.rot90(move_matrix, k)
            
            # Ghép lại
            rot_state = np.stack([rot_my, rot_opp])
            
            mr, mc = np.argwhere(rot_move == 1)[0]
            new_action = mr * BOARD_SIZE + mc
            augmented_data.append((rot_state.copy(), new_action))
            
            # Lật gương
            flip_my = np.fliplr(rot_my)
            flip_opp = np.fliplr(rot_opp)
            flip_move = np.fliplr(rot_move)
            
            flip_state = np.stack([flip_my, flip_opp])
            
            mr, mc = np.argwhere(flip_move == 1)[0]
            new_action_flip = mr * BOARD_SIZE + mc
            augmented_data.append((flip_state.copy(), new_action_flip))
            
    print(f"-> Tổng dữ liệu training: {len(augmented_data)} mẫu.")
    return augmented_data

def main():
    # XÓA FILE CŨ ĐỂ TRAIN LẠI TỪ ĐẦU CHO CHUẨN
    if os.path.exists(DATA_FILE):
        print("Đang tải dữ liệu cũ...")
        with open(DATA_FILE, 'rb') as f:
            dataset = pickle.load(f)
    else:
        raw_data = generate_data()
        dataset = augment_data(raw_data)
        with open(DATA_FILE, 'wb') as f:
            pickle.dump(dataset, f)

    # Khởi tạo và Train
    # Chú ý: Batch Size = 64, Epochs = 30
    ml_agent = MLAgent(BLACK)
    ml_agent.train_on_data(dataset, epochs=30, batch_size=64)

if __name__ == "__main__":
    main()