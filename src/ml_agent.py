import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.constants import *

# Định nghĩa mạng Neural (CNN)
class GomokuNet(nn.Module):
    def __init__(self):
        super(GomokuNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * BOARD_SIZE * BOARD_SIZE, 256)
        self.fc2 = nn.Linear(256, BOARD_SIZE * BOARD_SIZE) # Output 225 ô

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLAgent:
    def __init__(self, player_color, model_path="models/gomoku_model.pth"):
        self.player = player_color
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GomokuNet().to(self.device)
        self.model_path = model_path
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("ML Agent: Đã load model thành công!")
        else:
            print("ML Agent: Chưa có model, cần huấn luyện trước!")

    def get_move(self, board_obj):
        # Chuyển bàn cờ thành Tensor
        # Agent cần biết quân mình là 1, đối thủ là -1 (hoặc ngược lại) để chuẩn hóa
        board_input = board_obj.board.copy()
        if self.player == WHITE: # Nếu Agent là White (-1), đảo ngược lại để model dễ hiểu
             board_input = -board_input
             
        tensor_board = torch.FloatTensor(board_input).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor_board)
            # Lấy xác suất
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # Chọn nước đi hợp lệ có xác suất cao nhất
        valid_moves = board_obj.get_valid_moves()
        best_move = None
        best_prob = -1
        
        # reshape probs về 15x15
        probs_2d = probs.reshape(BOARD_SIZE, BOARD_SIZE)
        
        for move in valid_moves:
            prob = probs_2d[move[0]][move[1]]
            if prob > best_prob:
                best_prob = prob
                best_move = move
                
        return best_move if best_move else valid_moves[0]

    # Hàm huấn luyện (Behavioral Cloning)
    # Học theo Minimax hoặc Data có sẵn
    def train_on_data(self, dataset_moves, epochs=10):
        print("Bắt đầu train ML Model...")
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for board_state, best_move_idx in dataset_moves:
                # board_state: 15x15 matrix, best_move_idx: flat index (0-224)
                inputs = torch.FloatTensor(board_state).unsqueeze(0).unsqueeze(0).to(self.device)
                targets = torch.LongTensor([best_move_idx]).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset_moves):.4f}")
            
        # Lưu model
        if not os.path.exists("models"): os.makedirs("models")
        torch.save(self.model.state_dict(), self.model_path)
        print("Đã lưu model!")