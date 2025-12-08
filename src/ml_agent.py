import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.constants import *

# Định nghĩa mạng Neural (CNN) cải tiến
class GomokuNet(nn.Module):
    def __init__(self):
        super(GomokuNet, self).__init__()
        # Input 2 kênh: Kênh quân mình và Kênh quân địch (Thay vì 1 kênh 1/-1)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) # Giúp ổn định quá trình học
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * BOARD_SIZE * BOARD_SIZE, 512)
        self.dropout = nn.Dropout(0.3) # Chống học vẹt (Overfitting)
        self.fc2 = nn.Linear(512, BOARD_SIZE * BOARD_SIZE) # Output 225 ô

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLAgent:
    def __init__(self, player_color, model_path="models/gomoku_model.pth"):
        self.player = player_color
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GomokuNet().to(self.device)
        self.model_path = model_path
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print("ML Agent: Đã load model thành công!")
            except:
                print("ML Agent: Lỗi file model cũ, hãy train lại!")
        else:
            print("ML Agent: Chưa có model, cần huấn luyện trước!")

    def get_move(self, board_obj):
        # CHUẨN BỊ INPUT 2 KÊNH (Channel 0: Quân mình, Channel 1: Quân địch)
        # Cách này giúp CNN học tốt hơn nhiều so với 1, -1
        board_np = board_obj.board
        
        my_pieces = (board_np == self.player).astype(np.float32)
        opp_pieces = (board_np == -self.player).astype(np.float32)
        
        # Stack lại thành (2, 15, 15)
        input_state = np.stack([my_pieces, opp_pieces])
        
        tensor_board = torch.FloatTensor(input_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor_board)
            # Lấy xác suất
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # reshape probs về 15x15
        probs_2d = probs.reshape(BOARD_SIZE, BOARD_SIZE)
        
        # LỌC NƯỚC ĐI HỢP LỆ
        # Chỉ lấy nước đi có trong valid_moves và có xác suất cao nhất
        valid_moves = board_obj.get_valid_moves()
        best_move = None
        best_prob = -1
        
        for move in valid_moves:
            prob = probs_2d[move[0]][move[1]]
            if prob > best_prob:
                best_prob = prob
                best_move = move
                
        return best_move if best_move else valid_moves[0]

    # Hàm huấn luyện chuyên nghiệp (Batch Training + Shuffle)
    def train_on_data(self, dataset_moves, epochs=20, batch_size=64):
        print(f"Bắt đầu train ML Model (Batch Size: {batch_size})...")
        
        # CHUẨN BỊ DỮ LIỆU
        X_list = []
        y_list = []
        
        for board_state, move_idx in dataset_moves:
            X_list.append(board_state)
            y_list.append(move_idx)
            
        # Chuyển sang Tensor
        X_tensor = torch.FloatTensor(np.array(X_list)) # Shape: (N, 2, 15, 15)
        y_tensor = torch.LongTensor(np.array(y_list))  # Shape: (N,)
        
        # Tạo DataLoader để Shuffle và Batching
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        # Lưu model
        if not os.path.exists("models"): os.makedirs("models")
        torch.save(self.model.state_dict(), self.model_path)
        print("Đã lưu model!")