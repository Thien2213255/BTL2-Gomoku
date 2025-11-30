import random
import math
import numpy as np
from src.constants import *

class MinimaxAgent:
    def __init__(self, player_color, depth=2):
        self.player = player_color
        self.opponent = -player_color
        self.depth = depth

    def get_move(self, board_obj):
        # Nếu bàn cờ trống, đánh vào tâm (Thiên Nguyên)
        if len(board_obj.history) == 0:
            return (BOARD_SIZE // 2, BOARD_SIZE // 2)

        # Gọi Minimax
        # Alpha: Điểm tốt nhất cho Máy
        # Beta: Điểm tốt nhất cho Người (máy giả định người chơi hay nhất)
        _, best_move = self.minimax(board_obj, self.depth, -math.inf, math.inf, True)
        
        # Phòng trường hợp không tìm được nước đi (hiếm), trả về nước random hợp lệ
        if best_move is None:
            valid_moves = board_obj.get_valid_moves()
            if valid_moves:
                return valid_moves[0]
                
        return best_move

    def minimax(self, board_obj, depth, alpha, beta, is_maximizing):
        # 1. Kiểm tra trạng thái kết thúc (Terminal State)
        if board_obj.check_win(self.player):
            return WIN_SCORE, None
        if board_obj.check_win(self.opponent):
            return -WIN_SCORE, None
        if depth == 0 or board_obj.is_full():
            return self.evaluate_board(board_obj.board, self.player), None

        # 2. Lấy các nước đi khả thi (đã tối ưu vùng lân cận ở game_logic)
        valid_moves = board_obj.get_valid_moves()
        
        # Sắp xếp nước đi để cắt tỉa Alpha-Beta hiệu quả hơn
        # Ưu tiên các ô gần trung tâm bàn cờ hơn
        center = BOARD_SIZE // 2
        valid_moves.sort(key=lambda m: abs(m[0]-center) + abs(m[1]-center))

        best_move = valid_moves[0]

        if is_maximizing:
            max_eval = -math.inf
            for move in valid_moves:
                board_obj.make_move(move[0], move[1], self.player)
                current_eval, _ = self.minimax(board_obj, depth - 1, alpha, beta, False)
                board_obj.undo_move()

                if current_eval > max_eval:
                    max_eval = current_eval
                    best_move = move
                
                alpha = max(alpha, current_eval)
                if beta <= alpha: # Cắt tỉa
                    break
            return max_eval, best_move
        
        else: # Lượt người chơi (Minimizing)
            min_eval = math.inf
            for move in valid_moves:
                board_obj.make_move(move[0], move[1], self.opponent)
                current_eval, _ = self.minimax(board_obj, depth - 1, alpha, beta, True)
                board_obj.undo_move()

                if current_eval < min_eval:
                    min_eval = current_eval
                    best_move = move
                
                beta = min(beta, current_eval)
                if beta <= alpha: # Cắt tỉa
                    break
            return min_eval, best_move

    def evaluate_board(self, board, player):
        """
        Hàm Heuristic "xịn" hơn:
        - Quét cả 4 hướng: Ngang, Dọc, Chéo Chính, Chéo Phụ.
        - Đổi các hàng thành chuỗi string để so khớp mẫu (Pattern Matching).
        """
        score = 0
        
        # Chuyển bàn cờ về dạng list các chuỗi để dễ xử lý
        lines = self.get_all_lines(board)
        
        # Tính điểm cho Máy (Tấn công)
        score += self.evaluate_lines_score(lines, player)
        
        # Tính điểm cho Người (Phòng thủ)
        # Nhân hệ số cao hơn một chút để AI biết sợ thua
        score -= self.evaluate_lines_score(lines, -player) * 1.2 
        
        return score

    def get_all_lines(self, board):
        lines = []
        
        # 1. Hàng ngang
        for r in range(BOARD_SIZE):
            lines.append(board[r, :])
            
        # 2. Hàng dọc
        for c in range(BOARD_SIZE):
            lines.append(board[:, c])
            
        # 3. Đường chéo (Chỉ lấy các đường dài >= 5 ô)
        # Chéo chính (Top-left to Bottom-right)
        for k in range(-BOARD_SIZE + 5, BOARD_SIZE - 4):
            diag = np.diagonal(board, offset=k)
            if len(diag) >= 5:
                lines.append(diag)
                
        # Chéo phụ (Top-right to Bottom-left)
        # Lật ngược bàn cờ trái-phải rồi lấy chéo chính
        flipped_board = np.fliplr(board)
        for k in range(-BOARD_SIZE + 5, BOARD_SIZE - 4):
            diag = np.diagonal(flipped_board, offset=k)
            if len(diag) >= 5:
                lines.append(diag)
                
        return lines

    def evaluate_lines_score(self, lines, player):
        total_score = 0
        opponent = -player
        
        # Chuyển đổi giá trị quân cờ để dễ so khớp chuỗi
        # 1: Quân mình, 2: Quân địch, 0: Ô trống
        # Ví dụ: chuỗi [1, 1, 1, 0] -> "1110"
        
        for line in lines:
            # Chuyển numpy array thành string
            # map: player -> '1', opponent -> '2', empty -> '0'
            s = ""
            for cell in line:
                if cell == player: s += "1"
                elif cell == opponent: s += "2"
                else: s += "0"
            
            # --- SO KHỚP MẪU (PATTERN MATCHING) ---
            
            # 1. Check Win (5 con liên tiếp)
            if "11111" in s:
                return WIN_SCORE # Thắng luôn, không cần tính tiếp
            
            # 2. Open Four (4 con, 2 đầu trống: 011110) -> Không thể đỡ
            if "011110" in s:
                total_score += OPEN_FOUR
                
            # 3. Blocked Four (4 con, bị chặn 1 đầu: 211110 hoặc 011112 hoặc ra mép bàn)
            # Vì ta chỉ check substring nên check cả 2 trường hợp
            if "01111" in s or "11110" in s:
                total_score += BLOCKED_FOUR
            
            # 4. Open Three (3 con, 2 đầu trống: 01110)
            # Có thể phát triển thành Open Four
            if "01110" in s: # Mẫu chuẩn
                total_score += OPEN_THREE
            elif "010110" in s or "011010" in s: # Mẫu gãy (Broken three)
                total_score += OPEN_THREE * 0.8
                
            # 5. Blocked Three
            if "001112" in s or "211100" in s or "210110" in s or "011012" in s:
                total_score += BLOCKED_THREE
            
            # 6. Open Two (0110)
            if "001100" in s or "01010" in s:
                total_score += OPEN_TWO

        return total_score