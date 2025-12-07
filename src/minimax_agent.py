import random
import math
import numpy as np
from src.constants import *

class MinimaxAgent:
    def __init__(self, player_color, depth=3):
        self.player = player_color
        self.opponent = -player_color
        self.depth = depth
        
        # --- TRANSPOSITION TABLE (CACHE) ---
        # Lưu trữ các thế cờ đã tính
        # Key: Board string/bytes, Value: (score, depth, flag, best_move)
        self.transposition_table = {} 

    def get_move(self, board_obj):
        # Nếu bàn cờ trống, đánh vào tâm
        if len(board_obj.history) == 0:
            return (BOARD_SIZE // 2, BOARD_SIZE // 2)

        # Xóa Cache cũ nếu quá lớn để tránh tràn RAM (tùy chọn)
        if len(self.transposition_table) > 100000:
            self.transposition_table.clear()

        # Gọi Minimax
        # Lúc này có thể tự tin để depth cao hơn
        _, best_move = self.minimax(board_obj, self.depth, -math.inf, math.inf, True)
        
        if best_move is None:
            valid_moves = board_obj.get_valid_moves()
            if valid_moves:
                return valid_moves[0]
                
        return best_move

    def minimax(self, board_obj, depth, alpha, beta, is_maximizing):
        # 1. TẠO KEY CHO THẾ CỜ HIỆN TẠI
        # Dùng tobytes() của numpy cực nhanh để làm key hash
        board_key = board_obj.board.tobytes()
        
        # 2. KIỂM TRA TRANSPOSITION TABLE (CACHE)
        if board_key in self.transposition_table:
            entry_score, entry_depth, entry_flag, entry_move = self.transposition_table[board_key]
            # Chỉ dùng kết quả cache nếu độ sâu đã tính (entry_depth) >= độ sâu hiện tại (depth)
            if entry_depth >= depth:
                if entry_flag == 'EXACT':
                    return entry_score, entry_move
                elif entry_flag == 'LOWERBOUND':
                    alpha = max(alpha, entry_score)
                elif entry_flag == 'UPPERBOUND':
                    beta = min(beta, entry_score)
                
                if alpha >= beta:
                    return entry_score, entry_move

        # 3. KIỂM TRA TRẠNG THÁI KẾT THÚC
        if board_obj.check_win(self.player):
            return WIN_SCORE - (10 - depth), None # Thắng càng sớm càng tốt
        if board_obj.check_win(self.opponent):
            return -WIN_SCORE + (10 - depth), None # Thua càng muộn càng tốt
        if depth == 0 or board_obj.is_full():
            return self.evaluate_board(board_obj.board, self.player), None

        # 4. LẤY NƯỚC ĐI & SẮP XẾP
        valid_moves = board_obj.get_valid_moves()
        
        # Sắp xếp đơn giản theo khoảng cách tâm (Hiệu năng cao)
        center = BOARD_SIZE // 2
        valid_moves.sort(key=lambda m: abs(m[0]-center) + abs(m[1]-center))

        best_move = valid_moves[0]
        original_alpha = alpha # Lưu alpha gốc để xác định flag lưu cache

        if is_maximizing:
            max_eval = -math.inf
            for move in valid_moves:
                board_obj.make_move(move[0], move[1], self.player)
                
                # Truyền alpha, beta vào đệ quy
                eval_score, _ = self.minimax(board_obj, depth - 1, alpha, beta, False)
                
                board_obj.undo_move()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            final_score = max_eval
        else:
            min_eval = math.inf
            for move in valid_moves:
                board_obj.make_move(move[0], move[1], self.opponent)
                
                eval_score, _ = self.minimax(board_obj, depth - 1, alpha, beta, True)
                
                board_obj.undo_move()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            final_score = min_eval

        # 5. LƯU KẾT QUẢ VÀO TRANSPOSITION TABLE
        flag = 'EXACT'
        if final_score <= original_alpha:
            flag = 'UPPERBOUND'
        elif final_score >= beta:
            flag = 'LOWERBOUND'
            
        self.transposition_table[board_key] = (final_score, depth, flag, best_move)
        
        return final_score, best_move

    def evaluate_board(self, board, player):
        # Heuristic cải tiến: Nhanh và Chính xác
        score = 0
        lines = self.get_all_lines(board)
        score += self.evaluate_lines_score(lines, player)
        score -= self.evaluate_lines_score(lines, -player) * 1.5 # Tăng hệ số phòng thủ lên 1.5
        return score

    def get_all_lines(self, board):
        lines = []
        # Ngang
        for r in range(BOARD_SIZE):
            lines.append(board[r, :])
        # Dọc
        for c in range(BOARD_SIZE):
            lines.append(board[:, c])
        # Chéo (Dùng numpy lấy chéo siêu nhanh)
        # Chỉ lấy đường chéo có độ dài >= 5
        for k in range(-BOARD_SIZE + 5, BOARD_SIZE - 4):
            lines.append(np.diagonal(board, offset=k))
            lines.append(np.diagonal(np.fliplr(board), offset=k))
        return lines

    def evaluate_lines_score(self, lines, player):
        total_score = 0
        opponent = -player
        
        # Biến đếm số lượng pattern đặc biệt để phát hiện đánh đôi (Double Threat)
        count_open_four = 0
        count_open_three = 0
        
        for line in lines:
            if player not in line:
                continue

            # Vectorization: Map numpy array sang chuỗi ký tự
            # Player -> 1, Opponent -> 2, Empty -> 0
            arr = line.copy()
            s_arr = np.zeros_like(arr)
            s_arr[arr == player] = 1
            s_arr[arr == opponent] = 2
            s = "".join(s_arr.astype(str))
            
            # --- CHECK PATTERN TỪ MẠNH ĐẾN YẾU ---
            
            # 1. WIN (5 con): Thắng ngay lập tức
            if "11111" in s:
                return WIN_SCORE
            
            # 2. OPEN FOUR (011110)
            # Dùng s.count() để đề phòng trường hợp hiếm: 1 dòng có 2 cái open four
            c_o4 = s.count("011110")
            if c_o4 > 0:
                total_score += OPEN_FOUR * c_o4
                count_open_four += c_o4
                continue # Đã tìm thấy pattern mạnh nhất dòng này, bỏ qua pattern yếu hơn
            
            # 3. BLOCKED FOUR (011112, 11110...)
            if "11110" in s or "01111" in s:
                total_score += BLOCKED_FOUR
                continue
                
            # 4. OPEN THREE (01110)
            # Kiểm tra kỹ hơn: Open three thật sự phải không bị chặn lén ở xa
            # Tuy nhiên để tối ưu tốc độ, ta chấp nhận pattern "01110"
            c_o3 = s.count("01110")
            if c_o3 > 0:
                total_score += OPEN_THREE * c_o3
                count_open_three += c_o3
                continue
            
            # Check Broken Three (011010, 010110) - Loại này yếu hơn Open Three chuẩn
            if "011010" in s or "010110" in s:
                total_score += OPEN_THREE
                # Broken three cũng được tính vào count_open_three để tạo nước đôi
                count_open_three += 1 
                continue
                
            # 5. BLOCKED THREE
            if "11100" in s or "00111" in s or "11010" in s or "01011" in s:
                total_score += BLOCKED_THREE
                continue

            # 6. OPEN TWO
            if "0110" in s:
                total_score += OPEN_TWO
                continue
        
        # --- XỬ LÝ NƯỚC ĐÔI (DOUBLE THREATS) ---
        
        # Double Four: 2 đường 4 mở -> Không thể đỡ -> Điểm gần bằng thắng
        if count_open_four >= 2:
            total_score += WIN_SCORE // 2 
            
        # Double Three: 2 đường 3 mở -> Đối thủ chặn 1 thì mình đi cái kia thành 4 mở -> Thắng
        # Điểm này phải CAO HƠN Blocked Four (vì Blocked Four đối thủ còn đỡ được)
        elif count_open_three >= 2:
            total_score += WIN_SCORE // 4  # Khoảng 25.000.000 điểm -> Ưu tiên cực cao

        return total_score