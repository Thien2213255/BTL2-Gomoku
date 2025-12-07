import numpy as np
from src.constants import *

class Board:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.history = []
        self.winner = None
        self.turn = BLACK 

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.history = []
        self.winner = None
        self.turn = BLACK

    def switch_turn(self):
        self.turn = WHITE if self.turn == BLACK else BLACK

    def get_valid_moves(self):
        """
        TỐI ƯU 1: Giảm bán kính tìm kiếm từ 2 xuống 1 (Neighbor Heuristic).
        """
        # Nếu chưa có nước nào, đánh vào tâm
        if len(self.history) == 0:
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
        
        valid_moves = set()
        
        # Lấy tọa độ tất cả các quân đã đánh
        rows, cols = np.where(self.board != EMPTY)
        
        # Chỉ xét bán kính 1 ô (range -1 đến 2) thay vì 2
        # Giúp giảm hệ số nhánh cực lớn cho Minimax
        for r, c in zip(rows, cols):
            for dr in range(-1, 2): 
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0: continue
                    
                    nr, nc = r + dr, c + dc
                    
                    # Kiểm tra ô đó nằm trong bàn cờ và đang trống
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if self.board[nr, nc] == EMPTY:
                            valid_moves.add((nr, nc))
        
        # Phòng trường hợp hiếm hoi không còn ô lân cận nào trống (gần cuối game)
        # Thì trả về tất cả các ô trống còn lại
        if not valid_moves:
            empty_cells = np.argwhere(self.board == EMPTY)
            return [tuple(x) for x in empty_cells]

        return list(valid_moves)

    def make_move(self, row, col, player):
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and self.board[row][col] == EMPTY:
            self.board[row][col] = player
            self.history.append((row, col))
            return True
        return False

    def undo_move(self):
        if self.history:
            r, c = self.history.pop()
            self.board[r][c] = EMPTY
            self.winner = None

    def check_win(self, player):
        """
        TỐI ƯU 2: Chỉ kiểm tra thắng thua dựa trên nước đi cuối cùng (Last Move).
        Không quét lại cả bàn cờ.
        """
        # Nếu chưa có nước đi nào hoặc người chơi vừa đi không phải là người cần check
        if not self.history:
            return False
            
        last_r, last_c = self.history[-1]
        
        # Nếu quân ở nước đi cuối cùng không phải của player này 
        # (nghĩa là check xem đối thủ có thắng trước đó không - logic Minimax)
        if self.board[last_r][last_c] != player:
            # Fallback: Quét toàn bộ (chậm hơn nhưng an toàn cho logic check đối thủ)
            return self.check_win_full(player)

        # Kiểm tra 4 hướng xung quanh nước vừa đi
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self.check_direction(last_r, last_c, dr, dc, player):
                self.winner = player
                return True
        return False

    def check_win_full(self, player):
        # Quét toàn bộ bàn (Dùng khi cần thiết)
        rows, cols = np.where(self.board == player)
        for r, c in zip(rows, cols):
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dr, dc in directions:
                if self.check_direction(r, c, dr, dc, player):
                    self.winner = player
                    return True
        return False

    def check_direction(self, r, c, dr, dc, player):
        # Đếm số quân liên tiếp về 2 phía của hướng (dr, dc)
        count = 1 # Tính cả quân gốc
        
        # Hướng dương
        for i in range(1, 5):
            nr, nc = r + i*dr, c + i*dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr][nc] == player:
                count += 1
            else:
                break
        
        # Hướng âm (ngược lại)
        for i in range(1, 5):
            nr, nc = r - i*dr, c - i*dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr][nc] == player:
                count += 1
            else:
                break
                
        return count >= 5

    def is_full(self):
        return len(self.history) == BOARD_SIZE * BOARD_SIZE