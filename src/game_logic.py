import numpy as np
from src.constants import *

class Board:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.history = []
        self.winner = None
        self.turn = BLACK # Mặc định Đen đi trước

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.history = []
        self.winner = None
        self.turn = BLACK

    def switch_turn(self):
        self.turn = WHITE if self.turn == BLACK else BLACK

    def get_valid_moves(self):
        if len(self.history) == 0:
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
        
        valid_moves = set()
        rows, cols = np.where(self.board != EMPTY)
        
        # Chỉ xét các ô lân cận trong bán kính 2
        for r, c in zip(rows, cols):
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr, nc] == EMPTY:
                        valid_moves.add((nr, nc))
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
        rows, cols = np.where(self.board == player)
        for r, c in zip(rows, cols):
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dr, dc in directions:
                if self.check_direction(r, c, dr, dc, player):
                    self.winner = player
                    return True
        return False

    def check_direction(self, r, c, dr, dc, player):
        count = 0
        for i in range(5):
            nr, nc = r + i*dr, c + i*dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr][nc] == player:
                count += 1
            else:
                break
        return count == 5

    def is_full(self):
        return np.all(self.board != EMPTY)