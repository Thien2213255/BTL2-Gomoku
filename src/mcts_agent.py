import time
import math
import random
import copy
from src.constants import *

class MCTSNode:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state  # Board object copy
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.visits = 0
        self.wins = 0 # Từ góc nhìn của parent
        self.untried_moves = self.state.get_valid_moves()
        self.player = state.turn

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        # Công thức UCB1 để cân bằng giữa Khai thác (Exploitation) và Khám phá (Exploration)
        choices_weights = [
            (child.wins / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

class MCTSAgent:
    def __init__(self, player_color, simulation_time=3.0):
        self.player = player_color
        self.simulation_time = simulation_time # Thời gian suy nghĩ cho mỗi nước (giây)

    def get_move(self, board):
        # Gốc của cây tìm kiếm
        # Cần copy board để không làm hỏng bàn cờ thật
        root = MCTSNode(state=copy.deepcopy(board))

        end_time = time.time() + self.simulation_time
        
        # Vòng lặp MCTS: Chạy liên tục cho đến khi hết giờ
        while time.time() < end_time:
            node = root
            
            # 1. SELECTION (Chọn node lá)
            while not node.is_fully_expanded() and node.children:
                node = node.best_child()

            # 2. EXPANSION (Mở rộng node mới)
            if not node.is_fully_expanded():
                if node.untried_moves:
                    m = node.untried_moves.pop()
                    new_state = copy.deepcopy(node.state)
                    new_state.make_move(m[0], m[1], node.player)
                    new_state.switch_turn()
                    
                    new_node = MCTSNode(new_state, parent=node, parent_action=m)
                    node.children.append(new_node)
                    node = new_node
            
            # 3. SIMULATION (Mô phỏng ngẫu nhiên - Rollout)
            # Chơi ngẫu nhiên đến khi hết ván hoặc đạt giới hạn
            temp_state = copy.deepcopy(node.state)
            winner = self.rollout(temp_state)

            # 4. BACKPROPAGATION (Cập nhật kết quả ngược lên gốc)
            while node is not None:
                node.visits += 1
                if winner == node.state.turn: 
                    # Nếu người chơi tại node này thua (tức là đối thủ vừa đánh ở parent thắng)
                    # Logic MCTS hơi ngược một chút: 
                    # Node lưu trạng thái sau khi parent đánh.
                    pass 
                else:
                    # Nếu winner khác người chơi hiện tại, tức là nước đi dẫn đến node này tốt
                    if winner != 0: # Không hòa
                        node.wins += 1
                node = node.parent

        if not root.children:
            return random.choice(board.get_valid_moves())

        # Trả về node được thăm nhiều nhất (Robust Child)
        return root.best_child(c_param=0).parent_action

    def rollout(self, board):
        # Chơi ngẫu nhiên tối đa 30 nước nữa để tiết kiệm thời gian
        limit = 30 
        for _ in range(limit):
            if board.check_win(BLACK): return BLACK
            if board.check_win(WHITE): return WHITE
            if board.is_full(): return 0
            
            moves = board.get_valid_moves()
            if not moves: break
            
            # Heuristic uu tiên đánh gần tâm hơn là random hoàn toàn
            move = random.choice(moves)
            board.make_move(move[0], move[1], board.turn)
            board.switch_turn()
            
        return 0 # Coi như hòa nếu chưa phân thắng bại sau giới hạn