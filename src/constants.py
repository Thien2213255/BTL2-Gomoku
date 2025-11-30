# CẤU HÌNH HỆ THỐNG
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700 # Làm màn hình chữ nhật để có chỗ để menu bên cạnh hoặc thông báo

# Kích thước bàn cờ (Quay về 15x15 chuẩn Gomoku để dễ nhìn và tính toán)
BOARD_SIZE = 15 
CELL_SIZE = 40
# Tự động tính lề để bàn cờ nằm giữa theo chiều dọc
BOARD_PIXEL_SIZE = (BOARD_SIZE - 1) * CELL_SIZE
MARGIN_X = (SCREEN_WIDTH - BOARD_PIXEL_SIZE) // 2
MARGIN_Y = (SCREEN_HEIGHT - BOARD_PIXEL_SIZE) // 2

# GAME STATES
STATE_MENU = "menu"
STATE_PLAYING = "playing"
STATE_GAME_OVER = "game_over"

# MÀU SẮC (THEME GỖ SANG TRỌNG)
BG_COLOR = (240, 217, 181)       # Màu nền gỗ sáng
BOARD_COLOR = (220, 179, 92)     # Màu bàn cờ gỗ đậm
LINE_COLOR = (80, 50, 20)        # Màu đường kẻ nâu sẫm
HIGHLIGHT_COLOR = (255, 0, 0)    # Màu viền khi chọn

# Màu quân cờ (Giả lập 3D)
BLACK_COLOR = (20, 20, 20)
BLACK_SHADOW = (50, 50, 50)
WHITE_COLOR = (240, 240, 240)
WHITE_SHADOW = (200, 200, 200)

# UI Colors
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER = (100, 149, 237)
TEXT_COLOR = (50, 50, 50)

# GIÁ TRỊ GAME
EMPTY = 0
BLACK = 1
WHITE = -1

# Minimax Config
DEPTH_EASY = 1
DEPTH_HARD = 3

# ĐIỂM SỐ HEURISTIC (CẬP NHẬT MỚI)
# Điểm càng cao AI càng ưu tiên
WIN_SCORE = 100000000        # Thắng ngay lập tức (5 con)
OPEN_FOUR = 10000000         # 4 con, 2 đầu trống -> Chắc chắn thắng lượt sau
BLOCKED_FOUR = 1000000       # 4 con, bị chặn 1 đầu -> Phải đi nước này để thắng hoặc đỡ
OPEN_THREE = 100000          # 3 con, 2 đầu trống
BLOCKED_THREE = 10000        # 3 con, bị chặn 1 đầu
OPEN_TWO = 1000              # 2 con, 2 đầu trống
BLOCKED_TWO = 100            # 2 con, bị chặn 1 đầu