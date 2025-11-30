import pygame
from src.constants import *

class Button:
    def __init__(self, text, x, y, w, h, action=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action
        self.is_hovered = False

    def draw(self, screen, font):
        # Hiệu ứng đổi màu khi di chuột
        color = BUTTON_HOVER if self.is_hovered else BUTTON_COLOR
        
        # Vẽ nút phẳng 2D
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2, border_radius=8) # Viền trắng
        
        # Vẽ chữ
        text_surf = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

class UI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Gomoku AI - Professional Edition")
        
        # --- SỬA LỖI FONT TIẾNG VIỆT ---
        # Thử load các font hỗ trợ tiếng Việt tốt trên Windows
        # font.SysFont nhận vào tên font, nếu không có nó sẽ thử cái tiếp theo
        try:
            self.title_font = pygame.font.SysFont("segoeui, tahoma, arial", 50, bold=True)
            self.font = pygame.font.SysFont("segoeui, tahoma, arial", 22) # Tăng size lên 22 cho dễ đọc
        except:
            # Fallback nếu máy không có (ít khi xảy ra)
            self.title_font = pygame.font.Font(None, 50)
            self.font = pygame.font.Font(None, 22)
        
        # Tạo các nút bấm cho Menu
        center_x = SCREEN_WIDTH // 2 - 100
        # Căn chỉnh lại vị trí Y một chút cho cân đối
        self.btn_pvp = Button("Người vs Người", center_x, 260, 200, 50, "PVP")
        self.btn_pve_easy = Button("Máy (Dễ)", center_x, 330, 200, 50, "PVE_EASY")
        self.btn_pve_hard = Button("Máy (Khó)", center_x, 400, 200, 50, "PVE_HARD")
        self.btn_quit = Button("Thoát Game", center_x, 470, 200, 50, "QUIT")

        # Nút bấm cho Game Over
        self.btn_menu = Button("Về Menu", center_x - 110, 450, 100, 50, "MENU")
        self.btn_restart = Button("Chơi lại", center_x + 10, 450, 100, 50, "RESTART")

    def draw_menu(self):
        self.screen.fill(BG_COLOR)
        
        # Vẽ tiêu đề
        # Thêm bóng cho chữ tiêu đề nhìn cho nổi (tùy chọn)
        title_shadow = self.title_font.render("CỜ CARO AI", True, (100, 70, 40))
        self.screen.blit(title_shadow, (SCREEN_WIDTH//2 - title_shadow.get_width()//2 + 2, 102))
        
        title = self.title_font.render("CỜ CARO AI", True, (80, 50, 20)) # Màu nâu đậm
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 100))
        
        # Vẽ các nút
        mouse_pos = pygame.mouse.get_pos()
        for btn in [self.btn_pvp, self.btn_pve_easy, self.btn_pve_hard, self.btn_quit]:
            btn.check_hover(mouse_pos)
            btn.draw(self.screen, self.font)
        
        pygame.display.flip()

    def draw_game(self, board):
        self.screen.fill(BG_COLOR)
        
        # 1. Vẽ nền bàn cờ
        board_rect = (MARGIN_X - 20, MARGIN_Y - 20, BOARD_PIXEL_SIZE + 40, BOARD_PIXEL_SIZE + 40)
        pygame.draw.rect(self.screen, BOARD_COLOR, board_rect, border_radius=5)
        pygame.draw.rect(self.screen, LINE_COLOR, board_rect, 2, border_radius=5)

        # 2. Vẽ lưới
        for i in range(BOARD_SIZE):
            # Ngang
            start_x, start_y = MARGIN_X, MARGIN_Y + i * CELL_SIZE
            end_x, end_y = MARGIN_X + BOARD_PIXEL_SIZE, MARGIN_Y + i * CELL_SIZE
            pygame.draw.line(self.screen, LINE_COLOR, (start_x, start_y), (end_x, end_y), 1)
            # Dọc
            start_x, start_y = MARGIN_X + i * CELL_SIZE, MARGIN_Y
            end_x, end_y = MARGIN_X + i * CELL_SIZE, MARGIN_Y + BOARD_PIXEL_SIZE
            pygame.draw.line(self.screen, LINE_COLOR, (start_x, start_y), (end_x, end_y), 1)

        # 3. Vẽ điểm sao
        star_points = [3, 7, 11]
        for r in star_points:
            for c in star_points:
                cx = MARGIN_X + c * CELL_SIZE
                cy = MARGIN_Y + r * CELL_SIZE
                pygame.draw.circle(self.screen, LINE_COLOR, (cx, cy), 4)

        # 4. Vẽ quân cờ
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != EMPTY:
                    self.draw_piece(r, c, board[r][c])

        pygame.display.flip()

    def draw_piece(self, r, c, player):
        cx = MARGIN_X + c * CELL_SIZE
        cy = MARGIN_Y + r * CELL_SIZE
        # Giảm bán kính đi chút xíu để các quân không dính sát vào nhau
        radius = CELL_SIZE // 2 - 2 

        if player == BLACK:
            # --- QUÂN ĐEN 2D ---
            # Chỉ vẽ 1 hình tròn màu đen thuần
            pygame.draw.circle(self.screen, BLACK_COLOR, (cx, cy), radius)
        else:
            # --- QUÂN TRẮNG 2D ---
            # Vẽ hình tròn trắng
            pygame.draw.circle(self.screen, WHITE_COLOR, (cx, cy), radius)
            # Vẽ thêm viền đen mỏng (width=1) để quân trắng nổi bật trên nền gỗ
            pygame.draw.circle(self.screen, BLACK_COLOR, (cx, cy), radius, 1)

    def draw_game_over(self, winner_text):
        # Vẽ lớp phủ mờ
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180)) # Đậm hơn chút để dễ đọc chữ
        self.screen.blit(overlay, (0, 0))

        # Text kết quả
        # Vẽ viền đen cho chữ dễ đọc
        res_text = self.title_font.render(winner_text, True, (255, 215, 0)) # Màu vàng kim
        self.screen.blit(res_text, (SCREEN_WIDTH//2 - res_text.get_width()//2, 250))

        # Vẽ nút
        mouse_pos = pygame.mouse.get_pos()
        for btn in [self.btn_menu, self.btn_restart]:
            btn.check_hover(mouse_pos)
            btn.draw(self.screen, self.font)
        
        pygame.display.flip()

    def get_click_on_board(self, pos):
        x, y = pos
        # Kiểm tra click có trong vùng bàn cờ không
        if x < MARGIN_X - CELL_SIZE/2 or x > MARGIN_X + BOARD_PIXEL_SIZE + CELL_SIZE/2:
            return None
        if y < MARGIN_Y - CELL_SIZE/2 or y > MARGIN_Y + BOARD_PIXEL_SIZE + CELL_SIZE/2:
            return None

        # Tính tọa độ
        c = round((x - MARGIN_X) / CELL_SIZE)
        r = round((y - MARGIN_Y) / CELL_SIZE)
        
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            return r, c
        return None