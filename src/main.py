import pygame
import sys
from src.constants import *
from src.game_logic import Board
from src.ui import UI
from src.minimax_agent import MinimaxAgent
from src.ml_agent import MLAgent
def main():
    ui = UI()
    board = Board()
    
    # Trạng thái ban đầu
    current_state = STATE_MENU
    game_mode = "PVP" # PVP, PVE_EASY, PVE_HARD
    ai_agent = None
    
    # Game Loop
    clock = pygame.time.Clock()
    
    while True:
        clock.tick(60) # Giới hạn 60 FPS cho nhẹ máy
        
        # --- XỬ LÝ SỰ KIỆN ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                
                # 1. XỬ LÝ CLICK Ở MENU
                if current_state == STATE_MENU:
                    if ui.btn_pvp.is_clicked(pos):
                        game_mode = "PVP"
                        current_state = STATE_PLAYING
                        board.reset()
                    elif ui.btn_pve_easy.is_clicked(pos):
                        game_mode = "PVE_EASY"
                        ai_agent = MinimaxAgent(WHITE, depth=DEPTH_EASY)
                        current_state = STATE_PLAYING
                        board.reset()
                    elif ui.btn_pve_hard.is_clicked(pos):
                        game_mode = "PVE_HARD"
                        ai_agent = MLAgent(WHITE, model_path="models/gomoku_model.pth")
                        current_state = STATE_PLAYING
                        board.reset()
                    elif ui.btn_quit.is_clicked(pos):
                        pygame.quit()
                        sys.exit()

                # 2. XỬ LÝ CLICK KHI ĐANG CHƠI
                elif current_state == STATE_PLAYING:
                    # Nếu là lượt máy thì không nhận click
                    if "PVE" in game_mode and board.turn == WHITE:
                        continue
                        
                    move = ui.get_click_on_board(pos)
                    if move:
                        r, c = move
                        if board.make_move(r, c, board.turn):
                            if board.check_win(board.turn):
                                current_state = STATE_GAME_OVER
                            elif board.is_full():
                                current_state = STATE_GAME_OVER
                            else:
                                board.switch_turn()

                # 3. XỬ LÝ CLICK GAME OVER
                elif current_state == STATE_GAME_OVER:
                    if ui.btn_menu.is_clicked(pos):
                        current_state = STATE_MENU
                    elif ui.btn_restart.is_clicked(pos):
                        board.reset()
                        current_state = STATE_PLAYING

        # --- LOGIC AI (CHẠY TỰ ĐỘNG) ---
        if current_state == STATE_PLAYING and "PVE" in game_mode and board.turn == WHITE:
            # Vẽ bàn cờ trước để người chơi thấy nước mình vừa đi
            ui.draw_game(board.board)
            pygame.display.flip()
            
            # AI suy nghĩ
            move = ai_agent.get_move(board)
            if move:
                board.make_move(move[0], move[1], WHITE)
                if board.check_win(WHITE):
                    current_state = STATE_GAME_OVER
                elif board.is_full():
                    current_state = STATE_GAME_OVER
                else:
                    board.switch_turn()

        # --- VẼ MÀN HÌNH ---
        if current_state == STATE_MENU:
            ui.draw_menu()
        elif current_state == STATE_PLAYING:
            ui.draw_game(board.board)
        elif current_state == STATE_GAME_OVER:
            # Xác định text thắng thua (PHẢI Ở ĐÂY, trong cùng scope)
            winner_text = "Hòa cờ!"
            if board.winner == BLACK: 
                winner_text = "Bạn Thắng!" if "PVE" in game_mode else "Quân Đen Thắng!"
            elif board.winner == WHITE: 
                winner_text = "Máy Thắng!" if "PVE" in game_mode else "Quân Trắng Thắng!"
            
            # Vẽ tất cả trong 1 lần gọi để tránh chớp tắt
            ui.draw_game_over_combined(board.board, winner_text)

if __name__ == "__main__":
    main()