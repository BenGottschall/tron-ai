import pygame
from game_board import GameBoard
from player import Player
from rl_ai import RLAgent

def initialize_game():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Tron Game")
    return screen

def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True

def draw_game(screen, game_board, player):
    screen.fill((0, 0, 0))
    game_board.draw(screen)
    player.draw(screen)
    pygame.display.flip()

def run_game(ai_class, model_file=None):
    screen = initialize_game()
    game_board = GameBoard(40, 30)
    
    if ai_class == RLAgent:
        ai = RLAgent(state_size=7*7, action_size=4, model_file=model_file)
    else:
        ai = ai_class()
    
    player = Player(20, 15, (255, 0, 0), 1, ai)
    clock = pygame.time.Clock()

    running = True
    while running:
        running = handle_events()
        if running:
            player.move(game_board)
            if game_board.is_collision(player.x, player.y):
                running = False
            else:
                game_board.grid[player.y][player.x] = player.player_id
        draw_game(screen, game_board, player)
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    run_game(RLAgent, "tron_survival_model.pth")