import pygame
from game_board import GameBoard
from player import Player
from rl_ai import RLAgent
import torch

def initialize_game():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Tron Game")
    return screen

def handle_events(player):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                player.change_direction([0, -1])
            elif event.key == pygame.K_DOWN:
                player.change_direction([0, 1])
            elif event.key == pygame.K_RIGHT:
                player.change_direction([1, 0])
            elif event.key == pygame.K_LEFT:
                player.change_direction([-1, 0])
    return True

def draw_game(screen, game_board, player1, player2):
    screen.fill((0, 0, 0))
    game_board.draw(screen)
    player1.draw(screen)
    player2.draw(screen)
    pygame.display.flip()

def run_game(ai_class1, ai_class2, model_file1=None, model_file2=None):
    screen = initialize_game()
    game_board = GameBoard(40, 30)
    
    ai1 = RLAgent(action_size=4, player_id=1, grid_size=15, model_file=model_file1) if ai_class1 == RLAgent else ai_class1()
    ai2 = RLAgent(action_size=4, player_id=2, grid_size=15, model_file=model_file2) if ai_class2 == RLAgent else ai_class2()
    
    # Validate models after loading
    dummy_input = torch.zeros((1, 3, 15, 15)).to(ai1.device)
    with torch.no_grad():
        print(f"AI1 dummy output: {ai1.model(dummy_input)}")
        print(f"AI2 dummy output: {ai2.model(dummy_input)}")
    
    player1 = Player(10, 15, (0, 0, 255), 1, ai1)
    player2 = Player(30, 15, (255, 0, 0), 2, ai2)
    
    game_board.grid[player1.y][player1.x] = player1.player_id
    game_board.grid[player2.y][player2.x] = player2.player_id
    

    # Set opponents for both players
    player1.set_opponent(player2)
    player2.set_opponent(player1)
    
    print(player1.controller.epsilon)

    clock = pygame.time.Clock()

    running = True
    while running:
        running = handle_events(player2)
        if running:
            player1.move(game_board)
            player2.move(game_board)
            if game_board.is_collision(player1.x, player1.y) or game_board.is_collision(player2.x, player2.y):
                running = False
            else:
                game_board.grid[player1.y][player1.x] = player1.player_id
                game_board.grid[player2.y][player2.x] = player2.player_id
        draw_game(screen, game_board, player1, player2)
        clock.tick(5)

    pygame.quit()
    
def run_human_game(ai_class1, model_file1=None):
    screen = initialize_game()
    game_board = GameBoard(40, 30)
    
    ai = RLAgent(action_size=4, player_id=1, grid_size=15, model_file=model_file1) if ai_class1 == RLAgent else ai_class1()
    
    # Validate models after loading
    # dummy_input = torch.zeros((1, 3, 15, 15)).to(ai.device)
    # with torch.no_grad():
    #     print(f"AI1 dummy output: {ai.model(dummy_input)}")
    
    player1 = Player(10, 15, (0, 0, 255), 1, ai=None)
    player2 = Player(30, 15, (255, 0, 0), 2, ai)
    
    game_board.grid[player1.y][player1.x] = player1.player_id
    game_board.grid[player2.y][player2.x] = player2.player_id
    

    # Set opponents for both players
    player1.set_opponent(player2)
    player2.set_opponent(player1)
    
    # print(player1.controller.epsilon)

    clock = pygame.time.Clock()

    running = True
    while running:
        running = handle_events(player1)
        if running:
            player1.move(game_board)
            player2.move(game_board)
            if game_board.is_collision(player1.x, player1.y) or game_board.is_collision(player2.x, player2.y) or (player1.x == player2.x and player1.y == player2.y):
                if game_board.is_collision(player1.x, player1.y):
                    print(f"AI wins!!")
                elif game_board.is_collision(player2.x, player2.y):
                    print("Human Wins!!")
                else:
                    print("Tie!!!")
                running = False
            else:
                game_board.grid[player1.y][player1.x] = player1.player_id
                game_board.grid[player2.y][player2.x] = player2.player_id
        draw_game(screen, game_board, player1, player2)
        clock.tick(10)

    pygame.time.delay(2000)
    pygame.quit()

if __name__ == "__main__":
    # drew.pth or carl.pth
    run_human_game(RLAgent, "carl.pth")