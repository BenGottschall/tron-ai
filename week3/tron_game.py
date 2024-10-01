import pygame
from game_board import GameBoard
from player import Player
from config import *

def initialize_game():
    """
    Initialize Pygame and create the game window.
    :return: Pygame screen object
    """
    # TODO: Initialize Pygame
    # Create and return a Pygame screen object
    pygame.init()
    # Set up the game window
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("My First Tron Game")
    return screen
    

def handle_events(player1, player2):
    """
    Handle Pygame events, including player input.
    :param player: Player object to update based on input
    :return: False if the game should quit, True otherwise
    """
    # TODO: Loop through Pygame events
    # Handle QUIT event
    # Handle KEYDOWN events to change player direction
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                player1.change_direction([0, -1])
            elif event.key == pygame.K_DOWN:
                player1.change_direction([0, 1])
            elif event.key == pygame.K_RIGHT:
                player1.change_direction([1, 0])
            elif event.key == pygame.K_LEFT:
                player1.change_direction([-1, 0])
                
            elif event.key == pygame.K_w:
                player2.change_direction([0, -1])
            elif event.key == pygame.K_s:
                player2.change_direction([0, 1])
            elif event.key == pygame.K_d:
                player2.change_direction([1, 0])
            elif event.key == pygame.K_a:
                player2.change_direction([-1, 0])
            
    return True


def update_game_state(player1, player2, game_board):
    """
    Update the game state, including player movement and collision detection.
    :param player: Player object to update
    :param game_board: GameBoard object to check collisions against
    :return: False if the game is over (collision), True otherwise
    """
    # TODO: Move the player
    # Check for collisions with game_board
    # Update game_board with new player position
    player1.update_direction()
    player2.update_direction()
    
    print(f"Player1: [{player1.x}, {player1.y }]\nPlayer2: [{player2.x}, {player2.y}]")
    
    player1_collision = game_board.is_collision(player1.x_next, player1.y_next)
    player2_collision = game_board.is_collision(player2.x_next, player2.y_next)
    
    if player1_collision and player2_collision:
        print("Both")
        return False
    elif player1_collision:
        print("Player1")
        return False
    elif player2_collision:
        print("Player2")
        return False
    
    player1.move()
    player2.move()
    
    game_board.board[player1.x][player1.y] = 1
    game_board.board[player2.x][player2.y] = 1
    
    return True

    

def draw_game(screen, game_board, player1, player2):
    """
    Draw the current game state.
    :param screen: Pygame screen object to draw on
    :param game_board: GameBoard object to draw
    :param player: Player object to draw
    """
    # TODO: Clear the screen
    # Draw the game board
    # Draw the player
    # Update the display
    player1.draw(screen)
    player2.draw(screen)
    pygame.display.flip()

def main():
    """
    Main game loop.
    """
    # Initialize the game
    screen = initialize_game()
    
    # Create game objects (game_board, player1, player2)
    game_board = GameBoard(BOARD_WIDTH, BOARD_HEIGHT)
    player1 = Player(PLAYER1_START[0], PLAYER1_START[1], COLORS["player1"], 1)
    player2 = Player(PLAYER2_START[0], PLAYER2_START[1], COLORS["player2"], 2)
    
    # Draw the background grid
    game_board.draw(screen)

    # Run the game loop
    running = True
    while running:
        if not handle_events(player1, player2):
            running = False
        if not update_game_state(player1, player2, game_board):
            running = False
            # TODO: create end event
        
        draw_game(screen, game_board, player1, player2)
        
        pygame.time.delay(GAME_SPEED)

main()

# Quit Pygame
pygame.quit()