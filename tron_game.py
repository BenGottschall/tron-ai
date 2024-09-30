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
    

def handle_events(player):
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
                player.change_direction([0, -1])
            elif event.key == pygame.K_DOWN:
                player.change_direction([0, 1])
            elif event.key == pygame.K_RIGHT:
                player.change_direction([1, 0])
            elif event.key == pygame.K_LEFT:
                player.change_direction([-1, 0])
            
    return True


def update_game_state(player, game_board):
    """
    Update the game state, including player movement and collision detection.
    :param player: Player object to update
    :param game_board: GameBoard object to check collisions against
    :return: False if the game is over (collision), True otherwise
    """
    # TODO: Move the player
    # Check for collisions with game_board
    # Update game_board with new player position
    #player.update_direction()
    player.move()
    print(f"[{player.x // CELL_SIZE}, {player.y // CELL_SIZE }]")
    if game_board.is_collision(player.x // CELL_SIZE, player.y // CELL_SIZE):
        return False
    game_board.board[player.x // CELL_SIZE][player.y // CELL_SIZE] = 1
    
    return True

    

def draw_game(screen, game_board, player):
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
    #screen.fill((0, 0, 0))  # Clear the screen with black
    #game_board.draw(screen)
    player.draw(screen)
    pygame.display.flip()

def main():
    """
    Main game loop.
    """
    # TODO: Initialize the game
    # Create game objects (game_board, player)
    # Run the game loop:
    #   - Handle events
    #   - Update game state
    #   - Draw game
    #   - Control game speed
    screen = initialize_game()
    game_board = GameBoard(BOARD_WIDTH, BOARD_HEIGHT)
    player1 = Player(PLAYER_START[0], PLAYER_START[1], COLORS["player1"])
    
    game_board.draw(screen)

    
    running = True
    while running:
        if not handle_events(player1):
            running = False
        if not update_game_state(player1, game_board):
            running = False
            # create end event
        draw_game(screen, game_board, player1)
        pygame.time.delay(60)

main()

# Quit Pygame
pygame.quit()