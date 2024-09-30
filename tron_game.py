import pygame
from week2.game_board import GameBoard

# Initialize Pygame
pygame.init()


# Set up the game window
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("My First Tron Game")
GameBoard.__init__(GameBoard, 40, 40)


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            

    GameBoard.draw(GameBoard, screen)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()