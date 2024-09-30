import pygame
from config import *

class GameBoard:
    def __init__(self, width, height):
        """
        Initialize the game board.
        :param width: Width of the game board in grid cells
        :param height: Height of the game board in grid cells
        """
        self.width = width
        self.height = height
        self.board = [[0 for _ in range(height)] for _ in range(width)]
        print(f"Size of gameboard: [{len(self.board[0])} rows, {len(self.board)} columns]")
        

    def draw(self, screen):
        """
        Draw the game board on the screen.
        :param screen: Pygame screen object to draw on
        """
        BLACK = (0, 0, 0)
        WHITE = (200, 200, 200)
        screen.fill(BLACK)
        # Iterate through the 2D list and draw rectangles for each cell
        # Empty cells can be one color, trails another
        cell_width = screen.get_width() // self.width
        cell_height = screen.get_height() // self.height
        
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x * cell_width, y * cell_height, cell_width, cell_height)
                pygame.draw.rect(screen, WHITE, rect, 1)
                
                # if self.board[x][y] == 1:
                #     pygame.Surface.fill(screen, WHITE, rect)
                    

    def is_collision(self, x, y):
        """
        Check if the given coordinates collide with the board boundaries or a trail.
        :param x: X-coordinate to check
        :param y: Y-coordinate to check
        :return: True if collision, False otherwise
        """
        
        cell_x = x
        cell_y = y
        print(f"[{cell_x}, {cell_y}]")
        #print(f"{cell_x}, {cell_y}")
        # Check if x and y are within board boundaries
        # Also check if the cell at (x, y) is not empty (i.e., has a trail)
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            print("hit the border")
            return True
        if self.board[cell_x][cell_y]:
            print("collision")
            return True
        return False
        
        