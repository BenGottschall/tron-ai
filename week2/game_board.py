import pygame;

class GameBoard:
    def __init__(self, width, height):
        """
        Initialize the game board.
        :param width: Width of the game board in grid cells
        :param height: Height of the game board in grid cells
        """
        self.width = width
        self.height = height
        self.board = [[0 for _ in range(width)] for _ in range(height)]
        

    def draw(self, screen):
        """
        Draw the game board on the screen.
        :param screen: Pygame screen object to draw on
        """
        # TODO: Iterate through the 2D list and draw rectangles for each cell
        # Empty cells can be one color, trails another

    def is_collision(self, x, y):
        """
        Check if the given coordinates collide with the board boundaries or a trail.
        :param x: X-coordinate to check
        :param y: Y-coordinate to check
        :return: True if collision, False otherwise
        """
        # TODO: Check if x and y are within board boundaries
        # Also check if the cell at (x, y) is not empty (i.e., has a trail)