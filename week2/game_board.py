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
        BLACK = (0, 0, 0)
        WHITE = (200, 200, 200)
        screen.fill(BLACK)
        self.board[2][5] = 1
        # TODO: Iterate through the 2D list and draw rectangles for each cell
        # Empty cells can be one color, trails another
        cell_size = screen.get_width() // self.width
        for x in range(0, self.width):
            for y in range(0, self.height):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, WHITE, rect, 1)
                
                if self.board[x][y] == 1:
                    pygame.Surface.fill(screen, WHITE, rect)
                    

    def is_collision(self, x, y):
        """
        Check if the given coordinates collide with the board boundaries or a trail.
        :param x: X-coordinate to check
        :param y: Y-coordinate to check
        :return: True if collision, False otherwise
        """
        cell_x = x // self.width
        cell_y = y // self.height
        # TODO: Check if x and y are within board boundaries
        # Also check if the cell at (x, y) is not empty (i.e., has a trail)
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if self.board[cell_x][cell_y]
            return True
        return False
        
        