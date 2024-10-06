BOARD_WIDTH = 30 # number of cells horizontal
BOARD_HEIGHT = 15 # number of cells vertical
CELL_SIZE = 30 # pixel size of cells
SCREEN_WIDTH = BOARD_WIDTH * CELL_SIZE
SCREEN_HEIGHT = BOARD_HEIGHT * CELL_SIZE
GAME_SPEED = 100 # tick speed in miliseconds

PLAYER1_START = [BOARD_WIDTH - (BOARD_WIDTH // 4), BOARD_HEIGHT // 2]
PLAYER2_START = [(BOARD_WIDTH // 4), BOARD_HEIGHT // 2]

COLORS = {
    "background": (0, 0, 0),
    "player1": (28, 255, 51),
    "player2": (255, 135, 28)
}