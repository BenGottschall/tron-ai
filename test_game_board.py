import os
import sys
import pygame
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from game_board import GameBoard

def test_game_board_initialization():
    board = GameBoard(20, 15)
    assert hasattr(board, 'width') and board.width == 20, "Board width not set correctly"
    assert hasattr(board, 'height') and board.height == 15, "Board height not set correctly"
    print("GameBoard initialization test passed")

def test_draw():
    pygame.init()
    screen = pygame.Surface((400, 300))
    board = GameBoard(20, 15)
    try:
        board.draw(screen)
        print("Draw method executed without errors")
    except Exception as e:
        print(f"Error in draw method: {e}")

def test_is_collision():
    board = GameBoard(20, 15)
    assert board.is_collision(-1, 0) == True, "Should detect collision on left boundary"
    assert board.is_collision(20, 0) == True, "Should detect collision on right boundary"
    assert board.is_collision(0, -1) == True, "Should detect collision on top boundary"
    assert board.is_collision(0, 15) == True, "Should detect collision on bottom boundary"
    assert board.is_collision(10, 7) == False, "Should not detect collision inside the board"
    print("is_collision method tests passed")

def run_all_tests():
    test_game_board_initialization()
    test_draw()
    test_is_collision()
    print("All GameBoard tests passed!")

if __name__ == "__main__":
    run_all_tests()