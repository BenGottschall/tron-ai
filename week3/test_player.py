import pygame
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from player import Player

def test_player_initialization():
    player = Player(10, 10, (255, 0, 0))
    assert player.x == 10 and player.y == 10, "Player position not set correctly"
    assert player.color == (255, 0, 0), "Player color not set correctly"
    assert hasattr(player, 'direction'), "Player should have a direction attribute"
    assert hasattr(player, 'trail'), "Player should have a trail attribute"
    print("Player initialization test passed")

def test_move():
    player = Player(10, 10, (255, 0, 0))
    initial_x, initial_y = player.x, player.y
    player.move()
    assert (player.x != initial_x or player.y != initial_y), "Player should move"
    print("Move method test passed")

def test_change_direction():
    player = Player(10, 10, (255, 0, 0))
    initial_direction = player.direction.copy()
    player.change_direction([0, 1])  # Change to moving down
    assert player.direction != initial_direction, "Player direction should change"
    assert player.direction == [0, 1], "Player direction not changed correctly"
    print("change_direction method test passed")

def test_draw():
    pygame.init()
    screen = pygame.Surface((400, 300))
    player = Player(10, 10, (255, 0, 0))
    try:
        player.draw(screen)
        print("Draw method executed without errors")
    except Exception as e:
        print(f"Error in draw method: {e}")

def run_all_tests():
    test_player_initialization()
    test_move()
    test_change_direction()
    test_draw()
    print("All Player tests passed!")

if __name__ == "__main__":
    run_all_tests()