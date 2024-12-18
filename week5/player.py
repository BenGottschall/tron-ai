import pygame
from config import *
from collections import deque

class Player:
    def __init__(self, x, y, color, player_id, ai):
        """
        Initialize the player.
        :param x: Initial x-coordinate
        :param y: Initial y-coordinate
        :param color: Color of the player's trail
        :param player_id: ID of the player (1 or 2)
        :param ai: AI object that provides directions
        """
        # Set initial position, color, direction (e.g., [1, 0] for right)
        # Initialize an empty list for the player's trail
        self.x = x
        self.y = y
        self.color = color
        self.player_id = player_id
        self.trail = [[self.x, self.y]]
        self.direction_queue = deque()
        if player_id == 1:
            self.direction = [-1, 0]
        else:
            self.direction = [1, 0]
        self.ai = ai
        
    def move(self):
        """
        Move the player based on their current direction.
        """
        # Update the player's position based on their direction
        # Add the new position to the trail
        # if self.player_id == 2: 
        #     new_direction = self.ai.get_direction()
        #     self.change_direction(new_direction)
        new_direction = self.ai.get_direction()
        self.change_direction(new_direction)
        self.x += self.direction[0]
        self.y += self.direction[1]
        self.trail.append([self.x, self.y])
        
        
    def change_direction(self, direction):
        """
        Change the player's direction.
        :param direction: New direction as a list [dx, dy]
        """
        # TODO: Update the player's direction
        # Ensure the new direction is not opposite to the current direction
        if self.direction[0] * direction[0] + self.direction[1] * direction[1] == 0:
            # self.direction = direction
            self.direction_queue.append(direction)
            
        # old code below:
        # if (direction[0] != -self.direction[0] or direction[1] != -self.direction[1]):
        #     self.direction_queue.append(direction)
    
    def update_direction(self):
        if self.direction_queue:
            self.direction = self.direction_queue.popleft()
        self.x_next = self.x + self.direction[0]
        self.y_next = self.y + self.direction[1]
            
        
    def draw(self, screen):
        """
        Draw the player and their trail on the screen.
        :param screen: Pygame screen object to draw on
        """
        # TODO: Draw the player's current position and their entire trail
        rect = pygame.Rect(self.trail[-1][0] * CELL_SIZE, self.trail[-1][1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.Surface.fill(screen, self.color, rect)
    
    def reset(self, x, y):
        """
        Reset the player's position and trail.
        :param x: New x-coordinate
        :param y: New y-coordinate
        """
        self.x = x
        self.y = y
        self.direction = [1, 0] if self.player_id == 1 else [-1, 0]
        self.trail = [[x, y]]
        # check if these should be tuples or lists (different from given code)
        