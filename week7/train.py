import pygame
from game_board import GameBoard
from player import Player
from rl_ai import RLAgent
import matplotlib.pyplot as plt

def plot_rewards(rewards1, rewards2):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards1, label='Player 1')
    plt.plot(rewards2, label='Player 2')
    plt.title('Episode Rewards over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig('rewards_plot.png')
    plt.close()
    
def obstacle_in_front(player, game_board):
    front_x = player.x + player.direction[0]
    front_y = player.y + player.direction[1]
    obstacle = (
        front_x < 0 or front_x >= game_board.width or
        front_y < 0 or front_y >= game_board.height or
        game_board.grid[front_y][front_x] != 0
    )
    return obstacle

def train():
    game_board = GameBoard(40, 30)
    rl_agent1 = RLAgent(state_size=24, action_size=4, player_id=1)
    rl_agent2 = RLAgent(state_size=24, action_size=4, player_id=2)
    player1 = Player(10, 15, (255, 0, 0), 1, rl_agent1)
    player2 = Player(30, 15, (0, 0, 255), 2, rl_agent2)
    
    # Set opponents
    player1.set_opponent(player2)
    player2.set_opponent(player1)

    num_episodes = 1000
    rewards1 = []
    rewards2 = []

    for episode in range(num_episodes):
        game_board = GameBoard(40, 30)
        player1.reset(10, 15)
        player2.reset(30, 15)
        
        done = False
        total_reward1 = 0
        total_reward2 = 0
        
        obstacles_avoided1 = 0
        obstacles_avoided2 = 0

        while not done:
            
            obstacle_in_front1 = obstacle_in_front(player1, game_board)
            obstacle_in_front2 = obstacle_in_front(player2, game_board)
            
            # Player 1's turn
            collision1 = player1.move(game_board)

            # Player 2's turn
            collision2 = player2.move(game_board)
            
            reward1, reward2 = 1, 1 # default reward

            # Check for collisions
            if collision1 or game_board.is_collision(player1.x, player1.y):
                # print(f"Player 1 collided at ({player1.x}, {player1.y})")
                reward1, reward2 = -30, 10
                done = True
            elif collision2 or game_board.is_collision(player2.x, player2.y):
                # print(f"Player 2 collided at ({player2.x}, {player2.y})")
                reward1, reward2 = 10, -30
                done = True
            else:
                if obstacle_in_front1 and not collision1:
                    obstacles_avoided1 += 1
                    reward1 += 100
                    
                if obstacle_in_front2 and not collision2:
                    obstacles_avoided2 += 1
                    reward2 += 100
                    

            total_reward1 += reward1
            total_reward2 += reward2

            # Update game board
            if not done:
                game_board.grid[player1.y][player1.x] = player1.player_id
                game_board.grid[player2.y][player2.x] = player2.player_id

            # Get new states
            next_state1 = rl_agent1.get_state(game_board, player1, player2)
            next_state2 = rl_agent2.get_state(game_board, player2, player1)

            # Remember the transitions
            rl_agent1.remember(rl_agent1.get_state(game_board, player1, player2), player1.direction, reward1, next_state1, done)
            rl_agent2.remember(rl_agent2.get_state(game_board, player2, player1), player2.direction, reward2, next_state2, done)

            # Train the agents
            rl_agent1.replay(64)
            rl_agent2.replay(64)

        if player1.controller.epsilon > player1.controller.epsilon_min:
            player1.controller.epsilon *= player1.controller.epsilon_decay
        if player2.controller.epsilon > player2.controller.epsilon_min:
            player2.controller.epsilon *= player2.controller.epsilon_decay

        rewards1.append(total_reward1)
        rewards2.append(total_reward2)

        if episode % 10 == 0:
            print(f"Episode: {episode},\n Player 1 Reward: {total_reward1}, Obstacles avoided: {obstacles_avoided1}, epsilon: {round(player1.controller.epsilon, 3)}\n Player 2 Reward: {total_reward2}, Obstacles avoided: {obstacles_avoided2}, epsilon: {round(player2.controller.epsilon, 3)}")
            plot_rewards(rewards1, rewards2)
            
        if episode % 50 == 0:
            rl_agent1.save_model(f"tron_model_player1_checkpoint_{episode}.pth")
            rl_agent2.save_model(f"tron_model_player2_checkpoint_{episode}.pth")

    rl_agent1.save_model("tron_model_player1.pth")
    rl_agent2.save_model("tron_model_player2.pth")
    plot_rewards(rewards1, rewards2)

if __name__ == "__main__":
    train()