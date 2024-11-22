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

def train():
    game_board = GameBoard(40, 30)
    rl_agent1 = RLAgent(action_size=4, player_id=1, grid_size=15)
    rl_agent2 = RLAgent(action_size=4, player_id=2, grid_size=15)
    player1 = Player(5, 5, (255, 0, 0), 1, rl_agent1)
    player2 = Player(15, 5, (0, 0, 255), 2, rl_agent2)
    game_board.grid[player1.y][player1.x] = player1.player_id
    game_board.grid[player2.y][player2.x] = player2.player_id
    
    # Set opponents
    player1.set_opponent(player2)
    player2.set_opponent(player1)

    num_episodes = 10000
    rewards1 = []
    rewards2 = []
    epsilon_update_interval = 50
    target_update_interval = 100
    high_score1, high_score2 = 0, 0

    for episode in range(num_episodes):
        game_board = GameBoard(40, 30)
        assert game_board.grid[15][10] == 0, "Player 1 reset position conflicts with game board!"
        assert game_board.grid[15][30] == 0, "Player 2 reset position conflicts with game board!"

        player1.reset(10, 15)
        player2.reset(30, 15)
        
        done1, done2 = False, False
        max_steps_after_win = 500
        total_reward1 = 0
        total_reward2 = 0
        
        step_counter = 0

        while not (done1 and done2):
            reward1, reward2 = 0, 0
            # Player 1's turn
            if not done1:
                grid1, compact1 = rl_agent1.get_state(game_board, player1, player2)
                collision1 = player1.move(game_board)
                reward1 = 1 # default reward
                if collision1 or game_board.is_collision(player1.x, player1.y):
                    # print(f"Player 1 collided at ({player1.x}, {player1.y})")
                    reward1 += -10
                    reward2 += 10
                    done1 = True
            
            # Player 2's turn
            if not done2:
                grid2, compact2 = rl_agent2.get_state(game_board, player2, player1)
                collision2 = player2.move(game_board)
                reward2 = 1 # default reward
                if collision2 or game_board.is_collision(player2.x, player2.y):
                    # print(f"Player 1 collided at ({player1.x}, {player1.y})")
                    reward1 += 10
                    reward2 += -10
                    done2 = True
            
            # update gameboard, get next states, modfy total reward, add experience to memory
            if not done1:
                game_board.grid[player1.y][player1.x] = player1.player_id
                next_grid1, next_compact1 = rl_agent1.get_state(game_board, player1, player2)
                total_reward1 += reward1
                rl_agent1.remember((grid1, compact1), player1.direction, reward1, (next_grid1, next_compact1), done1)
            
            if not done2:
                game_board.grid[player2.y][player2.x] = player2.player_id
                next_grid2, next_compact2 = rl_agent2.get_state(game_board, player2, player1)
                total_reward2 += reward2
                rl_agent2.remember((grid2, compact2), player2.direction, reward2, (next_grid2, next_compact2), done2)
            
            if done1 and not done2:
                max_steps_after_win -= 1
                if max_steps_after_win <= 0:
                    done2 = True
            elif done2 and not done1:
                max_steps_after_win -= 1
                if max_steps_after_win <= 0:
                    done1 = True
            
            if step_counter % 10 == 0:
                rl_agent1.replay(64)
                rl_agent2.replay(64)
            
            step_counter += 1
        
        rewards1.append(total_reward1)
        rewards2.append(total_reward2)
        
        if(total_reward1 > high_score1):
            high_score1 = total_reward1
            rl_agent1.save_model(f"drew.pth")
        if(total_reward2 > high_score2):
            high_score2 = total_reward2
            rl_agent2.save_model(f"carl.pth")

        if episode % epsilon_update_interval == 0:
            player1.controller.update_epsilon(num_episodes, episode)
            player2.controller.update_epsilon(num_episodes, episode)
            
        if episode % target_update_interval == 0:
            rl_agent1.update_target_model()
            rl_agent2.update_target_model()

        if episode % 20 == 0:
            print(f"Episode: {episode}, HighScore1: {high_score1}, HighSchore2: {high_score2}\n Player 1 Reward: {total_reward1}, Epsilon: {player1.controller.epsilon}\n Player 2 Reward: {total_reward2}, Epsilon: {player2.controller.epsilon}")
            
        if episode % 100 == 0:
            rl_agent1.save_model(f"tron_model_player1_checkpoint_{episode}.pth")
            rl_agent2.save_model(f"tron_model_player2_checkpoint_{episode}.pth")
            plot_rewards(rewards1, rewards2)
            

    rl_agent1.save_model("tron_model_player1.pth")
    rl_agent2.save_model("tron_model_player2.pth")
    plot_rewards(rewards1, rewards2)

if __name__ == "__main__":
    train()