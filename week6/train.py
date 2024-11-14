import pygame
from game_board import GameBoard
from player import Player
from rl_ai import RLAgent
import matplotlib.pyplot as plt

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Episode Rewards over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('rewards_plot.png')
    plt.close()

def plot_steps(steps):
    plt.figure(figsize=(10, 5))
    plt.plot(steps)
    plt.title('Steps Survived over Time')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.savefig('steps_plot.png')
    plt.close()

def train():
    game_board = GameBoard(40, 30)
    rl_agent = RLAgent(state_size=7*7, action_size=4)
    player = Player(20, 15, (255, 0, 0), 1, rl_agent)

    num_episodes = 1000
    all_steps = []

    for episode in range(num_episodes):
        game_board = GameBoard(40, 30)
        player.reset(20, 15)
        
        total_reward, steps = rl_agent.train(game_board, player)
        all_steps.append(steps)
        
        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {rl_agent.epsilon:.2f}")
        
        if episode % 100 == 0:
            plot_rewards(rl_agent.episode_rewards)
            plot_steps(all_steps)

    rl_agent.save_model("tron_survival_model.pth")
    plot_rewards(rl_agent.episode_rewards)
    plot_steps(all_steps)

if __name__ == "__main__":
    train()