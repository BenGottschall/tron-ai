import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_shape, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size = 3, stride = 1)
        self.bn1 = nn.LayerNorm([32, 19, 19])
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride=1)
        self.bn2 = nn.LayerNorm([64, 17, 17])
        
        self.fc1 = nn.Linear(64 * 17 * 17, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the convolutional output
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class RLAgent:
    def __init__(self, action_size, player_id, grid_size = 21, model_file=None):
        self.grid_size = grid_size
        self.state_size = (3, grid_size, grid_size)
        self.action_size = action_size
        self.player_id = player_id
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9  # discount rate
        self.epsilon = 0.01 if model_file else 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.state_size, action_size).to(self.device)
        self.target_model = DQN(self.state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # Up, Down, Left, Right
        self.episode_rewards = []
        self.training_step = 0

        if model_file:
            self.load_model(model_file)
            self.model.eval()
            for name, param in self.model.named_parameters():
                assert not torch.isnan(param).any(), f"NaN in parameter {name} after loading model!"
                
        self.target_model.load_state_dict(self.model.state_dict())
            
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, game_board, player, opponent):
        grid_size = self.grid_size
        state = np.zeros((3, grid_size, grid_size))  # 3 channels: empty, player, opponent
        half_size = grid_size // 2
        for i in range(-half_size, half_size + 1):
            for j in range(-half_size, half_size + 1):
                x, y = player.x + i, player.y + j
                if 0 <= x < game_board.width and 0 <= y < game_board.height:
                    if game_board.grid[y][x] == 0:
                        state[0][i+3][j+3] = 1  # Empty
                    elif game_board.grid[y][x] == player.player_id:
                        state[1][i+3][j+3] = 1  # Player
                    else:
                        state[2][i+3][j+3] = 1  # Opponent
                else:
                    state[2][j+3][j+3] = 1  # Treat walls as opponent
        return state

    def get_valid_directions(self, current_direction):
        invalid_direction = [-current_direction[0], -current_direction[1]]
        return [d for d in self.directions if d != invalid_direction]

    def get_direction(self, game_board, player, opponent):
        state = self.get_state(game_board, player, opponent)
        valid_directions = self.get_valid_directions(player.direction)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            
        # print(f"State tensor: {state_tensor}")
        # assert not torch.isnan(state_tensor).any(), "State tensor contains NaN values."
        # assert torch.isfinite(state_tensor).all(), "State tensor contains infinite values."


            
        # Convert Q-values to a numpy array for manipulation
        q_values_np = q_values.cpu().numpy().squeeze()

        # Create a mask for valid actions
        action_mask = np.full(len(self.directions), -float('inf'))  # Initialize all actions as invalid
        num_directions = 0
        for d in valid_directions:
            new_x, new_y = player.x + d[0], player.y + d[1]
            
            if 0 <= new_x < game_board.width and 0 <= new_y < game_board.height:
                if game_board.grid[new_y][new_x] == 0:  # Empty space
                    num_directions += 1
                    action_index = self.directions.index(d)
                    action_mask[action_index] = q_values_np[action_index]  # Retain Q-value for valid action
            
        valid_indices = [i for i, q in enumerate(action_mask) if q > -float('inf')]
        
        # Fallback: If no valid moves exist, return a random valid direction
        if not valid_indices:
            return random.choice(valid_directions)

        # Exploration vs. Exploitation
        if np.random.rand() <= self.epsilon:
            # Choose a random action from valid directions that are not masked (-inf)
            action = self.directions[random.choice(valid_indices)]
            # print(f"chosen action: {action}")
            return action
        
        # Pick the best valid action
        best_action_index = np.argmax(action_mask)
        return self.directions[best_action_index]
        
        # valid_q_values = [q_values[0][self.directions.index(d)].item() for d in valid_directions]
        # best_valid_action = valid_directions[np.argmax(valid_q_values)]
        
        # if np.random.rand() <= self.epsilon:
        #     return random.choice(valid_directions)
        
        # return best_valid_action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards_tensor = torch.tensor(np.array(rewards)).to(self.device)
        dones_tensor = torch.tensor(np.array(dones)).to(self.device)
        
        action_map = {
        (1, 0): 0,
        (-1, 0): 1,
        (0, 1): 2,
        (0, -1): 3
        }
        
        actions_indices = [action_map[tuple(action)] for action in actions]
        actions_tensor = torch.tensor(actions_indices).to(self.device)
        
        current_q_values = self.model(states_tensor)
    
        # Get next Q-values (max Q-value for each next state)
        next_q_values = self.target_model(next_states_tensor)
        next_q_values_max = torch.max(next_q_values, dim=1)[0].unsqueeze(1)  # Get max Q-value for each next state
        
        # Calculate the target Q-values
        target_q_values = current_q_values.clone()
        
        dones_tensor = dones_tensor.float()
        
        
        # print(f"Batch size: {batch_size}")
        # print(f"Actions tensor shape: {actions_tensor.shape}")
        # print(f"Target Q-values shape: {target_q_values.shape}")
        # print(f"Rewards tensor shape: {rewards_tensor.shape}")
        # print(f"Dones tensor shape: {dones_tensor.shape}")
        # print(f"Next Q-values max shape: {next_q_values_max.squeeze(1).shape}")
        
        target_q_values[torch.arange(batch_size), actions_tensor] = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values_max.squeeze(1)
        
        # Compute loss (MSE between current and target Q-values)
        loss = nn.MSELoss()(current_q_values.gather(1, actions_tensor.unsqueeze(1)), target_q_values.gather(1, actions_tensor.unsqueeze(1)))
        

        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self, game_board, player, opponent):
        state = self.get_state(game_board, player, opponent)
        done = False
        total_reward = 0

        while not done:
            action = self.get_direction(game_board, player, opponent)
            old_direction = player.direction[:]
            player.change_direction(action)
            
            if player.direction == [-old_direction[0], -old_direction[1]]:
                player.direction = old_direction  # Revert to the old direction
            
            collision = player.move(game_board)

            next_state = self.get_state(game_board, player, opponent)
            reward = 1  # Reward for surviving one more step

            if collision or game_board.is_collision(player.x, player.y):
                reward = -10  # Penalty for collision
                done = True
            elif game_board.is_collision(opponent.x, opponent.y):
                reward = 10  # Reward for opponent's collision
                done = True

            total_reward += reward
            self.remember(state, action, reward, next_state, done)
            state = next_state

            if not done:
                game_board.grid[player.y][player.x] = player.player_id

            self.training_step += 1
            if self.training_step % 4 == 0:
                self.replay(32)

        self.episode_rewards.append(total_reward)
        return total_reward
    
    def update_epsilon(self, total_episodes, episode):
        decay_rate = 0.005  # Decay rate, you can experiment with different values

        # Exponential decay formula
        self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * math.exp(-decay_rate * episode)

    def save_model(self, filename):
        for name, param in self.model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN detected in {name} before saving!"
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()