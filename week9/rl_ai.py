import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque

class DQN(nn.Module):
    def __init__(self, grid_input_shape, compact_input_size, output_size):
        super(DQN, self).__init__()
        
        # Convolutional layers for grid data
        self.conv1 = nn.Conv2d(grid_input_shape[0], 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        
        # Flatten the convolutional output
        conv_output_size = self._get_conv_output_size(grid_input_shape)
        self.fc_grid = nn.Linear(conv_output_size, 128)

        # Fully connected layers for compact data
        self.fc_compact = nn.Linear(compact_input_size, 64)

        # Combine both outputs
        self.fc_combined = nn.Linear(128 + 64, 128)
        self.fc_output = nn.Linear(128, output_size)

    def _get_conv_output_size(self, input_shape):
        # Dummy forward pass to calculate the flattened size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = torch.relu(self.conv1(dummy_input))
            x = torch.relu(self.conv2(x))
            return x.numel()  # Total number of elements in the output

    def forward(self, grid, compact):
        # Process the grid data
        x_grid = torch.relu(self.conv1(grid))
        x_grid = torch.relu(self.conv2(x_grid))
        x_grid = x_grid.view(x_grid.size(0), -1)  # Flatten
        x_grid = torch.relu(self.fc_grid(x_grid))

        # Process the compact data
        x_compact = torch.relu(self.fc_compact(compact))

        # Combine both processed inputs
        x = torch.cat([x_grid, x_compact], dim=1)
        x = torch.relu(self.fc_combined(x))
        return self.fc_output(x)

class RLAgent:
    def __init__(self, action_size, player_id, grid_size = 15, model_file=None):
        self.grid_size = grid_size
        
        self.action_size = action_size
        self.player_id = player_id
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 0.01 if model_file else 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.grid_input_shape = (3, grid_size, grid_size)
        self.compact_input_size = 12 # 8 direction + 4 pos
        self.model = DQN(self.grid_input_shape, self.compact_input_size, action_size).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # Up, Down, Left, Right
        self.episode_rewards = []
        self.training_step = 0
        
        self.target_model = DQN(self.grid_input_shape, self.compact_input_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        if model_file:
            self.load_model(model_file)
            self.model.eval()
            for name, param in self.model.named_parameters():
                assert not torch.isnan(param).any(), f"NaN in parameter {name} after loading model!"
                
        
            
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, game_board, player, opponent):
        grid_size = self.grid_size
        half_size = grid_size // 2
        max_distance = max(game_board.width, game_board.height)
        
        # Grid Representation
        state = np.zeros((3, grid_size, grid_size))  # 3 channels: empty, player, opponent
        for i in range(-half_size, half_size + 1):
            for j in range(-half_size, half_size + 1):
                x, y = player.x + i, player.y + j
                if 0 <= x < game_board.width and 0 <= y < game_board.height:
                    if game_board.grid[y][x] == 0:
                        state[0][i+half_size][j+half_size] = 1  # Empty
                    elif game_board.grid[y][x] == player.player_id:
                        state[1][i+half_size][j+half_size] = 1  # Player
                    else:
                        state[2][i+half_size][j+half_size] = 1  # Opponent
                else:
                    state[2][j+half_size][j+half_size] = 1  # Treat walls as opponent
        
        # Directional Data
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        directional_data = []
        for dx, dy in directions:
            dist = 0
            x, y = player.x, player.y
            while 0 <= x + dx < game_board.width and 0 <= y + dy < game_board.height:
                x += dx
                y += dy
                dist += 1
                if game_board.grid[y][x] != 0:
                    break
            directional_data.append(min(dist, max_distance))
        
        directional_data = [d / max_distance for d in directional_data]
        
        # Positional Data
        positional_data = [
            player.x / game_board.width,
            player.y / game_board.height,
            opponent.x / game_board.width,
            opponent.y / game_board.height,
        ]
        
        return state, np.array(directional_data + positional_data)

    def get_valid_directions(self, current_direction):
        invalid_direction = [-current_direction[0], -current_direction[1]]
        return [d for d in self.directions if d != invalid_direction]

    def get_direction(self, game_board, player, opponent):
        grid, compact = self.get_state(game_board, player, opponent)
        valid_directions = self.get_valid_directions(player.direction)
        
        grid_tensor = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
        compact_tensor = torch.FloatTensor(compact).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(grid_tensor, compact_tensor)
            
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
        
        grids, compacts = [], []
        next_grids, next_compacts = [], []
        actions, rewards, dones = [], [], []

        # Extract data from the minibatch
        for (grid, compact), action, reward, (next_grid, next_compact), done in minibatch:
            grids.append(grid)
            compacts.append(compact)
            next_grids.append(next_grid)
            next_compacts.append(next_compact)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        # Convert data to tensors
        grid_tensors = torch.FloatTensor(np.array(grids)).to(self.device)
        compact_tensors = torch.FloatTensor(np.array(compacts)).to(self.device)
        next_grid_tensors = torch.FloatTensor(np.array(next_grids)).to(self.device)
        next_compact_tensors = torch.FloatTensor(np.array(next_compacts)).to(self.device)
        rewards_tensor = torch.tensor(np.array(rewards)).to(self.device)
        dones_tensor = torch.tensor(np.array(dones)).float().to(self.device)
        
        # Map actions to indices
        action_map = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
        
        actions_indices = [action_map[tuple(action)] for action in actions]
        actions_tensor = torch.tensor(actions_indices).to(self.device).unsqueeze(1)  # Shape: (batch_size, 1)

        # Forward pass for current and next Q-values
        current_q_values = self.model(grid_tensors, compact_tensors)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_grid_tensors, next_compact_tensors)
            next_q_values_max = torch.max(next_q_values, dim=1)[0]  # Max Q-value for each next state

        # Compute target Q-values
        targets = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values_max
        targets = targets.unsqueeze(1)
        
        #Compute loss
        predicted_q_values = current_q_values.gather(1, actions_tensor)
        loss = nn.MSELoss()(predicted_q_values, targets)

        #Backpropagation
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
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * math.exp(-decay_rate * episode)

    def save_model(self, filename):
        for name, param in self.model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN detected in {name} before saving!"
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()