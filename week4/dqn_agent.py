# dqn_agent.py
import numpy as np
import random
from collections import deque
import tensorflow as tf
from keras import models, layers, optimizers

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
<<<<<<< HEAD
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
=======
>>>>>>> 0ae100a (added these stupid files)

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, activation='relu', input_dim=self.state_size),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch]).reshape(-1, self.state_size)
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch]).reshape(-1, self.state_size)
        dones = np.array([i[4] for i in minibatch])

        targets = rewards + self.gamma * np.amax(self.target_model.predict(next_states, verbose=0), axis=1) * (1 - dones)
        targets_full = self.model.predict(states, verbose=0)
        targets_full[np.arange(batch_size), actions] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0, batch_size=batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)