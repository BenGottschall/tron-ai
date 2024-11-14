# main.py
import numpy as np
<<<<<<< HEAD
from week4.cart_pole_env import create_env
from week4.dqn_agent import DQNAgent
=======
from cart_pole_env import create_env
from dqn_agent import DQNAgent
>>>>>>> 0ae100a (added these stupid files)
import time
import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def process_state(state):
    if isinstance(state, tuple):
        return np.concatenate(state).ravel()
    return state

def visualize_agent(env, agent, episode):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = process_state(state)
    state = np.reshape(state, [1, state_size])
    
    total_reward = 0
    t = 0
    done = False
    
    while not done:
        env.render()
        action = np.argmax(agent.model.predict(state, verbose=0)[0])
        next_state, reward, done, truncated, _ = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = process_state(next_state)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        total_reward += reward
        t += 1
        
        time.sleep(0.02)  # Add a small delay to make the visualization visible
        
        if done or truncated:
            time.sleep(0.5)  # Pause briefly to show the final state
            break

    print(f"Visualization Episode {episode} lasted for {t} steps with total reward {total_reward}")
    env.close()  # Close the environment after the episode is done

if __name__ == "__main__":
    env = create_env()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    EPISODES = 1000

    for e in range(EPISODES):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = process_state(state)
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for cur_time in range(500):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            reward = reward if not done else -10
            next_state = process_state(next_state)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        agent.update_target_model()
        print(f"Episode: {e}/{EPISODES}, Score: {cur_time}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # Visualize the agent's performance every 5 episodes
        if e % 5 == 0:
            vis_env = create_env(render_mode="human")  # Create a new environment for each visualization
            visualize_agent(vis_env, agent, e)
            vis_env.close()  # Ensure the environment is closed after visualization

        if e % 50 == 0:
            agent.save(f"cartpole-dqn-{e}.weights.h5")

    print("Training completed.")