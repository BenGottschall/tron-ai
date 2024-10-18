# cart_pole_env.py
import gym
from gym.wrappers import FlattenObservation

def create_env(render_mode=None):
    env = gym.make('CartPole-v1', render_mode=render_mode)
    return FlattenObservation(env)