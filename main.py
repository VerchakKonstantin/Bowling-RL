import tensorflow as tf
import numpy as np
import cv2
import time
import gym
import pygame
from gym.utils.play import play
import playplot
import matplotlib.pyplot as plt
import config



from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from scipy.stats import entropy

env = gym.make('ALE/Bowling-v5', render_mode="rgb_array")
print(env.observation_space)
print(env.action_space)
print(env.unwrapped.get_action_meanings())

env.reset()
obs = env.render()
print(obs.shape)


def preprocess_obs(obs: np.ndarray, vertical_crop_start: int, vertical_crop_end: int,
                   horizontal_crop_start: int, horizontal_crop_end: int):
    cropped_obs = obs[vertical_crop_start:vertical_crop_end, horizontal_crop_start:horizontal_crop_end]
    cropped_obs = cv2.resize(cropped_obs, (80, 80), interpolation=cv2.INTER_AREA)
    cropped_obs = cv2.cvtColor(cropped_obs, cv2.COLOR_RGB2GRAY)
    cropped_obs = cv2.adaptiveThreshold(cropped_obs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    cropped_obs = cropped_obs / 255.
    return cropped_obs


cropped_obs = preprocess_obs(obs, config.vertical_crop_start, config.vertical_crop_end,
                             config.horizontal_crop_start, config.horizontal_crop_end)
plt.imshow(cropped_obs)
plt.show()

epsilon = config.clipping_parameter


def clipped_surrogate_objective(probability_ratio, advantages):
    unclipped_objectives = probability_ratio * advantages
    clipped_objectives = K.clip(probability_ratio, 1 - epsilon, 1 + epsilon) * advantages
    lower_bounds = K.min([unclipped_objectives, clipped_objectives], axis=0)
    return K.mean(lower_bounds)


def compute_A_t(rewards, values, dones, start, T, gamma, gae_lambda):
    A_t = rewards[start] - values[start]
    coefficient = 1
    for t in range(start + 1, T - 1):
        delta = rewards[t] + (gamma * (1 - dones[t + 1]) * values[t + 1]) - values[t]
        A_t += coefficient * delta
        coefficient = (gamma * gae_lambda) ** (T - t + 1)
    return A_t


def compute_advantage_estimates(rewards, values, dones, T):
    advantage_estimates = []
    for t in range(T):
        A_t = compute_A_t(rewards, values, dones, t, T)
        advantage_estimates.append(A_t)
    return np.array(advantage_estimates)


def compute_discounted_returns(rewards, dones, T, gamma):
    returns = []
    discounted_sum = 0
    for t in reversed(range(T)):
        discounted_sum = (1 - dones[t]) * rewards[t] + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    return returns





def play_for_timestamps(env, state, model, horizon, actions_available):
    states = []
    actions = []
    action_probabilities = []
    critic_values = []
    rewards = []
    dones = []

    # The timesteps for which we play is part of our hyperparameters
    # and is represented by the horizon
    for step in range(horizon):

        # We must call expand_dims to have the correct input size for the network
        t_state = tf.expand_dims(state, axis=0)
        actor_value, critic_value = model(t_state)

        # Normally the action distribution would be (1,6)
        # By squeezing we remove "one-dimensional dimensions"
        # so we get (6,)
        # https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
        action_probability = np.squeeze(actor_value)
        critic_value = np.squeeze(critic_value)

        action = np.random.choice(actions_available, p=action_probability)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        action_probabilities.append(action_probability)
        critic_values.append(critic_value)
        rewards.append(reward)
        dones.append(done)

        if done:
            state = env.reset()
        else:
            state = next_state

    return states, actions, action_probabilities, critic_values, rewards, dones
