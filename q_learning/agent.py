import random
import numpy as np
import tensorflow as tf

from .network import QNetwork
from .buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.replay = ReplayBuffer()
        self.update_target()

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Assuming 5 discrete actions
        q_vals = self.q_net(np.expand_dims(state, axis=0))
        return int(tf.argmax(q_vals[0]))

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        next_qs = self.target_net(next_states)
        max_next_q = np.max(next_qs, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        with tf.GradientTape() as tape:
            q_vals = self.q_net(states)
            one_hot_actions = tf.one_hot(actions, q_vals.shape[1])
            q_vals_selected = tf.reduce_sum(q_vals * one_hot_actions, axis=1)
            loss = self.loss_fn(targets, q_vals_selected)

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
