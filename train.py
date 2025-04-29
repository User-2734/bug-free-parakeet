import environment
import pygame
import numpy as np
import dqn
from typing import Callable

clock = pygame.time.Clock()

env = environment.RacingEnv()
state_dim = 34
action_dim = 3
agent = dqn.DQNAgent(state_dim, action_dim)
target_update_interval = 10
checkpoint_freq = 25

def reverse_leaky_relu(x: float) -> float:
    if x <= 0: return x
    return x * 0.9

def reward_alignment(env: environment.RacingEnv):
    return reverse_leaky_relu(np.cos(env.relative_angle))

def normal_reward(env: environment.RacingEnv):
    # calculate wehere we are
    grid_x = int(env.car.x // env.grid.section_width)
    grid_y = int(env.car.y // env.grid.section_height)

    # reward hitting the target
    if (grid_x, grid_y) == env.grid.target: return 1
    
    # penalize crashing
    if env.car.collision(env.grid): return -1
    return 0

def action_to_movement(action):
    steer, throttle = 0, 1
    if action == 1:
        steer, throttle = -1, 1
    elif action == 2:
        steer, throttle = 1, 1
    return steer, throttle

def movement_to_action(steer, throttle):
    actions = [(0, 1), (-1, 1), (1, 1)]
    return actions.index((steer, throttle))

def degrees(rad: float) -> int:
    return int((rad / np.pi) * 180)

def run_episode(seed: int, agent: dqn.DQNAgent, clutter: float, q_func: Callable, manual=False):
    total_reward = 0
    state = env.reset(seed, clutter)
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        
        throttle, steer = 0, 0
        if manual:
            throttle = 1
            if pygame.key.get_pressed()[pygame.K_a]: steer = -1
            if pygame.key.get_pressed()[pygame.K_d]: steer = 1
            action = movement_to_action(steer, throttle)
        else:
            action = agent.get_action(state)
            steer, throttle = action_to_movement(action)

        # calculate the state
        next_state, done = env.step(steer, throttle)

        reward = q_func(env)
        total_reward += reward
        env.render(
            info={
                "seed": seed,
                "epsilon": round(agent.epsilon, 3),
                "angle": degrees(env.relative_angle),
                "distange": round(env.distance, 2),
                "reward": round(reward, 3),
                "reward method": q_func.__name__
            }
            )

        agent.replay.add((state, action, reward, next_state, float(done)))
        agent.train_step()
        state = next_state

        if manual: clock.tick(30)
    
    print(f"Episode {episode} — Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    return reward



def save_checkpoint(episode):
    agent.target_net.save_weights(f"checkpoints/check{episode}.weights.h5")

episodes = 5000

for episode in range(episodes):
    run_episode(
        episode, 
        agent, 
        0 if episode < 200 else 0.1, # first 200 iterations have no obsticles
        reward_alignment if episode < 100 else normal_reward, # award alignment at first
        manual=False)

    if episode % target_update_interval == 0:
        agent.update_target()

    if episode % checkpoint_freq == 0 and episode != 0:
        ...#save_checkpoint(episode)

agent.target_net.save_weights('model.weights.h5')