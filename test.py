import environment
import pygame
from q_learning.agent import DQNAgent

env = environment.RacingEnv()

clock = pygame.time.Clock()

def action_to_movement(action):
    steer, throttle = 0, 1
    if action == 1:
        steer, throttle = -1, 1
    elif action == 2:
        steer, throttle = 1, 1
    return steer, throttle

def run_episode(num, agent=None):
    state = env.reset(num, (6, 6))
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        
        
        throttle, steer = 0, 0
        action = agent.get_action(state)
        steer, throttle = action_to_movement(action)

        # calculate the state
        state, done = env.step(steer, throttle)

        env.render()

        clock.tick(30)


env = environment.RacingEnv()
state_dim = 34
action_dim = 3
agent = DQNAgent(state_dim, action_dim)
agent.epsilon = 0
agent.epsilon_min = 0
agent.q_net.build(input_shape=(None, state_dim))
agent.q_net.load_weights("model.weights.h5")
episodes = 1000

for episode in range(episodes):
    run_episode(episode, agent)