import random
import numpy as np

from .utils import wrap_angle
from .maps import RandomGridMap
from .car import Car
from .graphics import SimWindow

class RacingEnv:
    def __init__(self, max_steps=500):
        self.max_steps = max_steps
        self.last_view = None

        self.window = SimWindow(512, 512, 'racing')

    def reset(self, seed, clutter):
        self.grid = RandomGridMap(seed, clutter, None)
        self.car = Car(self.grid.grid_size / 2, self.grid.grid_size / 2, random.uniform(0, np.pi * 2))
        self.steps = 0
        return self._get_state()
    
    @property
    def angle(self):
        # the relative angle from the car to the target
        dx = (self.grid.target[0] + 0.5) * self.grid.grid_size - self.car.x
        dy = (self.grid.target[1] + 0.5) * self.grid.grid_size - self.car.y
        return np.arctan2(dy, dx)
    
    @property
    def relative_angle(self):
        relative_angle = self.angle - self.car.direction
        return wrap_angle(relative_angle)
    
    @property
    def distance(self):
        dx = (self.grid.target[0] + 0.5) * self.grid.grid_size - self.car.x
        dy = (self.grid.target[1] + 0.5) * self.grid.grid_size - self.car.y
        distance = np.sqrt((dx ** 2) + (dy ** 2))
        return distance

    def _get_state(self):
        # view + velocity + goal vector
        self.last_view = self.car.view(self.grid)
        view = np.array(self.last_view) # 32
        goal_vector = np.array([self.relative_angle, self.distance]) # 2
        return np.concatenate([view, goal_vector]) # 34

    def step(self, steer, throttle):
        self.car.step(steer, throttle)

        self.steps += 1

        grid_x, grid_y = self.grid.to_grid(self.car.x, self.car.y)

        if self.steps >= self.max_steps:
            done = True
            print('Max steps exceeded')
        elif self.car.collision(self.grid):
            done = True
            print('Crashed')
        elif (grid_x, grid_y) == self.grid.target:
            done = True
            print('Hit target')
        else:
            done = False

        return self._get_state(), done

    def render(self, info={}):
        # draw the base map
        self.window.clear()
        self.window.camera.x = self.car.x
        self.window.camera.y = self.car.y
        self.grid.render(self.window, self.car.x, self.car.y)

        # draw the car's LIDAR
        for i, dist in enumerate(self.last_view):
            angle = self.car.direction - (70 * np.pi / 180) / 2 + (i / 31) * (70 * np.pi / 180)
            end_x = self.car.x + dist * np.cos(angle)
            end_y = self.car.y + dist * np.sin(angle)
            self.window.line(
                self.car.x, self.car.y, 
                end_x, end_y, 
                color=(255, 0, 0, 255), 
                thickness=1
                )

        # draw the angle to the target
        end_x = self.car.x + 30 * np.cos(self.angle)
        end_y = self.car.y + 30 * np.sin(self.angle)
        self.window.line(self.car.x, self.car.y, end_x, end_y, color=(0, 255, 0, 255), thickness=3)

        """
        # render info
        y = spacing
        for stat in info:
            text = f"{stat}: {info[stat]}"
            text_surface = font.render(text, True, (255, 255, 255))  # white text
            window_surface.blit(text_surface, (spacing, y))  # top-left corner
            y += font_size + spacing
        """
        self.window.update()