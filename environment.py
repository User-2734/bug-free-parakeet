import pygame
import random
import numpy as np

import hashlib

WIDTH = 512
HEIGHT = 512
CENTER = (WIDTH // 2, HEIGHT // 2)

pygame.init()
pygame.display.set_caption('Racing')
window_surface = pygame.display.set_mode((WIDTH, HEIGHT))

background = pygame.Surface((WIDTH, HEIGHT))
background.fill(pygame.Color('#000000'))


font_size = 12
spacing = 2
pygame.font.init()
font = pygame.font.SysFont("Arial", font_size)

def sha256(n: int, size: int = 16, output_raw=False, input_raw=False) -> int:
    if not input_raw:
        n = n.to_bytes(size, 'big', signed=True)
    x = hashlib.sha256(n).digest()
    if output_raw: return x
    return int.from_bytes(x, 'big', signed=True)

def pseudo(x: int, y: int, seed: int = 0, size: int = 16):
    x = sha256(x, output_raw=True)
    y = sha256(y, output_raw=True)
    seed = sha256(seed, output_raw=True)
    total = sha256(x + y + seed, input_raw=True)
    mask = (2 ** (size + 1)) - 1
    return (total & mask) / mask

class Grid:
    def __init__(self, seed, clutter):
        self.seed   = seed
        target_direction = pseudo(seed, seed, seed) * np.pi * 2 # pick a random direction
        target_distance  = (pseudo(seed + 1, seed, seed) * 10) + 10 # pick a distance between 10 and 20
        self.target = (
            int(np.cos(target_direction) * target_distance),
            int(np.sin(target_direction) * target_distance)
        )

        self.clutter = clutter

    def get_square(self, x, y) -> bool:
        if (x, y) == self.target: return False
        if (x, y) == (0, 0): return False
        return pseudo(x, y, self.seed) < self.clutter
    
    @property
    def section_width(self): 
        return WIDTH / 8
    
    @property
    def section_height(self):
        return HEIGHT / 8
    
    @property
    def vertical_lines(self):
        return np.linspace(0, WIDTH, self.size + 1)
    
    @property
    def horizontal_lines(self):
        return np.linspace(0, HEIGHT, self.size + 1)

    def render(self, surface, car: 'Car', r=3):
        car_grid_x = car.x // self.section_width
        car_grid_y = car.y // self.section_height
        for y_offset in range(-r, r + 1):
            for x_offset in range(-r, r + 1):
                grid_x = int(car_grid_x - x_offset)
                grid_y = int(car_grid_y - y_offset)
                x = (grid_x * self.section_width) - (car.x - CENTER[0])
                y = (grid_y * self.section_height) - (car.y - CENTER[1])
                if self.get_square(grid_x, grid_y):
                    pygame.draw.rect(surface, '#FFFFFF', pygame.Rect(x, y, self.section_width, self.section_height))

        pygame.draw.rect(
            surface,
            '#00FF00', 
            pygame.Rect(
                (self.target[0] * self.section_width) - (car.x - CENTER[0]), 
                (self.target[1] * self.section_height) - (car.y - CENTER[1]), 
                self.section_width, 
                self.section_height
                ), 
            4
            )

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class Car:
    def __init__(self, x, y, direction) -> None:
        self.x = x
        self.y = y
        self.direction = direction

        self.max_speed = 30
        self.vx = 0
        self.vy = 0
        self.steer_rate = (np.pi / 6) # 180 degrees per second (at 30 fps)

    def step(self, steer, throttle):
        # clamp throttle and steer
        throttle = np.clip(throttle, -1, 1)
        steer = np.clip(steer, -1, 1)

        # compute direction delta BEFORE velocity update
        speed = np.sqrt(self.vx**2 + self.vy**2)
        self.direction += steer * self.steer_rate * (speed / self.max_speed)

        rate = 10

        # throttle influences acceleration along the current direction
        ax = throttle * np.cos(self.direction) * rate
        ay = throttle * np.sin(self.direction) * rate

        self.vx += ax
        self.vy += ay

        # Apply friction (simulate drag)
        self.vx *= 0.5
        self.vy *= 0.5

        # Limit speed
        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vx *= scale
            self.vy *= scale

        # Update position
        self.x += self.vx
        self.y += self.vy

    def collision(self, grid: Grid):
        # Convert pixel to grid index
        grid_x = int(self.x // grid.section_width)
        grid_y = int(self.y // grid.section_height)

        # If the car hits a wall
        if grid.get_square(grid_x, grid_y):
            return True
        
        return False

    def view(self, grid: Grid):
        fov = 70 * (np.pi / 180)  # field of view in radians
        num_rays = 32
        max_depth = 200  # how far a ray can go

        start_angle = self.direction - (fov / 2)
        stop_angle  = self.direction + (fov / 2)

        ray_angles = np.linspace(start_angle, stop_angle, num_rays)
        ray_distances = []

        # chat GPT code
        for angle in ray_angles:
            sin_a = np.sin(angle)
            cos_a = np.cos(angle)

            # Step in small increments to simulate ray movement
            for depth in range(1, int(max_depth), 2):  # step by 2 pixels
                ray_x = self.x + depth * cos_a
                ray_y = self.y + depth * sin_a

                # Convert pixel to grid index
                grid_x = int(ray_x // grid.section_width)
                grid_y = int(ray_y // grid.section_height)

                # If the ray hits a wall
                if grid.get_square(grid_x, grid_y):
                    ray_distances.append(depth)
                    break
            else:
                # If the ray didn't hit anything
                ray_distances.append(max_depth)

        return ray_distances
  

class RacingEnv:
    def __init__(self, max_steps=500):
        pygame.init()

        self.max_steps = max_steps
        self.last_view = None

    def reset(self, seed, clutter):
        self.grid = Grid(seed, clutter)
        self.car = Car(self.grid.section_width / 2, self.grid.section_height / 2, random.uniform(0, np.pi * 2))
        self.steps = 0
        return self._get_state()
    
    @property
    def angle(self):
        # the relative angle from the car to the target
        dx = (self.grid.target[0] + 0.5) * self.grid.section_width - self.car.x
        dy = (self.grid.target[1] + 0.5) * self.grid.section_height - self.car.y
        return np.arctan2(dy, dx)
    
    @property
    def relative_angle(self):
        relative_angle = self.angle - self.car.direction
        return wrap_angle(relative_angle)
    
    @property
    def distance(self):
        dx = (self.grid.target[0] + 0.5) * self.grid.section_width - self.car.x
        dy = (self.grid.target[1] + 0.5) * self.grid.section_height - self.car.y
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

        grid_x = int(self.car.x // self.grid.section_width)
        grid_y = int(self.car.y // self.grid.section_height)

        if self.steps >= self.max_steps or self.car.collision(self.grid) or (grid_x, grid_y) == self.grid.target:
            done = True
        else:
            done = False

        return self._get_state(), done

    def render(self, info={}):
        # draw the base map
        window_surface.blit(background, (0, 0))
        self.grid.render(window_surface, self.car)

        # draw the car's LIDAR
        for i, dist in enumerate(self.last_view):
            angle = self.car.direction - (70 * np.pi / 180) / 2 + (i / 31) * (70 * np.pi / 180)
            end_x = CENTER[0] + dist * np.cos(angle)
            end_y = CENTER[1] + dist * np.sin(angle)
            pygame.draw.line(window_surface, '#FF0000', CENTER, (end_x, end_y), 1)
        
        # draw the angle to the target
        end_x = CENTER[0] + 30 * np.cos(self.angle)
        end_y = CENTER[1] + 30 * np.sin(self.angle)
        pygame.draw.line(window_surface, '#00FF00', CENTER, (end_x, end_y), 3)

        # render info
        y = spacing
        for stat in info:
            text = f"{stat}: {info[stat]}"
            text_surface = font.render(text, True, (255, 255, 255))  # white text
            window_surface.blit(text_surface, (spacing, y))  # top-left corner
            y += font_size + spacing
        pygame.display.update()