import numpy as np
from .maps import RandomGridMap


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

    def collision(self, grid: RandomGridMap):
        # Convert pixel to grid index
        grid_x, grid_y = grid.to_grid(self.x, self.y)

        # If the car hits a wall
        if grid.get_square(grid_x, grid_y):
            return True
        
        return False

    def view(self, grid: RandomGridMap):
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
                ray_x = int(self.x + depth * cos_a)
                ray_y = int(self.y + depth * sin_a)

                # Convert pixel to grid index
                grid_x, grid_y = grid.to_grid(ray_x, ray_y)

                # If the ray hits a wall
                if grid.get_square(grid_x, grid_y):
                    ray_distances.append(depth)
                    break
            else:
                # If the ray didn't hit anything
                ray_distances.append(max_depth)

        return ray_distances
  
