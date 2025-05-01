from .utils import pseudo
from .graphics import SimWindow

class RandomGridMap:
    def __init__(self, seed, fill_percent, target_radius):
        self.seed          = seed
        self.start         = (0, 0)
        self.fill_percent  = fill_percent
        self.target_radius = target_radius
        self.grid_size     = 64
        self.generate_target()

    def generate_target(self) -> None:
        self.target = (100, 100)

    def get_square(self, x: int, y: int) -> bool:
        if (x, y) == self.target: return False
        if (x, y) == (0, 0): return False
        return pseudo(x, y, self.seed) < self.fill_percent
    
    def to_grid(self, x: int, y: int) -> tuple[int, int]:
        return x // self.grid_size, y // self.grid_size

    def from_grid(self, x: int, y: int) -> tuple[float, float]:
        return x * self.grid_size, y * self.grid_size

    def render(self, window: SimWindow, x, y, r=3):
        # render all the squares in a set radius
        car_grid_x, car_grid_y = self.to_grid(x, y)
        for y_offset in range(-r, r + 1):
            for x_offset in range(-r, r + 1):
                # calculate the acual grid coordinates of the square
                grid_x = int(car_grid_x - x_offset)
                grid_y = int(car_grid_y - y_offset)

                # render the square if needed
                if self.get_square(grid_x, grid_y):
                    window.rect(*self.from_grid(grid_x, grid_y), self.grid_size, self.grid_size)
        
        window.rect(
            *self.from_grid(*self.target),
            self.grid_size, 
            self.grid_size,
            color=(0, 255, 0, 255)
            )
