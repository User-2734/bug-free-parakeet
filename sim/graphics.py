import pyglet
from pyglet import shapes
from typing import Callable

class Camera:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def translate_point(self, x, y):
        return (
            x - self.x,
            y - self.y
        )
    
    def scale_point(self, x, y):
        return (
            x * self.z,
            y * self.z
        )
    
    def transform_point(self, x, y) -> tuple[float, float]:
        return self.scale_point(*self.translate_point(x, y))

class SimWindow:
    def __init__(self, width, height, title):
        # initialize the window
        self.window = pyglet.window.Window(vsync=True, caption=title, width=width, height=height)
        self.window.set_visible(True)

        # handle the window being closed
        self.open = True
        @self.window.event
        def on_close():
            self.open = False
            self.window.close()

        # rendering stuff
        self.batches: list[pyglet.graphics.Batch] = []
        self.camera = Camera(0, 0, 1)

    @property
    def center_x(self):
        return self.window.width / 2
    
    @property
    def center_y(self):
        return self.window.height / 2
    
    def center(self, x, y):
        return x + self.center_x, y + self.center_y
    
    def rect(self, x, y, w, h, color=(255, 255, 255, 255)) -> None:
        """Draws a rectangle on the screen
        
        Args:
            x: the x coordinate for the top left corner
            y: the y coordinate for the top left corner
            w: the width of the rectangle
            h: the height of the rectangle
            color: an RGBA tuple for the color of the rectangle
        
        Returns:
            None"""
        if not self.open: raise Exception('Cannot draw on a closed window.')
        shapes.Rectangle(
            *self.center(*self.camera.transform_point(x, y)),
            *self.camera.scale_point(w, h),
            color
            ).draw()

    def line(self, x1, y1, x2, y2, color=(255, 255, 255, 255), thickness=1) -> None:
        """Draws a line on the screen
        
        Args:
            x1: the x coordinate of the start point
            y1: the y coordinate of the start point
            x2: the x coordinate of the end point
            y2: the y coordinate of the end point
            color: the color for the line
            thickness: the thickness of the line
        
        Returns:
            None"""
        if not self.open: raise Exception('Cannot draw on a closed window.')
        shapes.Line(
            *self.center(*self.camera.transform_point(x1, y1)), 
            *self.center(*self.camera.transform_point(x2, y2)), 
            thickness=thickness, 
            color=color
            ).draw()
        
    def clear(self):
        self.window.clear()

    def update(self):
        self.window.dispatch_events()
        for batch in self.batches:
              batch.draw()
        self.window.flip()
        