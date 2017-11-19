import numpy as np
import random as rand 


class DiamondSquare:

    def __init__(self, size, roughness):

        rand.seed()

        self.size = (2 ** size) + 1
        self.max = self.size - 1
        self.roughness = roughness
        self.make_grid(self.size)
        self.divide(self.max)
        self.norm_grid()

    # Sets x,y position in self.grid
    def set(self, x, y, val):
        self.grid[x + self.size * y] = val

    # Get's value of x, y in self.grid
    def get(self, x, y):
        if (x < 0 or x > self.max or y < 0 or y > self.max):
            return -1
        return self.grid[x + self.size * y]

    def divide(self, size):
        size = int(size)
        x = int(size / 2)
        y = int(size / 2)
        half = int(size / 2)
        scale = self.roughness * size

        if (half < 1):
            return

        # Square
        for y in range(half, self.max, size):
            for x in range(half, self.max, size):
                s_scale = rand.uniform(0, 1) * scale * 2 - scale
                self.square(x, y, half, s_scale)

        # Diamond
        for y in range(0, self.max + 1, half):
            for x in range((y + half) % size, self.max + 1, size):
                d_scale = rand.uniform(0, 1) * scale * 2 - scale
                self.diamond(x, y, half, d_scale)

        self.divide(size / 2) 

    def square(self, x, y, size, scale):

        top_left = self.get(x - size, y - size)
        top_right = self.get(x + size, y - size)
        bottom_left = self.get(x + size, y + size)
        bottom_right = self.get(x - size, y + size)

        average = ((top_left + top_right + bottom_left + bottom_right) / 4)
        self.set(x, y, average + scale)

    def diamond(self, x, y, size, scale):

        """
                T

            L   X   R

                B
        """

        top = self.get(x, y - size)
        right = self.get(x + size, y)
        bottom = self.get(x, y + size)
        left = self.get(x - size, y)

        average = ((top + right + bottom + left) / 4)
        self.set(x, y, average + scale)


    def make_grid(self, size):

        self.grid = []

        for x in range(size * size):
            self.grid.append(-1)

        self.set(0, 0, rand.random())
        self.set(self.max, 0, rand.random())
        self.set(self.max, self.max, rand.random())
        self.set(0, self.max, rand.random())

    
    def norm_grid(self):
        self.norm_grid = np.array(self.grid)
        self.norm_grid = (self.grid + np.amin(self.grid))
        self.norm_grid = (self.grid/np.amax(self.grid))
    
    def get_grid(self):
        return self.grid

    def get_2dgrid(self):
        self.grid2d = np.reshape(self.norm_grid, (self.size, self.size))
        return self.grid2d
