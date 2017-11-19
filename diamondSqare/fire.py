from PIL import Image
import matplotlib.pyplot as plt
from fire_methods import *

#value_map = DiamondSquare(9, 0.25).get_2d_grid()

class world:
    def __init__(self, size):
        self.height_map(size)
        self.make_water(0.2)

    def height_map(self, size):
        self.height_map = DiamondSquare(size, 0.5)

    def make_water(self, level):
        grid = self.height_map.get_2d_grid()
        gmax = np.amax(grid)
        print (gmax)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i,j] < level*gmax: self.height_map.set(i,j,0.0)

    def get_2d_grid(self):
        return self.height_map.get_2d_grid()

world = world(9)
world_grid = world.get_2d_grid()

plot = plt.imshow(world_grid, interpolation="bicubic", cmap = plt.get_cmap('BuGn'))
plt.show()

heightmap_im = Image.fromarray(world_grid*255).convert('RGB')
heightmap_im.save("map.jpeg")
#webbrowser.open("map.jpeg")
heightmap_im.show()