from perlin import *
from grid import *
from PIL import Image
import random as rand
import numpy as np

class map: 

	def __init__(self, size):
		rand.seed()
		self.height_rec = 5
		self.size = size
		self.height = perlin(size, self.height_rec)

		self.biomes_rec = 3
		#rain = perlin(self.size, self.biomes_rec)
		#temp = perlin(self.size, self.biomes_rec)
		
		self.image = grid((2**self.size)+1, (0,0,0))

		self.make_biomes()
		self.show_map()
		


	def tile(self, tile_dim):
		self.tiled_grid = self.image.get_2dgrid().tolist()

		for i in range(len(self.tiled_grid)):
			self.tiled_grid[i].extend(self.tiled_grid[i][::1])

		self.tiled_grid.append(self.tiled_grid[::1])
		#self.tiled_grid = np.array(self.tiled_grid)

	def make_biomes(self):
		grid = self.height.get_2d_grid()
		for i in range(len(grid)):
			for j in range(len(grid)):
				if grid[i][j] <= 0.35:
					self.image.set_val(i,j, (0,0,153))
				elif grid[i][j] <=0.45:
					self.image.set_val(i,j, (0, 128, 255))
				elif grid[i][j] <= 0.55:
					self.image.set_val(i,j,	(225,240,153))
				elif grid[i][j] <=0.70:
					self.image.set_val(i,j, (0,153,0))
				elif grid[i][j] <=0.85:
					self.image.set_val(i,j, (128,128,128))
				else:
					self.image.set_val(i,j, (255,255,255))

	def show_map(self):
		image = self.image.get_2dgrid()
		image = Image.fromarray(np.uint8(image))
		image.show()

map(10)