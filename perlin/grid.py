import numpy as np

class grid:

  def __init__ (self, size, init_val) :
    self.size = size
    self.make_grid(init_val)

  def make_grid(self, init_val):
    self.grid = []
    for x in range((self.size) * (self.size)):
      self.grid.append(init_val)

  def get_val(self, i,j):
    return self.grid[j + self.size*i]

  def add_val(self, i,j, val, scale):
    self.grid[j+ self.size*i] = self.grid[j + self.size*i] + (scale*val)

  def set_val(self, i,j, val):
    self.grid[j+self.size*i] = val

  def get_grid(self):
    return self.grid

  def get_norm_grid(self):
    if type(self.grid[0]) is int or type(self.grid[0]) is float:
      mi = min(self.grid)
      self.norm_grid = [i-mi for i in self.grid]
      ma = max(self.norm_grid)
      self.norm_grid = [i/ma for i in self.norm_grid]
      return self.norm_grid
    return self.grid

  def get_2dgrid(self):
    grid = self.get_norm_grid()
    if type(grid[0]) is float:
      return np.reshape(grid, (self.size, self.size))
    else:
      temp = np.reshape(grid, (self.size, self.size, len(grid[0])))
      return temp
