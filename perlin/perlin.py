import random as rand
import numpy as np
import math
from grid import *


class perlin:

    def __init__(self, size, max_rec):

        self.size = (2 ** size)+1
        self.max = self.size - 1
        self.max_rec = max_rec
        self.make_grids()
        self.cur_rec = 0
        self.divide()

    def get_2d_grid(self):
        return self.grid.get_2dgrid()
    def get_2d_vectors():
        return self.vectors.get_2dgrid()

    def generate_vectors(self):
        for i in range(0, self.max+1, int(self.size)):
            for j in range(0, self.max+1, int(self.size)):
                vector = self.normalize(
                    ( rand.uniform(-self.size,self.size),
                    rand.uniform(-self.size,self.size),
                    )
                )
                if self.cur_rec ==0:
                    vector = (abs(vector[0]), abs(vector[1]))
                self.vectors.set_val(i,j,vector)

    def make_grids(self):

        self.grid = grid(self.size, 0)
        self.vectors = grid(self.size, (0,0))

    def is_vector_point(self, x, y):
        if x % self.size == 0 and y % self.size == 0:
            return True
        else:
            return False

    def normalize(self, v):
        norm = math.sqrt(v[0]**2 + v[1]**2)
        return (v[0]/norm, v[1]/norm)

    def get_pos_vector(self, v1, v2):
        return ((v1[0]-v2[0])/self.size, (v1[1]-v2[1])/self.size)
        #return ((v2[0]-v1[0]), (v2[1]-v1[1]))

    def get_dot_product(self, v1, v2):
        return  (v1[0] * v2[0]) + (v1[1] * v2[1])

    def fade(self, t):
        return 6*(t**5)-15*(t**4) + 10*(t**3)

    def linterp(self, a,b,x):
        return a+ x*(b-a)

    def calculate(self, x, y):

        grid_points = [(math.floor(x/self.size)*self.size, math.floor(y/self.size)*self.size),
                        (math.ceil(x/self.size)*self.size, math.floor(y/self.size)*self.size),
                        (math.floor(x/self.size)*self.size, math.ceil(y/self.size)*self.size),
                        (math.ceil(x/self.size)*self.size, math.ceil(y/self.size)*self.size)]

        grid_vectors = []

        for i in grid_points:
            #print ("requesting vector at: ", i)
            grid_vectors.append(self.vectors.get_val(int(i[0]),int(i[1])))
            #print (" ")
        pos_vectors = []

        for i in grid_points:
            pos_vectors.append(self.get_pos_vector((x,y), i))

        dots = []

        for i in range(len(grid_vectors)):
            dots.append(
                self.get_dot_product(pos_vectors[i], grid_vectors[i]))

        unit = (pos_vectors[0][0], pos_vectors[0][1])

        u = self.fade(unit[0])
        v = self.fade(unit[1])

        temp1 = self.linterp(dots[0], dots[1], u)
        temp2 = self.linterp(dots[2], dots[3], u)

        avg = self.linterp(temp1, temp2, v)
        avg = (avg+1.0)/2.0

        self.grid.add_val(x, y, avg, self.cur_scale)
        #self.grid.set_val(x,y, avg)


    def divide(self):
        if self.cur_rec == self.max_rec:
            return

        self.size = self.max/(2**self.cur_rec)
        print ("current size: %i" % self.size)

        self.cur_scale = 1/(2**self.cur_rec)
        #self.cur_scale = 1/(self.cur_rec+1)
        print ("current scale: %2f" %self.cur_scale )

        self.generate_vectors()

        for i in range(self.max+1):
            for j in range(self.max+1):
                if not self.is_vector_point(i, j):
                    self.calculate(i,j)
                else:
                    self.grid.set_val(i,j, self.grid.get_val(i-1, j-1))


        self.cur_rec = self.cur_rec + 1
        self.divide()
