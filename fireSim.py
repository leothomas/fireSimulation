import numpy as np
import scipy as scp 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def gen_map(tsize):
	return norm_terrain(terrain_layer(tsize,tsize) + \
		terrain_layer(tsize,tsize/2) + terrain_layer(tsize,tsize/4))

def terrain_layer(tsize, res):
	terrain = np.zeros((tsize,tsize))

	#randx = (1- (-1))*np.random.random_sample((int(tsize/res)+1, int(tsize/res)+1))-1
	#randy = (1- (-1))*np.random.random_sample((int(tsize/res)+1, int(tsize/res)+1))-1

	randx = (1-(-1))*np.random.rand(int(tsize/res +1), int(tsize/res +1))-1
	randy = (1-(-1))*np.random.rand(int(tsize/res +1), int(tsize/res +1))-1

	#print (randx)
	#print (randy)

	for i in range(0, tsize):
		for j in range(0,tsize):
			point = ("(" + str(i) + "," + str(j) + ")")
			#print ("point: ", point)
			#terrain[i][j] = bilinear_interp(grid_points(i,j), scalars(i,j), i,j)
			terrain[i,j] = interp(grid_points(i,j,res), scalars(i,j,res,randx,randy), i, j)

	return terrain

def grid_points(i,j, res):
	x = [np.floor(i/res)*res, np.ceil(i/res)*res]
	y = [np.floor(j/res)*res, np.ceil(j/res)*res]

	return np.array([[x[0],y[0]], [x[0],y[1]], [x[1], y[0]], [x[1],y[1]] ])


def grid_vectors(i,j,res,randx,randy):
	vpoints = (grid_points(i,j,res)/res).astype(int)
	#print ("\n vpoints: ", [n for n in vpoints])
	vectors = np.array([[randx[n[0], n[1]], randy[n[0], n[1]]] for n in vpoints])
	norm_vectors = np.array([n/np.linalg.norm(n) for n in vectors])
	return vectors
	#return norm_vectors

def position_vector(i,j,res):
	pvectors = np.array([i,j] - grid_points(i,j, res))
	norm_pvectors = np.array([n/np.linalg.norm(n) for n in pvectors])
	#return norm_pvectors
	return pvectors

def scalars (i,j, res,randx,randy):
	gvectors = grid_vectors(i,j,res,randx,randy)
	pvectors = position_vector(i,j,res)
	scalars = np.zeros(len(gvectors))

	for n in range(len(gvectors)):
		scalars[n]= np.dot(gvectors[n], pvectors[n])
	
	return np.abs(scalars)
	#return scalars

def bilinear_interp(points, values, i,j):

	x1 = points[0][0]
	x2 = points[2][0]
	y1 = points[0][1]
	y2 = points[1][1]

	A = np.array([
		 [1, x1, y1, x1*y1],
		 [1, x1, y2, x1*y2],
		 [1, x2, y1, x2*y1],
		 [1, x2, y2, x2*y2]])

	B = values
	try: 
		x = np.linalg.solve(A,B)
	except: 
		x = np.linalg.lstsq(A,B)
		x = x[0]

	return (x[0] + x[1]*i + x[2]*j + x[3]*i*j)

def interp(points, values, i,j):
	for n in range(len(points)):
		if np.array_equal(points[n], [i,j]):
			return values[n]

	#return smooth_interp(points, values, i,j)
	return bilinear_interp(points,values, i,j)

# TODO: implement smooth interpolation (possibly using imshow builtin function)
def smooth_interp(points, values, i,j):
	x = [n[0] for n in points]
	y = [n[1] for n in points]
	#return scp.interpolate.griddata(points, values, [i,j], method = 'cubic')[0]
	f = scp.interpolate.interp2d(x, y, values, kind = 'cubic')
	return f (i,j)

def norm_terrain(terrain):
	tmax = np.amax(terrain)
	for i in range(len(terrain)):
		for j in range(len(terrain[0])):
			terrain[i,j] = terrain[i,j]/tmax
	return terrain

def probability(val):
	size=1000
	p = np.random.beta(5,3,size)
	plt.figure(4)
	count,bins,ignored=plt.hist(p,25, normed = False)
	prob = count[int(val/4)]
	if np.random.randint(low=0,high=101) <= prob:
		return True
	else:
		return False

def find_neighbors(i,j,tsize):
	
	neighbors = np.array([[i-1,j-1],[i-1,j],[i-1,j+1],[i,j-1],[i,j+1],[i+1,j-1],[i+1,j],[i+1,j+1]])
	neighbors = [[n[0],n[1]] for n in neighbors if n[0] in range(0,tsize) and n[1] in range(0,tsize)]
	return np.array(neighbors)
	
def neighbors_burning(neighbors,burn):
	burning=0
	for n in neighbors:
		if burn[n[0],n[1]] >= 0:
			burning = burning + 1

	return burning

def spreadFire(terrain, i,j):

	burn = np.zeros((len(terrain)+1,len(terrain)+1))
	burn[i,j] = burn[i,j]+1
	queue = find_neighbors(i,j,len(terrain))
	visited = np.array([[i,j]])

	size=1000
	p = np.random.beta(5,3,size)
	plt.figure(4)
	count,bins,ignored=plt.hist(p,100, normed = False)

	while len(queue) != 0:
		#print (queue)
		print (len(queue))
		print (len(visited))
		x=queue[0,0] 
		y=queue[0,1] 
		
		neighbors = find_neighbors(x,y, len(terrain))
		burning = neighbors_burning(neighbors,burn)
		
		for n in neighbors:
			if n not in visited and n not in queue:
				queue = np.append(queue,[n], axis=0)

		#print (queue)
		
		visited = np.append(visited, queue[0])
		
		#queue = np.array([queue[n] for n in range(1,len(queue))])
		queue = np.delete(queue, 0,0)
		#print (queue)
		
		prob = count[terrain[x,y]]

		if np.random.randint(low=0,high=101)<= (prob*burning):
			burn[x,y] = burn[x,y] + 1

	return burn
"""
tmap = gen_map(100)
plt.figure(1)
plot1 = plt.imshow(tmap, interpolation="bicubic", cmap = plt.get_cmap('BuGn'))
plt.figure(2)
plot2 = plt.imshow(tmap, interpolation="bicubic", cmap=plt.get_cmap('BrBG'))
plt.figure(3)
plot3 = plt.imshow(tmap, interpolation='bicubic')
#tmap.set_cmap('hot')

burn = spreadFire(tmap, 40,40)

plt.figure(5)
plot5 = plt.imshow(burn, cmap=plt.get_cmap('hot'))
print (probability(70))

"""
queue = np.array([[90,90],[91,90],[91,89],[92,89]])
print (queue)
n = [40,29]
queue = np.append(queue,[n],0)
print (queue)
if n not in queue:
	queueu = np.append(queue,[n],0)
print (queue)

plt.show()