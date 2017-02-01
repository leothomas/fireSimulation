import numpy as np
import scipy as scp 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def gen_map(tsize):
	return norm_terrain(terrain_layer(tsize,tsize) + \
		terrain_layer(tsize,tsize/2)\
		+ 0.5*terrain_layer(tsize,tsize/4)
		)

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
"""
def probability(val):
	size=1000
	p = np.random.beta(5,3,size)

	count,bins,ignored=plt.hist(p,25, normed = False)
	
	prob = count[int(val/4)]
	if np.random.randint(low=0,high=101) <= prob:
		return True
	else:
		return False
"""
def find_neighbors(i,j,tsize):
	
	neighbors = np.array([[i-1,j-1],[i-1,j],[i-1,j+1],[i,j-1],[i,j+1],[i+1,j-1],[i+1,j],[i+1,j+1]])
	neighbors = [[n[0],n[1]] for n in neighbors if n[0] in range(0,tsize) and n[1] in range(0,tsize)]
	return np.array(neighbors)
	
def neighbors_burning(neighbors,burn):
	burning=0
	for n in neighbors:
		if burn[n[0],n[1]] > 0:
			burning = burning + 1
	return burning

def spreadFire(terrain, i,j):

	burn = np.zeros((len(terrain)+1,len(terrain)+1))
	burn[i,j] = burn[i,j]+1
	queue = find_neighbors(i,j,len(terrain))
	for n in queue: burn[n[0],n[1]] = burn[n[0],n[1]] +1
	
	visited = np.zeros((len(terrain),len(terrain)))

	size=1000
	p = np.random.beta(7,3,size)
	plt.figure(1)
	count,bins,ignored=plt.hist(p,100, normed = False)

	#plt.ion()
	#plt.show()
	maxVisit = 50
	maxVisited = 0
	while len(queue) != 0:
		
		x=queue[0,0] 
		y=queue[0,1] 
		
		queue = np.delete(queue, 0,0)

		neighbors = find_neighbors(x,y, len(terrain))
		
		burning = neighbors_burning(neighbors,burn)
		
		if burning == 0: continue
		if visited[x,y] > maxVisit: continue

		#print ("point: ", (x,y))
		#print ("visited: ",visited)
		#print ("burning: ", burning)
		np.random.shuffle(neighbors)
		for n in neighbors:
			if n not in queue:
				queue = np.append(queue,[n], axis=0)
				queue = np.append(queue,[[x,y]], axis = 0)
		
		visited[x,y] = visited[x,y]+1

		if visited[x,y]> maxVisited: 
			#print ("Current Max Number of Visits: ", maxVisited)
			maxVisited = visited[x,y]
		#queue = np.array([queue[n] for n in range(1,len(queue))])
		
		prob = count[terrain[x,y]]

		if np.random.randint(low=0,high=50) <= (prob*burning):
		#if np.random.choice([True, False]) == True:
			burn[x,y] = burn[x,y] + 1
		
		for i in range(0,len(terrain)):
			for j in range(0,len(terrain)):
				if burn[i,j] > 0 : burn[i,j] = burn[i,j] + 1
 		
		"""
		plt.imshow(burn, cmap=plt.get_cmap('hot'))
		plt.draw()
		#title = (str((x,y)))
		#plt.title(title)
		#plt.pause(0.00001)
		"""

	return burn


loadMap = True
mapsize = 50

startCood = np.array([])

if(loadMap):
	try:
		print ("\n loading terrain map...\n ")
		tmap = np.load("TerrainMap.npy")
		
		print ("\n loading burn map... \n")
		burn = np.load("BurnMap.npy")
		startCoord = np.load("StartCoord.npy")

	except: 
		print ("\n unable to load terrain, generating a new one... \n")
		tmap = gen_map(mapsize)
		np.save("TerrainMap", tmap)

		print ("\n unable to load burn map, generating new one... \n")
		
		print ("\n Generating BurnMap...")
		startx = int(np.random.normal(mapsize/2, mapsize/4))
		starty = int(np.random.normal(mapsize/2, mapsize/4))

		print ("\n Starting point: (%d, %d)" % (startx, starty))
		burn = spreadFire(tmap,starty, startx)
		np.save("BurnMap", burn)
		np.save("StartCoord", [startx,starty])


else:
	print ("\n Generating terrain map...")
	tmap = gen_map(mapsize)
	np.save("TerrainMap", tmap)

	print ("\n Generating BurnMap...")
	startx = int(np.random.normal(mapsize/2, mapsize/4))
	starty = int(np.random.normal(mapsize/2, mapsize/4))

	print ("\n Starting point: (%d, %d)" % (startx, starty))

	burn = spreadFire(tmap,starty, startx)
	np.save("BurnMap", burn)
	np.save("StartCoord", [startx,starty])

plt.figure()
plot1 = plt.imshow(tmap, interpolation="bicubic", cmap = plt.get_cmap('BuGn'))

plt.figure()
plot2 = plt.imshow(tmap, interpolation="bicubic", cmap=plt.get_cmap('BrBG'))

plt.figure()
plot3 = plt.imshow(tmap, interpolation='bicubic')
clevels = np.linspace(0, np.amax(burn), num=3)
plt.contour(burn, levels= clevels, colors ='k')



plt.figure()
plot5 = plt.imshow(burn, cmap=plt.get_cmap('OrRd'))
plt.title("Starting point: (%d, %d) " %(startCoord[0], startCoord[1]))


plt.figure()
plt.contour(burn, levels = clevels)



plt.show()

