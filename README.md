# fireSimulation
Simulation of a forest fire spreading
  - Terrain with semi-random tree density generated using Perlin Noise
  - Forest fire spreading = BFS algorithm, allowing nodes to be re-visited a certain number of times
  - Each cell catches on fire with a probability based on its density and the number of its neighbor cells 
    that are on fire. (Beta distribution: highest probability around 70% density, goes to to 0 
    at 0% density and goes to approx half of peak probability at 100% density)
  
  To implement: 
   - Wind vector: neighbor's contribution to the probability of the cell catching fire should be multiplied 
     by the dot-product of the vector from the neighbor to the cell and the wind vector. If the dot product
     is negative, the contribution is simply zero. (ie: with wind blowing towards the left, the neighbor on 
     the left's contribution to a cell catching fire is 0)
   - Burn map at different time stamps to visualize the fire propagating
   
