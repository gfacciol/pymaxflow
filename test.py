import sys
import pymaxflow
import pylab
import scipy.misc
import numpy as np

if len(sys.argv) > 1:
	filename = sys.argv[1]
else:
	filename = 'a2.png'
	print('no parameter supplied, reading default file %s'%(filename))

im = scipy.misc.imread(filename).astype(np.float32) 
	
indices = np.arange(im.size).reshape(im.shape).astype(np.int32)
g = pymaxflow.PyGraph(im.size, im.size * 4)

g.add_node(im.size)

# adjacent left (equal weights)
diffs = np.abs(im[:, 1:] - im[:, :-1]).ravel()*0 + 50.0
e1 = indices[:, :-1].ravel()
e2 = indices[:,  1:].ravel()
g.add_edge_vectorized(e1, e2, diffs, diffs)

# adjacent down (equal weights)
diffs = np.abs(im[:-1, :] - im[1:, :]).ravel()*0 + 50.0
e1 = indices[:-1, :].ravel()
e2 = indices[1: , :].ravel()
g.add_edge_vectorized(e1, e2, diffs, diffs)

# link to source/sink
g.add_tweights_vectorized(indices.ravel(), im.ravel(), (255.0 - im.ravel()))

print("calling maxflow")
g.maxflow()

out = g.what_segment_vectorized()
pylab.imshow(out.reshape(im.shape))
pylab.show()
