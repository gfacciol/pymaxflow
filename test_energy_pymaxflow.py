import pymaxflow
import numpy as np

def solve_binary_problem(indices, D0, D1, 
                        e1, e2, V00, V01, V10, V11,
                        _e1=None, _e2=None, _V00=None, _V01=None, _V10=None, _V11=None):
    '''
    solve the labeling problem     
        min_x    \sum_i  D_i(x_i) + \sum_ij  V_ij(x_i,x_j)     
        with x_i \in {0,1}
    '''
    sz = D0.size
    sh = D0.shape
    e  = pymaxflow.PyEnergy(sz, sz*4)

    # variables
    first_node_idx = e.add_variable(sz)
    
    # add unary terms                     
    e.add_term1_vectorized(indices.ravel(), D0.ravel(), D1.ravel())
    
    # add binary terms                     
    e.add_term2_vectorized(e1.ravel() , e2.ravel() , 
                           V00.ravel(), V01.ravel(), 
                           V10.ravel(), V11.ravel())
    # add more binary terms                     
    if type(_V11) != type(None):
        e.add_term2_vectorized(_e1.ravel() , _e2.ravel() , 
                               _V00.ravel(), _V01.ravel(), 
                               _V10.ravel(), _V11.ravel())
    Emin = e.minimize()       
    out  = e.get_var_vectorized()
    return Emin, out.reshape(sh)




def test_binary_optimization(N=300):
   ''' 
   a simple Potts minimization problem
   \min_{x: binary}  \sum_i  |x_i - f_i|^2 + 
                     \lambda \sum_{ij: edges} (1 - \delta(x_i,x_j)
   '''

   f = np.random.rand(N,N/2).astype(np.float32); 
   
   sh = f.shape
   sz = f.size
   
   l0 = 0.0
   l1 = 1.0
   
   # Unary terms
   D0 = (l0 - f)**2
   D1 = (f - l1)**2 
   indices = np.arange(sz).reshape(sh).astype(np.int32)
   
   # Binary terms 
   lmbda = 0.5
   # in this example K is just something with the right size
   K = np.ones(sh) * lmbda   
   #K = np.abs(im[1:, 1:] - im[:-1, :-1]) # K could be image dependent
   
   # horizontal edges
   V00 = np.abs(K[:, 1:]*0).astype(np.float32)
   V01 = np.abs(K[:, 1:]*1).astype(np.float32)
   V10 = np.abs(K[:, 1:]*1).astype(np.float32)
   V11 = np.abs(K[:, 1:]*0).astype(np.float32)
   e1  = indices[:, :-1]
   e2  = indices[:, 1: ]
   
   # vertical edges
   _V00 = np.abs(K[1:, :]*0).astype(np.float32)
   _V01 = np.abs(K[1:, :]*1).astype(np.float32)
   _V10 = np.abs(K[1:, :]*1).astype(np.float32)
   _V11 = np.abs(K[1:, :]*0).astype(np.float32)
   _e1  = indices[:-1,:]
   _e2  = indices[1:, :]
   
   # The vertical edges are passed using the optional _?? edges. 
   # Concatenating them to the first parameters yield the same result.
   Emin, out = solve_binary_problem(indices, D0, D1
                                   ,e1,e2,V00,V01,V10,V11
                                   ,_e1,_e2,_V00,_V01,_V10,_V11
                                  )
   return Emin, f, out




###############################
###############################
###############################
###############################
Emin, f, out = test_binary_optimization()
print('Energy: ', Emin)

import matplotlib.pyplot as plt

fig = plt.figure()
plt.subplot(121)
plt.imshow(f)
plt.subplot(122)
plt.imshow(out)

#plt.subplot_tool()
plt.show()
