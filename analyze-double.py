#%% 
import torch, time, sys
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

EXPERIMENT_DIR = './experiment-double/'
sys.path.append(EXPERIMENT_DIR)

from data import get_dataset, get_field, dynamics_fn, hamiltonian_fn, arrange_data
from nn_models import MLP
from hnn import HNN, HNN_structure
from utils import L2_loss, from_pickle

#%% [markdown]
# ## Set some notebook constants


#%%
DPI = 300
FORMAT = 'png'
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2

def get_args():
    return {'input_dim': 2,
         'hidden_dim': 200,
         'learn_rate': 1e-3,
         'nonlinearity': 'tanh',
         'total_steps': 2000,
         'field_type': 'solenoidal',
         'print_every': 200,
         'name': 'pend',
         'gridsize': 10,
         'input_noise': 0.5,
         'seed': 0,
         'save_dir': './{}'.format(EXPERIMENT_DIR),
         'fig_dir': './figures',
         'num_points': 4,
         'gpu': 0}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

#%% [markdown]
# ## Inspect the dataset

args = ObjectView(get_args())
data = get_dataset('acrobot', args.save_dir, verbose=True, seed=args.seed)
print(data['train_x'].shape)
print(data['test_x'].shape)
#%%
plt.plot(data['train_x'][:,0,0])
plt.show()

#%%
plt.plot(data['train_x'][:,0,1])
plt.show()

#%%
plt.plot(data['train_x'][:,0,2])
plt.show()

#%%
plt.plot(data['train_x'][:,0,3])
plt.show()

#%%
plt.plot(data['train_x'][:,0,4])
plt.show()

#%%
plt.plot(data['train_x'][:,0,5])
plt.show()

#%%
plt.plot(data['train_x'][:,0,6])
plt.show()

#%%
plt.plot(data['train_x'][:,0,7])
plt.show()

#%%
plt.plot(data['train_x'][:,0,8])
plt.show()

#%%
plt.plot(data['train_x'][:,0,9])
plt.show()

#%%
plt.plot(data['train_x'][:,0,10])
plt.show()

#%%
plt.plot(data['train_x'][:,0,11])
plt.show()

#%%
