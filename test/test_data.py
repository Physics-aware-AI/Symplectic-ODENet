#%%
#%% 
import torch, time, sys
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

EXPERIMENT_DIR = './experiment-single/'
sys.path.append(EXPERIMENT_DIR)

from data import get_dataset
from nn_models import MLP
# from hnn import HNN, HNN_structure
# from utils import L2_loss, from_pickle

#%%
data = get_dataset(seed=0, gym=False, save_dir=None, verbose=True)

#%%
traj = data['x'][:,18,:]

x, y, theta, p_x, p_y, p_theta = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3], traj[:, 4], traj[:, 5]

H = p_x * p_x /2 + p_y * p_y /2 + 6 * p_theta * p_theta + 10 * y + 5
print(H)
#%%
plt.plot(y)

#%%
