#%% 
import torch, time, sys
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

EXPERIMENT_DIR = './experiment-single-force/'
sys.path.append(EXPERIMENT_DIR)

from data import get_dataset, get_trajectory, dynamics_fn, hamiltonian_fn, arrange_data
from nn_models import MLP, PSD, DampMatrix
from hnn import HNN, HNN_structure, HNN_structure_forcing
from utils import L2_loss, from_pickle

#%%
DPI = 300
FORMAT = 'png'
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2

def get_args():
    return {'input_dim': 6,
         'hidden_dim': 600,
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
         'num_points': 2,
         'gpu': 0}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())

#%% [markdown]
# ## Inspect the dataset
# We can either set initial condition to be the same or different for different forces
#%%
data = get_dataset(seed=args.seed, gym=False, save_dir=args.save_dir, us=[-1.0, 0.0, 1.0])
print(data['x'].shape)
q_01 = data['x'][0,0,:,0] ; p_01 = data['x'][0,0,:,1]
q_02 = data['x'][1,0,:,0] ; p_02 = data['x'][1,0,:,1]
q_03 = data['x'][2,0,:,0] ; p_03 = data['x'][2,0,:,1]

plt.subplot(1, 3, 1)
plt.scatter(q_01, p_01)

plt.subplot(1, 3, 2)
plt.scatter(q_02, p_02)

plt.subplot(1, 3, 3)
plt.scatter(q_03, p_03)

#%%
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
def get_model(args, baseline, structure, damping, num_points):
    if structure == False:
        output_dim = args.input_dim if baseline else 1
        nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity).to(device)
        g_net = MLP(int(args.input_dim/3), args.hidden_dim, int(args.input_dim/3)).to(device)
        model = HNN_structure_forcing(args.input_dim, H_net=nn_model, g_net=g_net, device=device, baseline=baseline)
    elif structure == True and baseline ==False:
    
        M_net = MLP(1, args.hidden_dim, 1).to(device)
        V_net = MLP(int(args.input_dim/3), 4, 1).to(device)
        g_net = MLP(int(args.input_dim/3), args.hidden_dim, int(args.input_dim/3)).to(device)
        model = HNN_structure_forcing(args.input_dim, M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=baseline, structure=True).to(device)
    else:
        raise RuntimeError('argument *baseline* and *structure* cannot both be true')

    return model