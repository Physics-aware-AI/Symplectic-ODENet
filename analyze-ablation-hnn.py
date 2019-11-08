# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# code structure follows the style of HNN by Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

# This file is a script version of 'analyze-ablation-hnn.ipynb'
# Cells are seperated by the vscode convention '#%%'

#%%
import torch, time, sys
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

EXPERIMENT_DIR = './experiment-single-force/'
sys.path.append(EXPERIMENT_DIR)

from data import get_dataset, get_trajectory, dynamics_fn, hamiltonian_fn, arrange_data, get_field
from nn_models import MLP, PSD
from symoden import SymODEN_R
from hnn import HNN
from utils import L2_loss, from_pickle

#%%
DPI = 600
FORMAT = 'png'
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 1

def get_args():
    return {'input_dim': 2,
         'learn_rate': 1e-3,
         'nonlinearity': 'tanh',
         'baseline': False,
         'field_type': 'solenoidal',
         'name': 'pend',
         'gridsize': 10,
         'seed': 0,
         'save_dir': './{}'.format(EXPERIMENT_DIR),
         'fig_dir': './figures',
         'num_points': 2,
         'gpu': 0,
         'solver': 'rk4',
         'rad': True,
         'gym': False}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())

from torchdiffeq import odeint
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

#%% [markdown]
# ## Load models
#%%
# the symoden model
def get_model(args, baseline, structure, damping, num_points, gym=False):
    if structure == False and baseline == True:
        nn_model = MLP(args.input_dim, 600, args.input_dim, args.nonlinearity).to(device)    
        model = SymODEN_R(args.input_dim, H_net=nn_model, device=device, baseline=True)
    elif structure == False and baseline == False:
        H_net = MLP(args.input_dim, 400, 1, args.nonlinearity).to(device)
        g_net = MLP(int(args.input_dim/2), 200, int(args.input_dim/2)).to(device)
        model = SymODEN_R(args.input_dim, H_net=H_net, g_net=g_net, device=device, baseline=False)
    elif structure == True and baseline ==False:
        M_net = MLP(int(args.input_dim/2), 300, int(args.input_dim/2))
        V_net = MLP(int(args.input_dim/2), 50, 1).to(device)
        g_net = MLP(int(args.input_dim/2), 200, int(args.input_dim/2)).to(device)
        model = SymODEN_R(args.input_dim, M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=False, structure=True).to(device)
    else:
        raise RuntimeError('argument *baseline* and *structure* cannot both be true')
    model_name = 'baseline_ode' if baseline else 'hnn_ode'
    struct = '-struct' if structure else ''
    rad = '-rad' if args.rad else ''
    path = '{}pend-{}{}-{}-p{}{}.tar'.format(args.save_dir, model_name, struct, args.solver, num_points, rad)
    model.load_state_dict(torch.load(path, map_location=device))
    path = '{}/pend-{}{}-{}-p{}-stats{}.pkl'.format(args.save_dir, model_name, struct, args.solver, num_points, rad)
    stats = from_pickle(path)
    return model, stats

symoden_ode_model, symoden_ode_stats = get_model(args, baseline=False, structure=False, damping=False, num_points=args.num_points, gym=args.gym)

#%%
# the hnn model
def get_model(args, baseline):
    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, 400, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)

    label = '-baseline' if args.baseline else '-hnn'
    label = label + '-rad' if args.rad else label
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)


    return model

hnn_model = get_model(args, baseline=False)

#%% [markdown]
# ## Dataset to get prediction error

#%%
us = [0.0]
data = get_dataset(seed=args.seed, timesteps=90, rad=args.rad,
            save_dir=args.save_dir, us=us, samples=50) #us=np.linspace(-2.0, 2.0, 20)

pred_x, pred_t_eval = data['x'], data['t']

#%%
# get prediction loss of symoden

def get_pred_loss(pred_x, pred_t_eval, model):
    pred_x = torch.tensor(pred_x, requires_grad=True, dtype=torch.float32).to(device) 
    pred_t_eval = torch.tensor(pred_t_eval, requires_grad=True, dtype=torch.float32).to(device)

    pred_loss = []
    for i in range(pred_x.shape[0]):
        pred_x_hat = odeint(model, pred_x[i, 0, :, :], pred_t_eval, method='rk4')            
        pred_loss.append((pred_x[i,:,:,:] - pred_x_hat)**2)
    
    pred_loss = torch.cat(pred_loss, dim=1)
    pred_loss_per_traj = torch.sum(pred_loss, dim=(0, 2))

    return pred_loss_per_traj.detach().cpu().numpy()

symoden_pred_loss = get_pred_loss(pred_x, pred_t_eval, symoden_ode_model)

#%%
# get prediction loss of hnn
def get_pred_loss(pred_x, pred_t_eval, model):
    pred_x = torch.tensor(pred_x, requires_grad=True, dtype=torch.float32).to(device) 
    pred_t_eval = torch.tensor(pred_t_eval, requires_grad=True, dtype=torch.float32).to(device)

    pred_loss = []
    for i in range(pred_x.shape[0]):
        pred_x_hat = odeint(model.int_wrapper, pred_x[i, 0, :, 0:2], pred_t_eval, method='rk4')            
        pred_loss.append((pred_x[i,:,:,0:2] - pred_x_hat)**2)
    
    pred_loss = torch.cat(pred_loss, dim=1)
    pred_loss_per_traj = torch.sum(pred_loss, dim=(0, 2))

    return pred_loss_per_traj.detach().cpu().numpy()

hnn_pred_loss = get_pred_loss(pred_x, pred_t_eval, hnn_model)


#%%
# can we get prediction loss of hnn?
# this function seems not supporting batch
def integrate_model(model, t_span, y0, **kwargs):
    
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,2)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

def integrate_models(x0=np.asarray([1, 0]), t_span=[0,5], t_eval=None):
    # integrate along ground truth vector field
    # kwargs = {'t_eval': t_eval, 'rtol': 1e-12}
    # true_path = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=x0, **kwargs)
    # true_x = true_path['y'].T

    # integrate along HNN vector field
    hnn_path = integrate_model(hnn_model, t_span, x0, **kwargs)
    hnn_x = hnn_path['y'].T
    _tmp = torch.tensor( true_x, requires_grad=True, dtype=torch.float32)
    
    return hnn_x