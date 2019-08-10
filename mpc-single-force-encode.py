#%% 
import torch, time, sys
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

EXPERIMENT_DIR = './experiment-single-force-encode/'
sys.path.append(EXPERIMENT_DIR)

from data import get_dataset, get_trajectory, dynamics_fn, hamiltonian_fn, arrange_data, get_field
from nn_models import MLP, PSD, DampMatrix
from hnn import HNN, HNN_structure, HNN_structure_forcing
from utils import L2_loss, from_pickle
from torchdiffeq import odeint 
from tqdm import tqdm
from time import time

#%%
DPI = 300
FORMAT = 'png'
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2

def get_args():
    return {'input_dim': 2,
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
         'gpu': 0,
         'rad': False}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())

#%%
# device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def get_model(args, baseline, structure, damping, num_points, gym=False):
    reg_net = MLP(3, 100, 2).to(device)
    obs_net = MLP(2, 100, 3).to(device)
    if structure == False:
        output_dim = args.input_dim if baseline else 1
        nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity).to(device)
        g_net = MLP(int(args.input_dim/2), args.hidden_dim, int(args.input_dim/2)).to(device)
        model = HNN_structure_forcing(args.input_dim, H_net=nn_model, g_net=g_net, device=device, baseline=baseline)
    elif structure == True and baseline ==False:
        M_net = MLP(int(args.input_dim/2), args.hidden_dim, int(args.input_dim/2)).to(device)
        V_net = MLP(int(args.input_dim/2), 50, 1).to(device)
        g_net = MLP(int(args.input_dim/2), args.hidden_dim, int(args.input_dim/2)).to(device)
        model = HNN_structure_forcing(args.input_dim, M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=baseline, structure=True).to(device)
    else:
        raise RuntimeError('argument *baseline* and *structure* cannot both be true')    
    
    model_name = 'baseline_ode' if baseline else 'hnn_ode'
    struct = '-struct' if structure else ''
    rad = '-rad' if args.rad else ''
    gym_data = '-gym' if gym else ''
    path = '{}pend-{}{}{}-p{}{}.tar'.format(args.save_dir, model_name, struct, gym_data, num_points, rad)
    model.load_state_dict(torch.load(path, map_location=device))
    path = '{}/reg-{}{}{}-p{}{}.tar'.format(args.save_dir, model_name, struct, gym_data, args.num_points, rad)
    reg_net.load_state_dict(torch.load(path, map_location=device))
    path = '{}/obs-{}{}{}-p{}{}.tar'.format(args.save_dir, model_name, struct, gym_data, args.num_points, rad)
    obs_net.load_state_dict(torch.load(path, map_location=device))
    return model, reg_net, obs_net

hnn_ode_struct_model, hnn_ode_struct_reg_net, hnn_ode_struct_obs_net = \
    get_model(args, baseline=False, structure=True, damping=False, num_points=args.num_points, gym=True)
hnn_ode_model, hnn_ode_reg_net, hnn_ode_obs_net = \
    get_model(args, baseline=False, structure=False, damping=False, num_points=args.num_points, gym=True)
# base_ode_model, base_ode_reg_net, base_ode_obs_net = \
#     get_model(args, baseline=True, structure=False, damping=False, num_points=args.num_points, gym=True)

#%%
class Discrete_wrapper(torch.nn.Module):
    def __init__(self, diff_model, reg_net, obs_net):
        super(Discrete_wrapper, self).__init__()
        self.diff_model = diff_model
        self.reg_net = reg_net
        self.obs_net = obs_net

    def forward(self, x, u): 
        # y0_u = torch.cat((x, u), dim = 1)
        with torch.enable_grad():
            obs = torch.zeros(1, 3, dtype=torch.float32, requires_grad=True, device=device)
            force = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True, device=device)
            obs[0, :] = x
            force[0, :] = u
            q_p = self.reg_net(obs)
            y0_u = torch.cat((q_p, force), dim=1)
            q_p_new = odeint(self.diff_model, y0_u, torch.linspace(0.0, 0.05, 2), method='rk4')
            q_p_new = q_p_new[-1,:,0:2]
            obs_new = self.obs_net(q_p_new)
        return obs_new

discrete_hnn_ode_struct_model = Discrete_wrapper(hnn_ode_struct_model.time_derivative,
                                    hnn_ode_struct_reg_net,
                                    hnn_ode_struct_obs_net)
discrete_hnn_ode_model = Discrete_wrapper(hnn_ode_model.time_derivative,
                                    hnn_ode_reg_net,
                                    hnn_ode_obs_net)
# discrete_base_ode_model = Discrete_wrapper(base_ode_model.time_derivative,
#                                     base_ode_reg_net,
#                                     base_ode_obs_net)

#%%
import gym
import myenv
from mpc import mpc
from mpc.mpc import QuadCost, GradMethods

# set gym env
env = gym.make('MyPendulum-v0')
# env = gym.wrappers.Monitor(env, './videos/' + str(time()) + '/') 
env.seed(0)

n_batch, n_state, n_ctrl, mpc_T, T = 1, 3, 1, 20, 60
u_lower = -2.0 * torch.ones(mpc_T, n_batch, n_ctrl, dtype=torch.float32, device=device)
u_upper = 2.0 * torch.ones(mpc_T, n_batch, n_ctrl, dtype=torch.float32, device=device)
u_init = None #0.0 * torch.ones(T, n_batch, n_ctrl, dtype=torch.float32, device=device, requires_grad=True)
x_init = torch.tensor([[np.cos(0), np.sin(0), 0]], dtype=torch.float32, device=device, requires_grad=True).view(n_batch, n_state)

# cost
C = torch.diag(torch.tensor([1.0, 1, 1, 0.01], dtype=torch.float32, device=device)).view(1, 1, 4, 4)
C = C.repeat(mpc_T, n_batch, 1, 1) 
c = torch.tensor([-1., 0.0, 0.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 4)
c = c.repeat(mpc_T, n_batch, 1)

# env.reset()
# env.state = np.array([0.0, 0.0], dtype=np.float32)
# x = env._get_obs()
x = x_init

actual_action = []
actual_states = []
actual_states.append(x)
for t in tqdm(range(T)):
    tensor_x = torch.tensor(x, dtype=torch.float32, device=device).view(1, 3)
    nominal_states, nominal_actions, nominal_objs = mpc.MPC(
        n_state=n_state, 
        n_ctrl=n_ctrl, 
        T=mpc_T,
        u_init=u_init,
        u_lower=u_lower,
        u_upper=u_upper,
        lqr_iter=50,
        verbose=1,
        n_batch=n_batch,
        exit_unconverged=False,
        detach_unconverged=False,
        grad_method=GradMethods.AUTO_DIFF,
        linesearch_decay=0.2, 
        max_linesearch_iter=5,
        eps=1e-2,
    )(tensor_x, QuadCost(C, c), discrete_hnn_ode_model)

    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, n_ctrl, device=device)), dim=0)
    next_action = nominal_actions[0].detach().cpu().numpy()
    # env.render()
    # x, _, _, _ = env.step(next_action[0])
    x = discrete_hnn_ode_struct_model(x, nominal_actions[0])

    actual_action.append(next_action[0])
    actual_states.append(x)



#%%
actual_action = np.stack(actual_action)
actual_states = np.stack(actual_states)

#%%
fig = plt.figure(figsize=[12, 12], dpi=DPI)
plt.subplot(3, 1, 1)
plt.plot(np.arctan2(-actual_states[:, 1], -actual_states[:, 0]))

plt.subplot(3, 1, 2)
plt.plot(actual_states[:, 2])

plt.subplot(3, 1, 3)
plt.plot(actual_action[:, 0])

model_name = "hnn_struct"
# model_name = "hnn"
# model_name = "base"
# fig.savefig('{}/mpc-{}-gym-p{}-new.{}'.format(args.fig_dir, model_name, args.num_points, FORMAT))

#%%
