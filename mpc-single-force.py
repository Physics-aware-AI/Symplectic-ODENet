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

from data import get_dataset, get_trajectory, dynamics_fn, hamiltonian_fn, arrange_data, get_field
from nn_models import MLP, PSD, DampMatrix
from hnn import HNN, HNN_structure, HNN_structure_forcing
from utils import L2_loss, from_pickle
from torchdiffeq import odeint 
from tqdm import tqdm

#%%
DPI = 300
FORMAT = 'png'
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2

def get_args():
    return {'input_dim': 3,
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
         'num_points': 4,
         'gpu': 0,
         'rad': False}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())

#%%
# device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def get_model(args, baseline, structure, damping, num_points):
    if structure == False:
        output_dim = args.input_dim-1 if baseline else 1
        nn_model = MLP(args.input_dim-1, args.hidden_dim, output_dim, args.nonlinearity).to(device)
        g_net = MLP(int(args.input_dim/3), args.hidden_dim, int(args.input_dim/3)).to(device)
        model = HNN_structure_forcing(args.input_dim, H_net=nn_model, g_net=g_net, device=device, baseline=baseline)
    elif structure == True and baseline ==False:
    
        M_net = MLP(1, args.hidden_dim, 1).to(device)
        V_net = MLP(int(args.input_dim/3), 50, 1).to(device)
        g_net = MLP(int(args.input_dim/3), args.hidden_dim, int(args.input_dim/3)).to(device)
        model = HNN_structure_forcing(args.input_dim, M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=baseline, structure=True).to(device)
    else:
        raise RuntimeError('argument *baseline* and *structure* cannot both be true')
    model_name = 'baseline_ode' if baseline else 'hnn_ode'
    struct = '-struct' if structure else ''
    rad = '-rad' if args.rad else ''
    path = '{}pend-{}{}-p{}{}.tar'.format(args.save_dir, model_name, struct, num_points, rad)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

hnn_ode_struct_model = get_model(args, baseline=False, structure=True, damping=False, num_points=args.num_points)


from mpc import mpc
from mpc.mpc import QuadCost, GradMethods
from mpc.env_dx import pendulum
params = torch.tensor((10., 1., 1.))
dx = pendulum.PendulumDx(params, simple=True)

n_batch, n_state, n_ctrl, mpc_T, T = 1, 2, 1, 50, 150
u_lower = -2.0 * torch.ones(mpc_T, n_batch, n_ctrl, dtype=torch.float32, device=device)
u_upper = 2.0 * torch.ones(mpc_T, n_batch, n_ctrl, dtype=torch.float32, device=device)
u_init = None #0.0 * torch.ones(T, n_batch, n_ctrl, dtype=torch.float32, device=device, requires_grad=True)
x_init = torch.tensor([[0.2, 0]], dtype=torch.float32, device=device, requires_grad=True).view(n_batch, n_state)

# cost
C = torch.diag(torch.tensor([1.0, 1, 1], dtype=torch.float32, device=device)).view(1, 1, 3, 3)
C = C.repeat(mpc_T, n_batch, 1, 1) 
c = torch.tensor([-np.pi, 0.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 3)
c = c.repeat(mpc_T, n_batch, 1)

class Discrete_wrapper(torch.nn.Module):
    def __init__(self, diff_model):
        super(Discrete_wrapper, self).__init__()
        self.diff_model = diff_model

    def forward(self, x, u): 
        # y0_u = torch.cat((x, u), dim = 1)
        with torch.enable_grad():
            y0_u = torch.zeros(1, 3, dtype=torch.float32, requires_grad=True, device=device)
            # y0_u = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True, device=device)
            y0_u[:, 0:2] = x
            y0_u[:, 2] = torch.squeeze(u)
            y_step = odeint(self.diff_model, y0_u, torch.linspace(0.0, 0.05, 2), method='rk4')
        return y_step[-1,:,0:2]


class True_discrete_dynamics(torch.nn.Module):
    def __init__(self):
        super(True_discrete_dynamics, self).__init__()
        self.assume_canonical_coords = True
        self.device = device
        self.M = self.permutation_tensor(2)
        

    def forward(self, x, u):
        # with torch.enable_grad():
            # q_p = torch.zeros(x.shape, dtype=torch.float32, requires_grad=True, device=device)
            # q_p[:,:] = x.data
            # q, p = torch.chunk(q_p,2, dim=1)
            # H = 5*(1-torch.cos(q)) + 1.5 * p**2
            # dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            # H_vector_field = torch.matmul(dH, self.M.t())
            # F_vector_field = torch.cat((torch.zeros_like(u), u), dim=1)
            # return x + (H_vector_field + F_vector_field) * 0.05
        q, p = torch.chunk(x, 2, dim=1)
        return x + torch.cat((3*p, -5*torch.sin(q)+u), dim=1)*0.05

    def permutation_tensor(self, n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1

            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M.to(self.device)

discrete_hnn_struct = Discrete_wrapper(hnn_ode_struct_model.time_derivative)
#%%
x = x_init
actual_action = []
actual_states = []
actual_states.append(x_init)
for t in tqdm(range(T)):

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
    )(x, QuadCost(C, c), discrete_hnn_struct)

    next_action = nominal_actions[0]
    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, n_ctrl, device=device)), dim=0)
    x = discrete_hnn_struct(x, next_action)

    actual_action.append(next_action)
    actual_states.append(x)

#%%
actual_action = torch.stack(actual_action).detach().cpu().numpy()
actual_states = torch.stack(actual_states).detach().cpu().numpy()
#%%
fig = plt.figure(figsize=[12, 12], dpi=DPI)
plt.subplot(3, 1, 1)
plt.plot(actual_states[:, 0, 0])

plt.subplot(3, 1, 2)
plt.plot(actual_states[:, 0, 1])

plt.subplot(3, 1, 3)
plt.plot(actual_action[:, 0, 0])
fig.savefig('{}/mpc.{}'.format(args.fig_dir, FORMAT))

#%%
