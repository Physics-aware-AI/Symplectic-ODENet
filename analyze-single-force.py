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

#%% [markdown]
# ## Inspect the dataset
# We can either set initial condition to be the same or different for different forces
#%%
# data = get_dataset(seed=args.seed, gym=False, save_dir=args.save_dir, us=[-1.0, 0.0, 1.0])
# print(data['x'].shape)

#%%
# t = 0
# q_01 = data['x'][0,t,:,0] ; p_01 = data['x'][0,t,:,1]
# q_02 = data['x'][1,t,:,0] ; p_02 = data['x'][1,t,:,1]
# q_03 = data['x'][2,t,:,0] ; p_03 = data['x'][2,t,:,1]

# for _ in range(0):
#     fig = plt.figure(figsize=[12,3], dpi=DPI)
#     plt.subplot(1, 3, 1)
#     plt.scatter(q_01, p_01)
#     plt.xlim(-3, 3)
#     plt.ylim(-3, 3)

#     plt.subplot(1, 3, 2)
#     plt.scatter(q_02, p_02)
#     plt.xlim(-3, 3)
#     plt.ylim(-3, 3)

#     plt.subplot(1, 3, 3)
#     plt.scatter(q_03, p_03)
#     plt.xlim(-3, 3)
#     plt.ylim(-3, 3)

# train_q = data['x'][:,:,:,0].reshape(-1,1)
# plt.hist(train_q, 20)
# %%
# this is an example of q goes to -28
# fig = plt.figure(figsize=[12, 3], dpi=DPI)
# plt.plot(data['x'][0,:, 4, 0])

#%%
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
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

base_ode_model = get_model(args, baseline=True, structure=False, damping=False, num_points=args.num_points)
hnn_ode_model = get_model(args, baseline=False, structure=False, damping=False, num_points=args.num_points)
hnn_ode_struct_model = get_model(args, baseline=False, structure=True, damping=False, num_points=args.num_points)

#%% [markdown]
# ## Integrate along vector fields

#%%
from torchdiffeq import odeint 
def integrate_model(model, t_span, y0, **kwargs):
    
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,3).to(device)
        dx = model.time_derivative(0, x).detach().cpu().numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

t_span = [0,28]
y0 = np.asarray([2.1, 0])
u0 = 1.0
y0_u = np.asarray([2.1, 0, u0])
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 1000), 'rtol': 1e-12}
base_ivp = integrate_model(base_ode_model, t_span, y0_u, **kwargs)
hnn_ivp = integrate_model(hnn_ode_model, t_span, y0_u, **kwargs)
hnn_struct_ivp = integrate_model(hnn_ode_struct_model, t_span, y0_u, **kwargs)


#%%
# get vector filed of different model
def get_vector_field(model, **kwargs):
    field = get_field(**kwargs)
    np_mesh_x = field['x']
    
    # run model
    mesh_x = torch.tensor( np_mesh_x, requires_grad=True, dtype=torch.float32).to(device)
    mesh_x_aug = torch.cat((mesh_x, torch.zeros_like(mesh_x)[:,0].view(-1, 1)), dim=1)
    mesh_dx_aug = model.time_derivative(0, mesh_x_aug)
    mesh_dx = mesh_dx_aug[:, 0:2]
    return mesh_dx.detach().cpu().numpy()

# get their vector fields
R = 3.6
kwargs = {'xmin': 0.0, 'xmax': 2*R, 'ymin': -R, 'ymax': R, 'gridsize': args.gridsize, 'u': 0}
field = get_field(**kwargs)
# data = get_dataset(radius=2.0)
base_field = get_vector_field(base_ode_model, **kwargs)
hnn_field = get_vector_field(hnn_ode_model, **kwargs)
hnn_struct_field = get_vector_field(hnn_ode_struct_model, **kwargs)

#%%
###### PLOT ######

for _ in range(1):
    fig = plt.figure(figsize=(16, 3.2), dpi=DPI)

    plt.subplot(1, 4, 1)
    x, y, t = get_trajectory(t_span=[0, 4], noise_std=0.0, y0=y0, radius=2.1, u=u0)
    N = len(x)
    point_colors = [(i/N, 0, 1-i/N) for i in range(N)]
    plt.scatter(x,y, s=14, label='data', c=point_colors)

    plt.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2))  

    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Data", pad=10)
    plt.xlim(0, 2*R)
    plt.ylim(-R, R)

    plt.subplot(1, 4, 2)
    for i, l in enumerate(np.split(base_ivp['y'].T, LINE_SEGMENTS)):
        color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
        plt.plot(l[:,0],l[:,1],color=color, linewidth=LINE_WIDTH)

    plt.quiver(field['x'][:,0], field['x'][:,1], base_field[:,0], base_field[:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.5,.5,.5))

    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Baseline ODE NN ({})".format(args.num_points), pad=10)
    plt.xlim(0, 2*R)
    plt.ylim(-R, R)

    plt.subplot(1, 4, 3)
    for i, l in enumerate(np.split(hnn_ivp['y'].T, LINE_SEGMENTS)):
        color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
        plt.plot(l[:,0],l[:,1],color=color, linewidth=LINE_WIDTH)

    plt.quiver(field['x'][:,0], field['x'][:,1], hnn_field[:,0], hnn_field[:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.5,.5,.5))

    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Hamiltonian ODE NN ({})".format(args.num_points), pad=10)
    plt.xlim(0, 2*R)
    plt.ylim(-R, R)

    plt.subplot(1, 4, 4)
    for i, l in enumerate(np.split(hnn_struct_ivp['y'].T, LINE_SEGMENTS)):
        color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
        plt.plot(l[:,0],l[:,1],color=color, linewidth=LINE_WIDTH)

    plt.quiver(field['x'][:,0], field['x'][:,1], hnn_struct_field[:,0], hnn_struct_field[:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.5,.5,.5))

    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Hamiltonian structured ODE NN ({})".format(args.num_points), pad=10)
    plt.xlim(0, 2*R)
    plt.ylim(-R, R)

#%%
# plot leanrt function

q = np.linspace(-5.0, 5.0, 40)
q_tensor = torch.tensor(q, dtype=torch.float32).view(40, 1).to(device)

for _ in range(1):
    fig = plt.figure(figsize=(20, 3.2), dpi=DPI)
    g_q = hnn_ode_model.g_net(q_tensor)
    plt.subplot(1, 4, 1)
    plt.plot(q, g_q.detach().cpu().numpy())
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$g_q$", rotation=0, fontsize=14)
    plt.title("g_q - Hamiltonian ODE NN ({})".format(args.num_points), pad=10, fontsize=14)


    g_q = hnn_ode_struct_model.g_net(q_tensor)
    plt.subplot(1, 4, 2)
    plt.plot(q, g_q.detach().cpu().numpy())
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$g_q$", rotation=0, fontsize=14)
    plt.title("g_q - Hamiltonian structured ODE NN ({})".format(args.num_points), pad=10, fontsize=14)

    M_q_inv = hnn_ode_struct_model.M_net(q_tensor)
    plt.subplot(1, 4, 3)
    plt.plot(q, M_q_inv.detach().cpu().numpy())
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$Mq_inv$", rotation=0, fontsize=14)
    plt.title("Mq_inv - Hamiltonian structured ODE NN ({})".format(args.num_points), pad=10, fontsize=14)

    V_q = hnn_ode_struct_model.V_net(q_tensor)
    plt.subplot(1, 4, 4)
    plt.plot(q, V_q.detach().cpu().numpy())
    plt.plot(q, -5. * np.cos(q))
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$V_q$", rotation=0, fontsize=14)
    plt.title("V_q - Hamiltonian structured ODE NN ({})".format(args.num_points), pad=10, fontsize=14)


#%%
# vanilla control
t_span = [0,20]
y0 = torch.tensor([0, 0, 0], requires_grad=True, device=device, dtype=torch.float32).view(1, 3)
t_eval = torch.linspace(t_span[0], t_span[1], 100)
rtol = 1e-12
y = y0
k_p = 1 ; k_d = 0.6
y_traj = []
y_traj.append(y)
for i in range(len(t_eval)-1):
    q, p, _ = torch.chunk(y, 3, dim=1)
    V_q = hnn_ode_struct_model.V_net(q)
    dV_q = torch.autograd.grad(V_q, q)[0]

    if p > 0:
        u = dV_q - k_p * (q - np.pi) - k_d * p
    else:
        u = dV_q - k_p * (q + np.pi) - k_d * p
    if u > 2.0:
        u = 2.0 * torch.ones_like(u)
    if u < -2.0:
        u = 2.0 * torch.ones_like(u)
    y0_u = torch.cat((q, p, u), dim = 1)
    y_step = odeint(hnn_ode_struct_model.time_derivative, y0_u, t_eval[i:i+2], method='rk4')
    y = y_step[-1,:,:]
    y_traj.append(y)

y_traj = torch.stack(y_traj).view(-1, 3).detach().cpu().numpy()




#%%
# plot control result
fig = plt.figure(figsize=[10, 10], dpi=DPI)
plt.subplot(3, 1, 1)
plt.plot(t_eval.numpy(), y_traj[:, 0])

plt.subplot(3, 1, 2)
plt.plot(t_eval.numpy(), y_traj[:, 1])

plt.subplot(3, 1, 3)
plt.plot(t_eval.numpy(), y_traj[:, 2])

#%%

# MPC control
from mpc import mpc
from mpc.mpc import QuadCost, GradMethods

n_batch, n_state, n_ctrl, T = 1, 2, 1, 40
u_lower = 2.0 * torch.ones(T, n_batch, n_ctrl, dtype=torch.float32, device=device)
u_upper = -2.0 * torch.ones(T, n_batch, n_ctrl, dtype=torch.float32, device=device)
u_init = None #0.0 * torch.ones(T, n_batch, n_ctrl, dtype=torch.float32, device=device, requires_grad=True)
x_init = torch.tensor([[0.2, 0]], dtype=torch.float32, device=device, requires_grad=True).view(n_batch, n_state)

# cost
C = torch.diag(torch.tensor([1.0, 1, 1], dtype=torch.float32, device=device)).view(1, 1, 3, 3)
C = C.repeat(T, n_batch, 1, 1) 
c = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 3)
c = c.repeat(T, n_batch, 1)

# class Discrete_wrapper(torch.nn.Module):
#     def __init__(self, diff_model):
#         super(Discrete_wrapper, self).__init__()
#         self.diff_model = diff_model

#     def forward(self, x, u): 
#         # y0_u = torch.cat((x, u), dim = 1)
#         with torch.enable_grad():
#             y0_u = torch.zeros(1, 3, dtype=torch.float32, requires_grad=True, device=device)
#             # y0_u = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True, device=device)
#             y0_u[:, 0:2] = x
#             y0_u[:, 2] = torch.squeeze(u)
#             y_step = odeint(self.diff_model, y0_u, torch.linspace(0.0, 0.1, 2), method='rk4')
#         return y_step[-1,:,0:2]


# class True_discrete_dynamics(torch.nn.Module):
#     def __init__(self):
#         super(True_discrete_dynamics, self).__init__()
#         self.assume_canonical_coords = True
#         self.device = device
#         self.M = self.permutation_tensor(2)
        

#     def forward(self, x, u):
#         # with torch.enable_grad():
#             # q_p = torch.zeros(x.shape, dtype=torch.float32, requires_grad=True, device=device)
#             # q_p[:,:] = x.data
#             # q, p = torch.chunk(q_p,2, dim=1)
#             # H = 5*(1-torch.cos(q)) + 1.5 * p**2
#             # dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
#             # H_vector_field = torch.matmul(dH, self.M.t())
#             # F_vector_field = torch.cat((torch.zeros_like(u), u), dim=1)
#             # return x + (H_vector_field + F_vector_field) * 0.05
#         q, p = torch.chunk(x, 2, dim=1)
#         return x + torch.cat((3*p, -5*torch.sin(q)+u), dim=1)*0.05

#     def permutation_tensor(self, n):
#         M = None
#         if self.assume_canonical_coords:
#             M = torch.eye(n)
#             M = torch.cat([M[n//2:], -M[:n//2]])
#         else:
#             '''Constructs the Levi-Civita permutation tensor'''
#             M = torch.ones(n,n) # matrix of ones
#             M *= 1 - torch.eye(n) # clear diagonals
#             M[::2] *= -1 # pattern of signs
#             M[:,::2] *= -1

#             for i in range(n): # make asymmetric
#                 for j in range(i+1, n):
#                     M[i,j] *= -1
#         return M.to(self.device)

# # discrete_hnn_struct = Discrete_wrapper(hnn_ode_struct_model.time_derivative)
# discrete_hnn_struct = True_discrete_dynamics().to(device)
# #%%
# nominal_states, nominal_actions, nominal_objs = mpc.MPC(
#     n_state=n_state, 
#     n_ctrl=n_ctrl, 
#     T=T,
#     u_init=u_init,
#     u_lower=u_lower,
#     u_upper=u_upper,
#     lqr_iter=50,
#     verbose=1,
#     n_batch=n_batch,
#     exit_unconverged=False,
#     detach_unconverged=False,
#     grad_method=GradMethods.AUTO_DIFF,
#     linesearch_decay=0.2, 
#     max_linesearch_iter=5,
#     eps=1e-2,
# )(x_init, QuadCost(C, c), discrete_hnn_struct)

#%%

# 3D input
n_batch, T, mpc_T = 1, 100, 20

torch.manual_seed(0)
th = torch.ones(n_batch, dtype=torch.float32) * 3.14
thdot = torch.zeros(n_batch, dtype=torch.float32)
xinit = torch.stack((torch.cos(th), torch.sin(th), thdot), dim=1)

x = xinit
u_init = None

goal_weights = torch.Tensor((1., 1., 0.1))
goal_state = torch.Tensor((1., 0. ,0.))
ctrl_penalty = 0.001
q = torch.cat((
    goal_weights,
    ctrl_penalty*torch.ones(1)
))
px = -torch.sqrt(goal_weights)*goal_state
p = torch.cat((px, torch.zeros(1)))
Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
    mpc_T, n_batch, 1, 1
)
p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)


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
        cos_q, sin_q, p = torch.chunk(x, 3, dim=1)
        q = torch.atan2(sin_q, cos_q)
        new_q = q + 3*p * 0.05
        new_p = p + (u - 5*torch.sin(q)) * 0.05
        return torch.cat((torch.cos(new_q), torch.sin(new_q), new_p), dim=1)

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

# discrete_hnn_struct = Discrete_wrapper(hnn_ode_struct_model.time_derivative)
discrete_hnn_struct = True_discrete_dynamics().to(device)
#%%
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
)(x, QuadCost(Q, p), discrete_hnn_struct)

#%%
