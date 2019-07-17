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

from data import get_dataset, get_trajectory, transform_q_p, dynamics_fn, hamiltonian_fn, arrange_data, hamiltonian_6d_fn
from nn_models import MLP, PSD, DampMatrix
from hnn import HNN, HNN_structure, HNN_structure_pend
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

args = ObjectView(get_args())
#%% [markdown]
# ## Inspect the dataset

#%%
R = 2.5
# field = get_field(xmin=-R, xmax=R, ymin=-R, ymax=R, gridsize=15)
# data = get_dataset()

# plot config
fig = plt.figure(figsize=(3, 3), facecolor='white', dpi=DPI)

x, y, dx, dy, t = get_trajectory(radius=2.4, y0=np.array([2,0]), noise_std=0)
plt.scatter(x,y,c=t,s=14, label='data')
# plt.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
#            cmap='gray_r', color=(.5,.5,.5))
plt.xlabel("$q$", fontsize=14)
plt.ylabel("p", rotation=0, fontsize=14)
plt.title("Dynamics")
plt.legend(loc='upper right')

plt.tight_layout() ; plt.show()

#%%
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
def get_model(args, baseline, structure, damping, num_points):
    M_net = PSD(int(args.input_dim/2), args.hidden_dim, int(args.input_dim/2)).to(device)
    V_net = MLP(int(args.input_dim/2), args.hidden_dim, 1).to(device)
    # A_net = ConstraintNet(int(args.input_dim/2), 50, 1, 2).to(device)
    F_net = MLP(int(args.input_dim/2)*3, args.hidden_dim, int(args.input_dim/2), bias_bool=False).to(device)
    model = HNN_structure_pend(args.input_dim, M_net, V_net, F_net, device).to(device)
    path = '{}pend-hnn_ode-p{}.tar'.format(args.save_dir, num_points)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

#%% [markdown]
# ## Integrate along vector fields

#%%
from torchdiffeq import odeint 
def integrate_model(model, t_span, y0, **kwargs):
    
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,6).to(device)
        dx = model.time_derivative(0, x).detach().cpu().numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

#%% [markdown]
# ## Run analysis

hnn_ode_model = get_model(args, baseline=False, structure=False, damping=False, num_points=args.num_points)

#%%
# t_span = [0,28]
# y0 = transform_q_p(np.array([2.1]), np.array([0.]))
# y0 = y0[0]
# #%%
# kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 1000), 'rtol': 1e-12}
# hnn_ivp = integrate_model(hnn_ode_model, t_span, y0, **kwargs)


# #%%
# hnn_ivp['y'].shape

# #%%
# x, y, theta, p_x, p_y, p_theta = np.split(hnn_ivp['y'], 6)
# x, y, theta, p_x, p_y, p_theta = x[0], y[0], theta[0], p_x[0], p_y[0], p_theta[0]
# #%%
# plt.plot(x)

# #%%
# plt.plot(y)

# #%%
# plt.plot(theta)

# #%%
# plt.plot(p_x)

# #%%
# plt.plot(p_y)

# #%%
# plt.plot(p_theta)

#%%
def integrate_models(x0=np.asarray([1, 0]), t_span=[0,5], t_eval=None):
    # integrate along ground truth vector field
    kwargs = {'t_eval': t_eval, 'rtol': 1e-7, 'method': 'LSODA'}
    true_path = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=x0, **kwargs)
    q, p = true_path['y'][0,:], true_path['y'][1,:]
    true_x = transform_q_p(q, p)

    # integrate along HNN vector field
    y0 = transform_q_p(np.array([x0[0]]), np.array([x0[1]]))
    y0 = y0[0]
    hnn_path = integrate_model(hnn_ode_model, t_span, y0, **kwargs)
    hnn_x = hnn_path['y'].T
    # hnn_x = true_x

    return true_x, hnn_x

#%%
x0 = np.asarray([2.1, 0])
t_span=[0,10]
t_eval = np.linspace(t_span[0], t_span[1], 100)
kwargs = {'t_eval': t_eval, 'rtol': 1e-7, 'method': 'RK23'}

# integrate along HNN vector field
y0 = transform_q_p(np.array([x0[0]]), np.array([x0[1]]))
y0 = y0[0]

#%%
hnn_path = integrate_model(hnn_ode_model, t_span, y0, **kwargs)
hnn_x = hnn_path['y'].T
hnn_x
#%%
x0 = np.asarray([2.1, 0])

# integration
t_span=[0,1]
t_eval = np.linspace(t_span[0], t_span[1], 100)
true_x, hnn_x = integrate_models(x0=x0, t_span=t_span, t_eval=t_eval)

#%%
tpad = 7

fig = plt.figure(figsize=[12,28], dpi=DPI)
plt.subplot(7,1,1)
plt.title("x and x_hat", pad=tpad) ; plt.xlabel('$t$') #; plt.ylabel('$x and x_hat$')
plt.plot(t_eval, true_x[:, 0], t_eval, hnn_x[:, 0], 'g-')

plt.subplot(7,1,2)
plt.title("y and y_hat", pad=tpad) ; plt.xlabel('$t$') #; plt.ylabel('$p_x and p_xhat$')
plt.plot(t_eval, true_x[:, 1], t_eval, hnn_x[:, 1], 'g-')

plt.subplot(7,1,3)
plt.title("theta and theta_hat", pad=tpad) ; plt.xlabel('$t$') #; plt.ylabel('$p_x and p_xhat$')
plt.plot(t_eval, true_x[:, 2], t_eval, hnn_x[:, 2], 'g-')

plt.subplot(7,1,4)
plt.title("p_x and p_xhat", pad=tpad) ; plt.xlabel('$t$') #; plt.ylabel('$p_x and p_xhat$')
plt.plot(t_eval, true_x[:, 3], t_eval, hnn_x[:, 3], 'g-')

plt.subplot(7,1,5)
plt.title("p_y and p_yhat", pad=tpad) ; plt.xlabel('$t$') #; plt.ylabel('$p_x and p_xhat$')
plt.plot(t_eval, true_x[:, 3], t_eval, hnn_x[:, 3], 'g-')

plt.subplot(7,1,6)
plt.title("p_theta and p_thetahat", pad=tpad) ; plt.xlabel('$t$') #; plt.ylabel('$p_x and p_xhat$')
plt.plot(t_eval, true_x[:, 3], t_eval, hnn_x[:, 3], 'g-')

plt.subplot(7,1,7)
plt.title("Total energy", pad=tpad)
plt.xlabel('Time step')
true_e = hamiltonian_6d_fn(true_x)
hnn_e = hamiltonian_6d_fn(hnn_x)
plt.plot(t_eval, true_e, t_eval, hnn_e, 'g-')


#%%
# fig.savefig('{}/pend-single-wo-force-p{}.{}'.format(args.fig_dir, args.num_points, FORMAT))


#%%
tensor_x = torch.tensor(true_x, dtype=torch.float32, requires_grad=True).to(device)

#%%
H_hat = hnn_ode_model(tensor_x)

#%%
plt.plot(H_hat.detach().cpu().numpy())

#%%
RHS = hnn_ode_model.time_derivative(_, tensor_x)
RHS = RHS.detach().cpu().numpy()
#%%
plt.plot(RHS[:,0])

#%%
plt.plot(RHS[:,1])

#%%
plt.plot(RHS[:,2])

#%%
plt.plot(RHS[:,3])

#%%
plt.plot(RHS[:,4])

#%%
plt.plot(RHS[:,5])

#%%
pend_hnn_stats = from_pickle(EXPERIMENT_DIR + 'pend-hnn_ode-p4-stats.pkl')
hnn_nfe = np.array(pend_hnn_stats['nfe'])
hnn_diff_nfe = hnn_nfe[1:] - hnn_nfe[:-1]
hnn_forward_time = np.array(pend_hnn_stats['forward_time'])
hnn_backward_time = np.array(pend_hnn_stats['backward_time'])
hnn_train_loss = np.array(pend_hnn_stats['train_loss'])
hnn_test_loss = np.array(pend_hnn_stats['test_loss'])
#%%
plt.plot(hnn_train_loss)

#%%
plt.plot(hnn_test_loss)

#%%
dHdq, dHdp, F, Fc, dHdp_Fc = hnn_ode_model.get_intermediate_value(_, tensor_x)


#%%
dHdq = dHdq.detach().cpu().numpy()
dHdp = dHdp.detach().cpu().numpy()
F = F.detach().cpu().numpy()
Fc = Fc.detach().cpu().numpy()
dHdp_Fc = dHdp_Fc.detach().cpu().numpy()


#%%
plt.plot(dHdp_Fc)

#%%
plt.plot(dHdp[:, 0])

#%%

plt.plot(dHdp[:, 1])

#%%
plt.plot(dHdp[:, 2])

#%%
plt.plot(Fc[:, 0])

#%%
plt.plot(Fc[:, 1])

#%%
plt.plot(Fc[:, 2])

#%%
