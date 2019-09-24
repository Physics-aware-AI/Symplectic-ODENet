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
DPI = 600
FORMAT = 'png'
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 1

def get_args():
    return {'input_dim': 2,
         'hidden_dim': 600,
         'learn_rate': 1e-3,
         'nonlinearity': 'tanh',
         'total_steps': 2000,
         'print_every': 200,
         'name': 'pend',
         'gridsize': 10,
         'input_noise': 0.5,
         'seed': 0,
         'save_dir': './{}'.format(EXPERIMENT_DIR),
         'fig_dir': './figures',
         'num_points': 5,
         'gpu': 3,
         'solver': 'dopri5',
         'rad': False,
         'gym': False}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())

#%% [markdown]
## Inspect the dataset
# We can either set initial condition to be the same or different for different forces
#%%
data = get_dataset(seed=args.seed, gym=True, save_dir=args.save_dir, us=[-2.0, 0.0, 2.0])
print(data['x'].shape)

print(data['t'])
#%%
# t = 1
# q_01 = data['x'][0,t,:,0] ; p_01 = data['x'][0,t,:,1]
# q_02 = data['x'][1,t,:,0] ; p_02 = data['x'][1,t,:,1]
# q_03 = data['x'][2,t,:,0] ; p_03 = data['x'][2,t,:,1]

i = 9
q_01 = data['x'][0,:,i,0] ; p_01 = data['x'][0,:,i,1]
q_02 = data['x'][1,:,i,0] ; p_02 = data['x'][1,:,i,1]
q_03 = data['x'][2,:,i,0] ; p_03 = data['x'][2,:,i,1]

for _ in range(0):
    fig = plt.figure(figsize=[12,3], dpi=DPI)
    plt.subplot(1, 3, 1)
    plt.plot(q_01)
    # plt.xlim(-3, 3)
    # plt.ylim(-3, 3)

    plt.subplot(1, 3, 2)
    plt.plot(p_01)
    # plt.xlim(-3, 3)
    # plt.ylim(-3, 3)

    plt.subplot(1, 3, 3)
    # plt.scatter(q_03, p_03)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

for _ in range(0):
    fig = plt.figure(figsize=[12,3], dpi=DPI)
    plt.subplot(1, 3, 1)
    plt.scatter(q_01, p_01)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.subplot(1, 3, 2)
    plt.scatter(q_02, p_02)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.subplot(1, 3, 3)
    plt.scatter(q_03, p_03)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

#%%
# train_p = data['x'][:,:,:,1].reshape(-1,1)
# plt.hist(train_p, 20)

#%%
# data['x'][2,:,i,2]
#%%
# this is an example of q goes to -28
# fig = plt.figure(figsize=[12, 3], dpi=DPI)
# plt.plot(data['x'][0,:, 4, 0])

#%%
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
def get_model(args, baseline, structure, damping, num_points, gym=False):
    if structure == False and baseline == True:
        nn_model = MLP(args.input_dim, 600, args.input_dim, args.nonlinearity).to(device)    
        model = HNN_structure_forcing(args.input_dim, H_net=nn_model, device=device, baseline=True)
    elif structure == False and baseline == False:
        H_net = MLP(args.input_dim, 400, 1, args.nonlinearity).to(device)
        g_net = MLP(int(args.input_dim/2), 200, int(args.input_dim/2)).to(device)
        model = HNN_structure_forcing(args.input_dim, H_net=H_net, g_net=g_net, device=device, baseline=False)
    elif structure == True and baseline ==False:
        # M_net = MLP(1, args.hidden_dim, 1).to(device)
        M_net = MLP(int(args.input_dim/2), 300, int(args.input_dim/2))
        V_net = MLP(int(args.input_dim/2), 50, 1).to(device)
        g_net = MLP(int(args.input_dim/2), 200, int(args.input_dim/2)).to(device)
        model = HNN_structure_forcing(args.input_dim, M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=False, structure=True).to(device)
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

base_ode_model, base_ode_stats = get_model(args, baseline=True, structure=False, damping=False, num_points=args.num_points, gym=args.gym)
hnn_ode_model, hnn_ode_stats = get_model(args, baseline=False, structure=False, damping=False, num_points=args.num_points, gym=args.gym)
hnn_ode_struct_model, hnn_ode_struct_stats = get_model(args, baseline=False, structure=True, damping=False, num_points=args.num_points, gym=args.gym)
#%%
# get number of parameters and final training loss
def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


print('Baseline_ode contains {} parameters'.format(get_model_parm_nums(base_ode_model)))
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(base_ode_stats['traj_train_loss']), np.std(base_ode_stats['traj_train_loss']),
        np.mean(base_ode_stats['traj_test_loss']), np.std(base_ode_stats['traj_test_loss'])))
print('')
print('HNN_ode contains {} parameters'.format(get_model_parm_nums(hnn_ode_model)))
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(hnn_ode_stats['traj_train_loss']), np.std(hnn_ode_stats['traj_train_loss']),
        np.mean(hnn_ode_stats['traj_test_loss']), np.std(hnn_ode_stats['traj_test_loss'])))
print('')
print('HNN_structure_ode contains {} parameters'.format(get_model_parm_nums(hnn_ode_struct_model)))
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(hnn_ode_struct_stats['traj_train_loss']), np.std(hnn_ode_struct_stats['traj_train_loss']),
        np.mean(hnn_ode_struct_stats['traj_test_loss']), np.std(hnn_ode_struct_stats['traj_test_loss'])))

#%%
# get prediction dataset
# us = [-2.0, -1.0, 0.0, 1.0, 2.0]
us = [0.0]
data = get_dataset(seed=args.seed, timesteps=40,
            save_dir=args.save_dir, us=us, samples=512) #us=np.linspace(-2.0, 2.0, 20)

pred_x, pred_t_eval = data['x'], data['t']

#%%
from torchdiffeq import odeint
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

base_pred_loss = get_pred_loss(pred_x, pred_t_eval, base_ode_model)
hnn_pred_loss = get_pred_loss(pred_x, pred_t_eval, hnn_ode_model)
hnn_struct_pred_loss = get_pred_loss(pred_x, pred_t_eval, hnn_ode_struct_model)


#%%

print('Baseline_ode')
print('Prediction loss {:.4e} +/- {:.4e}'
.format(np.mean(base_pred_loss), np.std(base_pred_loss)))
print('')
print('HNN_ode')
print('Prediction loss {:.4e} +/- {:.4e}'
.format(np.mean(hnn_pred_loss), np.std(hnn_pred_loss)))
print('')
print('HNN_structure_ode')
print('Prediction loss {:.4e} +/- {:.4e}'
.format(np.mean(hnn_struct_pred_loss), np.std(hnn_struct_pred_loss)))

#%% [markdown]
# ## Integrate along vector fields

#%%
from torchdiffeq import odeint 
def integrate_model(model, t_span, y0, **kwargs):
    
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,3).to(device)
        dx = model(0, x).detach().cpu().numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

t_span = [0,10]
y0 = np.asarray([1.8, 0])
u0 = 0.0
y0_u = np.asarray([1.8, 0, u0])
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 1000), 'rtol': 1e-12}
base_ivp = integrate_model(base_ode_model, t_span, y0_u, **kwargs)
hnn_ivp = integrate_model(hnn_ode_model, t_span, y0_u, **kwargs)
hnn_struct_ivp = integrate_model(hnn_ode_struct_model, t_span, y0_u, **kwargs)


#%%
# get vector field of different model
def get_vector_field(model, u=0, **kwargs):
    field = get_field(**kwargs)
    np_mesh_x = field['x']
    
    # run model
    mesh_x = torch.tensor( np_mesh_x, requires_grad=True, dtype=torch.float32).to(device)
    mesh_x_aug = torch.cat((mesh_x, u * torch.ones_like(mesh_x)[:,0].view(-1, 1)), dim=1)
    mesh_dx_aug = model(0, mesh_x_aug)
    mesh_dx = mesh_dx_aug[:, 0:2]
    return mesh_dx.detach().cpu().numpy()

# get their vector fields
R = 3.6
kwargs = {'xmin': -R, 'xmax': R, 'ymin': -R, 'ymax': R, 'gridsize': args.gridsize, 'u': u0}
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
    x, y, t = get_trajectory(timesteps=40, noise_std=0.0, y0=y0, radius=2.1, u=u0)
    N = len(x)
    point_colors = [(i/N, 0, 1-i/N) for i in range(N)]
    plt.scatter(x,y, s=14, label='data', c=point_colors)

    plt.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2))  

    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Data", pad=10)
    plt.xlim(-R, R)
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
    plt.xlim(-R, R)
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
    plt.xlim(-R, R)
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
    plt.xlim(-R, R)
    plt.ylim(-R, R)

# fig.savefig('{}/pend-single-force-p{}.{}'.format(args.fig_dir, args.num_points, FORMAT))
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

    # fig.savefig('{}/pend-single-learnt-fun-p{}.{}'.format(args.fig_dir, args.num_points, FORMAT))


#%%
# plot for the paper
fig = plt.figure(figsize=(9.6, 3.2), dpi=DPI)
plt.subplot(1, 3, 1)

x, y, t = get_trajectory(timesteps=200, noise_std=0.0, y0=y0, u=u0)

plt.plot(x,y, label='Ground Truth', color='k', linewidth=1)
plt.plot(base_ivp['y'][0,:],base_ivp['y'][1,:], 'y', label='Naive Baseline', linewidth=1.3)

plt.xlabel("$q$", fontsize=14)
plt.ylabel("$p$", rotation=0, fontsize=14)
plt.title("Trajectory Prediction", pad=10)
plt.xlim(-3, 3)
plt.ylim(-2.4, 3.6)
# plt.gca().set_aspect('equal', adjustable='box')
plt.legend(fontsize=10)

plt.subplot(1, 3, 2)

plt.plot(x,y, label='Ground Truth', color='k', linewidth=1)
plt.plot(hnn_ivp['y'][0,:],hnn_ivp['y'][1,:], 'g', label='Unstructured SymODEN', linewidth=1.3)

plt.xlabel("$q$", fontsize=14)
plt.ylabel("$p$", rotation=0, fontsize=14)
plt.title("Trajectory Prediction", pad=10)
plt.xlim(-3, 3)
plt.ylim(-2.4, 3.6)
# plt.gca().set_aspect('equal', adjustable='box')
plt.legend(fontsize=10)

plt.subplot(1, 3, 3)

plt.plot(x,y, label='Ground Truth', color='k', linewidth=1)
plt.plot(hnn_struct_ivp['y'][0,:],hnn_struct_ivp['y'][1,:], 'b', label='SymODEN', linewidth=1.3)

plt.xlabel("$q$", fontsize=14)
plt.ylabel("$p$", rotation=0, fontsize=14)
plt.title("Trajectory Prediction", pad=10)
plt.xlim(-3, 3)
plt.ylim(-2.4, 3.6)
# plt.gca().set_aspect('equal', adjustable='box')
plt.legend(fontsize=10)

plt.tight_layout()
fig.savefig('{}/fig-single-traj.{}'.format(args.fig_dir, FORMAT))

#%%
fig = plt.figure(figsize=(9.6, 2.5), dpi=DPI)

plt.subplot(1, 3, 1)

plt.plot(q, np.ones_like(q), label='Ground Truth', color='k', linewidth=2)
plt.plot(q, g_q.detach().cpu().numpy(), 'b--', linewidth=3, label=r'SymODEN $g_{\theta_3}(q)$')
plt.xlabel("$q$", fontsize=14)
# plt.ylabel("$g(q)$", rotation=0, fontsize=14)
plt.title("$g(q)$", pad=10, fontsize=14)
plt.xlim(-5, 5)
plt.ylim(0, 4)
plt.legend(fontsize=10)

M_q_inv = hnn_ode_struct_model.M_net(q_tensor)
plt.subplot(1, 3, 2)
plt.plot(q, 3 * np.ones_like(q), label='Ground Truth', color='k', linewidth=2)
plt.plot(q, M_q_inv.detach().cpu().numpy(), 'b--', linewidth=3, label=r'SymODEN $M^{-1}_{\theta_1}(q)$')
plt.xlabel("$q$", fontsize=14)
# plt.ylabel("$M^{-1}(q)$", rotation=0, fontsize=14)
plt.title("$M^{-1}(q)$", pad=10, fontsize=14)
plt.xlim(-5, 5)
plt.ylim(0, 4)
plt.legend(fontsize=10)

V_q = hnn_ode_struct_model.V_net(q_tensor)
plt.subplot(1, 3, 3)
plt.plot(q, 5.-5. * np.cos(q), label='Ground Truth', color='k', linewidth=2)
plt.plot(q, V_q.detach().cpu().numpy(), 'b--', linewidth=3, label=r'SymODEN $V_{\theta_2}(q)$')
plt.xlabel("$q$", fontsize=14)
# plt.ylabel("$V(q)$", rotation=0, fontsize=14)
plt.title("$V(q)$", pad=10, fontsize=14)
plt.xlim(-5, 5)
plt.ylim(-6, 21)
plt.legend(fontsize=10)
plt.tight_layout()
fig.savefig('{}/fig-single-pend.{}'.format(args.fig_dir, FORMAT))

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