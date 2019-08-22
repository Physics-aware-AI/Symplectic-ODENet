#%% 
import torch, time, sys
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

EXPERIMENT_DIR = './experiment-single-embed/'
sys.path.append(EXPERIMENT_DIR)

from data import get_dataset, get_trajectory, dynamics_fn, hamiltonian_fn, arrange_data, get_field
from nn_models import MLP, PSD, DampMatrix
from hnn import HNN, HNN_structure, HNN_structure_embed
from utils import L2_loss, from_pickle

#%%
DPI = 300
FORMAT = 'png'
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2

def get_args():
    return {'num_angle': 1,
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
         'num_points': 4,
         'gpu': 3,
         'solver': 'rk4'}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())

#%%
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
def get_model(args, baseline, structure, naive, damping, num_points):
    M_net = PSD(2*args.num_angle, 300, args.num_angle).to(device)
    g_net = MLP(2*args.num_angle, 200, args.num_angle).to(device)
    if structure == False:
        if naive and baseline:
            raise RuntimeError('argument *baseline* and *naive* cannot both be true')
        elif naive:
            input_dim = 4 * args.num_angle
            output_dim = 3 * args.num_angle
            nn_model = MLP(input_dim, 800, output_dim, args.nonlinearity).to(device)
            model = HNN_structure_embed(args.num_angle, H_net=nn_model, device=device, baseline=baseline, naive=naive)
        elif baseline:
            input_dim = 4 * args.num_angle
            output_dim = 2 * args.num_angle
            nn_model = MLP(input_dim, 600, output_dim, args.nonlinearity).to(device)
            model = HNN_structure_embed(args.num_angle, H_net=nn_model, M_net=M_net, device=device, baseline=baseline, naive=naive)
        else:
            input_dim = 3 * args.num_angle
            output_dim = 1
            nn_model = MLP(input_dim, 500, output_dim, args.nonlinearity).to(device)
            model = HNN_structure_embed(args.num_angle, H_net=nn_model, M_net=M_net, g_net=g_net, device=device, baseline=baseline, naive=naive)
    elif structure == True and baseline ==False and naive==False:
        V_net = MLP(2*args.num_angle, 50, 1).to(device)
        model = HNN_structure_embed(args.num_angle, M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=baseline, structure=True).to(device)
    else:
        raise RuntimeError('argument *structure* is set to true, no *baseline* or *naive*!')

    if naive:
        label = '-naive_ode'
    elif baseline:
        label = '-baseline_ode'
    else:
        label = '-hnn_ode'
    struct = '-struct' if structure else ''
    path = '{}/{}{}{}-{}-p{}.tar'.format(args.save_dir, args.name, label, struct, args.solver, args.num_points)
    model.load_state_dict(torch.load(path, map_location=device))
    path = '{}/{}{}{}-{}-p{}-stats.pkl'.format(args.save_dir, args.name, label, struct, args.solver, args.num_points)
    stats = from_pickle(path)
    return model, stats

# naive_ode_model, naive_ode_stats = get_model(args, baseline=False, structure=False, naive=True, damping=False, num_points=args.num_points)
base_ode_model, base_ode_stats = get_model(args, baseline=True, structure=False, naive=False, damping=False, num_points=args.num_points)
hnn_ode_model, hnn_ode_stats = get_model(args, baseline=False, structure=False, naive=False, damping=False, num_points=args.num_points)
hnn_ode_struct_model, hnn_ode_struct_stats = get_model(args, baseline=False, structure=True, naive=False, damping=False, num_points=args.num_points)

#%%
def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total

# get final traning loss
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

#%% [markdown]
# ## Integrate along vector fields

#%%
# from torchdiffeq import odeint_adjoint as ode_int 
from torchdiffeq import odeint
def integrate_model(model, t_span, y0, **kwargs):
    
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,4).to(device)
        dx = model(0, x).detach().cpu().numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

# time info for simualtion
time_step = 100 ; n_eval = 100
t_span = [0,time_step*0.05]
t_linspace_true = np.linspace(t_span[0], time_step, time_step)*0.05
t_linspace_model = np.linspace(t_span[0], t_span[1], n_eval)
# angle info for simuation
init_angle = 0.5
y0 = np.asarray([init_angle, 0])
u0 = 0.0
y0_u = np.asarray([np.cos(init_angle), np.sin(init_angle), 0, u0])
# simulate
kwargs = {'t_eval': t_linspace_model, 'rtol': 1e-12, 'method': 'RK45'}
# naive_ivp = integrate_model(naive_ode_model, t_span, y0_u, **kwargs)
base_ivp = integrate_model(base_ode_model, t_span, y0_u, **kwargs)
hnn_ivp = integrate_model(hnn_ode_model, t_span, y0_u, **kwargs)
hnn_struct_ivp = integrate_model(hnn_ode_struct_model, t_span, y0_u, **kwargs)

import gym 
import myenv
env = gym.make('MyPendulum-v0')
env.reset()
env.state = np.array([init_angle, 0.0], dtype=np.float32)
obs = env._get_obs()
obs_list = []
for _ in range(time_step):
    obs_list.append(obs)
    obs, _, _, _ = env.step([u0])
    
true_ivp = np.stack(obs_list, 1)
true_ivp = np.concatenate((true_ivp, np.zeros((1, time_step))), axis=0)

def get_qp(x):
    q = np.arctan2(-x[:, 1], -x[:, 0])
    p = x[:, 2] /3
    return np.stack((q, p), axis=1)

true_qp = get_qp(true_ivp.T)
# naive_qp = get_qp(naive_ivp.y.T)
base_qp = get_qp(base_ivp.y.T)
hnn_qp = get_qp(hnn_ivp.y.T)
hnn_struct_qp = get_qp(hnn_struct_ivp.y.T)

#%%
# comparing true trajectory and the estimated trajectory

plt.plot(t_linspace_model, hnn_struct_ivp.y[1,:], 'r')
plt.plot(t_linspace_true, true_ivp[1,:], 'g')

# %%
# sanity check of traj
# naive_1 = naive_ivp.y[0,:]**2 + naive_ivp.y[1,:]**2
base_1 = base_ivp.y[0,:]**2 + base_ivp.y[1,:]**2
hnn_1 = hnn_ivp.y[0,:]**2 + hnn_ivp.y[1,:]**2
hnn_struct_1 = hnn_struct_ivp.y[0,:]**2 + hnn_struct_ivp.y[1,:]**2

# plt.plot(t_linspace_model, naive_1)
plt.plot(t_linspace_model, base_1, 'r')
plt.plot(t_linspace_model, hnn_1, 'b')
plt.plot(t_linspace_model, hnn_struct_1, 'g')

#%%
#%%
# get vector field of different models
def get_vector_field(model, u=0, **kwargs):
    field = get_field(**kwargs)
    np_mesh_x = field['x']
    
    # run model
    mesh_x = torch.tensor( np_mesh_x, requires_grad=True, dtype=torch.float32).to(device)
    # transform q, p into cosq, sinq, q_dot and add u, be careful about the angle convention
    input_x = torch.stack((-torch.cos(mesh_x[:, 0]), -torch.sin(mesh_x[:, 0]), mesh_x[:, 1]*3 , u * torch.ones_like(mesh_x)[:,0]), dim=1)
    vector_field = model(0, input_x)
    mesh_dx = model.get_dqdp(input_x)
    # match the vector field with the behavior of the estimated trajectory
    mesh_dx[:, 1] = vector_field[:, 2] /3
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
####### PLOT VECTOR FIELD ########
for _ in range(1):
    fig = plt.figure(figsize=(16, 3.2), dpi=DPI)

    plt.subplot(1, 4, 1)
    for i, l in enumerate(np.split(true_qp, LINE_SEGMENTS)):
        color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
        plt.plot(l[:,0],l[:,1],color=color, linewidth=LINE_WIDTH)
    plt.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2))  
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Data", pad=10)
    plt.xlim(-R, R)
    plt.ylim(-R, R)

    plt.subplot(1, 4, 2)
    for i, l in enumerate(np.split(base_qp, LINE_SEGMENTS)):
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
    for i, l in enumerate(np.split(hnn_qp, LINE_SEGMENTS)):
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
    for i, l in enumerate(np.split(hnn_struct_qp, LINE_SEGMENTS)):
        color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
        plt.plot(l[:,0],l[:,1],color=color, linewidth=LINE_WIDTH)
    plt.quiver(field['x'][:,0], field['x'][:,1], hnn_struct_field[:,0], hnn_struct_field[:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.5,.5,.5))
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Hamiltonian structured ODE NN ({})".format(args.num_points), pad=10)
    plt.xlim(-R, R)
    plt.ylim(-R, R)

    # plt.tight_layout() ; plt.show()
    # fig.savefig('{}/pend-single-embed-p{}.{}'.format(args.fig_dir, args.num_points, FORMAT))


#%%
# plot learnt function
q = np.linspace(-5.0, 5.0, 40)
q_tensor = torch.tensor(q, dtype=torch.float32).view(40, 1).to(device)
cos_q_sin_q = torch.cat((-torch.cos(q_tensor), -torch.sin(q_tensor)), dim=1)

for _ in range(1):
    fig = plt.figure(figsize=(20, 6.4), dpi=DPI)
    g_q = hnn_ode_model.g_net(cos_q_sin_q)
    # plt.subplot(2, 4, 1)
    # plt.plot(q, g_q.detach().cpu().numpy())
    # plt.xlabel("$q$", fontsize=14)
    # plt.ylabel("$g_q$", rotation=0, fontsize=14)
    # plt.title("g_q - Hamiltonian ODE NN ({})".format(args.num_points), pad=10, fontsize=14)


    g_q = hnn_ode_struct_model.g_net(cos_q_sin_q)
    plt.subplot(2, 4, 2)
    plt.plot(q, g_q.detach().cpu().numpy())
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$g_q$", rotation=0, fontsize=14)
    plt.title("g_q - Hamiltonian structured ODE NN ({})".format(args.num_points), pad=10, fontsize=14)

    M_q_inv = hnn_ode_struct_model.M_net(cos_q_sin_q)
    plt.subplot(2, 4, 3)
    plt.plot(q, M_q_inv.detach().cpu().numpy())
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$Mq_inv$", rotation=0, fontsize=14)
    plt.title("Mq_inv - Hamiltonian structured ODE NN ({})".format(args.num_points), pad=10, fontsize=14)

    V_q = hnn_ode_struct_model.V_net(cos_q_sin_q)
    plt.subplot(2, 4, 4)
    plt.plot(q, V_q.detach().cpu().numpy())
    plt.plot(q, -5. * np.cos(q))
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$V_q$", rotation=0, fontsize=14)
    plt.title("V_q - Hamiltonian structured ODE NN ({})".format(args.num_points), pad=10, fontsize=14)

    M_q_inv = hnn_ode_model.M_net(cos_q_sin_q)
    # plt.subplot(2, 4, 5)
    # plt.plot(q, M_q_inv.detach().cpu().numpy())
    # plt.xlabel("$q$", fontsize=14)
    # plt.ylabel("$Mq_inv$", rotation=0, fontsize=14)
    # plt.title("Mq_inv - Hamiltonian ODE NN ({})".format(args.num_points), pad=10, fontsize=14)

    M_q_inv = base_ode_model.M_net(cos_q_sin_q)
    # plt.subplot(2, 4, 6)
    # plt.plot(q, M_q_inv.detach().cpu().numpy())
    # plt.xlabel("$q$", fontsize=14)
    # plt.ylabel("$Mq_inv$", rotation=0, fontsize=14)
    # plt.title("Mq_inv - Baseline ODE NN ({})".format(args.num_points), pad=10, fontsize=14)
    
    # plt.tight_layout() ; plt.show()
    # fig.savefig('{}/pend-single-embed-learnt-p{}.{}'.format(args.fig_dir, args.num_points, FORMAT))





#%%
# comapring the estimated p and the true p
hnn_struct_traj = torch.tensor(hnn_struct_ivp.y.T, dtype=torch.float32).to(device)
hnn_traj = torch.tensor(hnn_ivp.y.T, dtype=torch.float32).to(device)
base_traj = torch.tensor(base_ivp.y.T, dtype=torch.float32).to(device)
true_traj = torch.tensor(true_ivp.T, dtype=torch.float32).to(device)
cos_q_sin_q, q_dot, u = torch.split(hnn_struct_traj, [2, 1, 1], dim=1)
M_q_inv = hnn_ode_struct_model.M_net(cos_q_sin_q)
hnn_struct_p = q_dot / M_q_inv
true_p = true_ivp.T[:, 2] / 3
plt.plot(t_linspace_model , hnn_struct_p.detach().cpu().numpy())
plt.plot(t_linspace_true, true_p)

#%%
# comparing the energy

E_true = true_ivp.T[:, 2]**2 / 6 + 5 * (1 + true_ivp.T[:, 0])
E_base = base_ivp.y.T[:, 2]**2 / 6 + 5 * (1 + base_ivp.y.T[:, 0])
E_hnn = hnn_ivp.y.T[:, 2]**2 / 6 + 5 * (1 + hnn_ivp.y.T[:, 0])
E_hnn_struct = hnn_struct_ivp.y.T[:, 2]**2 / 6 + 5 * (1 + hnn_struct_ivp.y.T[:, 0])

plt.plot(t_linspace_true, E_true, 'r')
plt.plot(t_linspace_model, E_base, 'b')
plt.plot(t_linspace_model, E_hnn, 'g')
plt.plot(t_linspace_model, E_hnn_struct, 'y')
# the Open AI gym data is generate by explicit Euler method, 
# thus strictly speaking the data are not energy conserved.
# this plot must add naive baseline for comparison.
#%%
# check conserved quantity of hnn and hnn_struct
struct_H_hnn_struct, _ = hnn_ode_struct_model.get_H(hnn_struct_traj)
struct_H_base, _ = hnn_ode_struct_model.get_H(base_traj)
struct_H_hnn, _ = hnn_ode_struct_model.get_H(hnn_traj)
struct_H_true, _ = hnn_ode_struct_model.get_H(true_traj)
plt.plot(t_linspace_model, struct_H_hnn_struct.detach().cpu().numpy())
plt.plot(t_linspace_model, struct_H_hnn.detach().cpu().numpy())
plt.plot(t_linspace_model, struct_H_base.detach().cpu().numpy())
plt.plot(t_linspace_true, struct_H_true.detach().cpu().numpy())
#%%
H_hnn_struct, _ = hnn_ode_model.get_H(hnn_struct_traj)
H_base, _ = hnn_ode_model.get_H(base_traj)
H_hnn, _ = hnn_ode_model.get_H(hnn_traj)
H_true, _ = hnn_ode_model.get_H(true_traj)
plt.plot(t_linspace_model, H_hnn_struct.detach().cpu().numpy())
plt.plot(t_linspace_model, H_hnn.detach().cpu().numpy())
plt.plot(t_linspace_model, H_base.detach().cpu().numpy())
plt.plot(t_linspace_true, H_true.detach().cpu().numpy())



#%%
# vanilla control
# time info for simualtion
time_step = 100 ; n_eval = 100
t_span = [0,time_step*0.05]
t_eval = torch.linspace(t_span[0], t_span[1], n_eval)
# t_linspace_true = np.linspace(t_span[0], time_step, time_step)*0.05
# t_linspace_model = np.linspace(t_span[0], t_span[1], n_eval)
# angle info for simuation
init_angle = 3.14
u0 = 0.0

# generate initial condition from gym
import gym 
import myenv
env = gym.make('MyPendulum-v0')
env.reset()
env.state = np.array([init_angle, u0], dtype=np.float32)
obs = env._get_obs()
y = torch.tensor([obs[0], obs[1], obs[2], u0], requires_grad=True, device=device, dtype=torch.float32).view(1, 4)
# manually generate initial condition
# y0 = torch.tensor([np.cos(init_angle), np.sin(init_angle), 0, u0], requires_grad=True, device=device, dtype=torch.float32).view(1, 4)
# rtol = 1e-12
# y = y0

k_p = 1 ; k_d = 3
y_traj = []
y_traj.append(y)
for i in range(len(t_eval)-1):
    cos_q_sin_q, q_dot, _ = torch.split(y, [2, 1, 1], dim=1)
    cos_q, sin_q = torch.chunk(cos_q_sin_q, 2, dim=1)
    V_q = hnn_ode_struct_model.V_net(cos_q_sin_q)
    dV = torch.autograd.grad(V_q, cos_q_sin_q)[0]
    dVdcos_q, dVdsin_q= torch.chunk(dV, 2, dim=1)
    dV_q = - dVdcos_q * sin_q + dVdsin_q * cos_q
    M_inv = hnn_ode_struct_model.M_net(cos_q_sin_q)
    q = torch.atan2(sin_q, cos_q)

    # u = (dV_q - k_p * (cos_q - 1) - k_p * (sin_q) - k_d * q_dot)
    # u = M_inv * (dV_q - k_p * q - k_d * q_dot)
    u = (2*dV_q  - k_d * q_dot)

    # use openai simulator
    u = u.detach().cpu().numpy()
    obs, _, _, _ = env.step(u)
    y = torch.tensor([obs[0], obs[1], obs[2], u], requires_grad=True, device=device, dtype=torch.float32).view(1, 4)
    # use learnt model
    # y0_u = torch.cat((cos_q_sin_q, q_dot, u), dim = 1)
    # y_step = odeint(hnn_ode_struct_model, y0_u, t_eval[i:i+2], method='rk4')
    # y = y_step[-1,:,:]

    y_traj.append(y)

y_traj = torch.stack(y_traj).view(-1, 4).detach().cpu().numpy()




#%%
# plot control result
fig = plt.figure(figsize=[10, 10], dpi=DPI)
plt.subplot(4, 1, 1)
plt.plot(t_eval.numpy(), y_traj[:, 0])
plt.ylabel('$cos(q)$', fontsize=14)

plt.subplot(4, 1, 2)
plt.plot(t_eval.numpy(), y_traj[:, 1])
plt.ylabel('$sin(q)$', fontsize=14)

plt.subplot(4, 1, 3)
plt.plot(t_eval.numpy(), y_traj[:, 2])
plt.ylabel('$\dot{q}$', fontsize=14)

plt.subplot(4, 1, 4)
plt.plot(t_eval.numpy(), y_traj[:, 2])
plt.ylabel('$u$', fontsize=14)

plt.tight_layout() ; plt.show()
fig.savefig('{}/pend-single-embed-ctrl-p{}.{}'.format(args.fig_dir, args.num_points, FORMAT))


#%%
plt.plot(y_traj[:, 0], y_traj[:, 1])

#%%
