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
from nn_models import MLP, PSD, DampMatrix, Decoder
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

#%% [markdown]
## Inspect the dataset
# We can either set initial condition to be the same or different for different forces
#%%
data = get_dataset(seed=args.seed, gym=True, 
        save_dir=args.save_dir, us=[-1.0, 0.0, 1.0], samples=50, timesteps=100)
print(data['x'].shape)

print(data['t'])
#%%
t = 0
cos_q_01 = data['x'][0,t,:,0] ; sin_q_01 = data['x'][0,t,:,1] ; p_01 = data['x'][0,t,:,2]
cos_q_02 = data['x'][1,t,:,0] ; sin_q_02 = data['x'][1,t,:,1] ; p_02 = data['x'][1,t,:,2]
cos_q_03 = data['x'][2,t,:,0] ; sin_q_03 = data['x'][2,t,:,1] ; p_03 = data['x'][2,t,:,2]

# i = 3
# cos_q_01 = data['x'][0,:,i,0] ; sin_q_01 = data['x'][0,:,i,1] ; p_01 = data['x'][0,:,i,2]
# cos_q_02 = data['x'][1,:,i,0] ; sin_q_02 = data['x'][1,:,i,1] ; p_02 = data['x'][1,:,i,2]
# cos_q_03 = data['x'][2,:,i,0] ; sin_q_03 = data['x'][2,:,i,1] ; p_03 = data['x'][2,:,i,2]


for _ in range(0):
    fig = plt.figure(figsize=[12,3], dpi=DPI)
    plt.subplot(1, 3, 1)
    plt.plot(np.arctan2(-sin_q_03, -cos_q_03))
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

for _ in range(1):
    fig = plt.figure(figsize=[12,3], dpi=DPI)
    plt.subplot(1, 3, 1)
    plt.scatter(np.arctan2(sin_q_01, cos_q_01), p_01)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.subplot(1, 3, 2)
    plt.scatter(np.arctan2(sin_q_02, cos_q_02), p_02)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.subplot(1, 3, 3)
    plt.scatter(np.arctan2(sin_q_03, cos_q_03), p_03)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
#%%
# train_p = data['x'][:,:,:,2].reshape(-1,1)
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
    reg_net = MLP(3, 100, 2).to(device)
    obs_net = Decoder(2, 100, 3).to(device)
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
base_ode_model, base_ode_reg_net, base_ode_obs_net = \
    get_model(args, baseline=True, structure=False, damping=False, num_points=args.num_points, gym=True)

#%%
# check the performance of the encoder

import gym

env: gym.wrappers.time_limit.TimeLimit = gym.make('Pendulum-v0')
env.seed(0)

obs_list = []
env.reset()
env.state = np.array([0.0, 0.0], dtype=np.float32)


total_t = 50
for _ in range(total_t):
    obs, _, _, _ = env.step([1.])
    obs_list.append(obs)

obs = np.stack(obs_list)
true_q = np.arctan2(-obs[:, 1], -obs[:, 0])
true_p = obs[:, 2] / 3

obs = torch.tensor(obs, dtype=torch.float32, device=device)

# plot the performance of the encoder alone
for _ in range(1):
    q_p_struct_hat = hnn_ode_struct_reg_net(obs).detach().cpu().numpy()
    q_struct_hat = q_p_struct_hat[:, 0]
    p_struct_hat = q_p_struct_hat[:, 1]

    q_p_hnn_hat = hnn_ode_reg_net(obs).detach().cpu().numpy()
    q_hnn_hat = q_p_hnn_hat[:, 0]
    p_hnn_hat = q_p_hnn_hat[:, 1]

    q_p_base_hat = base_ode_reg_net(obs).detach().cpu().numpy()
    q_base_hat = q_p_base_hat[:, 0]
    p_base_hat = q_p_base_hat[:, 1]


    plt.subplot(2, 3, 1)
    plt.plot(range(total_t), true_q, range(total_t), q_base_hat)
    plt.subplot(2, 3, 4)
    plt.plot(range(total_t), true_p, range(total_t), p_base_hat)

    plt.subplot(2, 3, 2)
    plt.plot(range(total_t), true_q, range(total_t), q_hnn_hat)
    plt.subplot(2, 3, 5)
    plt.plot(range(total_t), true_p, range(total_t), p_hnn_hat)

    plt.subplot(2, 3, 3)
    plt.plot(range(total_t), true_q, range(total_t), q_struct_hat)
    plt.subplot(2, 3, 6)
    plt.plot(range(total_t), true_p, range(total_t), p_struct_hat)

#%%


# plot the performance of the encoder and the decoder
for _ in range(1):
    obs_struct_hat = hnn_ode_struct_obs_net(hnn_ode_struct_reg_net(obs)).detach().cpu().numpy()
    obs_hnn_hat = hnn_ode_obs_net(hnn_ode_reg_net(obs)).detach().cpu().numpy()
    obs_base_hat = base_ode_obs_net(base_ode_reg_net(obs)).detach().cpu().numpy()

    q_base_hat = np.arctan2(-obs_base_hat[:, 1], -obs_base_hat[:, 0])
    p_base_hat = obs_base_hat[:, 2] / 3
    plt.subplot(2, 3, 1)
    plt.plot(range(total_t), true_q, range(total_t), q_base_hat)
    plt.subplot(2, 3, 4)
    plt.plot(range(total_t), true_p, range(total_t), p_base_hat)

    q_hnn_hat = np.arctan2(-obs_hnn_hat[:, 1], -obs_hnn_hat[:, 0])
    p_hnn_hat = obs_hnn_hat[:, 2] / 3
    plt.subplot(2, 3, 2)
    plt.plot(range(total_t), true_q, range(total_t), q_hnn_hat)
    plt.subplot(2, 3, 5)
    plt.plot(range(total_t), true_p, range(total_t), p_hnn_hat)

    q_struct_hat = np.arctan2(-obs_struct_hat[:, 1], -obs_struct_hat[:, 0])
    p_struct_hat = obs_struct_hat[:, 2] / 3
    plt.subplot(2, 3, 3)
    plt.plot(range(total_t), true_q, range(total_t), q_struct_hat)
    plt.subplot(2, 3, 6)
    plt.plot(range(total_t), true_p, range(total_t), p_struct_hat)

#%%
# check the result of the vector field
# get vector filed of different model
def get_vector_field(model, u=0, **kwargs):
    field = get_field(**kwargs)
    np_mesh_x = field['x']
    
    # run model
    mesh_x = torch.tensor( np_mesh_x, requires_grad=True, dtype=torch.float32).to(device)
    mesh_x_aug = torch.cat((mesh_x, u * torch.ones_like(mesh_x)[:,0].view(-1, 1)), dim=1)
    mesh_dx_aug = model.time_derivative(0, mesh_x_aug)
    mesh_dx = mesh_dx_aug[:, 0:2]
    return mesh_dx.detach().cpu().numpy()

# get their vector fields
R = 3.6
kwargs = {'xmin': -R, 'xmax': R, 'ymin': -R, 'ymax': R, 'gridsize': args.gridsize, 'u': -2.0}
field = get_field(**kwargs)
hnn_struct_field = get_vector_field(hnn_ode_struct_model, **kwargs)
hnn_field = get_vector_field(hnn_ode_model, **kwargs)
base_field = get_vector_field(base_ode_model, **kwargs)

#%%
####### PLOT VECTOR FIELD ########
for _ in range(1):
    fig = plt.figure(figsize=(16, 3.2), dpi=DPI)

    plt.subplot(1, 4, 1)
    plt.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2))  
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Data", pad=10)
    plt.xlim(-R, R)
    plt.ylim(-R, R)

    plt.subplot(1, 4, 2)
    plt.quiver(field['x'][:,0], field['x'][:,1], base_field[:,0], base_field[:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.5,.5,.5))
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Hamiltonian ODE NN ({})".format(args.num_points), pad=10)
    plt.xlim(-R, R)
    plt.ylim(-R, R)

    plt.subplot(1, 4, 3)
    plt.quiver(field['x'][:,0], field['x'][:,1], hnn_field[:,0], hnn_field[:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.5,.5,.5))
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Hamiltonian ODE NN ({})".format(args.num_points), pad=10)
    plt.xlim(-R, R)
    plt.ylim(-R, R)

    plt.subplot(1, 4, 4)
    plt.quiver(field['x'][:,0], field['x'][:,1], hnn_struct_field[:,0], hnn_struct_field[:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.5,.5,.5))
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Hamiltonian structured ODE NN ({})".format(args.num_points), pad=10)
    plt.xlim(-R, R)
    plt.ylim(-R, R)

#%%
########### check the physical meaning of M and V #############

q = np.linspace(-5.0, 5.0, 40)
q_tensor = torch.tensor(q, dtype=torch.float32).view(40, 1).to(device)

for _ in range(1):
    fig = plt.figure(figsize=(20, 3.2), dpi=DPI)
    # g_q = hnn_ode_model.g_net(q_tensor)
    plt.subplot(1, 4, 1)
    # plt.plot(q, g_q.detach().cpu().numpy())
    # plt.xlabel("$q$", fontsize=14)
    # plt.ylabel("$g_q$", rotation=0, fontsize=14)
    # plt.title("g_q - Hamiltonian ODE NN ({})".format(args.num_points), pad=10, fontsize=14)


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
