# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# code structure follows the style of HNN by Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

# This file is a script version of 'analyze-effect-tau.ipynb'
# Cells are seperated by the vscode convention '#%%'

#%% 
import torch, time, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

EXPERIMENT_DIR = './experiment-single-embed/'
sys.path.append(EXPERIMENT_DIR)

from data import get_dataset, arrange_data, get_field
from nn_models import MLP, PSD
from symoden import SymODEN_T
from utils import L2_loss, from_pickle
import imageio

#%%
DPI = 600
FORMAT = 'png'

def get_args():
    return {'num_angle': 1,
         'nonlinearity': 'tanh',
         'name': 'pend',
         'seed': 0,
         'save_dir': './{}'.format(EXPERIMENT_DIR),
         'fig_dir': './figures',
         'num_points': 5,
         'gpu': 0,
         'solver': 'dopri5'}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())

#%%
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
def get_model(args, baseline, structure, naive, num_points, solver):
    M_net = PSD(2*args.num_angle, 300, args.num_angle).to(device)
    g_net = MLP(2*args.num_angle, 200, args.num_angle).to(device)
    if structure == False:
        if naive and baseline:
            raise RuntimeError('argument *baseline* and *naive* cannot both be true')
        elif naive:
            input_dim = 4 * args.num_angle
            output_dim = 3 * args.num_angle
            nn_model = MLP(input_dim, 800, output_dim, args.nonlinearity).to(device)
            model = SymODEN_T(args.num_angle, H_net=nn_model, device=device, baseline=baseline, naive=naive)
        elif baseline:
            input_dim = 4 * args.num_angle
            output_dim = 2 * args.num_angle
            nn_model = MLP(input_dim, 600, output_dim, args.nonlinearity).to(device)
            model = SymODEN_T(args.num_angle, H_net=nn_model, M_net=M_net, device=device, baseline=baseline, naive=naive)
        else:
            input_dim = 3 * args.num_angle
            output_dim = 1
            nn_model = MLP(input_dim, 500, output_dim, args.nonlinearity).to(device)
            model = SymODEN_T(args.num_angle, H_net=nn_model, M_net=M_net, g_net=g_net, device=device, baseline=baseline, naive=naive)
    elif structure == True and baseline ==False and naive==False:
        V_net = MLP(2*args.num_angle, 50, 1).to(device)
        model = SymODEN_T(args.num_angle, M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=baseline, structure=True).to(device)
    else:
        raise RuntimeError('argument *structure* is set to true, no *baseline* or *naive*!')

    if naive:
        label = '-naive_ode'
    elif baseline:
        label = '-baseline_ode'
    else:
        label = '-hnn_ode'
    struct = '-struct' if structure else ''
    path = '{}/{}{}{}-{}-p{}.tar'.format(args.save_dir, args.name, label, struct, solver, num_points)
    model.load_state_dict(torch.load(path, map_location=device))
    path = '{}/{}{}{}-{}-p{}-stats.pkl'.format(args.save_dir, args.name, label, struct, solver, num_points)
    stats = from_pickle(path)
    return model, stats

model_rk4_p2, stats_rk4_p2 = get_model(args, baseline=False, structure=True, naive=False, num_points=2, solver='rk4')
model_rk4_p3, stats_rk4_p3 = get_model(args, baseline=False, structure=True, naive=False, num_points=3, solver='rk4')
model_rk4_p4, stats_rk4_p4 = get_model(args, baseline=False, structure=True, naive=False, num_points=4, solver='rk4')
model_rk4_p5, stats_rk4_p5 = get_model(args, baseline=False, structure=True, naive=False, num_points=5, solver='rk4')
model_rk4_p6, stats_rk4_p6 = get_model(args, baseline=False, structure=True, naive=False, num_points=6, solver='rk4')
model_dopri5_p2, stats_dopri5_p2 = get_model(args, baseline=False, structure=True, naive=False, num_points=2, solver='dopri5')
model_dopri5_p3, stats_dopri5_p3 = get_model(args, baseline=False, structure=True, naive=False, num_points=3, solver='dopri5')
model_dopri5_p4, stats_dopri5_p4 = get_model(args, baseline=False, structure=True, naive=False, num_points=4, solver='dopri5')
model_dopri5_p5, stats_dopri5_p5 = get_model(args, baseline=False, structure=True, naive=False, num_points=5, solver='dopri5')
model_dopri5_p6, stats_dopri5_p6 = get_model(args, baseline=False, structure=True, naive=False, num_points=6, solver='dopri5')

print('stats_rk4_p2')
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(stats_rk4_p2['traj_train_loss']), np.std(stats_rk4_p2['traj_train_loss']),
        np.mean(stats_rk4_p2['traj_test_loss']), np.std(stats_rk4_p2['traj_test_loss'])))
print('stats_rk4_p3')
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(stats_rk4_p3['traj_train_loss']), np.std(stats_rk4_p3['traj_train_loss']),
        np.mean(stats_rk4_p3['traj_test_loss']), np.std(stats_rk4_p3['traj_test_loss'])))
print('stats_rk4_p4')
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(stats_rk4_p4['traj_train_loss']), np.std(stats_rk4_p4['traj_train_loss']),
        np.mean(stats_rk4_p4['traj_test_loss']), np.std(stats_rk4_p4['traj_test_loss'])))
print('stats_rk4_p5')
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(stats_rk4_p5['traj_train_loss']), np.std(stats_rk4_p5['traj_train_loss']),
        np.mean(stats_rk4_p5['traj_test_loss']), np.std(stats_rk4_p5['traj_test_loss'])))
print('stats_rk4_p6')
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(stats_rk4_p6['traj_train_loss']), np.std(stats_rk4_p6['traj_train_loss']),
        np.mean(stats_rk4_p6['traj_test_loss']), np.std(stats_rk4_p6['traj_test_loss'])))
print('')
print('stats_dopri5_p2')
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(stats_dopri5_p2['traj_train_loss']), np.std(stats_dopri5_p2['traj_train_loss']),
        np.mean(stats_dopri5_p2['traj_test_loss']), np.std(stats_dopri5_p2['traj_test_loss'])))
print('stats_dopri5_p3')
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(stats_dopri5_p3['traj_train_loss']), np.std(stats_dopri5_p3['traj_train_loss']),
        np.mean(stats_dopri5_p3['traj_test_loss']), np.std(stats_dopri5_p3['traj_test_loss'])))
print('stats_dopri5_p4')
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(stats_dopri5_p4['traj_train_loss']), np.std(stats_dopri5_p4['traj_train_loss']),
        np.mean(stats_dopri5_p4['traj_test_loss']), np.std(stats_dopri5_p4['traj_test_loss'])))
print('stats_dopri5_p5')
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(stats_dopri5_p5['traj_train_loss']), np.std(stats_dopri5_p5['traj_train_loss']),
        np.mean(stats_dopri5_p5['traj_test_loss']), np.std(stats_dopri5_p5['traj_test_loss'])))
print('stats_dopri5_p6')
print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
.format(np.mean(stats_dopri5_p6['traj_train_loss']), np.std(stats_dopri5_p6['traj_train_loss']),
        np.mean(stats_dopri5_p6['traj_test_loss']), np.std(stats_dopri5_p6['traj_test_loss'])))

#%%
#%%
us = [0.0]
data = get_dataset(seed=args.seed, timesteps=40,
            save_dir=args.save_dir, us=us, samples=128) 

pred_x, pred_t_eval = data['x'], data['t']

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


rk4_p2_pred_loss = get_pred_loss(pred_x, pred_t_eval, model_rk4_p2)
rk4_p3_pred_loss = get_pred_loss(pred_x, pred_t_eval, model_rk4_p3)
rk4_p4_pred_loss = get_pred_loss(pred_x, pred_t_eval, model_rk4_p4)
rk4_p5_pred_loss = get_pred_loss(pred_x, pred_t_eval, model_rk4_p5)
rk4_p6_pred_loss = get_pred_loss(pred_x, pred_t_eval, model_rk4_p6)
dopri5_p2_pred_loss = get_pred_loss(pred_x, pred_t_eval, model_dopri5_p2)
dopri5_p3_pred_loss = get_pred_loss(pred_x, pred_t_eval, model_dopri5_p3)
dopri5_p4_pred_loss = get_pred_loss(pred_x, pred_t_eval, model_dopri5_p4)
dopri5_p5_pred_loss = get_pred_loss(pred_x, pred_t_eval, model_dopri5_p5)
dopri5_p6_pred_loss = get_pred_loss(pred_x, pred_t_eval, model_dopri5_p6)

#%%
print('stats_rk4_p2')
print('Prediction loss {:.4e}'.format(np.mean(rk4_p2_pred_loss)))
print('stats_rk4_p3')
print('Prediction loss {:.4e}'.format(np.mean(rk4_p3_pred_loss)))
print('stats_rk4_p4')
print('Prediction loss {:.4e}'.format(np.mean(rk4_p4_pred_loss)))
print('stats_rk4_p5')
print('Prediction loss {:.4e}'.format(np.mean(rk4_p5_pred_loss)))
print('stats_rk4_p6')
print('Prediction loss {:.4e}'.format(np.mean(rk4_p6_pred_loss)))
print('stats_dopri5_p2')
print('Prediction loss {:.4e}'.format(np.mean(dopri5_p2_pred_loss)))
print('stats_dopri5_p3')
print('Prediction loss {:.4e}'.format(np.mean(dopri5_p3_pred_loss)))
print('stats_dopri5_p4')
print('Prediction loss {:.4e}'.format(np.mean(dopri5_p4_pred_loss)))
print('stats_dopri5_p5')
print('Prediction loss {:.4e}'.format(np.mean(dopri5_p5_pred_loss)))
print('stats_dopri5_p6')
print('Prediction loss {:.4e}'.format(np.mean(dopri5_p6_pred_loss)))
# %%
