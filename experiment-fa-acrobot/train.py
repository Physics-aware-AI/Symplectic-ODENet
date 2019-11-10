# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# code structure follows the style of HNN by Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP, PSD, MatrixNet
from symoden import SymODEN_T
from data import get_dataset, arrange_data
from utils import L2_loss, to_pickle

import time

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_angle', default=2, type=int, help='number of generalized coordinates')
    parser.add_argument('--learn_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=1000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='acrobot', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=2, help='number of evaluation points by the ODE solver, including the initial point')
    parser.add_argument('--structure', dest='structure', action='store_true', help='using a structured Hamiltonian')
    parser.add_argument('--naive', dest='naive', action='store_true', help='use a naive baseline')
    parser.add_argument('--solver', default='rk4', type=str, help='type of ODE Solver for Neural ODE')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def train(args):
    # import ODENet
    from torchdiffeq import odeint_adjoint as odeint
    # from torchdiffeq import odeint

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # reproducibility: set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init model and optimizer
    if args.verbose:
        print("Start training with num of points = {} and solver {}.".format(args.num_points, args.solver))

    M_net = PSD(2*args.num_angle, 400, args.num_angle).to(device)
    g_net = MatrixNet(2*args.num_angle, 300, 4, shape=(2,2)).to(device)
    if args.structure == False:
        if args.naive and args.baseline:
            raise RuntimeError('argument *baseline* and *naive* cannot both be true')
        elif args.naive:
            input_dim = 3 * args.num_angle + 1
            output_dim = 3 * args.num_angle
            nn_model = MLP(input_dim, 1200, output_dim, args.nonlinearity).to(device)
            model = SymODEN_T(args.num_angle, H_net=nn_model, device=device, baseline=args.baseline, naive=args.naive, u_dim=2)
        elif args.baseline:
            input_dim = 3 * args.num_angle + 1
            output_dim = 2 * args.num_angle
            nn_model = MLP(input_dim, 800, output_dim, args.nonlinearity).to(device)
            model = SymODEN_T(args.num_angle, H_net=nn_model, M_net=M_net, device=device, baseline=args.baseline, naive=args.naive, u_dim=2)
        else:
            input_dim = 3 * args.num_angle
            output_dim = 1
            nn_model = MLP(input_dim, 600, output_dim, args.nonlinearity).to(device)
            model = SymODEN_T(args.num_angle, H_net=nn_model, M_net=M_net, g_net=g_net, device=device, baseline=args.baseline, naive=args.naive, u_dim=2)
    elif args.structure == True and args.baseline ==False and args.naive==False:
        V_net = MLP(2*args.num_angle, 300, 1).to(device)
        model = SymODEN_T(args.num_angle, M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=args.baseline, structure=True, u_dim=2).to(device)
    else:
        raise RuntimeError('argument *structure* is set to true, no *baseline* or *naive*!')

    num_parm = get_model_parm_nums(model)
    print('model contains {} parameters'.format(num_parm))

    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=5e-5)

    # arrange data
    us = [[0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 2.0], [0.0, -2.0],
            [1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]
    data = get_dataset(seed=args.seed, timesteps=20,
                save_dir=args.save_dir, us=us, samples=128)
    train_x, t_eval = arrange_data(data['x'], data['t'], num_points=args.num_points)
    test_x, t_eval = arrange_data(data['test_x'], data['t'], num_points=args.num_points)

    train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)

    # training loop
    stats = {'train_loss': [], 'test_loss': [], 'forward_time': [], 'backward_time': [],'nfe': []}
    for step in range(args.total_steps+1):
        train_loss = 0
        test_loss = 0
        for i in range(train_x.shape[0]):
            t = time.time()
            train_x_hat = odeint(model, train_x[i, 0, :, :], t_eval, method=args.solver) 
            forward_time = time.time() - t
            train_loss_mini = L2_loss(train_x[i,:,:,:], train_x_hat)
            train_loss = train_loss + train_loss_mini 

            t = time.time()
            train_loss_mini.backward()
            optim.step() ; optim.zero_grad()
            backward_time = time.time() - t

            # run test data
            test_x_hat = odeint(model, test_x[i, 0, :, :], t_eval, method=args.solver)
            test_loss_mini = L2_loss(test_x[i,:,:,:], test_x_hat)
            test_loss = test_loss + test_loss_mini

        # logging
        stats['train_loss'].append(train_loss.item())
        stats['test_loss'].append(test_loss.item())
        stats['forward_time'].append(forward_time)
        stats['backward_time'].append(backward_time)
        stats['nfe'].append(model.nfe)
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, train_loss.item(), test_loss.item()))

    # calculate loss mean and std for each traj.
    train_x, t_eval = data['x'], data['t']
    test_x, t_eval = data['test_x'], data['t']

    train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)

    train_loss = []
    test_loss = []
    for i in range(train_x.shape[0]):
        train_x_hat = odeint(model, train_x[i, 0, :, :], t_eval, method=args.solver)            
        train_loss.append((train_x[i,:,:,:] - train_x_hat)**2)

        # run test data
        test_x_hat = odeint(model, test_x[i, 0, :, :], t_eval, method=args.solver)
        test_loss.append((test_x[i,:,:,:] - test_x_hat)**2)

    train_loss = torch.cat(train_loss, dim=1)
    train_loss_per_traj = torch.sum(train_loss, dim=(0,2))

    test_loss = torch.cat(test_loss, dim=1)
    test_loss_per_traj = torch.sum(test_loss, dim=(0,2))

    print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
    .format(train_loss_per_traj.mean().item(), train_loss_per_traj.std().item(),
            test_loss_per_traj.mean().item(), test_loss_per_traj.std().item()))

    stats['traj_train_loss'] = train_loss_per_traj.detach().cpu().numpy()
    stats['traj_test_loss'] = test_loss_per_traj.detach().cpu().numpy()
    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save 
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    if args.naive:
        label = '-naive_ode'
    elif args.baseline:
        label = '-baseline_ode'
    else:
        label = '-hnn_ode'
    struct = '-struct' if args.structure else ''
    path = '{}/{}{}{}-{}-p{}.tar'.format(args.save_dir, args.name, label, struct, args.solver, args.num_points)
    torch.save(model.state_dict(), path)
    path = '{}/{}{}{}-{}-p{}-stats.pkl'.format(args.save_dir, args.name, label, struct, args.solver, args.num_points)
    to_pickle(stats, path)
