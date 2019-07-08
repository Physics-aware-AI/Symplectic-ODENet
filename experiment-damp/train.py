# code structure follows Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import torch, argparse, os
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP, PSD, DampMatrix
from hnn import HNN, HNN_structure
from data import get_dataset, arrange_data
from utils import L2_loss, to_pickle, abs_loss

import time

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='pend', type=str, help='name of the task')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=2, help='number of evaluation points by the ODE solver, including the initial point')
    parser.add_argument('--structure', dest='structure', action='store_true', help='using a structured Hamiltonian')
    parser.add_argument('--damp', dest='damp', action='store_true', help='adding the damping term')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args):
    # import ODENet
    from torchdiffeq import odeint

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # reproducibility: set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init model and optimizer
    if args.verbose:
        print("Training baseline ODE model with num of points = {}:".format(args.num_points) if args.baseline 
            else "Training HNN ODE model with num of points = {}:".format(args.num_points))
        if args.structure:
            print("using the structured Hamiltonian")
        if args.damp:
            print("using the Hamiltonian with the damping term")

    if args.structure == False:
        # Neural net without structure
        if args.baseline:
            assert args.damp == False

        output_dim = args.input_dim if args.baseline else 1
        nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity).to(device)
        # damp_model = PSD(input_dim=args.input_dim, hidden_dim=30, 
        #             diag_dim=args.input_dim, nonlinearity=args.nonlinearity).to(device)
        damp_model = DampMatrix(input_dim=args.input_dim, hidden_dim=10,
                    diag_dim=args.input_dim, device=device, nonlinearity=args.nonlinearity).to(device)
        model = HNN(args.input_dim, differentiale_model=nn_model, device=device, 
                baseline=args.baseline, damp=args.damp, dampNet=damp_model).to(device)
    elif args.structure and args.baseline == False:
        pass
    else:
        raise RuntimeError('argument *baseline* and *structure* cannot both be true')
    
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    # arrange data
    data = get_dataset(seed=args.seed)
    train_x, t_eval = arrange_data(data['x'], data['t'], num_points=args.num_points)
    test_x, t_eval = arrange_data(data['test_x'], data['t'], num_points=args.num_points)

    train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float32).to(device) # (45, 25, 2)
    test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)

    # training loop
    stats = {'train_loss': [], 'test_loss': [], 'forward_time': [], 'backward_time': [],'nfe': []}
    for step in range(args.total_steps+1):
        # train step
        t = time.time()
        train_x_hat = odeint(model.time_derivative, train_x[0, :, :], t_eval, method='dopri5')
        forward_time = time.time() - t
        loss = L2_loss(train_x, train_x_hat)

        t = time.time()
        loss.backward() ; optim.step() ; optim.zero_grad()
        backward_time = time.time() - t
        
        # run test data
        test_x_hat = odeint(model.time_derivative, test_x[0, :, :], t_eval, method='dopri5')
        test_loss = abs_loss(test_x, test_x_hat)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        stats['forward_time'].append(forward_time)
        stats['backward_time'].append(backward_time)
        stats['nfe'].append(model.nfe)
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

            # label = '-baseline' if args.baseline else '-hnn_ode'
            # path = '{}/{}{}-stats.pkl'.format(args.save_dir, args.name, label)
            # to_pickle(stats, path)

    train_x_hat = odeint(model.time_derivative, train_x[0, :, :], t_eval, method='dopri5')
    train_dist = (train_x - train_x_hat)**2
    test_x_hat = odeint(model.time_derivative, test_x[0, :, :], t_eval, method='dopri5')
    test_dist = (test_x - test_x_hat)**2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
        .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]*train_dist.shape[1]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0]*test_dist.shape[1])))

    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save 
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-baseline_ode' if args.baseline else '-hnn_ode'
    struct = '-struct' if args.structure else ''
    damp = '-damp' if args.damp else ''

    path = '{}/{}{}{}{}-p{}.tar'.format(args.save_dir, args.name, label, struct, damp, args.num_points)
    torch.save(model.state_dict(), path)

    path = '{}/{}{}{}{}-p{}-stats.pkl'.format(args.save_dir, args.name, label, struct, damp, args.num_points)
    to_pickle(stats, path)