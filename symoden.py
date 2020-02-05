# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# code structure follows the style of HNN by Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import torch
import numpy as np

from nn_models import MLP

class SymODEN_R(torch.nn.Module):
    '''
    Architecture for input (q, p, u), 
    where q and p are tensors of size (bs, n) and u is a tensor of size (bs, 1)
    '''
    def __init__(self, input_dim, H_net=None, M_net=None, V_net=None, g_net=None, device=None,
                    assume_canonical_coords=True, baseline=False, structure=False, damp_net=None):
        super(SymODEN_R, self).__init__()
        self.baseline = baseline
        self.structure = structure
        if self.structure:
            self.M_net = M_net
            self.V_net = V_net
            self.g_net = g_net
        else:
            self.H_net = H_net
            self.g_net = g_net

        self.device = device
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)
        self.nfe = 0
        self.input_dim = input_dim
        self.damp_net = damp_net

    def forward(self, t, x):
        with torch.enable_grad():
            one = torch.tensor(1, dtype=torch.float32, device=self.device, requires_grad=True)
            x = one * x
            self.nfe += 1
            bs = x.shape[0]
            zero_vec = torch.zeros(bs, 1, dtype=torch.float32, device =self.device)
            q, p, u = torch.split(x, [self.input_dim//2, self.input_dim//2, 1], dim=1)
            q_p = torch.cat((q,p), dim=1)
            if self.baseline:
                dq, dp=  torch.chunk(self.H_net(q_p), 2, dim=1)
                return torch.cat((dq, dp, zero_vec), dim=1) # damping doesn't affect this term
            if self.structure:
                q, p = torch.chunk(q_p, 2, dim=1)
                V_q = self.V_net(q)
                M_q_inv = self.M_net(q)
                if self.input_dim == 2:
                    H = p * p * M_q_inv  / 2.0 + V_q
                else:
                    p_aug = torch.unsqueeze(p, dim=2)
                    H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
            else:
                H = self.H_net(q_p)
            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            H_vector_field = torch.matmul(dH, self.M.t())
            # dHdq, dHdp= torch.chunk(dH, 2, dim=1)
            # H_vector_field = torch.cat((dHdp, -dHdq, torch.zeros_like(dHdq)), dim=1)
            H_vector_field = torch.cat((H_vector_field, torch.zeros_like(H_vector_field)[:,0].view(-1,1)), dim=1)
            g_q = self.g_net(q)

            F = g_q * u
            F_vector_field = torch.cat((torch.zeros_like(F), F, zero_vec), dim=1)
            if self.damp_net:
                D_vector_field = torch.squeeze(torch.matmul(dH.unsqueeze(1), self.damp_net(q))) # should be self.damp_net transpose, but symmetric
                D_vector_field = torch.cat((D_vector_field, zero_vec), dim=1)
                return H_vector_field + F_vector_field - D_vector_field

            return H_vector_field + F_vector_field

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


class SymODEN_T(torch.nn.Module):
    '''
    Architecture for input (cos q, sin q, q_dot, u), 
    where q represent angles, a tensor of size (bs, n),
    cos q, sin q and q_dot are tensors of size (bs, n), and
    u is a tensor of size (bs, 1).
    '''
    def __init__(self, input_dim, H_net=None, M_net=None, V_net=None, g_net=None,
            device=None, baseline=False, structure=False, naive=False, u_dim=1, damp_net=None):
        super(SymODEN_T, self).__init__()
        self.baseline = baseline
        self.structure = structure
        self.naive = naive
        self.M_net = M_net
        self.u_dim = u_dim
        if self.structure:
            self.V_net = V_net
            self.g_net = g_net
        else:
            self.H_net = H_net
            self.g_net = g_net

        self.device = device
        self.nfe = 0
        self.input_dim = input_dim
        self.damp_net = damp_net

    def forward(self, t, x):
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            zero_vec = torch.zeros(bs, self.u_dim, dtype=torch.float32, device =self.device)

            if self.naive:
                return torch.cat((self.H_net(x), zero_vec), dim=1) # damping won't affect naive baseline

            cos_q_sin_q, q_dot, u = torch.split(x, [2*self.input_dim, 1*self.input_dim, self.u_dim], dim=1)
            M_q_inv = self.M_net(cos_q_sin_q)
            if self.input_dim == 1:
                p = q_dot / M_q_inv
            else:
                # assert 1==0
                q_dot_aug = torch.unsqueeze(q_dot, dim=2)
                p = torch.squeeze(torch.matmul(torch.inverse(M_q_inv), q_dot_aug), dim=2)
            cos_q_sin_q_p = torch.cat((cos_q_sin_q, p), dim=1)
            cos_q_sin_q, p = torch.split(cos_q_sin_q_p, [2*self.input_dim, 1*self.input_dim], dim=1)
            M_q_inv = self.M_net(cos_q_sin_q)
            cos_q, sin_q = torch.chunk(cos_q_sin_q, 2,dim=1)

            # M_q_inv = 3 * torch.ones_like(u)

            if self.baseline:
                dq, dp=  torch.chunk(self.H_net(x), 2, dim=1) # damping won't affect the term in baseline model
            else:
                if self.structure:
                    V_q = self.V_net(cos_q_sin_q)   
                    if self.input_dim == 1:
                        H = p * p * M_q_inv/ 2.0 + V_q
                    else:
                        p_aug = torch.unsqueeze(p, dim=2)
                        H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
                else:
                    H = self.H_net(cos_q_sin_q_p)
                dH = torch.autograd.grad(H.sum(), cos_q_sin_q_p, create_graph=True)[0]
                dHdcos_q, dHdsin_q, dHdp= torch.split(dH, [self.input_dim, self.input_dim, self.input_dim], dim=1)
                g_q = self.g_net(cos_q_sin_q)
                
                if self.u_dim == 1:
                    # broadcast multiply when angle is more than 1
                    F = g_q * u
                else:
                    F = torch.squeeze(torch.matmul(g_q, torch.unsqueeze(u, dim=2)))

                dHdq = - sin_q * dHdcos_q + cos_q * dHdsin_q
                if self.damp_net:
                    D_vector_field = torch.matmul(torch.cat((dHdq, dHdp), dim=1), self.damp_net) # should be self.damp_net transpose, but symmetric
                    D_vector_field_q, D_vector_field_p = D_vector_field.chunk(2, dim=1)
                    dq = dHdp - D_vector_field_q
                    dp = -dHdq + F - D_vector_field_p
                else:
                    dq = dHdp
                    dp = -dHdq + F

            if self.input_dim==1:
                dM_inv = torch.autograd.grad(M_q_inv.sum(), cos_q_sin_q, create_graph=True)[0]
                dM_inv_dt = (dM_inv * torch.cat((-sin_q * dq, cos_q * dq), dim=1)).sum(-1).view(-1, 1)
                ddq =  M_q_inv * dp  + dM_inv_dt * p
            else:
                dM_inv_dt = torch.zeros_like(M_q_inv)
                for row_ind in range(self.input_dim):
                    for col_ind in range(self.input_dim):
                        dM_inv = torch.autograd.grad(M_q_inv[:, row_ind, col_ind].sum(), cos_q_sin_q, create_graph=True)[0]
                        dM_inv_dt[:, row_ind, col_ind] = (dM_inv * torch.cat((-sin_q * dq, cos_q * dq), dim=1)).sum(-1)
                ddq = torch.squeeze(torch.matmul(M_q_inv, torch.unsqueeze(dp, dim=2)), dim=2) \
                        + torch.squeeze(torch.matmul(dM_inv_dt, torch.unsqueeze(p, dim=2)), dim=2)
            

            return torch.cat((-sin_q * dq, cos_q * dq, ddq, zero_vec), dim=1)


    def get_H(self, x):
        self.nfe += 1
        bs = x.shape[0]
        zero_vec = torch.zeros(bs, 1, dtype=torch.float32, device =self.device)

        assert self.naive == False
        assert self.baseline == False

        cos_q_sin_q, q_dot, u = torch.split(x, [2*self.input_dim, 1*self.input_dim, 1], dim=1)
        M_q_inv = self.M_net(cos_q_sin_q)
        if self.input_dim == 1:
            p = q_dot / M_q_inv
        else:
            # assert 1==0
            q_dot_aug = torch.unsqueeze(q_dot, dim=2)
            p = torch.squeeze(torch.matmul(torch.inverse(M_q_inv), q_dot_aug), dim=2)
        cos_q_sin_q_p = torch.cat((cos_q_sin_q, p), dim=1)
        cos_q_sin_q, p = torch.split(cos_q_sin_q_p, [2*self.input_dim, 1*self.input_dim], dim=1)
        M_q_inv = self.M_net(cos_q_sin_q)
        cos_q, sin_q = torch.chunk(cos_q_sin_q, 2,dim=1)

        # M_q_inv = 3 * torch.ones_like(u)


        if self.structure:
            V_q = self.V_net(cos_q_sin_q)   
            if self.input_dim == 1:
                H = p * p * M_q_inv/ 2.0 + V_q
            else:
                p_aug = torch.unsqueeze(p, dim=2)
                H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
        else:
            H = self.H_net(cos_q_sin_q_p)
        dH = torch.autograd.grad(H.sum(), cos_q_sin_q_p, create_graph=True)[0]

        return H, dH


class SymODEN_R1_T1(torch.nn.Module):
    '''
    Architecture for the cartpole system (x, cos q, sin q, x_dot, q_dot, u), 
    where x, cos q, sin q, x_dot, q_dot and u are all tensors of size (bs, 1)
    '''
    def __init__(self, input_dim, H_net=None, M_net=None, V_net=None, g_net=None,
            device=None, baseline=False, structure=False, naive=False, u_dim=1, damp_net=None):
        super(SymODEN_R1_T1, self).__init__()
        self.baseline = baseline
        self.structure = structure
        self.naive = naive
        self.M_net = M_net
        self.u_dim = u_dim
        if self.structure:
            self.V_net = V_net
            self.g_net = g_net
        else:
            self.H_net = H_net
            self.g_net = g_net

        self.device = device
        self.nfe = 0
        self.input_dim = input_dim
        self.damp_net = damp_net

    def forward(self, t, y):
        with torch.enable_grad():
            self.nfe += 1
            bs = y.shape[0]
            zero_vec = torch.zeros(bs, self.u_dim, dtype=torch.float32, device =self.device)

            if self.naive:
                return torch.cat((self.H_net(y), zero_vec), dim=1) # damping won't affect naive model

            x_cos_q_sin_q, x_dot_q_dot, u = torch.split(y, [3, 2, self.u_dim], dim=1)
            M_q_inv = self.M_net(x_cos_q_sin_q)

            x_dot_q_dot_aug = torch.unsqueeze(x_dot_q_dot, dim=2)
            p = torch.squeeze(torch.matmul(torch.inverse(M_q_inv), x_dot_q_dot_aug), dim=2)
            x_cos_q_sin_q_p = torch.cat((x_cos_q_sin_q, p), dim=1)
            x_cos_q_sin_q, p = torch.split(x_cos_q_sin_q_p, [3, 2], dim=1)
            M_q_inv = self.M_net(x_cos_q_sin_q)
            _, cos_q, sin_q = torch.chunk(x_cos_q_sin_q, 3,dim=1)

            # M_q_inv = 3 * torch.ones_like(u)

            if self.baseline:
                dx, dq, dp=  torch.split(self.H_net(y), [1, 1, 2], dim=1) # damping won't affect baseline model
            else:
                if self.structure:
                    V_q = self.V_net(x_cos_q_sin_q)   
                    p_aug = torch.unsqueeze(p, dim=2)
                    H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
                else:
                    H = self.H_net(x_cos_q_sin_q_p)
                dH = torch.autograd.grad(H.sum(), x_cos_q_sin_q_p, create_graph=True)[0]
                dHdx, dHdcos_q, dHdsin_q, dHdp= torch.split(dH, [1, 1, 1, 2], dim=1)
                g_q = self.g_net(x_cos_q_sin_q)
                
                if self.u_dim == 1:
                    # broadcast multiply when angle is more than 1
                    F = g_q * u
                else:
                    F = torch.squeeze(torch.matmul(g_q, torch.unsqueeze(u, dim=2)))

                dHdq = - sin_q * dHdcos_q + cos_q * dHdsin_q
                dp_q = - dHdq
                dp_x = - dHdx
                
                if not self.damp_net:
                    D_vector_field = torch.matmul(torch.cat((dHdx, dHdq, dHdp), dim=1), self.damp_net) # should be self.damp_net transpose, but symmetric
                    D_vector_field_xq, D_vector_field_p = torch.split(D_vector_field, [2, 2], dim=1)
                    dx, dq = torch.split(dHdp - D_vector_field_xq, [1, 1], dim=1)
                    dp = torch.cat((dp_x, dp_q), dim=1) + F - D_vector_field_p
                else:
                    dx, dq = torch.split(dHdp, [1, 1], dim=1)
                    dp = torch.cat((dp_x, dp_q), dim=1) + F


            dM_inv_dt = torch.zeros_like(M_q_inv)
            for row_ind in range(self.input_dim):
                for col_ind in range(self.input_dim):
                    dM_inv = torch.autograd.grad(M_q_inv[:, row_ind, col_ind].sum(), x_cos_q_sin_q, create_graph=True)[0]
                    dM_inv_dt[:, row_ind, col_ind] = (dM_inv * torch.cat((dx, -sin_q * dq, cos_q * dq), dim=1)).sum(-1)
            ddq = torch.squeeze(torch.matmul(M_q_inv, torch.unsqueeze(dp, dim=2)), dim=2) \
                    + torch.squeeze(torch.matmul(dM_inv_dt, torch.unsqueeze(p, dim=2)), dim=2)
        

            return torch.cat((dx, -sin_q * dq, cos_q * dq, ddq, zero_vec), dim=1)