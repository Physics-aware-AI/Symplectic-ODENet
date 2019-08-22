# code borrowed from Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import torch
import numpy as np

from nn_models import MLP


class HNN(torch.nn.Module):
    def __init__(self, input_dim, differentiale_model, device,
                    baseline=False, assume_canonical_coords=True,
                    damp=False, dampNet=None):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiale_model = differentiale_model
        self.assume_canonical_coords = assume_canonical_coords
        self.device = device
        self.M = self.permutation_tensor(input_dim)
        self.nfe = 0
        self.damp = damp

        if damp:
            self.dampNet = dampNet
        

    def forward(self, x):
        if self.baseline:
            return self.differentiale_model(x)
        else:
            y = self.differentiale_model(x)
            # assert y.dim() == 2 and y.shape[1] == 1
            return y

    # note that the input of this function has changed to meet the ODENet requirement.
    def time_derivative(self, t, x):
        self.nfe += 1
        if self.baseline:
            return self.differentiale_model(x)
        else:
            H = self.forward(x) # the Hamiltonian
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            H_vector_field = torch.matmul(dH, self.M.t())
            if not self.damp:
                return H_vector_field
            else:
                D = self.dampNet(x) # (100, 2, 2)
                D_vector_field = torch.squeeze(
                    torch.matmul(
                        torch.unsqueeze(dH, 1),
                        D # should be transpose but symmetric
                    )
                )
                return H_vector_field - D_vector_field

    def Hamiltonian_vector(self, x):
        H = self.forward(x) # the Hamiltonian
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        H_vector_field = torch.matmul(dH, self.M.t())
        return H_vector_field


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


class HNN_structure(torch.nn.Module):
    def __init__(self, input_dim, L_net, V_net, device,
                    assume_canonical_coords=True):
        super(HNN_structure, self).__init__()
        self.L_net = L_net
        self.V_net = V_net
        self.device = device
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)
        self.nfe = 0

    def forward(self, x):
        q, p = torch.chunk(x, 2, dim=1)
        L = self.L_net(q)
        V_q = self.V_net(q)
        M_q = L * L + 0.1

        H = p * p / M_q /2 + V_q
        return H
        # return p * p/ (2*self.L_net(q)*self.L_net(q)) + self.V_net(q)

    def time_derivative(self, t, x):
        self.nfe += 1
        H = self.forward(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        vector_field = torch.matmul(dH, self.M.t())
        return vector_field


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

class HNN_structure_pend(torch.nn.Module):
    def __init__(self, input_dim, H_net=None, M_net=None, V_net=None, F_net=None, device=None,
                    assume_canonical_coords=True, baseline=False, structure=False):
        super(HNN_structure_pend, self).__init__()
        self.baseline = baseline
        self.structure = structure
        if self.structure:
            self.M_net = M_net
            self.V_net = V_net
            self.F_net = F_net
        else:
            self.H_net = H_net
            self.F_net = F_net

        self.device = device
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)
        self.nfe = 0

    def forward(self, x):
        if self.baseline:
            return self.H_net(x)
        if self.structure:
            bs = x.shape[0]

            q, p, p_dot = torch.chunk(x, 3, dim=1)
            V_q = self.V_net(q)
            # M_q_inv = self.M_net(q)
            M_q_inv = torch.tensor([1.0, 1.0, 12.0], dtype=torch.float32, device=self.device)
            M_q_inv = torch.unsqueeze(torch.diag_embed(M_q_inv), dim=0)
            M_q_inv = M_q_inv.repeat(bs,1,1)

            p_aug = torch.unsqueeze(p, dim=2)
            H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
        else:
            H = self.H_net(x)

        return H

    def time_derivative(self, t, x):
        self.nfe += 1
        if self.baseline:
            return self.H_net(x)
        if self.structure:
            bs = x.shape[0]

            q, p, p_dot = torch.chunk(x, 3, dim=1)
            V_q = self.V_net(q)
            # M_q_inv = self.M_net(q)
            M_q_inv = torch.tensor([1.0, 1.0, 12.0], dtype=torch.float32, device=self.device)
            M_q_inv = torch.unsqueeze(torch.diag_embed(M_q_inv), dim=0)
            M_q_inv = M_q_inv.repeat(bs,1,1)

            p_aug = torch.unsqueeze(p, dim=2)
            H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
        else:
            H = self.H_net(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        H_vector_field = torch.matmul(dH[:, 0:6], self.M.t())
        dHdq, dHdp, _ = torch.chunk(dH, 3, dim=1)

        # dHdq1 = torch.autograd.grad(V_q.sum(), q, create_graph=True)[0]
        # dHdp1 = torch.squeeze(torch.matmul(M_q_inv, p_aug), dim=2)

        F = self.F_net(x)
        num = torch.squeeze(torch.bmm(torch.unsqueeze(dHdp, 1), torch.unsqueeze(F, 2)), dim=2)
        num = num * dHdp
        den = torch.squeeze(torch.bmm(torch.unsqueeze(dHdp, 1), torch.unsqueeze(dHdp, 2)), dim=2)
        div = num/den
        div = torch.where(torch.isnan(div), torch.zeros_like(div), div)

        Fc = F - div

        Fc_vector_field = torch.cat((torch.zeros_like(Fc), Fc), dim=1)

        return torch.cat((H_vector_field + Fc_vector_field, torch.zeros_like(Fc)), dim=1)

        # A_q = self.A_net(q)
        # A_T_q = torch.transpose(A_q, 1, 2)

        # # calculate the matrix to be differentiate
        # A_T_M_inv_p = torch.matmul(A_T_q, torch.matmul(M_q_inv, torch.unsqueeze(p, dim=2)))
        # dA_T_M_inv_p_dq = self.get_jacobian(q, A_T_M_inv_p)

        # # calculate lambda
        # RHS = torch.matmul(A_T_q, torch.matmul(M_q_inv, torch.unsqueeze(dHdq, dim=2))) \
        #         - torch.matmul(dA_T_M_inv_p_dq, torch.unsqueeze(dHdp, dim=2))
        # LHS = torch.matmul(A_T_q, torch.matmul(M_q_inv, A_q))
        # LHS = LHS + torch.ones_like(LHS) * 0.1
        # lambd = torch.matmul(torch.inverse(LHS), RHS)

        # con_force = torch.squeeze(torch.matmul(A_q, lambd))
        # con_force_vector_field = torch.cat((torch.zeros_like(con_force), con_force), dim=1)

    # def get_jacobian(self, input, output):
    #     output = torch.squeeze(output)
    #     output_sum = output.sum(0)

    #     jac = []
    #     for i in range(len(output_sum)):
    #         jac.append(torch.autograd.grad(output_sum[i], input, create_graph=True)[0])
    #     return torch.stack(jac, dim=1)

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

    def get_intermediate_value(self, t, x):
        bs = x.shape[0]

        q, p = torch.chunk(x, 2, dim=1)
        V_q = self.V_net(q)
        # M_q_inv = self.M_net(q)
        M_q_inv = torch.tensor([1.0, 1.0, 12.0], dtype=torch.float32, device=self.device)
        M_q_inv = torch.unsqueeze(torch.diag_embed(M_q_inv), dim=0)
        M_q_inv = M_q_inv.repeat(bs,1,1)

        p_aug = torch.unsqueeze(p, dim=2)
        H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)

        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        H_vector_field = torch.matmul(dH, self.M.t())
        dHdq, dHdp = torch.chunk(dH, 2, dim=1)

        # dHdq1 = torch.autograd.grad(V_q.sum(), q, create_graph=True)[0]
        # dHdp1 = torch.squeeze(torch.matmul(M_q_inv, p_aug), dim=2)

        F = self.F_net(torch.cat((q, p, dHdq), dim=1))
        num = torch.squeeze(torch.bmm(torch.unsqueeze(dHdp, 1), torch.unsqueeze(F, 2)), dim=2)
        num = num * dHdp
        den = torch.squeeze(torch.bmm(torch.unsqueeze(dHdp, 1), torch.unsqueeze(dHdp, 2)), dim=2)
        Fc = F - num/den

        Fc_vector_field = torch.cat((torch.zeros_like(Fc), Fc), dim=1)

        dHdp_Fc = torch.squeeze(torch.bmm(torch.unsqueeze(dHdp, 1), torch.unsqueeze(Fc, 2)), dim=2)
        return dHdq, dHdp, F, Fc, dHdp_Fc

class HNN_structure_forcing(torch.nn.Module):
    def __init__(self, input_dim, H_net=None, M_net=None, V_net=None, g_net=None, device=None,
                    assume_canonical_coords=True, baseline=False, structure=False):
        super(HNN_structure_forcing, self).__init__()
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

    def forward(self, t, x):
        with torch.enable_grad():
            one = torch.tensor(1, dtype=torch.float32, device=self.device, requires_grad=True)
            x = one * x
            self.nfe += 1
            bs = x.shape[0]
            zero_vec = torch.zeros(bs, 1, dtype=torch.float32, device =self.device)
            q, p, u = torch.chunk(x, 3, dim=1)
            q_p = torch.cat((q,p), dim=1)
            if self.baseline:

                dq, dp=  torch.chunk(self.H_net(q_p), 2, dim=1)
                return torch.cat((dq, dp, zero_vec), dim=1)
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


class HNN_structure_embed(torch.nn.Module):
    def __init__(self, input_dim, H_net=None, M_net=None, V_net=None, g_net=None,
            device=None, baseline=False, structure=False, naive=False):
        super(HNN_structure_embed, self).__init__()
        self.baseline = baseline
        self.structure = structure
        self.naive = naive
        self.M_net = M_net
        if self.structure:
            self.V_net = V_net
            self.g_net = g_net
        else:
            self.H_net = H_net
            self.g_net = g_net

        self.device = device
        self.nfe = 0
        self.input_dim = input_dim

    def forward(self, t, x):
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            zero_vec = torch.zeros(bs, 1, dtype=torch.float32, device =self.device)

            if self.naive:
                return torch.cat((self.H_net(x), zero_vec), dim=1)

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

            if self.baseline:
                dq, dp=  torch.chunk(self.H_net(x), 2, dim=1)
            else:
                if self.structure:
                    V_q = self.V_net(cos_q_sin_q)   
                    if self.input_dim == 1:
                        H = p * p * M_q_inv/ 2.0 + V_q
                    else:
                        # assert 1 == 0
                        p_aug = torch.unsqueeze(p, dim=2)
                        H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
                else:
                    H = self.H_net(cos_q_sin_q_p)
                dH = torch.autograd.grad(H.sum(), cos_q_sin_q_p, create_graph=True)[0]
                dHdcos_q, dHdsin_q, dHdp= torch.split(dH, [self.input_dim, self.input_dim, self.input_dim], dim=1)
                g_q = self.g_net(cos_q_sin_q)
                # broadcast multiply when angle is more than 1
                F = g_q * u
                # F = 3*u
                # F = u

                dq = dHdp
                dp = sin_q * dHdcos_q - cos_q * dHdsin_q + F

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
                # assert 1 == 0
                p_aug = torch.unsqueeze(p, dim=2)
                H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
        else:
            H = self.H_net(cos_q_sin_q_p)
        dH = torch.autograd.grad(H.sum(), cos_q_sin_q_p, create_graph=True)[0]

        return H, dH
    
    
    def get_dqdp(self, x):
        # assert 1==0
        self.nfe += 1
        bs = x.shape[0]
        zero_vec = torch.zeros(bs, 1, dtype=torch.float32, device =self.device)

        if self.naive:
            raise RuntimeError('*naive* ode does not support vector field.')

        cos_q_sin_q, q_dot, u = torch.split(x, [2, 1, 1], dim=1)
        M_q_inv = self.M_net(cos_q_sin_q)
        p = q_dot / M_q_inv
        cos_q_sin_q_p = torch.cat((cos_q_sin_q, p), dim=1)
        cos_q_sin_q, p = torch.split(cos_q_sin_q_p, [2, 1], dim=1)
        M_q_inv = self.M_net(cos_q_sin_q)
        cos_q, sin_q = torch.chunk(cos_q_sin_q, 2,dim=1)

        if self.baseline:
            dq, dp=  torch.chunk(self.H_net(x), 2, dim=1)
        else:
            if self.structure:
                V_q = self.V_net(cos_q_sin_q)   
                if self.input_dim == 1:
                    H = p * p * M_q_inv / 2.0 + V_q
                else:
                    assert 1 == 0
                    p_aug = torch.unsqueeze(p, dim=2)
                    H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
            else:
                H = self.H_net(cos_q_sin_q_p)
            dH = torch.autograd.grad(H.sum(), cos_q_sin_q_p, create_graph=True)[0]
            dHdcos_q, dHdsin_q, dHdp= torch.chunk(dH, 3, dim=1)
            g_q = self.g_net(cos_q_sin_q)
            F = g_q * u

            dq = dHdp
            dp = sin_q * dHdcos_q - cos_q * dHdsin_q + F

        return torch.cat((dq, dp), dim=1)


class HNN_structure_cart_embed(torch.nn.Module):
    def __init__(self, input_dim, H_net=None, M_net=None, V_net=None, g_net=None,
            device=None, baseline=False, structure=False, naive=False):
        super(HNN_structure_cart_embed, self).__init__()
        self.baseline = baseline
        self.structure = structure
        self.naive = naive
        self.M_net = M_net
        if self.structure:
            self.V_net = V_net
            self.g_net = g_net
        else:
            self.H_net = H_net
            self.g_net = g_net

        self.device = device
        self.nfe = 0
        self.input_dim = input_dim

    def forward(self, t, y):
        with torch.enable_grad():
            self.nfe += 1
            bs = y.shape[0]
            zero_vec = torch.zeros(bs, 1, dtype=torch.float32, device =self.device)

            if self.naive:
                return torch.cat((self.H_net(y), zero_vec), dim=1)

            x_cos_q_sin_q, x_dot_q_dot, u = torch.split(y, [3, 2, 1], dim=1)
            M_q_inv = self.M_net(x_cos_q_sin_q)

            x_dot_q_dot_aug = torch.unsqueeze(x_dot_q_dot, dim=2)
            p = torch.squeeze(torch.matmul(torch.inverse(M_q_inv), x_dot_q_dot_aug), dim=2)
            x_cos_q_sin_q_p = torch.cat((x_cos_q_sin_q, p), dim=1)
            x_cos_q_sin_q, p = torch.split(x_cos_q_sin_q_p, [3, 2], dim=1)
            M_q_inv = self.M_net(x_cos_q_sin_q)
            _, cos_q, sin_q = torch.chunk(x_cos_q_sin_q, 3,dim=1)

            # M_q_inv = 3 * torch.ones_like(u)

            if self.baseline:
                dx, dq, dp=  torch.split(self.H_net(y), [1, 1, 2], dim=1)
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
                # broadcast multiply when angle is more than 1
                F = g_q * u
                # F = 3*u
                # F = u

                dx, dq = torch.split(dHdp, [1, 1], dim=1)
                dp_q = sin_q * dHdcos_q - cos_q * dHdsin_q
                dp_x = - dHdx
                dp = torch.cat((dp_x, dp_q), dim=1) + F


            dM_inv_dt = torch.zeros_like(M_q_inv)
            for row_ind in range(self.input_dim):
                for col_ind in range(self.input_dim):
                    dM_inv = torch.autograd.grad(M_q_inv[:, row_ind, col_ind].sum(), x_cos_q_sin_q, create_graph=True)[0]
                    dM_inv_dt[:, row_ind, col_ind] = (dM_inv * torch.cat((dx, -sin_q * dq, cos_q * dq), dim=1)).sum(-1)
            ddq = torch.squeeze(torch.matmul(M_q_inv, torch.unsqueeze(dp, dim=2)), dim=2) \
                    + torch.squeeze(torch.matmul(dM_inv_dt, torch.unsqueeze(p, dim=2)), dim=2)
        

            return torch.cat((dx, -sin_q * dq, cos_q * dq, ddq, zero_vec), dim=1)