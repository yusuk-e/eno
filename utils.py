import os, sys
import pdb
import numpy as np
import scipy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
torch.set_default_dtype(torch.float32)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
sys.path.append(THIS_DIR+'/t_package')
from t_io import standard_vis as std_vis

class kdv():
    def __init__(self, N, Nx, dx):
        super().__init__()
        self.a = -6.
        self.b = 1.
        self.N = N
        self.Nx = Nx
        self.dx = dx
        self.S1 = self.symp_mat1()
        self.S2 = self.symp_mat2()

    def forward(self, t, u):
        u = u.reshape([int(u.size / self.Nx), self.Nx])
        dH = (-(self.a / 6.) * 3 * u**2 + self.b * u @ self.S2.T) @ self.S1.T
        return dH.flatten()

    def dx2(self, u):
        ids = np.arange(self.Nx) + 1
        ids[self.Nx-1] = 0
        return (u[:,:,ids] - u)**2 / (self.dx**2)
        
    def H(self, u):
        u = (u.T).reshape([-1, self.N, self.Nx])
        dx2 = self.dx2(u)
        local_energy = -(self.a / 6.) * u**3 - self.b / 2. * dx2
        H = (local_energy.sum(axis=2) * (self.dx)).T
        return H

    def get_init(self, N, x, X):
        a = self.a
        np.sech = lambda a: 1 / np.cosh(a)
        u0 = []
        D1 = 0.3
        D2 = 0.3
        for i in range(N):
            k1 = np.random.uniform(0.5, 1.0)
            k2 = np.random.uniform(1.5, 2.0)
            d1 = D1
            d2 = d1 + D2
            u0_elem = (-6. / a) * 2 * k1**2 * np.sech(k1 * (x - X * d1))**2
            u0_elem += (-6. / a) * 2 * k2**2 * np.sech(k2 * (x - X * d2))**2
            u0.append(u0_elem)
        return np.stack(u0)

    def symp_mat1(self):
        S = np.zeros([self.Nx,self.Nx])
        upper = np.eye(self.Nx-1)
        upper[0,-1]=-1
        lower = np.eye(self.Nx-1) * -1
        lower[-1,0]=1
        S[:-1,1:] += upper
        S[1:,:-1] += lower
        return S / (2*self.dx)

    def symp_mat2(self):
        S = -2 * np.eye(self.Nx,self.Nx)
        upper = np.eye(self.Nx-1)
        upper[0,-1]=1
        lower = np.eye(self.Nx-1)
        lower[-1,0]=1
        S[:-1,1:] += upper
        S[1:,:-1] += lower
        return S / (self.dx**2)


class ch():
    def __init__(self, N, Nx, dx):
        super().__init__()
        self.a = 1.
        self.b = 1.
        self.gamma = 0.0005
        self.N = N
        self.Nx = Nx
        self.dx = dx
        self.S1 = self.symp_mat1()
        self.S2 = self.symp_mat2()

    def forward(self, t, u):
        u = u.reshape([int(u.size / self.Nx), self.Nx])
        dH = ( -self.a * u + self.b * u**3 - self.gamma * u @ self.S2.T ) @ self.S2.T
        return dH.flatten()

    def dx2(self, u):
        ids = np.arange(self.Nx) + 1
        ids[self.Nx-1] = 0
        return (u[:,:,ids] - u)**2 / (self.dx**2)
        
    def H(self, u):
        u = (u.T).reshape([-1, self.N, self.Nx])
        dx2 = self.dx2(u)
        local_energy = -(self.a / 2.) * u**2 + (self.b / 4.) * u**4 + self.gamma / 2. * dx2
        H = (local_energy.sum(axis=2) * (self.dx)).T
        return H

    def get_init(self, N, x, X):
        degree = 5
        bases = []
        for d in range(degree):
            bases.append(scipy.special.eval_chebyt(d, x/X))
        bases = np.stack(bases)
        u0 = []
        for i in range(N):
            w = np.random.uniform(low=0., high=0.05, size=degree)
            u0.append( (bases * np.stack([w]*len(x)).T).sum(axis=0) )
        return np.stack(u0)

    def symp_mat1(self):
        S = np.zeros([self.Nx,self.Nx])
        upper = np.eye(self.Nx-1)
        upper[0,-1]=-1
        lower = np.eye(self.Nx-1) * -1
        lower[-1,0]=1
        S[:-1,1:] += upper
        S[1:,:-1] += lower
        return S / (2*self.dx)

    def symp_mat2(self):
        S = -2 * np.eye(self.Nx,self.Nx)
        upper = np.eye(self.Nx-1)
        upper[0,-1]=1
        lower = np.eye(self.Nx-1)
        lower[-1,0]=1
        S[:-1,1:] += upper
        S[1:,:-1] += lower
        return S / (self.dx**2)


def visualization(save_dir, t, u, e, m, T, X):
    for i in range(len(u)):
        u_elem = u[i]
        e_elem = e[i]
        m_elem = m[i]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6., 6.), facecolor='white')
        Nx = u.shape[-1]
        y = np.arange(Nx) / Nx
        t_grid, y_grid = np.meshgrid(t, y)
        vmin = u_elem.min()
        vmax = u_elem.max()
        c = ax1.pcolormesh(t_grid, y_grid, u_elem.T, cmap='seismic', vmin=-vmax, vmax=vmax)
        fig.colorbar(c, ax=ax1, label='colorbar label')
        ax1.set_aspect('auto')
        ax1.set_yticks((0 - .5 / Nx, 1 - .5 / Nx))
        ax1.set_yticklabels((0, X))
        ax2.plot(t, e_elem)
        ax3.plot(t, m_elem)
        ax3.set_xticks([0, T])
        ax3.set_xticklabels([0, T])
        fig.savefig('{}/data_{:02d}.png'.format(save_dir, i))
        plt.close()

        
def visualization_comp(save_dir, ts, us, es, ms, pred_us, pred_es, pred_ms, end_t, end_x, name):

    import seaborn as sns
    cmap = sns.palplot(sns.diverging_palette(220, 20, n=24))
    cmap = 'seismic'
    for i in range(len(us)):
        u = us[i]
        e = es[i]
        m = ms[i]
        p_u = pred_us[i]
        p_e = pred_es[i]
        p_m = pred_ms[i]
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, figsize=(6., 10.), facecolor='white')
        Nx = us.shape[-1]
        y = np.arange(Nx) / Nx
        T, Y = np.meshgrid(ts, y)
        if name == 'kdv':
            vmax = 5.2
            A = 4
        elif name == 'ch':
            vmax = 1.2
            A = 1
        c = ax1.pcolormesh(T, Y, u.T, cmap=cmap, vmin=-vmax, vmax=vmax)
        c = ax2.pcolormesh(T, Y, p_u.T, cmap=cmap, vmin=-vmax, vmax=vmax)
        c = ax3.pcolormesh(T, Y, A*(u.T - p_u.T), cmap=cmap, vmin=-vmax, vmax=vmax)
        fig.colorbar(c, ax=ax1, label='colorbar label')
        fig.colorbar(c, ax=ax2, label='colorbar label')
        fig.colorbar(c, ax=ax3, label='colorbar label')
        ax1.set_aspect('auto')
        ax1.set_yticks((0 - .5 / Nx, 1 - .5 / Nx))
        ax1.set_yticklabels((0, end_x))
        ax2.set_aspect('auto')
        ax2.set_yticks((0 - .5 / Nx, 1 - .5 / Nx))
        ax2.set_yticklabels((0, end_x))
        ax3.set_aspect('auto')
        ax3.set_yticks((0 - .5 / Nx, 1 - .5 / Nx))
        ax3.set_yticklabels((0, end_x))
        ax4.plot(ts, e, 'black')
        ax4.plot(ts, p_e, 'red')
        if name == 'kdv':
            ax4.set_ylim([e[0]-e[0]*0.2, e[0]+e[0]*0.2])
        ax5.plot(ts, m, 'black')
        ax5.plot(ts, p_m, 'red')
        if name == 'kdv':
            ax5.set_ylim([m[0]-m[0]*0.2, m[0]+m[0]*0.2])
        ax5.set_xticks([0, end_t])
        ax5.set_xticklabels([0, end_t])
        fig.savefig('{}/data_{:02d}.png'.format(save_dir, i))
        plt.close()
