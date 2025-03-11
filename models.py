import sys
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math
import time
torch.set_default_dtype(torch.float32)


class MLP(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight)
      
    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    return self.linear3(h)

  
class ENO(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, Hamilton, device, col_size, G):
    super(ENO, self).__init__()
    self.device = device
    self.col_size = col_size
    self.G = G
    self.Hamiltonian = MLP(2, hidden_dim, 1)
    self.operator = MLP(input_dim, hidden_dim, 1)
    self.Hamilton = Hamilton

  def solve(self, t, x, u0):
    Nx = len(x); Nt = len(t); N = len(u0)
    query_t = torch.flatten(torch.vstack([t]*Nx).T)
    query_x = torch.hstack([x]*Nt)
    query = torch.vstack([torch.stack([query_x, query_t]).T]*N)
    aug_u0 = torch.hstack([u0]*Nx*Nt).reshape([-1, u0.shape[1]])
    return self.operator(torch.hstack([query, aug_u0])).squeeze()

  def Hamilton_eq(self, t, x, u0):
      Nx = len(x); Nt = len(t); N = len(u0)
      query_t = (torch.rand(N*self.col_size, requires_grad=True, device=self.device)) * t.max()
      query_x = (torch.rand(N*self.col_size, requires_grad=True, device=self.device)) * x.max()
      query = torch.stack([query_x, query_t]).T
      aug_u0 = torch.hstack([u0]*self.col_size).reshape([-1, Nx])
      pred_u = self.operator(torch.hstack([query, aug_u0])).squeeze()
      pred_du = torch.autograd.grad(pred_u.sum(), query, create_graph=True)[0]
      pred_dudx = pred_du[:,0]
      pred_dudt = pred_du[:,1]
      inputs = torch.stack([pred_u, pred_dudx]).T
      H = self.Hamiltonian(inputs)
      dH = torch.autograd.grad(H.sum(), inputs, create_graph=True)[0].T
      if self.G == 0: #identity
        dHu_dx = torch.autograd.grad(dH[0].sum(), query_x, create_graph=True)[0]
        H_eq = -dH[0] + dHu_dx
      elif self.G == 1: #du/dx
        dHu_dx = torch.autograd.grad(dH[0].sum(), query_x, create_graph=True)[0]
        dHux_dx = torch.autograd.grad(dH[1].sum(), query_x, create_graph=True)[0]
        dHux_dxx = torch.autograd.grad(dHux_dx.sum(), query_x, create_graph=True)[0]
        H_eq = dHu_dx - dHux_dxx
      elif self.G == 2: #du2/dx2
        dHu_dx = torch.autograd.grad(dH[0].sum(), query_x, create_graph=True)[0]
        dHu_dxx = torch.autograd.grad(dHu_dx.sum(), query_x, create_graph=True)[0]
        dHux_dx = torch.autograd.grad(dH[1].sum(), query_x, create_graph=True)[0]
        dHux_dxx = torch.autograd.grad(dHux_dx.sum(), query_x, create_graph=True)[0]
        dHux_dxxx = torch.autograd.grad(dHux_dxx.sum(), query_x, create_graph=True)[0]
        H_eq = dHu_dxx - dHux_dxxx
      return pred_dudt, H_eq

  def loss(self, t, x, u, xi):
    u0 = u[:,0,:]
    pred_u = self.solve(t, x, u0)
    data_loss = ((pred_u - torch.flatten(u))**2).mean()
    if self.Hamilton:
      pred_dudt, symp_grad = self.Hamilton_eq(t, x, u0)
      hamiltonian_loss = ((pred_dudt - symp_grad)**2).mean()
      loss = data_loss + xi * hamiltonian_loss
    else:
      hamiltonian_loss = torch.tensor([0])
      loss = data_loss      
    return loss, data_loss, hamiltonian_loss

class ENO_fixed(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, Hamilton, device, col_size, G):
    super(ENO_fixed, self).__init__()
    self.device = device
    self.col_size = col_size
    self.G = G
    self.Hamiltonian = MLP(2, hidden_dim, 1)
    self.operator = MLP(input_dim, hidden_dim, 1)
    self.Hamilton = Hamilton

  def solve(self, t, x, u0):
    Nx = len(x); Nt = len(t); N = len(u0)
    query_t = torch.flatten(torch.vstack([t]*Nx).T)
    query_x = torch.hstack([x]*Nt)
    query = torch.vstack([torch.stack([query_x, query_t]).T]*N)
    aug_u0 = torch.hstack([u0]*Nx*Nt).reshape([-1, u0.shape[1]])
    return self.operator(torch.hstack([query, aug_u0])).squeeze()

  def Hamilton_eq(self, t, x, u0):
      Nx = len(x); Nt = len(t); N = len(u0)
      query = torch.cartesian_prod(t,x)
      if len(query) > self.col_size:
        ids = torch.randperm(len(query))[:self.col_size]
        query = query[ids]
      else:
        self.col_size = len(query)
      query_t = torch.hstack([query[:,0]]*N)
      query_t.requires_grad=True
      query_x = torch.hstack([query[:,1]]*N)
      query_x.requires_grad=True
      query = torch.stack([query_x, query_t]).T

      aug_u0 = torch.hstack([u0]*self.col_size).reshape([-1, Nx])
      pred_u = self.operator(torch.hstack([query, aug_u0])).squeeze()
      pred_du = torch.autograd.grad(pred_u.sum(), query, create_graph=True)[0]
      pred_dudx = pred_du[:,0]
      pred_dudt = pred_du[:,1]
      inputs = torch.stack([pred_u, pred_dudx]).T
      H = self.Hamiltonian(inputs)
      dH = torch.autograd.grad(H.sum(), inputs, create_graph=True)[0].T
      if self.G == 0: #identity
        dHu_dx = torch.autograd.grad(dH[0].sum(), query_x, create_graph=True)[0]
        H_eq = -dH[0] + dHu_dx
      elif self.G == 1: #du/dx
        dHu_dx = torch.autograd.grad(dH[0].sum(), query_x, create_graph=True)[0]
        dHux_dx = torch.autograd.grad(dH[1].sum(), query_x, create_graph=True)[0]
        dHux_dxx = torch.autograd.grad(dHux_dx.sum(), query_x, create_graph=True)[0]
        H_eq = dHu_dx - dHux_dxx
      elif self.G == 2: #du2/dx2
        dHu_dx = torch.autograd.grad(dH[0].sum(), query_x, create_graph=True)[0]
        dHu_dxx = torch.autograd.grad(dHu_dx.sum(), query_x, create_graph=True)[0]
        dHux_dx = torch.autograd.grad(dH[1].sum(), query_x, create_graph=True)[0]
        dHux_dxx = torch.autograd.grad(dHux_dx.sum(), query_x, create_graph=True)[0]
        dHux_dxxx = torch.autograd.grad(dHux_dxx.sum(), query_x, create_graph=True)[0]
        H_eq = dHu_dxx - dHux_dxxx
      return pred_dudt, H_eq
  
  def loss(self, t, x, u, xi):
    u0 = u[:,0,:]
    pred_u = self.solve(t, x, u0)
    data_loss = ((pred_u - torch.flatten(u))**2).mean()
    if self.Hamilton:
      pred_dudt, symp_grad = self.Hamilton_eq(t, x, u0)
      hamiltonian_loss = ((pred_dudt - symp_grad)**2).mean()
      loss = data_loss + xi * hamiltonian_loss
    else:
      hamiltonian_loss = torch.tensor([0])
      loss = data_loss      
    return loss, data_loss, hamiltonian_loss
  

class DeepONet(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, Hamilton, device, col_size, embed_dim):
    super(DeepONet, self).__init__()
    self.device = device
    self.col_size = col_size
    self.Hamiltonian = MLP(2, hidden_dim, 1)
    self.nn_trunk = MLP(2, hidden_dim, embed_dim)
    self.nn_branch = MLP(input_dim, hidden_dim, embed_dim)
    self.Hamilton = Hamilton

  def solve(self, t, x, u0):
    Nx = len(x); Nt = len(t); N = len(u0)
    query_t = torch.flatten(torch.vstack([t]*Nx).T)
    query_x = torch.hstack([x]*Nt)
    query = torch.vstack([torch.stack([query_x, query_t]).T]*N)
    latent_q = self.nn_trunk(query)
    aug_u0 = torch.hstack([u0]*Nx*Nt).reshape([-1, u0.shape[1]])
    latent_u0 = self.nn_branch(aug_u0)
    return (latent_q * latent_u0).sum(axis=1)

  def loss(self, t, x, u, xi, name):
    u0 = u[:,0,:]
    pred_u = self.solve(t, x, u0)
    data_loss = ((pred_u - torch.flatten(u))**2).mean()
    hamiltonian_loss = torch.tensor([0])
    loss = data_loss      
    return loss, data_loss, hamiltonian_loss


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        #x_ft = x_ft.real

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

  
class FNO(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, Hamilton, device, col_size, mode, nl, dh):
    super(FNO, self).__init__()
    self.device = device
    self.col_size = col_size
    self.fc0 = nn.Linear(input_dim+2, hidden_dim)
    self.fc0_2 = nn.Linear(hidden_dim, hidden_dim)

    self.conv0 = SpectralConv2d(hidden_dim, hidden_dim, mode, mode)
    self.conv1 = SpectralConv2d(hidden_dim, hidden_dim, mode, mode)
    self.conv2 = SpectralConv2d(hidden_dim, hidden_dim, mode, mode)
    self.conv3 = SpectralConv2d(hidden_dim, hidden_dim, mode, mode)
    self.w0 = nn.Conv2d(hidden_dim, hidden_dim, 1)
    self.w1 = nn.Conv2d(hidden_dim, hidden_dim, 1)
    self.w2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
    self.w3 = nn.Conv2d(hidden_dim, hidden_dim, 1)
    
    self.fc1 = nn.Linear(hidden_dim, dh)
    self.fc2 = nn.Linear(dh, 1)

    self.Hamiltonian = MLP(2, hidden_dim, 1)
    self.Hamilton = Hamilton
    self.mode = mode
    self.nonlinearity = choose_nonlinearity(nl)
    self.dh = dh

  def solve(self, t, x, u0):
    Nx = len(x); Nt = len(t); N = len(u0)
    query_t = torch.flatten(torch.vstack([t]*Nx).T)
    query_x = torch.hstack([x]*Nt)
    query = torch.vstack([torch.stack([query_x, query_t]).T]*N)
    aug_u0 = torch.hstack([u0]*Nx*Nt).reshape([-1, u0.shape[1]])
    x = self.fc0(torch.hstack([query, aug_u0])).squeeze()
    x = x.reshape([N,Nt,Nx,-1]).permute(0, 3, 1, 2)

    x1 = self.conv0(x)
    x2 = self.w0(x)
    x = x1 + x2
    x = self.nonlinearity(x)

    x1 = self.conv1(x)
    x2 = self.w1(x)
    x = x1 + x2
    x = self.nonlinearity(x)

    x1 = self.conv2(x)
    x2 = self.w2(x)
    x = x1 + x2
    x = self.nonlinearity(x)

    x1 = self.conv3(x)
    x2 = self.w3(x)
    x = x1 + x2

    x = x.permute(0, 2, 3, 1)
    x = self.fc1(x)
    x = self.nonlinearity(x)
    x = self.fc2(x)
    
    return x.squeeze()

  def loss(self, t, x, u, xi, name):
    u0 = u[:,0,:]
    pred_u = self.solve(t, x, u0)
    data_loss = ((pred_u - u)**2).mean()
    hamiltonian_loss = torch.tensor([0])
    loss = data_loss      
    return loss, data_loss, hamiltonian_loss


class EnerReg(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, Hamilton, device, col_size):
    super(EnerReg, self).__init__()
    self.device = device
    self.col_size = col_size
    self.Hamiltonian = MLP(2, hidden_dim, 1)
    self.operator = MLP(input_dim, hidden_dim, 1)
    self.Hamilton = Hamilton

  def solve(self, t, x, u0):
    Nx = len(x); Nt = len(t); N = len(u0)
    query_t = torch.flatten(torch.vstack([t]*Nx).T)
    query_x = torch.hstack([x]*Nt)
    query = torch.vstack([torch.stack([query_x, query_t]).T]*N)
    aug_u0 = torch.hstack([u0]*Nx*Nt).reshape([-1, u0.shape[1]])
    return self.operator(torch.hstack([query, aug_u0])).squeeze()

  def Hamilton_eq(self, t, x, u0, name):
      Nx = len(x); Nt = len(t); N = len(u0)
      query_t = (torch.rand(N*self.col_size, requires_grad=True, device=self.device)) * t.max()
      query_x = (torch.rand(N*self.col_size, requires_grad=True, device=self.device)) * x.max()
      query = torch.stack([query_x, query_t]).T
      aug_u0 = torch.hstack([u0]*self.col_size).reshape([-1, Nx])
      pred_u = self.operator(torch.hstack([query, aug_u0])).squeeze()
      pred_du = torch.autograd.grad(pred_u.sum(), query, create_graph=True)[0]
      pred_dudx = pred_du[:,0]
      pred_dudt = pred_du[:,1]
      inputs = torch.stack([pred_u, pred_dudx]).T
      if name == 'kdv':
        H = -(-6 / 6.) * inputs[:,0]**3 - 1 / 2. * inputs[:,1]**2
      elif name == 'ch':
        H = -(1 / 2.) * inputs[:,0]**2 + (1 / 4.) * inputs[:,0]**4 + 0.0005 / 2. * inputs[:,1]**2
      dH = torch.autograd.grad(H.sum(), inputs, create_graph=True)[0].T
      if name == 'kdv':
        dHu_dx = torch.autograd.grad(dH[0].sum(), query_x, create_graph=True)[0]
        dHux_dx = torch.autograd.grad(dH[1].sum(), query_x, create_graph=True)[0]
        dHux_dxx = torch.autograd.grad(dHux_dx.sum(), query_x, create_graph=True)[0]
        H_eq = dHu_dx - dHux_dxx
      elif name == 'ch':
        dHu_dx = torch.autograd.grad(dH[0].sum(), query_x, create_graph=True)[0]
        dHu_dxx = torch.autograd.grad(dHu_dx.sum(), query_x, create_graph=True)[0]
        dHux_dx = torch.autograd.grad(dH[1].sum(), query_x, create_graph=True)[0]
        dHux_dxx = torch.autograd.grad(dHux_dx.sum(), query_x, create_graph=True)[0]
        dHux_dxxx = torch.autograd.grad(dHux_dxx.sum(), query_x, create_graph=True)[0]
        H_eq = dHu_dxx - dHux_dxxx
      return pred_dudt, H_eq
  
  def loss(self, t, x, u, xi, name):
    u0 = u[:,0,:]
    pred_u = self.solve(t, x, u0)
    data_loss = ((pred_u - torch.flatten(u))**2).mean()
    if self.Hamilton:
      pred_dudt, symp_grad = self.Hamilton_eq(t, x, u0, name)
      hamiltonian_loss = ((pred_dudt - symp_grad)**2).mean()
      loss = data_loss + xi * hamiltonian_loss
    else:
      hamiltonian_loss = torch.tensor([0])
      loss = data_loss      
    return loss, data_loss, hamiltonian_loss
  

def choose_nonlinearity(name):
  nl = None
  if name == 'tanh':
    nl = torch.tanh
  elif name == 'relu':
    nl = torch.relu
  elif name == 'sigmoid':
    nl = torch.sigmoid
  elif name == 'softplus':
    nl = torch.nn.functional.softplus
  elif name == 'selu':
    nl = torch.nn.functional.selu
  elif name == 'elu':
    nl = torch.nn.functional.elu
  elif name == 'gelu':
    nl = torch.nn.functional.gelu
  elif name == 'swish':
    nl = lambda x: x * torch.sigmoid(x)
  else:
    raise ValueError("nonlinearity not recognized")
  return nl
