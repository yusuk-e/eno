import sys
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math
import time
torch.set_default_dtype(torch.float64)


class MLP(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight)
      #torch.nn.init.normal_(l.weight)
      #torch.nn.init.kaiming_uniform_(l.weight, mode='fan_in')#, nonlinearity=nonlinearity)
      
    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    return self.linear3(h)

  
class HNO(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, Hamilton, device, col_size, G):
    super(HNO, self).__init__()
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
        H_eq = dH[0] + dHu_dx
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
