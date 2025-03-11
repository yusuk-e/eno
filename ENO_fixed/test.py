import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
num_threads = '1'
os.environ['OMP_NUM_THREADS'] = num_threads
os.environ['MKL_NUM_THREADS'] = num_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_threads

import copy
import pdb
import time
import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.autograd import detect_anomaly
torch.set_default_dtype(torch.float32)

sys.path.append(PARENT_DIR+'/t_package')
from t_io import standard_io as std_io
from t_io import standard_vis as std_vis

from models import ENO_fixed
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--cuda_no', default=0, type=str)
    parser.add_argument('--name', default='kdv', type=str)
    parser.add_argument('--s', default=0, type=int, help='dataset index')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--data_no', default=0, type=int)
    parser.add_argument('--setting_no', default=0, type=int)
    parser.add_argument('--Hamilton', action='store_true')
    parser.add_argument('--lam', default=1e-4, type=float, help='hyperparameter')
    return parser.parse_args()


if __name__ == "__main__":

    # setting
    args = get_args()
    device = torch.device('cuda:' + str(args.cuda_no) if torch.cuda.is_available() else 'cpu')
    tmp_dir = '../data/' + args.name + '/result_' + str(args.data_no) + '/' + str(args.s)
    if args.Hamilton:
        i_dir = tmp_dir + '/ENO_fixed/st_' + str(args.setting_no) +'/' + str(args.lam)
    else:
        i_dir = tmp_dir + '/MLP/st_' + str(args.setting_no)
    save_dir = i_dir + '/test'
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    
    # input
    filename = '../data/' + args.name + '/test/data.pkl'
    data = std_io.pkl_read(filename)
    filename = '../data/' + args.name + '/test/' + str(args.name) + '.json'
    with open(filename) as f:
        vs = json.load(f)
    filename = '../data/' + args.name + '/train_' + str(args.data_no) + '/' + str(args.name) + '.json'
    with open(filename) as f:
        train_vs = json.load(f)

    # test data
    u = torch.tensor(data['u']).squeeze().float().to(device)
    t = torch.tensor(data['t']).float().to(device)
    x = torch.tensor(data['x']).float().to(device)
    e = data['e']
    m = data['m']
    dx = vs['X'] / vs['Nx']
    x_ratio = int((vs['Nx']-1) / (train_vs['Nx']-1))
    x_ids = [i for i in range(len(data['x']))][::x_ratio]
    tmp = u[:,0,:]
    u0 = tmp[:,x_ids]

    # learned model
    json_open = open(i_dir + '/model.json', 'r')
    json_model = json.load(json_open)
    input_dim = 2 + train_vs['Nx']
    hidden_dim = json_model['hidden_dim']
    Hamilton = json_model['Hamilton']
    col_size = json_model['col_size']
    G = json_model['G']
    model = ENO_fixed(input_dim, hidden_dim, Hamilton, device, col_size, G).to(device)
    model.load_state_dict(torch.load(i_dir + '/model.tar', map_location=torch.device('cpu')), False)

    # simulation
    t0 = time.time()
    pred_u = []
    with torch.no_grad():
        for line in u0:
            line = line.unsqueeze(0)
            solution = model.solve(t, x, line)
            pred_u.append(solution)
    pred_u = torch.stack(pred_u)

    #pred_u = pred_u.cpu().numpy()
    sim_time = time.time() - t0
    path = '{}/simulation_time.csv'.format(save_dir)
    std_io.csv_write(path, np.array(['simulation_time', sim_time/u.shape[0]]))
    if args.name == 'kdv':
        from utils import kdv
        pde = kdv(vs['N'], vs['Nx'], dx)
    elif args.name == 'ch':
        from utils import ch
        pde = ch(vs['N'], vs['Nx'], dx)

    pred_u = pred_u.reshape([u0.shape[0], len(t), len(x)]).cpu().numpy()
    #pred_e = pde.H(torch.permute(pred_u, (2,0,1)))    
    pred_e = pde.H(pred_u.transpose(1,0,2).T)
    pred_m = pred_u.sum(-1)
    sim = {}
    sim['u'] = pred_u
    sim['e'] = pred_e
    sim['m'] = pred_m
    sim['t'] = t
    sim['x'] = x

    loss = ((pred_u - u.cpu().numpy())**2).mean()
    #loss = (np.abs( (pred_us - us.cpu().numpy())/us.cpu().numpy() )).mean()
    #print(loss)

    mse_e = ((e - sim['e'])**2).mean()
    #mse_es = (np.abs( (data['es'] - sim['es'])/data['es'] )).mean()
    #print(mse_es)

    mse_m = ((m - sim['m'])**2).mean()
    #mse_ms = (np.abs( (data['ms'] - sim['ms'])/data['ms'] )).mean()
    #print(mse_ms)

    path = '{}/test_result.csv'.format(save_dir)
    std_io.csv_write(path, np.array(['mse_u', loss.item(), 'mse_e', mse_e.item(), 'mse_m', mse_m.item()]))

    visualization_comp(save_dir, t.cpu(), u.cpu(), e, m, pred_u, pred_e, pred_m, vs['T'], vs['X'], args.name)

