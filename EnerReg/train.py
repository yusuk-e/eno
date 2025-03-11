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

from models import EnerReg


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--cuda_no', default=0, type=str)
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--lr_op', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lr_H', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--wd_op', default=0, type=float, help='weight decay')
    parser.add_argument('--wd_H', default=0, type=float, help='weight decay')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--epochs', default=10000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=1, type=int, help='number of epochs for prints')
    parser.add_argument('--name', default='kdv', type=str)
    parser.add_argument('--s', default=0, type=int, help='dataset index')
    parser.add_argument('--batch_size', default=30, type=int, help='batch size')
    parser.add_argument('--col_size', default=200, type=int, help='collocation points')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--data_no', default=0, type=int)
    parser.add_argument('--setting_no', default=0, type=int)
    parser.add_argument('--Hamilton', action='store_true')
    parser.add_argument('--lam', default=1e-4, type=float, help='hyperparameter')
    return parser.parse_args()


def train():
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init model and optimizer
    input_dim = 2 + vs['Nx']
    model = EnerReg(input_dim, args.hidden_dim, args.Hamilton, device, args.col_size).to(device)
    optim = torch.optim.Adam([
        {'params': model.operator.parameters(), 'lr':args.lr_op, 'weight_decay':args.wd_op},
        {'params': model.Hamiltonian.parameters(), 'lr':args.lr_H, 'weight_decay':args.wd_H}
    ])

    # train loop
    stats = {'train_loss': [], 'val_loss': [], 'data_loss': [], 'hamiltonian_loss': []}
    t0 = time.time()
    best_val_loss = 1e+10
    for epoch in range(args.epochs+1):            
        ids = torch.randperm(u.shape[0])
        batch_ids = torch.stack([ids[i:i+args.batch_size] for i in range(0, len(ids), args.batch_size)])
        for ids in batch_ids:
            loss, d_loss, h_loss = model.loss(t, x, u[ids], args.lam, args.name)
            loss.backward(); optim.step(); optim.zero_grad()
        with torch.no_grad():
            u0 = val_u[:,0,:]
            pred_u = model.solve(t, x, u0)
            val_loss = ((pred_u - torch.flatten(val_u))**2).mean()
            
        # logging
        stats['train_loss'].append(loss.item())
        stats['data_loss'].append(d_loss.item())
        stats['hamiltonian_loss'].append(h_loss.item())
        stats['val_loss'].append(val_loss.item())
        if epoch % args.print_every == 0:
            print("epoch {}, time {:.5e}, train_loss {:.2e}, val_loss {:.2e}, h_loss {:.2e}"
                  .format(epoch, time.time()-t0, loss.item(), val_loss.item(), h_loss.item()))
            t0 = time.time()
        if val_loss < best_val_loss:
            best_model = copy.deepcopy(model)
            best_epoch = epoch; best_val_loss = val_loss.item(); best_train_loss = loss.item()

    return best_epoch, best_model, best_train_loss, best_val_loss, stats


if __name__ == "__main__":

    # setting
    args = get_args()
    device = torch.device('cuda:' + str(args.cuda_no) if torch.cuda.is_available() else 'cpu')
    print(device,args.data_no)
    i_dir = '../data/' + args.name + '/train_' + str(args.data_no) + '/' + str(args.s)
    tmp_dir = '../data/' + args.name + '/result_' + str(args.data_no)
    if args.Hamilton:
        save_dir = tmp_dir + '/' + str(args.s) + '/EnerReg/st_' + str(args.setting_no) + '/' + str(args.lam)
    else:
        save_dir = tmp_dir + '/' + str(args.s) + '/MLP/st_' + str(args.setting_no)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None

    # input
    filename = i_dir + '/data.pkl'
    data = std_io.pkl_read(filename)
    filename = '../data/' + args.name + '/train_' + str(args.data_no) + '/' + str(args.name) + '.json'
    with open(filename) as f:
        vs = json.load(f)

    # training data
    u = torch.tensor(data['u']).squeeze().float().to(device)
    t = torch.tensor(data['t']).float().to(device)
    x = torch.tensor(data['x']).float().to(device)

    # validation data
    val_u = torch.tensor(data['val_u']).squeeze().float().to(device)

    # learning
    t0 = time.time()
    epoch, model, train_loss, val_loss, stats = train()
    train_time = time.time() - t0
    
    # save
    path = '{}/model.tar'.format(save_dir)
    torch.save(model.state_dict(), path)
    
    path = '{}/model.json'.format(save_dir)
    with open(path, 'w') as f:
        json.dump(vars(args), f)

    path = '{}/result.csv'.format(save_dir)
    std_io.csv_write(path, np.array(['val_epoch',epoch,'train_loss',train_loss,
                                     'val_loss',val_loss,'train_time',train_time]))

    # learning curve
    filename = save_dir + '/learning_curve.pdf'
    x = np.arange(len(stats['train_loss']))
    std_vis.plot2(filename, x, [stats['train_loss'],stats['val_loss']],
                  'iteration','loss', ['train','validation'])

    filename = save_dir + '/dh_curve.pdf'
    std_vis.plot2(filename, x, [stats['data_loss'],stats['hamiltonian_loss']],
                  'iteration','loss', ['data','hamiltonian'])
