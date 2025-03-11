import pdb
import os, sys
import json
import argparse
import scipy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
from utils import *
from t_io import standard_io as std_io

#os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
#num_threads = '1'
#os.environ['OMP_NUM_THREADS'] = num_threads
#os.environ['MKL_NUM_THREADS'] = num_threads
#os.environ['NUMEXPR_NUM_THREADS'] = num_threads


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--name', default='kdv', type=str)
    parser.add_argument('--N', default=25, type=int)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--Nt', default=101, type=int)
    parser.add_argument('--X', default=10., type=float)
    parser.add_argument('--Nx', default=50, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--S', default=5, type=int)
    parser.add_argument('--val_rate', default=0.2, type=float)
    parser.add_argument('--data_no', default=0, type=int)
    return parser.parse_args()

def gen_data():
    data = {}
    u, e, m = [], [], []
    t = np.array(args.T * np.arange(args.Nt) / (args.Nt-1))
    x = args.X * np.arange(args.Nx) / (args.Nx-1)
    u0 = pde.get_init(args.N, x, args.X)
    result = scipy.integrate.solve_ivp(fun=pde.forward, t_span=[0,args.T], y0=u0.flatten(),
                                       method='DOP853', t_eval=t, rtol=1e-12, atol=1e-14)
    u = result.y.reshape([args.N, args.Nx, args.Nt]).transpose(0,2,1)
    du = pde.forward(t, u)
    du = du.reshape([args.N, args.Nt, args.Nx])
    e = pde.H(result.y)
    m = u.sum(axis=2)

    data['t'] = t
    data['x'] = x
    data['u'] = u
    data['du'] = du
    data['e'] = e
    data['m'] = m
    return data

def split(s):
    train_split_id = int(args.N * (1-args.val_rate))
    ids = [i for i in range(args.N)]
    np.random.seed(s)
    np.random.shuffle(ids)
    train_ids = ids[:train_split_id]; val_ids = ids[train_split_id:]
    split_data = {}
    for k in ['u', 'du', 'e', 'm']:
        split_data['val_' + k], split_data[k] = data[k][val_ids], data[k][train_ids]
    for k in ['t', 'x']:
        split_data[k] = data[k]
    return split_data


if __name__ == "__main__":

    args = get_args()
    if args.name == 'kdv':
        pde = kdv(args.N, args.Nx, args.X/args.Nx)
    elif args.name == 'ch':
        pde = ch(args.N, args.Nx, args.X/args.Nx)

    data = gen_data()

    if args.train:
        save_dir = args.name + '/train_' + str(args.data_no)
        os.makedirs(save_dir) if not os.path.exists(save_dir) else None
        for s in range(args.S):
            print(s)
            s_data = split(s)
            save_dir_s = save_dir + '/' + str(s)
            os.makedirs(save_dir_s) if not os.path.exists(save_dir_s) else None
            std_io.pkl_write(save_dir_s + '/data.pkl', s_data)
            
            save_dir_v = save_dir_s + '/vis'
            os.makedirs(save_dir_v) if not os.path.exists(save_dir_v) else None
            visualization(save_dir_v, s_data['t'], s_data['u'][:10],
                          s_data['e'][:10], s_data['m'][:10], args.T, args.X)

            save_dir_v = save_dir_s + '/vis_val'
            os.makedirs(save_dir_v) if not os.path.exists(save_dir_v) else None
            visualization(save_dir_v, s_data['t'], s_data['val_u'][:10],
                          s_data['val_e'][:10], s_data['val_m'][:10], args.T, args.X)
            
        filename = '{}/{}.json'.format(save_dir, args.name)
        with open(filename, 'w') as f:
            json.dump(vars(args), f)
    else:
        save_dir = args.name + '/test'
        os.makedirs(save_dir) if not os.path.exists(save_dir) else None
        std_io.pkl_write(save_dir + '/data.pkl', data)
        visualization(save_dir, data['t'], data['u'][:10], data['e'][:10], data['m'][:10], args.T, args.X)
        filename = '{}/{}.json'.format(save_dir, args.name)
        with open(filename, 'w') as f:
            json.dump(vars(args), f)
