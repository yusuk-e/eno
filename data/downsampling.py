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

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--name', default='kdv', type=str)
    return parser.parse_args()


def down_sampling():
    ds_data = {}
    t_ratio = int((vs['Nt']-1) / (Nt-1))
    t_ids = [i for i in range(len(data['t']))][::t_ratio]
    x_ratio = int((vs['Nx']-1) / (Nx-1))
    x_ids = [i for i in range(len(data['x']))][::x_ratio]
    ds_data['x'] = data['x'][x_ids]
    ds_data['t'] = data['t'][t_ids]
    for k in ['val_e', 'e', 'val_m', 'm']:
        ds_data[k] = data[k][:,t_ids]
    for k in ['val_u', 'u', 'val_du', 'du']:
        tmp = data[k][:,t_ids,:]
        ds_data[k] = tmp[:,:,x_ids]
    return ds_data
    

if __name__ == "__main__":

    # setting
    args = get_args()
    for s in range(5):
        i_dir = args.name + '/train_0' + '/' + str(s)
        filename = i_dir + '/data.pkl'
        data = std_io.pkl_read(filename)
        filename = args.name + '/train_0/' + str(args.name) + '.json'
        with open(filename) as f:
            vs = json.load(f)

        counter = 0
        for Nt, Nx in [[11,11],[16,16],[26,26]]:
            counter += 1
            ds_data = down_sampling()
            ds_vs = vs.copy()
            ds_vs['Nt'] = ds_data['u'].shape[1]
            ds_vs['Nx'] = ds_data['u'].shape[2]

            save_dir = args.name + '/train_' + str(counter)
            os.makedirs(save_dir) if not os.path.exists(save_dir) else None

            save_dir_s = save_dir + '/' + str(s)
            os.makedirs(save_dir_s) if not os.path.exists(save_dir_s) else None
            std_io.pkl_write(save_dir_s + '/data.pkl', ds_data)

            save_dir_v = save_dir_s + '/vis'
            os.makedirs(save_dir_v) if not os.path.exists(save_dir_v) else None
            visualization(save_dir_v, ds_data['t'], ds_data['u'][:10],
                          ds_data['e'][:10], ds_data['m'][:10], vs['T'], vs['X'])

            save_dir_v = save_dir_s + '/vis_val'
            os.makedirs(save_dir_v) if not os.path.exists(save_dir_v) else None
            visualization(save_dir_v, ds_data['t'], ds_data['val_u'][:10],
                          ds_data['val_e'][:10], ds_data['val_m'][:10], vs['T'], vs['X'])

            filename = '{}/{}.json'.format(save_dir, args.name)
            with open(filename, 'w') as f:
                json.dump(ds_vs, f)

        
