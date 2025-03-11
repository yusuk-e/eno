#!/bin/bash

python gen_data.py --name kdv --N 100 --T 0.5 --Nt 1001 --X 10. --Nx 101
python gen_data.py --name kdv --N 1000 --T 0.5 --Nt 1001 --X 10. --Nx 101 --train --S 5 --val_rate 0.1 --data_no 0
python downsampling.py --name kdv

python gen_data.py --name ch --N 100 --T 0.05 --Nt 1001 --X 1. --Nx 101
python gen_data.py --name ch --N 1000 --T 0.05 --Nt 1001 --X 1. --Nx 101 --train --S 5 --val_rate 0.1 --data_no 0
python downsampling.py --name ch
