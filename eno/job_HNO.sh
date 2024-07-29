#!/bin/bash
#$-l rt_AG.small=1
#$-l h_rt=48:00:00
#$-j y
#$-cwd
source ~/.pyenv/versions/miniforge3-latest/etc/profile.d/conda.sh
conda activate hno
cd ~/ENO/ENO

for s in 0 1 2
do
    for d in 1 2 3
    do
	for name in kdv ch
	do
	    python train.py --name "${name}" --s "${s}" --data_no "${d}" --epochs 5000 --lr_op $1 --wd_op $2 --lr_H $3 --wd_H $4 --hidden_dim 200 --Hamilton --lam $5 --batch_size 30 --col_size 200 --setting_no $6 --G $7
	    python test.py --name "${name}" --s "${s}" --data_no "${d}" --Hamilton --lam $5 --setting_no $6
	done
    done
done
