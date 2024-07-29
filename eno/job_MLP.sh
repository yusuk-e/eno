#!/bin/bash
#$-l rt_G.small=1
#$-l h_rt=24:00:00
#$-j y
#$-cwd
source ~/.pyenv/versions/miniforge3-latest/etc/profile.d/conda.sh
conda activate hno
cd ~ENO/ENO

for s in 0 1 2
do
    for d in 1 2 3
    do
	for name in kdv ch
	do
	    python train.py --name "${name}" --s "${s}" --data_no "${d}" --epochs 5000 --lr_op $1 --wd_op $2 --hidden_dim 200 --batch_size 30 --setting_no $3
	    python test.py --name "${name}" --s "${s}" --data_no "${d}" --setting_no $3
	done
    done
done
