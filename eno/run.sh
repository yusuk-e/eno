#!/bin/bash

rm -r out
mkdir out

st=0
for lr_op in 1e-3
do
    for wd_op in 0
    do
	qsub -o ./out -g gcf51167 job_MLP.sh "${lr_op}" "${wd_op}" "${st}" &
	((st++))
    done
done

st=0
for lr_op in 1e-3
do
    for wd_op in 0
    do
	for lr_H in 1e-4
	do
	    for wd_H in 0
	    do
		for G in 0 1 2
		do
		    for lam in 1e-4 1e-5 1e-6 1e-7 1e-8
		    do
			qsub -o ./out -g gcf51167 job_HNO.sh "${lr_op}" "${wd_op}" "${lr_H}" "${wd_H}" "${lam}" "${st}" "${G}" &
		    done
		    ((st++))		    
		done
	    done
	done
    done
done
