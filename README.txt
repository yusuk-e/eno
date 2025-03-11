Create emvironment:
conda create --name eno --file eno.yml


To generate data in 'data' directory:
bash job.sh


To run ENO in 'ENO' directory:
python train.py --name kdv --s 0 --data_no 1 --Hamilton --lam 1e-5 --G 0 --setting_no 0
       --name: system
       --s: dataset id
       --data_no: 1 (Nx:10,Nt:10), 2 (Nx:15,Nt:15), 3 (Nx:25,Nt:25)
       --lam: hyperparameter
       --G: differential operator 0 (-1), 1 (\partial/\partial x), 2 (\partial^2/\partial x^2)
       --setting_no: change as needed       
python test.py --name kdv --s 0 --data_no 1 --Hamilton --lam 1e-5 --setting_no 0


To run Vanilla NO in 'ENO' directory:
python train.py --name kdv --s 0 --data_no 1 --lam 1e-5 --G 0 --setting_no 0
       --name: system
       --s: dataset id
       --data_no: 1 (Nx:10,Nt:10), 2 (Nx:15,Nt:15), 3 (Nx:25,Nt:25)
       --lam: hyperparameter
       --G: differential operator 0 (-1), 1 (\partial/\partial x), 2 (\partial^2/\partial x^2)
       --setting_no: change as needed       
python test.py --name kdv --s 0 --data_no 1 --lam 1e-5 --setting_no 0


To run ENO (fixied) in 'ENO_fixed' directory:
python train.py --name kdv --s 0 --data_no 1 --Hamilton --lam 1e-5 --G 0 --setting_no 0
       --name: system
       --s: dataset id
       --data_no: 1 (Nx:10,Nt:10), 2 (Nx:15,Nt:15), 3 (Nx:25,Nt:25)
       --lam: hyperparameter
       --G: differential operator 0 (-1), 1 (\partial/\partial x), 2 (\partial^2/\partial x^2)
       --setting_no: change as needed       
python test.py --name kdv --s 0 --data_no 1 --Hamilton --lam 1e-5 --setting_no 0


To run DeepONet in 'DeepNet' directory:
python train.py --name kdv --s 0 --data_no 1 --setting_no 0
       --name: system
       --s: dataset id
       --data_no: 1 (Nx:10,Nt:10), 2 (Nx:15,Nt:15), 3 (Nx:25,Nt:25)
       --setting_no: change as needed       
python test.py --name kdv --s 0 --data_no 1 --setting_no 0


To run FNO in 'FNO' directory:
python train.py --name kdv --s 0 --data_no 1 --setting_no 0
       --name: system
       --s: dataset id
       --data_no: 1 (Nx:10,Nt:10), 2 (Nx:15,Nt:15), 3 (Nx:25,Nt:25)
       --setting_no: change as needed       
python test.py --name kdv --s 0 --data_no 1 --setting_no 0


To run EnerReg in 'EnerReg' directory:
python train.py --name kdv --s 0 --data_no 1 --Hamilton --lam 1e-5 --setting_no 0
       --name: system
       --s: dataset id
       --data_no: 1 (Nx:10,Nt:10), 2 (Nx:15,Nt:15), 3 (Nx:25,Nt:25)
       --lam: hyperparameter
       --setting_no: change as needed
python test.py --name kdv --s 0 --data_no 1 --Hamilton --lam 1e-5 --setting_no 0

