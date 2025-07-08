#!/bin/bash
#SBATCH --job-name=hseSPHTest                # create a short name for your job
#SBATCH --nodes=1                            # node count
#SBATCH --ntasks=1                           # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=4                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1                         # number of gpus per node
#SBATCH --mem-per-cpu=8G                     # memory per cpu-core (4G is default)
#SBATCH --time=100:00:00                      # total run time limit (HH:MM:SS)






## Fix Persmission error
cd $HOME/DualSPHysics
bash *.sh

## Run DualSPHysics Model
cd $OLDPWD
bash run.sh
