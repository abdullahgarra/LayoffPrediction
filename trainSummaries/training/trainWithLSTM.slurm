#!/bin/bash

#SBATCH --job-name=LSTM             
#SBATCH --nodes=1                                 
#SBATCH --partition=killable 
#SBATCH --nodelist=n-350
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10                      
#SBATCH --ntasks-per-node=1                     
#SBATCH --output=trainWithLSTM.out         
#SBATCH --error=trainWithLSTM.err  

python3 trainWithLSTM.py