#!/bin/bash

#SBATCH --job-name=trainAll             
#SBATCH --nodes=1                                 
#SBATCH --partition=killable 
#SBATCH --nodelist=n-601
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10                      
#SBATCH --ntasks-per-node=2                       
#SBATCH --output=trainAll.out         
#SBATCH --error=trainAll.err  

python3 trainAll.py