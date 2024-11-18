#!/bin/bash

#SBATCH --job-name=trainModel             
#SBATCH --nodes=1                                 
#SBATCH --partition=killable 
#SBATCH --nodelist=n-601
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10                      
#SBATCH --ntasks-per-node=1                       
#SBATCH --output=trainModel.out         
#SBATCH --error=trainModel.err  

python3 trainModel.py