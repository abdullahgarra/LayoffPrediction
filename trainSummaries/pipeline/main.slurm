#!/bin/bash

#SBATCH --job-name=Embedding             
#SBATCH --nodes=1                                 
#SBATCH --partition=killable 
#SBATCH --nodelist=n-602
#SBATCH --gpus=1
#SBATCH --cpus-per-task=50                      
#SBATCH --ntasks-per-node=2                       
#SBATCH --output=Embedding.out         
#SBATCH --error=Embedding.err  

python3 main.py