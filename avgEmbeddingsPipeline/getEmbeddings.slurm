#!/bin/bash

#SBATCH --job-name=Embedding             
#SBATCH --nodes=1                                 
#SBATCH --partition=killable 
#SBATCH --nodelist=n-601
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20                      
#SBATCH --ntasks-per-node=1                    
#SBATCH --output=getEmbedding.out         
#SBATCH --error=getEmbedding.err  

python3 getEmbeddings.py