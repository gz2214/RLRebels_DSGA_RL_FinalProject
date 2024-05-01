#!/bin/bash
#SBATCH --nodes=1                      
#SBATCH --cpus-per-task=8                
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=6:00:00
#SBATCH --mem=55GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ki2130@nyu.edu
#SBATCH --job-name=mlp-bkout-pong
#SBATCH --output=mlp-bkout-pong.out


module purge;
source ../rl3-venv/bin/activate;
export OMP_NUM_THREADS=1;
python3 ../transferLearning.py pretrained_model_ALE_Breakout-v5_DQN_MLP_1000 ALE/Pong-v5 1000

