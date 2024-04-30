#!/bin/bash
#SBATCH --nodes=1                      
#SBATCH --cpus-per-task=8                
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=16:00:00
#SBATCH --mem=55GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ki2130@nyu.edu
#SBATCH --job-name=cnn-pong-bkout
#SBATCH --output=cnn-pong-bkout.out


module purge;
source ../rl3-venv/bin/activate;
export OMP_NUM_THREADS=1;
python3 ../transferLearning.py pretrained_model_ALE_Pong-v5_DQN_CONV_10000 ALE/Breakout-v5 1000

