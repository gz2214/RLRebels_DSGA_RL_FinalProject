#!/bin/bash
#SBATCH --nodes=1                      
#SBATCH --cpus-per-task=8                
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=16:00:00
#SBATCH --mem=55GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ki2130@nyu.edu
#SBATCH --job-name=cnn-bkout-pong
#SBATCH --output=cnn-bkout-pong.out


module purge;
source ../rl3-venv/bin/activate;
export OMP_NUM_THREADS=1;
python3 ../transferLearning.py pretrained_model_ALE_Breakout-v5_DQN_CONV_10000 ALE/Pong-v5 1000

