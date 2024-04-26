#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --time 4:00:00
#SBATCH --output=dqn_base_pong.out
#SBATCH --exclude=a100-4003
#SBATCH --mail-type=ALL
#SBATCH --partition=a100_dev
#SBATCH --mail-user=sh7008@nyu.edu
#SBATCH --job-name=dqn_pong


module purge;
module load condaenvs/gpu/a100_machinelearning;
source /gpfs/share/apps/anaconda3/gpu/5.2.0/etc/profile.d/conda.sh
conda activate rl;
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$scratch/miniconda3/lib/;
#export PATH=/gpfs/scratch/wz727/miniconda3/bin:$PATH;
#echo "Which Python: $(which python)"
#python -c "import sys; print(sys.path)"
python3 train.py "$1" "$2" "$3"
