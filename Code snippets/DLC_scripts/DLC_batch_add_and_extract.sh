#!/bin/bash
#
#SBATCH --job-name=DLC_extract_test
#SBATCH --output=output/R-%x.%j.txt
#SBATCH --error=output/ERR-%x.%j.txt
#SBATCH --account=behavior
#
#SBATCH --partition=debug-gpu
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#
#
module load EasyBuild/2022a

module load devel/python/Anaconda3-2022.05
conda activate DLC_dev
module load CUDA/11.7.0 TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

cd /gpfs/projects/acad/behavior/softs/DLC_scripts

python DLC_add_videos.py $1
python DLC_extract_frames.py $1