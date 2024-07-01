#!/bin/bash
#SBATCH --partition=gpulong
#SBATCH --job-name=mdiabetesMujoco$1
#SBATCH --output=/dev/null
#SBATCH --error=ERRORMUJOCO.err
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M
#SBATCH --constraint=gpu-40gb
#SBATCH --account=ailab
module load miniconda
module load racs-eb/1
module load Mesa/17.2.5-foss-2017b
conda activate mdiabetes
python /home/abutler9/ailab/mdiabetes-behavior-modeling/mujoco_exp.py --cuda True --env $1 --logging True --numHidden $2 --consecHidden $3 --seed $4 --statepred True --statemodel $5
