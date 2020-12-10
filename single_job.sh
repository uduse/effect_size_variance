#!/bin/bash
#SBATCH --time=16
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-16
#SBATCH --output=tmp-%A_%a.out
##SBATCH --output=/dev/null

# set -e

cd ~/project/effect_size_variance
source ~/spinup/bin/activate
JOB_TIME=$(date +"_%m_%d___%H_%M")
DIR_NAME=$SLURM_ARRAY_JOB_ID
if [[ $SLURM_ARRAY_TASK_ID -le 1 ]]
then 
	echo "DIR_NAME=$DIR_NAME"
	mkdir -p log/$DIR_NAME
	cp script.py job.sh log/$DIR_NAME
	touch log/$DIR_NAME/$JOB_TIME.txt
else
	while [[ ! -d "log/$DIR_NAME" ]]
	do
		echo "Sleeping..."
		sleep 1
	done
fi
python script.py --algo vpg --seed $SLURM_ARRAY_JOB_ID  --seed_weight_init $SLURM_ARRAY_TASK_ID > log/$DIR_NAME/$SLURM_ARRAY_TASK_ID.out