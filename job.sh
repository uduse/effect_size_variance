#!/bin/bash
#SBATCH --time=10
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-4
#SBATCH --output=tmp-%A_%a.out
##SBATCH --output=/dev/null

# set -e

cd ~/project/effect_size_variance
source ~/spinup/bin/activate
JOB_TIME=$(date +"%m_%d___%H_%M")
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

EPOCHS=10
STEPS_PER_EPOCH=4000
ENV_SEED=$((${SLURM_ARRAY_JOB_ID} + ${SLURM_ARRAY_TASK_ID}))

python script.py --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --algo vpg --seed $END_SEED  \ 
				--seed_weight_init $SLURM_ARRAY_TASK_ID --output_dir log/$DIR_NAME --output_fname ${SLURM_ARRAY_TASK_ID}_vpg.out  
python script.py --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --algo ppo --seed $END_SEED  \
				--seed_weight_init $SLURM_ARRAY_TASK_ID --output_dir log/$DIR_NAME --output_fname ${SLURM_ARRAY_TASK_ID}_ppo_c.out
python script.py --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --algo ppo --seed $END_SEED  \
				--seed_weight_init $((${SLURM_ARRAY_TASK_ID} + 1000)) --output_dir log/$DIR_NAME --output_fname ${SLURM_ARRAY_TASK_ID}_ppo_uc.out
