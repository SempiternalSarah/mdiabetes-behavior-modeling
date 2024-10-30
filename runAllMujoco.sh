hiddens=(3 2 1)
# hiddens=(0)
# envs=("Ant-v2" "HalfCheetah-v2" "Hopper-v2" "Walker2d-v2")
# envs=("Ant-v2" "HalfCheetah-v2")
envs=("Humanoid-v2")
# envs=("HalfCheetah-v2")
# envs=("Ant-v2")
statepred=(1)
cp mujoco_exp.py NOTOUCH_mujoco_exp.py
# hiddenTime=(5)
for hidden in "${hiddens[@]}"
do
	for env in "${envs[@]}"
	do
		for sp in "${statepred[@]}"
		do
			for seed in {6..10}
			do
				sbatch runMujocoGPU.srun $env $hidden $seed $sp
			done
			sleep 1
		done
	done
done
