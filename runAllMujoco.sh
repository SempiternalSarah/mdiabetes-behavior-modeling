hiddens=(3 2 1 0)
# hiddens=(0)
# envs=("Hopper-v2" "Walker2d-v2")
envs=("Ant-v2" "Humanoid-v2")
statepred=(0)
cp mujoco_exp.py NOTOUCH_mujoco_exp.py
# hiddenTime=(5)
for hidden in "${hiddens[@]}"
do
	for env in "${envs[@]}"
	do
		for sp in "${statepred[@]}"
		do
			for seed in {1..5}
			do
				sbatch runMujoco.srun $env $hidden $seed $sp
			done
			sleep 1
		done
	done
done
