hiddens=(3)
# hiddens=(0)
# envs=("Hopper-v2" "Walker2d-v2")
envs=("Ant-v2")
statepred=(1)
# startLearning=(10000 50000 100000)
startLearning=(20000)
# qlrs=(0.0003 0.0006 0.0001)
# alrs=(0.0003 0.0006 0.0001)
qlrs=(0.0003)
alrs=(0.0003)
hsizes=(64 128 256)
slr=.001
contexts=(20 50 75)
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
				for qlr in "${qlrs[@]}"
				do
					for alr in "${qlrs[@]}"
					do
						for context in "${contexts[@]}"
						do
							for hsize in "${hsizes[@]}"
							do
								for sl in "${startLearning[@]}"
								do
									sbatch runOneAnt.srun $env $hidden $seed $sp $qlr $alr $slr $sl $hsize $context
								done
							done
						done
					done
					sleep 1
				done
			done
		done
	done
done
