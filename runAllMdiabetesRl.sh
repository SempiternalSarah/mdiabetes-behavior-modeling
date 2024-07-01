statepred=(0 1)
for sp in "${statepred[@]}"
do
	for seed in {1..10}
	do
		sbatch runMdiabetesRl.srun $seed $sp
	done
	sleep 1
done
