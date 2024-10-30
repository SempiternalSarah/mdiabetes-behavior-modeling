statepred=(0 1)
eqs=(1)
for sp in "${statepred[@]}"
do
	for eq in "${eqs[@]}"
	do
		for seed in {1..40}
		do
			sbatch runMdiabetesRl.srun $seed $sp $eq
		done
		sleep 1
	done
done
