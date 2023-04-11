# models=("BasicNN" "AdaptableLSTM" "LogisticRegressor")
# lr=(0.003 0.005 0.001)
models=("BasicNN")
lr=(0.007)
# models=("AdaptableLSTM")
# lr=(0.005)
numWeeksList=(2)
estatesList=(True)
includeStates=True
fullQsList=(False)
insertPredsList=(False)
splitQsList=(False)
splitMsList=(True)
catHistList=(False)
knowEpochsList=(500)
physEpochsList=(500)
conEpochsList=(500)
smooth=0.0
noiseList=(0.00)
for noise in "${noiseList[@]}"
do
	for numWeeks in "${numWeeksList[@]}"
	do
		for catHist in "${catHistList[@]}"
		do
			for fullQs in "${fullQsList[@]}"
			do
				for estates in "${estatesList[@]}"
				do
					for insertPreds in "${insertPredsList[@]}"
					do
						for splitQs in "${splitQsList[@]}"
						do
							for splitMs in "${splitMsList[@]}"
							do
								for i in "${!models[@]}"
								do
									sbatch runOne.srun ${models[i]} $numWeeks $estates $includeStates $fullQs $insertPreds $splitQs $splitMs $catHist ${knowEpochsList[i]}  ${physEpochsList[i]}  ${conEpochsList[i]} $smooth $noise ${lr[i]}
									# echo ${models[i]} $numWeeks $estates $includeStates $fullQs $insertPreds $splitQs $splitMs $catHist ${knowEpochsList[i]}  ${physEpochsList[i]}  ${conEpochsList[i]} $smooth $noise ${lr[i]}
								done
							done
						done
					done
				done
			done
		done
	done
done
