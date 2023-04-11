models=("BasicNN" "AdaptableLSTM" "LogisticRegressor")
lr=(0.003 0.005 0.001)
# models=("BasicNN" "LogisticRegressor")
# lr=(0.003 0.001)
# models=("AdaptableLSTM")
# lr=(0.005)
numWeeksList=(2)
estatesList=(True False)
includeStates=True
fullQsList=(True False)
insertPredsList=(True)
splitQsList=(True False)
splitMsList=(True False)
catHistList=(True False)
knowEpochsList=(40 30 40)
physEpochsList=(20 30 20)
conEpochsList=(60 90 220)
smooth=0.0
noiseList=(0.00 0.04 0.10)
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
