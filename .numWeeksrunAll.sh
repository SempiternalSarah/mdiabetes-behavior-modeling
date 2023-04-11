models=("BasicNN" "AdaptableLSTM" "LogisticRegressor")
# models=("AdaptableLSTM")
numWeeksList=(1 2 3 4 5 30)
estates=True
includeStates=True
fullQs=True
insertPreds=True
splitQs=True
splitMs=True
catHist=True
knowEpochsList=(40 30 40)
physEpochsList=(20 30 20)
conEpochsList=(60 90 220)
smooth=0.0
noise=0.07
lr=(0.003 0.005 0.001)
for numWeeks in "${numWeeksList[@]}"
do
	for i in "${!models[@]}"
	do
		sbatch runOne.srun ${models[i]} $numWeeks $estates $includeStates $fullQs $insertPreds $splitQs $splitMs $catHist ${knowEpochsList[i]}  ${physEpochsList[i]}  ${conEpochsList[i]} $smooth $noise ${lr[i]}
	done
done
