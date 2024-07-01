models=("BasicNN" "AdaptableLSTM")
lr=(0.00025 0.0002)
# models=("AdaptableLSTM")
# lr=(0.001)
numWeeks=3
estates=True
includeStates=True
fullQs=True
insertPreds=False
sep=False
splitQs=True
splitMsList=(True False)
catHistList=(False)
# knowEpochsList=(150 60)
# physEpochsList=(70 30)
# conEpochsList=(150 60)
knowEpochsList=(240 260)
physEpochsList=(180 200)
conEpochsList=(240 260)
regressionList=(False)
# knowEpochsList=(1000 1000)
# physEpochsList=(1000 1000)
# conEpochsList=(1000 1000)
smooth=0
hierList=(None)
noise=0.05
transformer=True
nClusterList=(2 3 4 5)
clusterMethodList=("Gaussian" "Kmeans" "Spectral")
clusterByList=("Demographics" "Initial")
numSeeds=5
nrc=False
for regression in "${regressionList[@]}"
do
	for hier in "${hierList[@]}"
	do
		for catHist in "${catHistList[@]}"
		do
			for clusterBy in "${clusterByList[@]}"
			do
				for numClusters in "${nClusterList[@]}"
				do
					for clusterMethod in "${clusterMethodList[@]}"
					do
                        for splitMs in "${splitMsList[@]}"
                        do
                            for i in "${!models[@]}"
                            do
                                sbatch runOne.srun ${models[i]} $numWeeks $estates $includeStates $fullQs $insertPreds $splitQs $splitMs $catHist ${knowEpochsList[i]}  ${physEpochsList[i]}  ${conEpochsList[i]} $smooth $noise ${lr[i]} $hier $sep $regression $transformer $numSeeds $numClusters $clusterBy $clusterMethod $nrc

									# echo ${models[i]} $numWeeks $estates $includeStates $fullQs $insertPreds $splitQs $splitMs $catHist ${knowEpochsList[i]}  ${physEpochsList[i]}  ${conEpochsList[i]} $smooth $noise ${lr[i]}
                            done
                        done
					done
					sleep 1
				done
			done
		done
	done
done
