from experiment import Experiment
from utils.behavior_data import BehaviorData
import torch
import numpy as np
from utils.state_data import StateData
import argparse
import os


parser = argparse.ArgumentParser()

def toBool(x):
    return (str(x).lower() in ['true', '1', 't'])

# expanded states
parser.add_argument('-model', type=str, default="BasicNN")
parser.add_argument('-numWeeks', type=int, default=1)
parser.add_argument('-estate', type=toBool, default=False)
parser.add_argument('-includeState', type=toBool, default=False)
parser.add_argument('-fullQ', type=toBool, default=False)
parser.add_argument('-insertPreds', type=toBool, default=False)
parser.add_argument('-splitQ', type=toBool, default=False)
parser.add_argument('-splitM', type=toBool, default=False)
parser.add_argument('-catHist', type=toBool, default=False)
parser.add_argument('-knowEpochs', type=int, default=30)
parser.add_argument('-physEpochs', type=int, default=50)
parser.add_argument('-conEpochs', type=int, default=100)
parser.add_argument('-smooth', type=float, default=0)
parser.add_argument('-noise', type=float, default=0.07)
parser.add_argument('-learning_rate', type=float, default=0.07)

args = parser.parse_args()

model = args.model

respond_perc = .5

conSched = [args.conEpochs]
knowSched = [args.knowEpochs]
physSched = [args.physEpochs]

learning_rate = args.learning_rate

smooth, noise = args.smooth, args.noise

splitQ, splitM = args.splitQ, args.splitM

catHist = args.catHist

numWeeks = args.numWeeks

insertPreds = args.insertPreds

loss_fn = "CrossEntropyLoss"

include_state, estate, fullq = args.includeState, args.estate, args.fullQ

epochs=max([args.conEpochs, args.knowEpochs, args.physEpochs])

if "LSTM" in model:
    stateweek = 1
else:
    stateweek = 500

if epochs > 400:
    hiddenSize = 100
    lrmult = 1.0
else:
    hiddenSize = 25
    lrmult = 0.9

for seed in range(30):
    np.random.seed(seed)
    torch.manual_seed(seed)
    e = Experiment(
        modelSplit = splitM,
        numValFolds = 5,
        epochsToUpdateLabelMods = 10,
        knowSchedule = knowSched,
        consumpSchedule = conSched,
        physSchedule = physSched,
        data_kw={"minw": 2,
                "maxw": 31,
                "include_state": include_state,
                "include_pid": False,
                "expanded_states": estate,
                "top_respond_perc": respond_perc,
                "full_questionnaire": fullq,
                "num_weeks_history": numWeeks,
                "insert_predictions": insertPreds,
                "split_model_features": splitM,
                "split_weekly_questions": splitQ,
                "category_specific_history": catHist,
                "max_state_week": stateweek,
                },
        model=model,
        model_kw={
            "lossfn": loss_fn,
            # "lossfn": "NDCG",
            # "lossfn": "CrossEntropyLoss",
            "hidden_size": hiddenSize, 
            "lr_step_mult": lrmult, 
            "lr_step_epochs": 60,
            "opt_kw": {
                "lr": learning_rate
            },
            "labelSmoothPerc": smooth,
            "gaussianNoiseStd": noise,
            "splitModel": splitM,
            "splitWeeklyQuestions": splitQ
        },
        train_kw={
            "epochs": epochs,
            "n_subj": 500,
            "rec_every": 5,
        })
    # torch.autograd.set_detect_anomaly(True)
    report = e.run()



    individual_test_scores, labels = e.report_scores_individual_test()
    individual_train_scores, labels = e.report_scores_individual_train()


    if epochs > 400:
        dire = f"/home/abutler9/ailab/mdiabetes-behavior-modeling/experiment_output_long/{model}/"
    else:
        dire = f"/home/abutler9/ailab/mdiabetes-behavior-modeling/experiment_output/{model}/"

    if (not os.path.exists(dire)):
        os.makedirs(dire)


    fileprefix = f"W{numWeeks}LR{learning_rate}Resp{respond_perc}States{int(include_state)}Expanded{int(estate)}Full{int(fullq)}CHist{int(catHist)}Pred{int(insertPreds)}Smooth{smooth}Noise{noise}Split{int(splitQ)}{int(splitM)}"
    np.savetxt(f"{dire}TRAINMETRICS-{fileprefix}S{seed}.csv", report["train_metrics"], delimiter = ',', header = ','.join(report['metric_labels']))
    np.savetxt(f"{dire}TESTMETRICS-{fileprefix}S{seed}.csv", report["test_metrics"], delimiter = ',', header = ','.join(report['metric_labels']))
    np.savetxt(f"{dire}IDVDTESTMETRICS-{fileprefix}S{seed}.csv", individual_test_scores, delimiter = ',', header = ','.join(report['metric_labels']))
    np.savetxt(f"{dire}IDVDTRAINMETRICS-{fileprefix}S{seed}.csv", individual_train_scores, delimiter = ',', header = ','.join(report['metric_labels']))
    np.savetxt(f"{dire}TRAINLOSSES-{fileprefix}S{seed}.csv", report["loss"], delimiter = ',')

    preds1, preds2, preds3 = e.get_class_predictions(False)


    np.savetxt(f"{dire}TRAINPREDS1-{fileprefix}S{seed}.csv", preds1, delimiter = ',')
    np.savetxt(f"{dire}TRAINPREDS2-{fileprefix}S{seed}.csv", preds2, delimiter = ',')
    np.savetxt(f"{dire}TRAINPREDS3-{fileprefix}S{seed}.csv", preds3, delimiter = ',')


    preds1, preds2, preds3 = e.get_class_predictions(True)

    np.savetxt(f"{dire}TESTPREDS1-{fileprefix}S{seed}.csv", preds1, delimiter = ',')
    np.savetxt(f"{dire}TESTPREDS2-{fileprefix}S{seed}.csv", preds2, delimiter = ',')
    np.savetxt(f"{dire}TESTPREDS3-{fileprefix}S{seed}.csv", preds3, delimiter = ',')



    writer = open(f"{dire}FINALTRAINMETRICS-{fileprefix}.csv", "a")
    writer.write(",".join([str(loss) for loss in report["train_metrics"][-1, :]]))
    writer.write("\n")
    writer.close()

    writer = open(f"{dire}FINALTESTMETRICS-{fileprefix}.csv", "a")
    writer.write(",".join([str(loss) for loss in report["test_metrics"][-1, :]]))
    writer.write("\n")
    writer.close()



