from utils.behavior_data import BehaviorData
import torch
import numpy as np
import importlib

class Experiment:
    
    def __init__(self, data_kw={}, model="BasicLSTM", model_kw={}, train_kw={}, numValFolds=5, epochsToUpdateLabelMods=5, stateZeroEpochs=0, modelSplit=False):
        # data_kw:  dict of keyword arguments to BehaviorData instance
        # model_kw: dict of keyword arguments for Model instance
        # train_kw: dict of keyword arguments for training loop
        self.numValFolds = numValFolds
        self.stateZeroEpochs = stateZeroEpochs
        self.data_kw = data_kw
        self.model_name = model
        self.model_kw = model_kw
        self.train_kw = train_kw
        # similar to DQN - update label modifications based on network predictions
        # every x epochs
        # used to replace non responses with predicted responses
        self.epochsToUpdateLabelMods = epochsToUpdateLabelMods

        self.bd = BehaviorData(**data_kw)
        if modelSplit and "LSTM" not in model:
            self.modelSplit = True
            self.physicalModel = self._get_model()(
                input_size=self.bd.dimensions[0],
                output_size=self.bd.dimensions[1] // 2,
                **model_kw,
            )
            self.knowledgeModel = self._get_model()(
                input_size=self.bd.dimensions[0],
                output_size=self.bd.dimensions[1] // 2,
                **model_kw,
            )
            self.consumptionModel = self._get_model()(
                input_size=self.bd.dimensions[0],
                output_size=self.bd.dimensions[1] // 2,
                **model_kw,
            )
            # define one model for shared functionality (like score reporting and optimizer creation)
            self.model = self.physicalModel
        else:
            self.modelSplit = False
            self.model = self._get_model()(
                input_size=self.bd.dimensions[0],
                output_size=self.bd.dimensions[1],
                **model_kw,
            )
        

    def runValidation(self):
        # Train the model, report parameters and results
        rep = self.__dict__
        rep = {k: v for k,v in rep.items() if "_kw" in k}
        train_loss, train_metrics, val_metrics, labels = self.train_validation()
        # results = self.evaluate()
        rep["params"] = {
            "data_kw": self.data_kw,
            "model_name": self.model_name,
            "model_kw": self.model_kw,
            "train_kw": self.train_kw,
        }
        # rep["results"] = results
        rep["loss"] = train_loss
        rep["train_metrics"] = train_metrics
        rep["metric_labels"] = labels
        rep["test_metrics"] = val_metrics
        return rep

    def run(self):
        # Train the model, report parameters and results
        rep = self.__dict__
        rep = {k: v for k,v in rep.items() if "_kw" in k}
        train_loss, train_metrics, test_metrics, labels = self.train()
        # results = self.evaluate()
        epochs = self.train_kw.get("epochs", 1)
        rec_every = self.train_kw.get("rec_every", 5)
        rep["rec_epochs"] = np.arange(start=0, stop=epochs, step=rec_every)
        rep["rec_epochs"] = np.append(rep["rec_epochs"], epochs - 1)
        rep["params"] = {
            "data_kw": self.data_kw,
            "model_name": self.model_name,
            "model_kw": self.model_kw,
            "train_kw": self.train_kw,
        }
        # rep["results"] = results
        rep["loss"] = train_loss
        rep["train_metrics"] = train_metrics
        rep["metric_labels"] = labels
        rep["test_metrics"] = test_metrics
        return rep

    def train_validation(self):
        # Loop over data and train model on each batch
        # Returns matrix of loss for each participant
        stored_losses = []
        train_metrics = []
        test_metrics = []
        epochs = self.train_kw.get("epochs", 1)
        rec_every = self.train_kw.get("rec_every", 5)
        foldsets = np.array_split(np.random.permutation(self.bd.train), self.numValFolds)
        for fold in range(self.numValFolds):
            valSet = foldsets[fold]
            trainSet = [i for i in self.bd.train if i not in valSet]
            stored_losses.append([])
            train_metrics.append([])
            test_metrics.append([])
            self.model = self._get_model()(
                input_size=self.bd.dimensions[0],
                output_size=self.bd.dimensions[1],
                **self.model_kw,
            )
            opt, sched = self.model.make_optimizer()
            for e in range(epochs):
                lh = self.train_epoch_val(opt, trainSet)
                if (e%rec_every) == 0 or e == epochs - 1:
                    print(f'{e:}\t', lh)
                    stored_losses[fold].append(lh)
                    metrics, labels = self.report_scores_subset(trainSet)
                    train_metrics[fold].append(metrics)
                    tmetrics, tlabels = self.report_scores_subset(valSet)
                    test_metrics[fold].append(tmetrics)
                sched.step()
        return np.mean(stored_losses, axis=0), np.mean(train_metrics, axis=0), np.mean(test_metrics, axis=0), labels

    def get_class_predictions(self, testset):
        if (testset):
            toUse = self.bd.test
        else:
            toUse = self.bd.train
        preds = None
        for indx in toUse:
            # extract one participants data
            data  = self.bd.get_features(indx)
            label = self.bd.chunkedLabels[indx]
            # important to note that we are not using batching here
            # one participants data is ONE sequence
            pred = self.getPrediction(data)
            # we have to predict with one sequence at a time and then
            # stitch together the predictions and labels to calculate our metrics
            if (preds == None):
                preds = pred
                labels = label
            else: 
                preds = torch.cat([preds, pred], dim = 0)
                labels = torch.cat([labels, label], dim = 0)
        preds = torch.cat([preds[:, :3], preds[:, 3:]], 0)
        labels = torch.cat([labels[:, :3 + 1], labels[:, 3 + 1:]], 0)
        preds = preds.argmax(dim=1)
        p1 = preds[labels[:, 1] == 1]
        p2 = preds[labels[:, 2] == 1]
        p3 = preds[labels[:, 3] == 1]
        return p1, p2, p3

    # TODO: make sure this is correct and/or come up with more efficient method
    def getPrediction(self, datas):
        if not self.modelSplit:
            pred = self.model.forward(datas)
        else:
            pred = torch.zeros([datas.shape[0] * 2, self.model.output_size])
            pred.requires_grad = True
            # separate data by category for each weekly question
            # then, recombine the predictions using the indices
            # consider computing these indices ahead of time and storing values for all participants?
            consumptionRows2 = (torch.where(datas[:, -1] == 0, 1, 0) * torch.where(datas[:, -2] == 0, 1, 0)).nonzero()
            if consumptionRows2.numel() > 0:
                consumptionRows2 = consumptionRows2[0, :]
                cpred2 = self.consumptionModel.predict(datas[consumptionRows2])
                pred.index_add(0, consumptionRows2, cpred2)
            knowledgeRows2 = (torch.where(datas[:, -1] == 1, 1, 0) * torch.where(datas[:, -2] == 0, 1, 0)).nonzero()
            if knowledgeRows2.numel() > 0:
                knowledgeRows2 = knowledgeRows2[0, :]
                kpred2 = self.knowledgeModel.predict(datas[knowledgeRows2])
                pred.index_add(0, knowledgeRows2, kpred2)
            physRows2 = (torch.where(datas[:, -1] == 0, 1, 0) * torch.where(datas[:, -2] == 1, 1, 0)).nonzero()
            if physRows2.numel() > 0:
                physRows2 = physRows2[0, :]
                ppred2 = self.physicalModel.predict(datas[physRows2])
                pred.index_add(0, physRows2, ppred2)
            consumptionRows1 = (torch.where(datas[:, -3] == 0, 1, 0) * torch.where(datas[:, -4] == 0, 1, 0)).nonzero()
            if consumptionRows1.numel() > 0:
                consumptionRows1 = consumptionRows1[0, :]
                cpred1 = self.consumptionModel.predict(datas[consumptionRows1])
                pred.index_add(0, consumptionRows1, cpred1)
            knowledgeRows1 = (torch.where(datas[:, -3] == 1, 1, 0) * torch.where(datas[:, -4] == 0, 1, 0)).nonzero()
            if knowledgeRows1.numel() > 0:
                knowledgeRows1 = knowledgeRows1[0, :]
                kpred1 = self.knowledgeModel.predict(datas[knowledgeRows1])
                pred.index_add(0, knowledgeRows1, kpred1)
            physRows1 = (torch.where(datas[:, -3] == 0, 1, 0) * torch.where(datas[:, -4] == 1, 1, 0)).nonzero()
            if physRows1.numel() > 0:
                physRows1 = physRows1[0, :]
                ppred1 = self.physicalModel.predict(datas[physRows1])
                pred.index_add(0, physRows1, ppred1)

            # reshape to match single model output
            pred = torch.cat((pred[0:datas.shape[0]], pred[datas.shape[0]:]), dim = 1)
        
        return pred


    def train_epoch_val(self, opt, trainSet):
        # feed through training data one time
        loss = []
        preds = None
        labels = None
        opt.zero_grad()
        for indx in trainSet:
            # extract one participants data
            data  = self.bd.get_features(indx)
            label = self.bd.chunkedLabels[indx]
            # important to note that we are not using batching here
            # one participants data is ONE sequence
            pred = self.getPrediction(data)
            # we have to predict with one sequence at a time and then
            # stitch together the predictions and labels to calculate our metrics
            if (preds == None):
                preds = pred
                labels = label
            else: 
                preds = torch.cat([preds, pred], dim = 0)
                labels = torch.cat([labels, label], dim = 0)
            
        loss1 = self.model.train_step(preds, labels)
        if (loss1 != None):
            loss.append(loss1)
        loss1.backward()
        opt.step()
        loss = [l1.item() for l1 in loss]
        return loss
    
    # update feature modifications for both train and test data
    def update_all_feature_mods(self):
        with torch.no_grad():
            for set in [self.bd.train, self.bd.test]:
                for indx in set:
                    # extract one participants data
                    data = self.bd.get_features(indx)
                    pred = self.getPrediction(data)
                    self.bd.set_feature_response_mods(indx, pred.detach().numpy())

        
    def train(self):
        # Loop over data and train model on each batch
        # Returns matrix of loss for each participant
        stored_losses = []
        train_metrics = []
        test_metrics = []
        opts = []
        scheds = []
        if (self.modelSplit):
            for model in [self.consumptionModel, self.knowledgeModel, self.physicalModel]:
                opt, sched = model.make_optimizer()
                opts.append(opt)
                scheds.append(sched)
        else:
            opt, sched = self.model.make_optimizer()
            opts.append(opt)
        epochs = self.train_kw.get("epochs", 1)
        rec_every = self.train_kw.get("rec_every", 5)
        for e in range(epochs):
            
            # enable the zeroing of state features once appropriate
            if (self.stateZeroEpochs > 0 and e == self.stateZeroEpochs):
                self.bd.zeroStateFeatures = True

            lh = self.train_epoch(opts)

            # update our predictions as features when appropriate
            if (self.bd.insert_predictions):
                if (e > 0 and e % self.epochsToUpdateLabelMods == 0) or e == epochs - 1:
                    self.update_all_feature_mods()

            # record metrics every rec_every epochs
            if (e%rec_every) == 0 or e == epochs - 1:
                stored_losses.append(lh)
                metrics, labels = self.report_scores_train()
                train_metrics.append(metrics)
                tmetrics, tlabels = self.report_scores()
                test_metrics.append(tmetrics)
                print(f'{e}\t', f"train loss: {lh[0]:.4f}", f"train acc: {metrics[labels.index('Acc')]:.3%}", f"test acc: {tmetrics[labels.index('Acc')]:.3%}")
            for sched in scheds:
                sched.step()
        return np.array(stored_losses), np.array(train_metrics), np.array(test_metrics), labels

    def calcFinalGradients(self):
        # Loop over data and train model on each batch
        # Returns matrix of loss for each participant
        grads = self.train_epoch_no_step()

        return grads

    def train_epoch_no_step(self):
        # feed through training data one time
        preds = None
        labels = None
        opt = torch.optim.SGD(self.bd.chunkedFeatures, lr=1)
        for indx in self.bd.train:
            # extract one participants data
            data  = self.bd.get_features(indx)
            data.requires_grad = True
            label = self.bd.chunkedLabels[indx]
            # important to note that we are not using batching here
            # one participants data is ONE sequence
            pred = self.getPrediction(data)
            # we have to predict with one sequence at a time and then
            # stitch together the predictions and labels to calculate our metrics
            if (preds == None):
                preds = pred
                labels = label
            else: 
                preds = torch.cat([preds, pred], dim = 0)
                labels = torch.cat([labels, label], dim = 0)
            
        loss1 = preds.sum()
        loss1.backward()
        grads = None
        for indx in self.bd.train:
            grad  = self.bd.get_features(indx).grad
            # we have to predict with one sequence at a time and then
            # stitch together the predictions and labels to calculate our metrics
            if (grads == None):
                grads = grad
            else: 
                grads = torch.cat([grads, grad], dim = 0)
        return grads.mean(dim=0)
    
    def train_epoch(self, opts):
        # feed through training data one time
        loss = []
        preds = None
        labels = None
        for opt in opts:
            opt.zero_grad()
        for indx in self.bd.train:
            # extract one participants data
            data  = self.bd.get_features(indx)
            label = self.bd.chunkedLabels[indx]

            # important to note that we are not using batching here (FOR THE LSTM)
            # one participants data is ONE sequence
            # for non LSTM models, the participants data is batched by week
            # no special trick here - the dimensions work to be non batched in LSTM (as desired)
            # and batched by week for non LSTM models (also as desired)
            pred = self.getPrediction(data)
            

            # we have to predict with one sequence at a time and then
            # stitch together the predictions and labels to calculate our metrics
            if (preds == None):
                preds = pred
                labels = label
            else: 
                preds = torch.cat([preds, pred], dim = 0)
                labels = torch.cat([labels, label], dim = 0)
            
        loss1 = self.model.train_step(preds, labels)
        if (loss1 != None):
            loss.append(loss1)
        loss1.backward()
        for opt in opts:
            opt.step()
        loss = [l1.item() for l1 in loss]
        return loss
    
    def evaluate(self):
        # Evaluate the trained models predictions
        evals = []
        for indx in self.bd.test:
            # extract one participants data
            data  = self.bd.get_features(indx)
            label = self.bd.chunkedLabels[indx]
            # important to note that we are not using batching here
            # one participants data is ONE sequence
            pred = self.getPrediction(data)
            pred = pred.view(label.shape)
            evals.append(self.diff_matrix(label, pred))
        return evals
    
    def report_scores(self):
        with torch.no_grad():
            preds, labels = None, None
            for indx in self.bd.test:
                # extract one participants data
                data = self.bd.get_features(indx)
                label = self.bd.chunkedLabels[indx]
                pred = self.getPrediction(data)
                if (preds == None):
                    preds = pred
                    labels = label
                else: 
                    preds = torch.cat([preds, pred], dim = 0)
                    labels = torch.cat([labels, label], dim = 0)
            scores, label = self.model.report_scores_min(labels, preds)
            return scores, label
    
    def report_scores_individual_test(self):
        with torch.no_grad():
            scores = []
            for indx in self.bd.test:
                # extract one participants data
                data  = self.bd.get_features(indx)
                label = self.bd.chunkedLabels[indx]
                pred = self.getPrediction(data)
                score, label = self.model.report_scores_min(label, pred)
                if (len(score) > 0):
                    scores.append(score)
            return np.array(scores), label

    def report_scores_individual_train(self):
        with torch.no_grad():
            scores = []
            for indx in self.bd.train:
                # extract one participants data
                data  = self.bd.get_features(indx)
                label = self.bd.chunkedLabels[indx]
                pred = self.getPrediction(data)
                score, label = self.model.report_scores_min(label, pred)
                if (len(score) > 0):
                    scores.append(score)
            return np.array(scores), label
    
    
    def report_scores_train(self):
        with torch.no_grad():
            preds, labels = None, None
            for indx in self.bd.train:
                # extract one participants data
                data  = self.bd.get_features(indx)
                label = self.bd.chunkedLabels[indx]
                pred = self.getPrediction(data)
                if (preds == None):
                    preds = pred
                    labels = label
                else: 
                    preds = torch.cat([preds, pred], dim = 0)
                    labels = torch.cat([labels, label], dim = 0)
            scores, label = self.model.report_scores_min(labels, preds)
            return scores, label

    def report_scores_subset(self, subset):
        with torch.no_grad():
            preds, labels = None, None
            for indx in subset:
                # extract one participants data
                data  = self.bd.get_features(indx)
                label = self.bd.chunkedLabels[indx]
                pred = self.getPrediction(data)
                if (preds == None):
                    preds = pred
                    labels = label
                else: 
                    preds = torch.cat([preds, pred], dim = 0)
                    labels = torch.cat([labels, label], dim = 0)
            scores, label = self.model.report_scores_min(labels, preds)
            return scores, label
        
            
    def diff_matrix(self, true, pred):
        # Build a matrix showing the error in predicted responses
        diff = np.zeros((2, true.shape[0]))
        diff[0] = (true[:,:4].argmax(1)-pred[:,:4].argmax(1))
        diff[1] = (true[:,-4:].argmax(1)-pred[:,-4:].argmax(1))
        return diff
        
    def totensor(self, a):
        # Convert a to a batched tensor
        a = torch.Tensor(a)
        a = a.view(a.shape[0], 1, a.shape[1])
        return a
    
    def forceBatch(self, a):
        # Convert a to a batched tensor
        a = a.view(1, 1, a.shape[0])
        return a
        
    def _get_model(self):
        mod = importlib.import_module(f"models.{self.model_name}")
        return getattr(mod, self.model_name)
