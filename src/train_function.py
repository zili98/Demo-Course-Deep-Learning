import torch
import pandas as pd
import numpy as np
import sys
import copy
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from models.losses import LogitAdjustLoss, FocalLoss, CrossEntropyLoss, instance_weighted_loss, DiscriminativeLoss
from utils.help_functions import Voting, compute_metrics_from_confusion_matrix, set_seeds
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import mlflow
import mlflow.pytorch

    
class TrainModule():
    def __init__(self, cfg, model, train_loader, val_loader, loss_func, use_instance_weight=True,
                posthoc_adjustment=False):
        self._cfg = cfg
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._posthoc_adjustment = posthoc_adjustment

        # self._losses = {
        #     loss['NAME']: getattr(sys.modules[__name__], loss['NAME'])(**loss.get('ARGS', {}))
        #     for loss in cfg.LOSSES
        # }
        # self._loss = self._losses[cfg.MODEL.LOSS_FUNC]
        # self._loss_feat = DiscriminativeLoss(delta_var=0.5, delta_dist=5)
        self._loss = loss_func
        self._use_instance_weight = use_instance_weight
        
    def train(self, validate_interval=1, verbose=10):

        set_seeds(self._cfg.SEED)
        self._model = self._model.to(self._cfg.DEVICE)
        
        optimizer, scheduler = self.configure_optimizers()

        best_val_loss=1e5
        best_val_UAR = -1e5
        best_model_wts=None
        early = 0
        ######### Start ##############################################################
        print('START TRAINING...')
        for epoch in range(0, self._cfg.TRAIN_ARGS.NUM_EPOCHS):
            # Cyclical Learning Rate
            if epoch % self._cfg.TRAIN_ARGS.CYCLICAL_EPOCHS == 0:
                optimizer, scheduler = self.configure_optimizers()
            # Early Stopping
            if (self._cfg.TRAIN_ARGS.EARLY_STOPPING_PATIENCE > 0) and (early >= self._cfg.TRAIN_ARGS.EARLY_STOPPING_PATIENCE):
                break
            #### training ########################################################
            self._model.train()

            loss_train = 0
            num_samples = 0
            matrix_train = np.zeros((self._cfg.MODEL.NUM_CLASSES,self._cfg.MODEL.NUM_CLASSES))

            for batch_idx, (batch_samples, ins_weights) in enumerate(self._train_loader):
                ecg, label = batch_samples['ecg'].to(self._cfg.DEVICE, dtype=torch.float32), batch_samples['label'].to(self._cfg.DEVICE)
                label = label.squeeze()
                optimizer.zero_grad()
                pred = self._model(ecg)
                num_classes=pred.size(1)
    
                if self._use_instance_weight:
                    loss = self._loss(pred, label, ins_weights)
                else:
                    loss = self._loss(pred, label)
                loss.backward()
                optimizer.step()
                # loss
                loss_train += loss.item()*len(label)
                num_samples += len(label)
                # recall
                matrix_train += confusion_matrix(label.reshape(1, -1).squeeze().cpu().numpy(), 
                                           torch.argmax(pred, dim=1).reshape(1, -1).squeeze().cpu().numpy(),
                                           labels=range(self._cfg.MODEL.NUM_CLASSES))

            loss_train = loss_train / num_samples
            UAR_train, acc_train, metrics_train = compute_metrics_from_confusion_matrix(matrix_train)
            scheduler.step()

            mlflow.log_metric(f"train_loss", loss_train, step=epoch)
            mlflow.log_metric(f"train_uar", UAR_train, step=epoch)
            mlflow.log_metric(f"train_acc", acc_train, step=epoch)
            for _i in range(len(metrics_train['recall'])):
                mlflow.log_metric(f"train_recall_{_i}", metrics_train['recall'][_i], step=epoch)
            
            mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=epoch)
        
            if epoch % verbose == 0:
                print('Training\tEpoch: {}\tLoss: {:.3f}\tUAR: {:.3f}\t{} subjects'.format(
                        epoch, loss_train, UAR_train, num_samples))
            del pred, label, batch_samples, num_samples
            ############ Validation #################################################################################################################
            if epoch % validate_interval == 0:
                self._model.eval()

                loss_val = 0
                num_samples = 0
                matrix_val = np.zeros((self._cfg.MODEL.NUM_CLASSES,self._cfg.MODEL.NUM_CLASSES))
                logit_0 = torch.zeros(num_classes)
                logit_1 = torch.zeros(num_classes)
                num_0 = 0
                num_1 = 0
                logit_0_orig = torch.zeros(num_classes)
                logit_1_orig = torch.zeros(num_classes)
                with torch.no_grad():
                    for batch_idx, (batch_samples, ins_weights) in enumerate(self._val_loader):
                        ecg,label= batch_samples['ecg'].to(self._cfg.DEVICE,dtype=torch.float32),batch_samples['label'].to(self._cfg.DEVICE)
                        label = label.squeeze()
                        optimizer.zero_grad()

                        pred= self._model(ecg)
                        if self._use_instance_weight:
                            loss = self._loss(pred, label, ins_weights)
                        else:
                            loss = self._loss(pred, label)
                        loss_val += loss.item()*len(label)

                        if self._posthoc_adjustment:
                            base_probs = torch.tensor([0.9, 0.1])
                            tau = torch.tensor(1.0) 
                            pred_orig = pred
                            pred = pred - torch.log(torch.Tensor(base_probs**tau + 1e-12).to(self._cfg.DEVICE,dtype=torch.float32))

                            matrix_val_orig = matrix_val.copy()
                            matrix_val_orig += confusion_matrix(label.reshape(1, -1).squeeze().cpu().numpy(), 
                                              torch.argmax(pred_orig, dim=1).reshape(1, -1).squeeze().cpu().numpy(),
                                              labels=range(self._cfg.MODEL.NUM_CLASSES))                       
                            logit_0_orig += pred_orig[label==0].sum(dim=0)
                            logit_1_orig += pred_orig[label==1].sum(dim=0)
                        num_samples += len(label)

                        matrix_val += confusion_matrix(label.reshape(1, -1).squeeze().cpu().numpy(), 
                                           torch.argmax(pred, dim=1).reshape(1, -1).squeeze().cpu().numpy(),
                                           labels=range(self._cfg.MODEL.NUM_CLASSES))

                        logit_0 += pred[label==0].sum(dim=0)
                        logit_1 += pred[label==1].sum(dim=0)
                        num_0 += len(label==0)
                        num_1 += len(label==1)
                # validation loss
                loss_val = loss_val / num_samples
                UAR_val, acc_val, metrics_val = compute_metrics_from_confusion_matrix(matrix_val)            

                # UAR
                mlflow.log_metric(f"neg_logit", (logit_0[0]-logit_0[1])/num_0, step=epoch)
                mlflow.log_metric(f"pos_logit", (logit_1[0]-logit_1[1])/num_1, step=epoch)
                mlflow.log_metric(f"neg_logit_orig", (logit_0_orig[0]-logit_0_orig[1])/num_0, step=epoch) 
                mlflow.log_metric(f"pos_logit_orig", (logit_1_orig[0]-logit_1_orig[1])/num_1, step=epoch)       
                mlflow.log_metric(f"val_loss", loss_val, step=epoch)
                mlflow.log_metric(f"val_uar", UAR_val, step=epoch)
                mlflow.log_metric(f"val_acc", acc_val, step=epoch)
                for _i in range(len(metrics_val['recall'])):
                    mlflow.log_metric(f"val_recall_{_i}", metrics_val['recall'][_i], step=epoch)

                if self._posthoc_adjustment:
                    UAR_val_orig, acc_val_orig, metrics_val_orig = compute_metrics_from_confusion_matrix(matrix_val_orig)
                    mlflow.log_metric(f"val_uar_origin", UAR_val_orig, step=epoch)
                    mlflow.log_metric(f"val_acc_origin", acc_val_orig, step=epoch)
                    for _i in range(len(metrics_val_orig['recall'])):
                        mlflow.log_metric(f"val_origin_recall_{_i}", metrics_val_orig['recall'][_i], step=epoch)

                ################################################################################
                if (epoch > 5) and (loss_val <= best_val_loss):
                    best_model_wts = copy.deepcopy(self._model.state_dict())
                    best_val_loss = loss_val
                    best_val_UAR = UAR_val
                    best_val_acc = acc_val
                    best_metrics = metrics_val['recall']
                    best_matrix = matrix_val
                    best_train_UAR = UAR_train
                    best_epoch = epoch
                    early = 0
                else:
                    early += 1
                
                if epoch % verbose == 0:
                    print('Validate\tEpoch: {}\tLoss: {:.3f}\tUAR: {:.3f}\tEarly: {}\t{} subjects'.format(
                        epoch, loss_val, UAR_val, early, num_samples),'\n')
                del pred, label, batch_samples

        print('\nFinished TRAINING.')
        
        self._model.load_state_dict(best_model_wts)
        
        # mlflow.pytorch.log_state_dict(
        #     best_model_wts, artifact_path="epoch_{}-uar_{:.2f}".format(best_epoch, best_val_UAR*100)
        # )

        results = {'best_val_uar': np.round(best_val_UAR,3),
              'best_val_acc': np.round(best_val_acc,3),
              'epoch_end': best_epoch,
            }
        mlflow.log_params(results)
    
        print('Epoch: {}\tVal UAR: {:.3f}\tTrain UAR: {:.3f}'.format(
            best_epoch, best_val_UAR, best_train_UAR),'\n')

        return self._model, best_val_UAR, best_matrix

    def test(self, test_loader):
        self._model.eval()

        loss_test = 0
        num_samples = 0
        pred_all = [None]*len(test_loader)
        label_all = [None]*len(test_loader)
        pred_orig_all = [None]*len(test_loader)
        matrix_test = np.zeros((self._cfg.MODEL.NUM_CLASSES,self._cfg.MODEL.NUM_CLASSES))
        with torch.no_grad():
            for batch_idx, (batch_samples, ins_weights) in tqdm(enumerate(test_loader)):
                ecg,label= batch_samples['ecg'].to(self._cfg.DEVICE,dtype=torch.float32),batch_samples['label'].to(self._cfg.DEVICE)
                pred= self._model(ecg)
                if self._use_instance_weight:
                    loss = self._loss(pred, label, ins_weights)
                else:
                    loss = self._loss(pred, label)
                loss_test += loss.item()

                if self._posthoc_adjustment:
                    base_probs = torch.tensor([0.9, 0.1])
                    tau = torch.tensor(1.0) 
                    pred_orig = pred
                    pred = pred - torch.log(torch.Tensor(base_probs**tau + 1e-12).to(self._cfg.DEVICE,dtype=torch.float32))

                    pred_orig_all[batch_idx] = torch.argmax(pred_orig).cpu().tolist()
                    # matrix_test_orig = matrix_test.copy()
                    # matrix_test_orig += confusion_matrix(label.reshape(1, -1).squeeze().cpu().numpy(), 
                    #                                     torch.argmax(pred_orig, dim=1).reshape(1, -1).squeeze().cpu().numpy(),
                    #                                     labels=range(self._cfg.MODEL.NUM_CLASSES))

                num_samples += 1
                pred_all[batch_idx] = torch.argmax(pred).cpu().tolist()
                label_all[batch_idx] = label.squeeze().cpu().tolist()
                # matrix_test += confusion_matrix(label.reshape(1, -1).squeeze().cpu().numpy(), 
                #                     torch.argmax(pred).reshape(1, -1).squeeze().cpu().numpy(),
                #                     labels=range(self._cfg.MODEL.NUM_CLASSES))

        # test loss
        loss_test = loss_test / num_samples
        matrix_test = confusion_matrix(np.asarray(label_all), 
                          np.asarray(pred_all),
                          labels=range(self._cfg.MODEL.NUM_CLASSES))
        UAR_test, acc_test, metrics_test, fig = compute_metrics_from_confusion_matrix(matrix_test, visualize=True)            
        results = {'test_uar': np.round(UAR_test,3),
              'test_acc': np.round(acc_test,3),
                   'test_loss': np.round(loss_test,3),
                }

        if self._posthoc_adjustment:
            matrix_test_orig = confusion_matrix(np.asarray(label_all), 
                            np.asarray(pred_orig_all),
                            labels=range(self._cfg.MODEL.NUM_CLASSES))
            UAR_test_orig, acc_test_orig, metrics_test_orig, fig_2 = compute_metrics_from_confusion_matrix(matrix_test_orig, visualize=True)
            results['original_test_uar']=np.round(UAR_test_orig,3)
            results['original_test_acc']=np.round(acc_test_orig,3)
            mlflow.log_figure(fig_2, "original_test_confusion_matrix.png")

        mlflow.log_params(results)
        mlflow.log_figure(fig, "test_confusion_matrix.png")
        return pred_all, label_all

    def configure_optimizers(self):    
        if self._cfg.TRAIN_ARGS.OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD(
                        self._model.parameters(), 
                        lr=self._cfg.TRAIN_ARGS.BASE_LR,
                        momentum=0.9,
                        weight_decay=self._cfg.TRAIN_ARGS.WEIGHT_DECAY)

        elif self._cfg.TRAIN_ARGS.OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(
                        self._model.parameters(),
                        lr=self._cfg.TRAIN_ARGS.BASE_LR,
                        weight_decay=self._cfg.TRAIN_ARGS.WEIGHT_DECAY
                        )
        elif self._cfg.TRAIN_ARGS.OPTIMIZER == 'Adam':
            optimizer = torch.optim.Adam(
                        self._model.parameters(),
                        lr=self._cfg.TRAIN_ARGS.BASE_LR,
                        weight_decay=self._cfg.TRAIN_ARGS.WEIGHT_DECAY
                        )
        else:
            raise RuntimeError(f'Unsupported Optimizer {self._cfg.TRAIN_ARGS.OPTIMIZER}')

        if self._cfg.TRAIN_ARGS.LR_SCHEDULER == 'ReduceLROnPlateau':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=self._cfg.TRAIN_ARGS.LR_SCHEDULER_FACTOR,
                    patience=self._cfg.TRAIN_ARGS.LR_SCHEDULER_PATIENCE,
                    min_lr=self._cfg.TRAIN_ARGS.MIN_LR,
                    verbose=True,
                ),
                'monitor': 'val_loss',
            }

        elif self._cfg.TRAIN_ARGS.LR_SCHEDULER == 'CosineAnnealingLR':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self._cfg.TRAIN_ARGS.MAX_EPOCHS,
                )
            }
        elif self._cfg.TRAIN_ARGS.LR_SCHEDULER == 'CosineAnnealingWarmRestarts':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self._cfg.TRAIN_ARGS.WARM_RESTART_EPOCH
                )
            }
        elif self._cfg.TRAIN_ARGS.LR_SCHEDULER == 'LinearWarmupCosineAnnealingLR':
            scheduler = {
                'scheduler': LinearWarmupCosineAnnealingLR(
                    optimizer, 
                    warmup_epochs=self._cfg.TRAIN_ARGS.WARM_UP_EPOCH, 
                    max_epochs=self._cfg.TRAIN_ARGS.MAX_EPOCHS,
                    warmup_start_lr=self._cfg.TRAIN_ARGS.MIN_LR,
                    eta_min=self._cfg.TRAIN_ARGS.MIN_LR
                )
            } 
        elif self._cfg.TRAIN_ARGS.LR_SCHEDULER == 'StepLR':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self._cfg.TRAIN_ARGS.LR_SCHEDULER_PATIENCE,
                    gamma=self._cfg.TRAIN_ARGS.LR_SCHEDULER_FACTOR
                )
            }       
        else:
            raise RuntimeError(f'Unsupported LR scheduler {self._cfg.TRAIN_ARGS.LR_SCHEDULER}')

        return optimizer, scheduler['scheduler']