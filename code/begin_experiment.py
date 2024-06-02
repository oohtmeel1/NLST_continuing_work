#!/usr/bin/env python3
"""This is going to create a tensorboard logger, save models automatically"""
import argparse
from typing import NamedTuple
import os
import pandas as pd
import torch
from model_architecture import Modelo
from loading_data_files import Loading_train
from loading_data_files import Loading_val
from loading_data_files import Loading_test
from torch.utils.data import DataLoader
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.metrics import Accuracy
from ignite.metrics import Precision, Recall
from ignite.handlers import ModelCheckpoint
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator  
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Timer, BasicTimeProfiler, HandlersTimeProfiler    
from ignite.engine import Events
from ignite.metrics import Accuracy
from ignite.metrics import Precision, Recall
from torch.optim.lr_scheduler import LinearLR
import warnings
warnings.filterwarnings('ignore')
from ignite.handlers import create_lr_scheduler_with_warmup

torch.cuda.empty_cache()

### This portion just inputs the information 

l_ = input('please specify the type of device to use: (cuda or cpu)',)
if l_ == 'cuda':
    device = torch.device('cuda')
    print(device)
else:
    device = torch.device('cpu')
    print(device)
l_ = input('going to load model to device)(y/n)',)
##This stores the model to the device after calling the correct function
if l_ == 'y':
    model = Modelo()
    model.to(device)
else:
    pass
print(device)
## This loads the training, validation and testing data sets
train = Loading_train()
val = Loading_val()
test = Loading_test()
print('loading data')
print("Number of files in training ",len(train))
print("Number of files in validation ",len(val))
print("Number of files in testing ",len(test))

q = int(input('please input a batch size',))
a = q
m = int(input('please input a batch size',))
n = m
train_dataloader = DataLoader(train, a, True)
val_dataloader = DataLoader(val, n, True)
test_dataloader = DataLoader(test,16,True)
    
loss_fn_ = input('Please enter a loss function ending in (), else nn.BCEWithLogitsLoss() will be used',)
optimizer = input('Please enter an optimizer function, else torch.optim.AdamW will be used,',)
x_ = input('Please enter a learning rate speed,else 0.00005 will be used',)
from torch import nn
if loss_fn_ == "":
    loss_fn_ = nn.BCEWithLogitsLoss()
else:
    loss_fn_ = f"{loss_fn_}"
if x_ == "":
    x_ = 0.00005
else:
    x_ = x_
if optimizer == "":
    optimizer = torch.optim.AdamW(model.parameters(), lr = x_)
else:
    optimizer = (f"torch.optim{optimizer}(model.parameters(), lr = {x_})")
print(loss_fn_)
print(optimizer) 
x_ = input('Begin training? (y/n)',)
if l_ == 'y':
    pass
else:
    pass
def update_model(engine, batch):
    model.train()
    data,label = batch['data'].float(),batch['label']
    optimizer.zero_grad()
    data=data.to(device)
    label=label.to(device)
    outputs,_=(model(data),label)
    outputs=outputs.squeeze()
    loss = loss_fn_(outputs, label.float())
    loss.backward()
    optimizer.step()
    return loss.item()

trainer = Engine(update_model)




def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch['data'].float(),batch['label']
        x=x.to(device)
        y=y.to(device)
        outputs,_=(model(x),y)
        outputs = outputs.squeeze()
        outputs=torch.sigmoid(outputs)
        outputs=outputs.cpu().detach()
        outputs=outputs.round()
        y=y.cpu().detach().long()

    return outputs.long(), y
    
    
evaluator = Engine(validation_step)
Accuracy().attach(evaluator, "accuracy")
Precision().attach(evaluator,'precision')
Recall(average='weighted').attach(evaluator,'recall')
val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(loss_fn_)
}
torch_lr_scheduler = LinearLR(optimizer=optimizer,start_factor=0.001, end_factor=1.0)  
scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
                                            warmup_start_value=0.00004,
                                            warmup_end_value=0.0001,
                                            warmup_duration=20)


precision = Precision()


@trainer.on(Events.ITERATION_COMPLETED(every=100))
def log_training(engine):
    batch_loss = engine.state.output
    lr = optimizer.param_groups[0]['lr']
    e = engine.state.epoch
    n = engine.state.max_epochs
    i = engine.state.iteration
    print("Epoch {}/{} : {} - batch loss: {}, lr: {}".format(e, n, i, batch_loss, lr))
def print_lr():
    print(optimizer.param_groups[0]["lr"])


validate_every = 2

@trainer.on(Events.EPOCH_COMPLETED(every=validate_every))
def run_validation():
    evaluator.run(val_dataloader)
    
@trainer.on(Events.EPOCH_COMPLETED(every=validate_every))
def log_validation():
    ugh=[]
    metrics = evaluator.state.metrics
    print(f"Epoch: {trainer.state.epoch},  Accuracy: {metrics['accuracy']},  Precision: {metrics['precision']}, recall: {metrics['recall']}")


def score_function(engine):
    return engine.state.metrics["accuracy"]
abc = int(input('Please enter how many models to save ',))
model_checkpoint = ModelCheckpoint(
    "checkpoint",
    n_saved=abc,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer),
)

evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

tb_logger = TensorboardLogger(log_dir="tb-logger")

tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

for tag, evaluator in [("training", trainer), ("validation", evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )
trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)    
epxn_ = int(input('Please enter how many epochs you would like to train ',))

trainer.run(train_dataloader, max_epochs= epxn_)
accuracy10=[]

from torcheval.metrics.functional import binary_accuracy
from torcheval.metrics.functional.classification import binary_recall
from torcheval.metrics.functional.aggregation.auc import auc
from torcheval.metrics.functional import binary_precision
from torcheval.metrics.functional import binary_accuracy
from torcheval.metrics.functional import binary_f1_score
from torcheval.metrics.functional.classification import binary_recall
from torcheval.metrics.functional import binary_accuracy
from sklearn import metrics
from sklearn.metrics import roc_curve
x_ = input('Run Testing? (y/n)',)
if l_ == 'y':
    for j in os.listdir(os.getcwd()+'\\checkpoint'):
        PATH=os.path.join(os.getcwd()+"\\checkpoint",j)
        model.load_state_dict(torch.load(PATH))
        for i, batch in enumerate(test_dataloader):
            model.eval()
            data,label = batch['data'].float(),batch['label']
            data=data.to(device)
            label=label.to(device)
            y_logits,label=(model(data),label)
            test_pred1=y_logits.squeeze(-1)
            test_pred=torch.sigmoid(test_pred1)
            test_pred=test_pred.cpu().detach()
            test_pred=test_pred.round()
            label=label.cpu().detach()
            arfs=binary_accuracy(test_pred, label, threshold=0.7)
            test_pred2=arfs.detach().numpy()
            f1_score=binary_f1_score(test_pred, label, threshold=0.7)
            f1_rec=f1_score.detach().numpy()
            ats=binary_recall(test_pred, label, threshold=0.7)
            bin_rec=ats.detach().numpy()
            prec1=binary_precision(test_pred, label, threshold=0.7)
            prec_scr=prec1.detach().numpy()
            accuracy10.append([test_pred2,bin_rec,f1_rec,prec_scr,j,test_pred.numpy(),label.numpy()])#bin_rec,f1_rec,prec_scr])
            ahh=pd.DataFrame(accuracy10,columns=['Accuracy','Recall','F1','Precision','Model Name','Predictions','Labels'])
            ahh.to_csv('results.csv',index=False)
print("F1 Score ",f1_score, "Precision ",prec1, "Recall ",bin_rec)
x_ = input('Would you like to print full resuls? ',)
if x_ == 'y':
    print("F1: ",np.mean(ahh['F1']),"Recall: ", np.mean(ahh['Recall']),"Precision: ", np.mean(ahh['Precision']), "Accuracy: ", np.mean(ahh['Accuracy']))



# --------------------------------------------------
