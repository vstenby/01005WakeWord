from tqdm import trange, tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
import numpy as np
import pickle
import os

def train_cnn(model, criterion, optimizer, train_loader, val_loader, device, nepoch, silent=False, path = None):
    '''
    Training loop!
    '''
    
    model = model.to(device)
    model = model.eval()
    
    train_statistics = {}
    val_statistics   = {}
    
    #Preallocate statistics.
    train_statistics['loss'] = []
    train_statistics['tn']   = []
    train_statistics['fp']   = []
    train_statistics['fn']   = []
    train_statistics['tp']   = []
    train_statistics['f1']   = []


    val_statistics['loss']   = []
    val_statistics['tn']     = []
    val_statistics['fp']     = []
    val_statistics['fn']     = []
    val_statistics['tp']     = []
    val_statistics['f1']     = []

    opt_f1_val = 0
    
    for epoch in trange(nepoch, unit = 'epoch', desc = 'Epoch'):
        #Epochs start @ 1
        epoch += 1
        
        #Progress string
        progress_string = ''

        #Set it up for training.
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets     = []
        
        for minibatch_no, data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, paths = data

            #Get that stuff on the GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())

            #loss = criterion(outputs, labels.reshape(-1,).long())
            
            #If the loss is BCEWithLogitsLoss.
            loss = criterion(outputs, labels.float())
            
            loss.backward()
            optimizer.step()

            #Add the batch loss
            train_loss += loss.item()

            #Save predictions and targets
            #train_predictions += torch.softmax(outputs.cpu().detach(), axis=-1).tolist()
            train_predictions += (torch.sigmoid(outputs.cpu().detach()) >= 0.5).tolist()
            train_targets     += labels.long().tolist()
    
        #Set it up for evaluation on validation set.
        model.eval()  
        val_loss = 0
        val_predictions = []
        val_targets     = []

        with torch.no_grad():
            for minibatch_no, data in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, paths = data
            
                #Get that stuff on the GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
        
                # forward pass
                outputs = model(inputs.float())

                #Calculate the loss
                #loss = criterion(outputs, labels.long())
                
                #If the loss is BCEWithLogitsLoss
                loss = criterion(outputs, labels.float())
            
                val_loss += loss.item()

                #Save predictions and targets
                #val_predictions += torch.softmax(outputs.cpu().detach(), axis=-1).tolist()
                val_predictions += (torch.sigmoid(outputs.cpu().detach()) >= 0.5).tolist()
                val_targets     += labels.long().tolist()

        #Get the average observation loss.
        train_loss = train_loss / len(train_loader.dataset)
        val_loss   = val_loss   / len(val_loader.dataset)

        #Calculate the train statistics.
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(train_targets, train_predictions, labels=[0,1]).ravel()
        f1_train = f1_score(train_targets, train_predictions)

        #Calculate the validation statistics.
        tn_val, fp_val, fn_val, tp_val = confusion_matrix(val_targets, val_predictions, labels=[0,1]).ravel()
        f1_val = f1_score(val_targets, val_predictions)

        #Update the train statistics.
        train_statistics['loss'].append(train_loss)
        train_statistics['tn'].append(tn_train)
        train_statistics['fp'].append(fp_train)
        train_statistics['fn'].append(fn_train)
        train_statistics['tp'].append(tp_train)
        train_statistics['f1'].append(f1_train)
        
        
        val_statistics['loss'].append(val_loss)
        val_statistics['tn'].append(tn_val)
        val_statistics['fp'].append(fp_val)
        val_statistics['fn'].append(fn_val)
        val_statistics['tp'].append(tp_val)
        val_statistics['f1'].append(f1_val)
        
        #Dump the statistics.
        with open(os.path.join(path, 'statistics.p'), 'wb') as f:
            pickle.dump([train_statistics, val_statistics], f)

        #Save the newest model 
        torch.save(model.state_dict(), os.path.join(path, 'model.pth'))

        if opt_f1_val <= f1_val:
            #Save it as the optimal model!
            torch.save(model.state_dict(), os.path.join(path, 'opt_model.pth'))
            opt_f1_val = f1_val

    return