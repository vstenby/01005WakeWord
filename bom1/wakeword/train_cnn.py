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
        train_paths       = []
    
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

            loss = criterion(outputs, labels.reshape(-1,).long())
            loss.backward()
            optimizer.step()

            #Add the batch loss
            train_loss += loss.item()

            #Save predictions and targets
            train_predictions += torch.softmax(outputs.cpu().detach(), axis=-1).tolist()
            train_targets     += labels.long().tolist()
            train_paths       += list(paths)
    
        #Set it up for evaluation on validation set.
        model.eval()  
        val_loss = 0
        val_predictions = []
        val_targets     = []
        val_paths       = []

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
                loss = criterion(outputs, labels.long())
                val_loss += loss.item()

                #Save predictions and targets
                val_predictions += torch.softmax(outputs.cpu().detach(), axis=-1).tolist()
                val_targets     += labels.long().tolist()
                val_paths       += list(paths)

        #Get the per-batch loss.
        train_loss = train_loss / len(train_loader)
        val_loss   = val_loss   / len(val_loader)

        #Calculate the train statistics.
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(np.array(train_targets), np.array(train_predictions).argmax(axis=1), labels=[0,1]).ravel()
        f1_train = f1_score(np.array(train_targets), np.array(train_predictions).argmax(axis=1))

        #Calculate the validation statistics.
        tn_val, fp_val, fn_val, tp_val = confusion_matrix(np.array(val_targets), np.array(val_predictions).argmax(axis=1), labels=[0,1]).ravel()
        f1_val = f1_score(np.array(val_targets), np.array(val_predictions).argmax(axis=1))

        #Update statistics
        train_statistics[epoch] = {'avg_batch_loss' : train_loss, 'tn' : tn_train, 'fp' : fp_train, 'fn' : fn_train, 'tp' : tp_train, 'f1' : f1_train}
        val_statistics[epoch]   = {'avg_batch_loss' : val_loss, 'tn' : tn_val, 'fp' : fp_val, 'fn' : fn_val, 'tp' : tp_val, 'f1' : f1_val}
        
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