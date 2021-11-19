from tqdm import trange, tqdm
from sklearn.metrics import accuracy_score
import torch
import numpy as np

def train_cnn(model, criterion, optimizer, train_loader, device, nepoch, val_loader = None, silent=False):
    '''
    Training loop!
    '''
    
    model = model.to(device)
    model = model.eval()
    
    train_losses = []
    train_accs   = []
    
    val_losses   = []
    val_accs     = []
    
    for epoch in trange(nepoch, unit = 'epoch', desc = 'Epoch'):
        #Epochs start @ 1
        epoch += 1
        
        #Progress string
        progress_string = ''

        #Set it up for training.
        model.train()
        train_loss = 0
        predictions = []
        targets = []
        for minibatch_no, data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):

            # get the inputs; data is a list of [inputs, labels]
            _, _, inputs, labels, paths, _ = data

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
            train_loss += loss.item() / len(paths) #per observation loss.

            #Save predictions and targets
            predictions += outputs.argmax(axis=1).tolist()
            targets     += labels.long().tolist()

        train_acc = accuracy_score(targets, predictions)
    
        #Append the statistics
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        progress_string += '[%d]: Train: [%.4f | %.3f]' % (epoch, train_loss, train_acc)

        if val_loader is not None:
            #Set it up for evaluation on validation set.
            model.eval()  
            predictions = []
            targets = []
            val_loss = 0

            with torch.no_grad():
                for minibatch_no, data in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):

                    # get the inputs; data is a list of [inputs, labels]
                    _, _, inputs, labels, paths, _ = data
                
                    #Get that stuff on the GPU
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs.float())

                    #Calculate the loss
                    loss = criterion(outputs, labels.long())
                    val_loss += loss.item() / len(paths) #per observation loss.

                    #Save predictions and targets
                    predictions += outputs.argmax(axis=1).tolist()
                    targets     += labels.long().tolist()
    
                val_acc = accuracy_score(targets, predictions)
        
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            progress_string += ' Validation: [%.4f | %.3f]' % (val_loss, val_accs)

        if not silent: print(progress_string)
        
    return np.arange(1, nepoch+1), train_losses, train_accs, val_losses, val_accs