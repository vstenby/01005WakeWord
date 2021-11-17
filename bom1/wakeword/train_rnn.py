import torch
import numpy as np

def train_rnn(model, criterion, optimizer, train_loader, device, nepoch, val_loader = None, silent=False, scheduler = None):
    '''
    Training loop!
    '''
    
    model = model.to(device)
    
    train_losses = []
    val_losses   = []

    print(f'Training for {nepoch} epochs.')
    
    for epoch in range(nepoch):
        #Epochs start @ 1
        epoch += 1
        
        #Set it up for training.
        model.train()
        train_loss = 0
        
        for data in train_loader:

            # get the inputs; data is a list of [inputs, labels]
            _, _, inputs, targets, paths, _ = data
            
            #Get that stuff on the GPU
            inputs  = inputs.to(device)
            targets = targets.to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float()) 

            loss = criterion(outputs.permute(0,2,1), targets.long()) #Permute according to https://discuss.pytorch.org/t/loss-functions-for-batches/20488/6

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            #Add the batch loss
            train_loss += loss.item() / len(paths) #per observation loss.

        #Append the statistics
        train_losses.append(train_loss)

        if val_loader is not None:
            with torch.no_grad():    
                #Set it up for evaluation on validation set.
                model.eval()  
                val_loss = 0

                for data in val_loader:

                    # get the inputs; data is a list of [inputs, labels]
                    _, _, inputs, targets, paths, _ = data
                    
                    #Get that stuff on the GPU
                    inputs  = inputs.to(device)
                    targets = targets.to(device)
                
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs.float()) 

                    #Calculate the loss
                    loss = criterion(outputs.permute(0,2,1), targets.long()) #Permute according to https://discuss.pytorch.org/t/loss-functions-for-batches/20488/6
                    val_loss += loss.item() / len(paths) #per observation loss.

            val_losses.append(val_loss)
        
        if not silent and val_loader is None: print('[%d]: Train: [%.4f]' % (epoch, train_loss))
        
        if not silent and val_loader is not None: print('[%d]: Train: [%.4f] Validation: [%.4f]' % (epoch, train_loss, val_loss))
        
    return np.arange(1, nepoch+1), train_losses, val_losses