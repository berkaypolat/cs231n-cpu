from utils import load_checkpoint
import torch.optim as optim
import torch

class CheXpertTrainer():

    def train (self, model, dataLoaderTrain, nnClassCount, trMaxEpoch, checkpoint, use_cuda):
        
        #SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
                
        #SETTINGS: LOSS
        loss = torch.nn.BCELoss(size_average = True)
        
        #LOAD CHECKPOINT 
        if checkpoint != None:
            load_checkpoint(checkpoint, model, optimizer, use_cuda)

        budget = 0.3
        
        #TRAIN THE NETWORK
        
        for epochID in range(0, trMaxEpoch):
            
            
            batchs, losst = CheXpertTrainer.epochTrain(model, dataLoaderTrain, optimizer, trMaxEpoch, nnClassCount, loss, budget)
         
        return batchs, losst #, losse        
    #-------------------------------------------------------------------------------- 
       
    def epochTrain(model, dataLoader, optimizer, epochMax, classCount, loss, budget):
        
        batch = []
        losstrain = []
        losseval = []
        
        lmbda = 0.1    #start with reasonable value
        
        model.train()

        for batchID, (varInput, target) in enumerate(dataLoader):
            
            batch.append(batchID)
            varTarget = torch.stack(target).float().transpose(0,1).to(device)
            print(varTarget.shape)
            #varTarget = target.cuda(non_blocking = True)
            
            #varTarget = target.cuda()         

            bs, c, h, w = varInput.size()
            varInput = varInput.view(-1, c, h, w)

            varOutput, confidence = model(varInput)
            confidence = torch.sigmoid(confidence)
            
            
            # prevent any numerical instability
            eps = 1e-12
            varOutput = torch.clamp(varOutput, 0. + eps, 1. - eps)
            confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
            
            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).to(device)
            conf = confidence * b + (1 - b)
            pred_new = varOutput * conf + varTarget * (1 - conf)
            
            first_loss = loss(pred_new, varTarget)
            second_loss = torch.mean(torch.mean(-torch.log(confidence),1))
            
            loss_value = first_loss + lmbda * second_loss
            
            if budget > second_loss.item():
                lmbda = lmbda / 1.01
            elif budget <= second_loss.item():
                lmbda = lmbda / 0.99
            
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            l = loss_value.item()
            losstrain.append(l)
            print(l)
            
        return batch, losstrain