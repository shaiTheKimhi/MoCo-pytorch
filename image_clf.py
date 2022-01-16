import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageClassifier(nn.Module):
    def __init__(self, pretask_model, feature_dim=10):
        super(ImageClassifier, self).__init__()
        self.feature_extractor = pretask_model.backbone #the feature extractor of the pretrained model
        #make the feature extractor untrainable
        for params in self.feature_extractor.parameters():
            params.require_grad = False
        
        self.predictor = nn.Sequential(nn.Linear(2048, 10, bias=True))
       
       
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        out = self.predictor(x)
        return out #will be mixed with classification loss

def train_eval(classifier, train_dl, valid_dl, epochs=20, lr=0.01, weight_decay=0.001):
    torch.autograd.set_detect_anomaly(True)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=0, verbose=True)
    loss_func = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max = 0.0
    classifier = classifier.to(device)
    train_losses, train_acc, valid_acc = [], [] , []
    for epoch in range(epochs):
        
        t_acc, avg_loss_train = train_epoch(classifier, train_dl, optimizer, loss_func, epoch, epochs)
        
        v_acc = valid_epoch(classifier, valid_dl, epoch, epochs)
       
        epoch_loss = np.mean(avg_loss_train)
        train_losses.append(epoch_loss)
        train_acc.append(t_acc)
        valid_acc.append(v_acc)
        if v_acc >= max: #save best classifier for further statistic check
            max = v_acc
            torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, 'results/classifier.pt')
        
        scheduler.step()
        
        
    return train_losses, train_acc, valid_acc

  
def train_epoch(classifier, train_dl, optimizer, loss_func, epoch, epochs):
    avg_loss_train = []
    tot_loss, tot_samples, t_acc = 0.0, 0, 0
    bar = tqdm(train_dl)
    for x, aug_x, y in bar:
        torch.cuda.empty_cache()
        classifier.train()
        optimizer.zero_grad()
        
        x, aug_x, y = x.to(device), aug_x.to(device), y.to(device)

        # Run through network
        res = classifier(x)
        t_acc += torch.sum(torch.eq(torch.argmax(res, dim=1), y)) #accuracy is calculated only over original images
        # Calculate loss and backprop
        loss = loss_func(res, y)
        avg_loss_train.append(loss.item())

        loss.backward()
        optimizer.step()
        #run again with augmented image
        if aug_x.shape == x.shape:
            optimizer.zero_grad()
            res = classifier(aug_x)
            loss = loss_func(res, y)
            avg_loss_train.append(loss.item())
            loss.backward()
            optimizer.step()

        tot_samples += x.shape[0]
        tot_loss += loss.item() * x.shape[0]
        bar.set_description(f'TEpoch:[{epoch+1}/{epochs}]  Accuracy:{t_acc / tot_samples}', refresh=True)
    
    return t_acc / tot_samples, avg_loss_train
    
def valid_epoch(classifier, valid_dl, epoch, epochs):
    v_acc, tot_samples, vbar = 0.0, 0, tqdm(valid_dl)
    for x, _, y in vbar:
        classifier.eval()

        x, y = x.to(device), y.to(device)

        res = classifier(x)
        v_acc += torch.sum(torch.eq(torch.argmax(res, dim=1), y))
        tot_samples += x.shape[0]
        vbar.set_description(f'VEpoch: [{epoch+1}/{epochs}] Acc: {v_acc / tot_samples}', refresh=True)
    
    return v_acc / tot_samples


'''
Present the model's accuracy over test/validation set and it's confusion matrix
'''
def present_accuracy(model, dataloader, classes=10, show=True):
    model.eval() # put in evaluation mode
    total_correct = 0
    total = 0
    confusion_matrix = np.zeros([classes,classes], int)
    with torch.no_grad():
        vbar = tqdm(dataloader)
        for X, _, y in vbar:
            X, y = X.to(device), y.to(device)
            y_pred_log_proba = model(X)
            predicted = torch.argmax(y_pred_log_proba, dim=1)
            total += X.shape[0]
            total_correct += (predicted == y).sum().item()
            for i, l in enumerate(y):
                confusion_matrix[l.item(), predicted[i].item()] += 1 

    model_accuracy = total_correct / total * 100
    print("Test accuracy: {:.3f}%".format(model_accuracy))
    if show:
        labels = tuple([str(i) for i in range(classes)])
        
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
        plt.ylabel('Actual Category')
        plt.yticks(range(classes), labels)
        plt.xlabel('Predicted Category')
        plt.xticks(range(classes), labels)
        plt.savefig("./results/confusion.png")
    return model_accuracy

def print_stats(model, dataloader, classes=10):
    model.eval() # put in evaluation mode
    trues = []
    preds = []
    vbar = tqdm(dataloader)
    with torch.no_grad():
        for X, _, y in vbar:
            X, y = X.to(device), y.to(device)
            trues+=list(y.cpu())
            y_pred_log_proba = model(X)
            predicted = torch.argmax(y_pred_log_proba, dim=1)
            preds+= list(predicted.cpu())            
    file = open("results/stats.txt")
    file.write(metrics.classification_report(trues, preds, digits=classes))
    file.close()