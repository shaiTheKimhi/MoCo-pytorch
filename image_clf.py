import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm



class ImageClassifier(nn.Module):
    def __init__(self, pretask_model, feature_dim=10):
        super(ImageClassifier, self).__init__()
        self.feature_extractor = pretask_model.backbone #the feature extractor of the pretrained model
        #make the feature extractor untrainable
        for params in self.feature_extractor.parameters():
            params.require_grad = False
        
        #TODO: choose between these two
        #self.predictor = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
        #                              nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        self.predictor = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                                nn.Linear(512, 64, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
                                nn.ReLU(inplace=True), nn.Linear(64, feature_dim, bias=True)) #can check different architecture
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        out = self.predictor(x)
        return out #will be mixed with classification loss

def train_eval(classifier, train_dl, valid_dl, epochs=20, lr=0.01, weight_decay=0.001):
    torch.autograd.set_detect_anomaly(True)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)
    train_losses, train_acc, valid_acc = [], [] , []

    for epoch in range(epochs):
        avg_loss_train, avg_acc_train, avg_acc_val = [], [], []
        tot_loss, tot_samples, t_acc = 0.0, 0, 0
        bar = tqdm(train_dl)
        for x, aug_x, y in bar:
            classifier.train()
            optimizer.zero_grad()
            
            x, aug_x, y = x.to(device), aug_x, y.to(device)

            # Run through network
            res = classifier(x)
            t_acc += torch.sum(torch.eq(torch.argmax(res, dim=1), y)) #accuracy is calculated only over original images
            # Calculate loss and backprop
            loss = loss_func(res, y)
            avg_loss_train.append(loss.item())

            loss.backward()
            optimizer.step()
            #run again with augmented image
            res = classifier(aug_x)
            loss = loss_func(res, y)
            avg_loss_train.append(loss.item())
            loss.backward()
            optimizer.step()

            

            tot_samples += x.shape[0]
            tot_loss += loss.item() * x.shape[0]
            bar.set_description(f'Train Epoch: [{epoch+1}/{epochs}] Loss: {tot_loss / tot_samples} Accuracy: {t_acc / tot_samples}')

        v_acc, tot_samples, vbar = 0.0, 0, tqdm(valid_dl)
        for x, _, y in vbar:
            classifier.eval()

            x, y = x.to(device), y.to(device)

            res = classifier(x)
            loss = loss_func(res, y)
            v_acc += torch.sum(torch.eq(torch.argmax(res, dim=1), y))
            tot_samples += x.shape[0]
            avg_acc_val.append(acc)
            vbar.set_description(f'Test Epoch: [{epoch+1}/{epochs}] Accuracy: {v_acc / tot_samples}')
        
        epoch_loss = np.mean(avg_loss_train)
        train_losses.append(epoch_loss)
        train_acc.append(t_acc / tot_samples)
        valid_acc.append(v_acc / tot_samples)
        #TODO: save best model
        
        return train_losses, train_acc, valid_acc

        #status_str = f'Epoch = {epoch + 1}, Loss = {epoch_loss}, Accuracy: {acc / tot_samples}'
        #f.write(status_str + '\n')
        '''
        torch.save({
            'epoch': epoch,
            'model_state_dict': linear_clf.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, moco_path + '/clf_checkpoint.pt')

    f.close()
    plt.figure()
    plt.plot(np.array(train_loss_list))
    plt.plot(np.array(val_loss_list))
    plt.title('Linear Classifier Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig(moco_path + '/clf_loss_graph.png')

    plt.figure()
    plt.plot(np.array(top_1_acc))
    plt.title(f'Linear Classifier Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(moco_path + '/clf_acc_graph.png')
    '''

