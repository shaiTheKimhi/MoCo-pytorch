import torchvision
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json


from dataset import ImagenetteDataset
from moco_model import MOCO
from image_clf import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = '../imagenette2/'
#data_path = os.path.join('..','imagenette2','imagenette2')

#current_time = datetime.now().strftime("%H_%M_%S")
res_path = './results/Moco'
#os.makedirs(res_path, exist_ok=True)

def train_moco():

    start_epoch,epochs = 0, 1000
    print_every = 10
    q_size = 4096
    batch_size = 256    
    contrast_momentum = 0.999
    # for gamble softmax
    T = 0.07
    # optimizer
    lr = 0.001
    wd = 0.0001

    #moshe: if implement distributed training, remember to change shuffle...

    train_ds = ImagenetteDataset(data_path, crop_size=112, train=True, augment=2, num_augmentations=2)
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size, shuffle=True)

    # Model
    Q_enc = MOCO().to(device=device)

    Q_enc.load_state_dict(torch.load('./moco_checkpoint_fq.pt', map_location=device)['model_state_dict']) #load model from file (only when exists) #TODO: remove this loadings when done training

    # Create K_enc and make sure not to track any gradient... note there is no optimizer but we don't even want to use
    # too much memory
    #K_enc = copy.deepcopy(Q_enc).to(device) #TODO: return this line and remove the two lines below
    K_enc =  MOCO().to(device=device)
    K_enc.load_state_dict(torch.load('./moco_checkpoint_fk.pt', map_location=device)['model_state_dict'])
    for param in K_enc.parameters():
        param.requires_grad = False

    # optimizers
    optimizer = torch.optim.Adam(Q_enc.parameters(), lr=lr, weight_decay=wd)
    #todo: add scedulare (LRSTep or CosineAniling...)

    loss_func = torch.nn.CrossEntropyLoss()
    loss_list = []

    # initialize queue of augmented data
    queue = F.normalize(torch.randn(128, q_size), dim=0).to(device)

    #   log file
    f = open(res_path + '/moco_log.txt', "a+")

    for epoch in range(start_epoch,epochs):
        # Training
        Q_enc.train()
        K_enc.train()
        avg_loss = []

        bar = tqdm(train_loader)

        i, tot_loss, tot_samples = 0, 0.0, 0
        #labels = torch.zeros(b_size, dtype=torch.int64).to(device)
        for q_batch, k_batch, labels in bar:
            optimizer.zero_grad()
            q_batch, k_batch = q_batch.to(device), k_batch.to(device)

            _, q_emb_b = Q_enc(q_batch.type(torch.float))
            _, k_emb_b = K_enc(k_batch.type(torch.float))

            k_emb_b = k_emb_b.detach()


            #todo: use sqeeze and unsqeeze (1 for both)
            b_size = k_emb_b.shape[0]
            f_size = k_emb_b.shape[1]
            l_pos = torch.bmm(q_emb_b.view(b_size, 1, f_size), k_emb_b.view(b_size, f_size, 1))
            l_neg = torch.mm(q_emb_b.view(b_size, f_size), queue)
            logits = torch.cat([l_pos.view(-1, 1), l_neg], dim=1)/T

            labels = torch.zeros(b_size, dtype=torch.int64).to(device)

            loss = loss_func(logits, labels)
            avg_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            enc_params = zip(Q_enc.parameters(), K_enc.parameters())
            for q_parameters, k_parameters in enc_params:
                k_parameters.data = k_parameters.data * contrast_momentum + q_parameters.data * (1. - contrast_momentum)

            queue = torch.cat([queue, k_emb_b.T], dim=1).to(device)
            queue = queue[:, k_emb_b.T.shape[1]:]

        #   Save status
        epoch_log = "Epoch: "+str(epoch+1)+", Iter: "+str(i+1)+', Loss: '+str(loss.item())
        if (epoch+1) % print_every == 0:
            print(epoch_log)
        f.write(epoch_log + '\n')

        tot_samples += k_emb_b.shape[0]
        tot_loss += loss.item() * k_emb_b.shape[0]
        bar.set_description(f'Train Epoch: [{epoch+1}/{epochs}] Loss: {tot_loss / tot_samples}')

        epoch_loss = np.mean(avg_loss)
        loss_list.append(epoch_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': Q_enc.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, res_path + '/moco_checkpoint_fq.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': K_enc.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, res_path + '/moco_checkpoint_fk.pt')

    f.close()
    print_losses(loss_list)



def print_losses(loss_list, graph_name='moco_loss_path', ylab='Accuracy', path=res_path):
    plt.figure()
    plt.plot(np.array(loss_list))
    plt.title(graph_name.replace('_',' '))
    plt.xlabel('Epoch')
    plt.ylabel(ylab)
    plt.savefig(res_path + '/' + graph_name +'.png')



def train_classifier():
    epochs = 100
    batch_size = 32
    # for gamble softmax
    T = 0.07
    # optimizer
    lr = 0.001
    momentum = 0.9
    wd = 0.0001

    train_ds = ImagenetteDataset(data_path, crop_size=288, train=True, augment=1, num_augmentations=3) #return an original image and an augmented image
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size, shuffle=True)

    val_ds = ImagenetteDataset(data_path, crop_size=288, train=False, augment=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)    

    pretask = MOCO()
    #load parameters
    pretask.load_state_dict(torch.load('./moco_checkpoint_fq.pt', map_location=device)['model_state_dict'])
    classifier = ImageClassifier(pretask_model=pretask).to(device)

    losses = train_eval(classifier, train_loader, val_loader, epochs, lr, wd) #train loss, train acc, validation acc

    #save losses logs and plot graph
    for idx, name in enumerate(["train_loss", "train_accuracy", "validation_accuracy"]):
        f = open(f"./results/{name}.txt", "w")
        f.write(json.dumps(torch.tensor(losses[idx]).tolist()))
        f.close()
        print_losses(loss_list=losses[idx], graph_name=name, path="./results/")


def analyze_classifier():
    batch_size = 32
    pretask = MOCO().to(device)
    #load parameters
    pretask.load_state_dict(torch.load('./moco_checkpoint_fq.pt', map_location=device)['model_state_dict'])
    classifier = ImageClassifier(pretask_model=pretask).to(device)
    classifier.predictor.load_state_dict(torch.load("./classifier.pt", map_location=device)["model_state_dict"])

    val_ds = ImagenetteDataset(data_path, crop_size=288, train=False, augment=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)    

    clf = nn.Sequential(nn.Linear(2048, 10, bias=True)).to(device)
    clf.load_state_dict(torch.load("./classifier.pt", map_location=device)["model_state_dict"])
    #losses = train_eval(nn.Sequential(pretask.backbone, clf), val_loader, val_loader, 20, 0.001, 0.0001) #train loss, train acc, validation acc

    print_stats(nn.Sequential(pretask.backbone, clf), val_loader)

    #present_accuracy(classifier, val_loader)

if __name__ == '__main__':
    #train_moco()
    #train_classifier()

    analyze_classifier()
    
