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

from dataset import ImagenetteDataset
from moco_model import MOCO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = '../imagenette2/imagenette2/'
#data_path = os.path.join('..','imagenette2','imagenette2')

current_time = datetime.now().strftime("%H_%M_%S")
res_path = './results/moco_' + current_time
os.makedirs(res_path, exist_ok=True)

def main():

    start_epoch,epochs =0, 1000
    print_every = 50
    q_size = 4096
    batch_size = 256
    contrast_momentum = 0.999
    # for gamble softmax
    T = 0.07
    # optimizer
    lr = 0.001
    momentum = 0.9
    wd = 0.0001

    # load ds, todo: put in diffrent function
    #todo: check with defult crop_size(299)
    #moshe: if implement distributed training, remember to change shuffle...

    train_ds = ImagenetteDataset(data_path, crops_size=112, train=True, augment=2)
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size, shuffle=True)

    mem_ds = ImagenetteDataset(data_path, crops_size=112, train=True, augment=0) #moshe: need augment=1?
    mem_loader = torch.utils.data.DataLoader(mem_ds, batch_size=batch_size, shuffle=False)

    val_ds = ImagenetteDataset(data_path, crops_size=112, train=False, augment=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    Q_enc = MOCO().to(device=device)

    # Create K_enc and make sure not to track any gradient... note there is no optimizer but we don't even want to use
    # too much memory
    K_enc = copy.deepcopy(Q_enc).to(device)
    for param in K_enc.parameters():
        param.requires_grad = False

    # optimizers
    # optimizer = torch.optim.SGD(f_q.parameters(), lr=lr, momentum=SGD_momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(Q_enc.parameters(), lr=lr, weight_decay=wd)
    #todo: add scedulare (LRSTep or CosineAniling...)

    loss_func = torch.nn.CrossEntropyLoss()
    loss_list = []

    # initialize queue of augmented data
    queue = F.normalize(torch.randn(128, q_size), dim=0).to(device)

    #   log file
    f = open(res_path + '/moco_log.txt', "a")

    for epoch in range(start_epoch,epochs):
        # Training
        Q_enc.train()
        K_enc.train()
        avg_loss = []

        bar = tqdm(train_loader)

        i, tot_loss, tot_samples = 0, 0.0, 0
        labels = torch.zeros(b_size, dtype=torch.int64).to(device)
        for q_batch, k_batch, _ in bar:

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
            if i % print_every ==0:
                print(epoch_log)
            f.write(epoch_log + '\n')

            tot_samples += k_emb_b.shape[0]
            tot_loss += loss.item() * k_emb_b.shape[0]
            bar.set_description(f'Train Epoch: [{epoch+1}/{max_epochs}] Loss: {tot_loss / tot_samples}')

        epoch_loss = np.mean(avg_loss)
        loss_list.append(epoch_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': f_q.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, res_path + '/moco_checkpoint_fq.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': f_k.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, res_path + '/moco_checkpoint_fk.pt')

    f.close()
    print_losses(loss_list)

def print_losses(loss_list):
    plt.figure()
    plt.plot(np.array(loss_list))
    plt.title('MoCo Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(res_path + '/moco_loss_graph.png')


if __name__ == '__main__':
    main()