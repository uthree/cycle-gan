import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from gan_revunet import Generator, Discriminator
from imagedataset import ImageDataset
import cv2

class CycleGan(nn.Module):
    """Some Information about CycleGan"""
    def __init__(self):
        super(CycleGan, self).__init__()
        self.g_a2b = Generator()
        self.g_b2a = Generator()

        self.d_a = Discriminator()
        self.d_b = Discriminator()


def main():
    # hyperparameters
    batch_size = 6
    num_epochs = 100
    model_path = './model.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Dataset
    a_dataset = ImageDataset('./vangogh2photo/trainA/')
    b_dataset = ImageDataset('./vangogh2photo/trainB/')

    a_dataloader = torch.utils.data.DataLoader(a_dataset, batch_size=batch_size, shuffle=True)
    b_dataloader = torch.utils.data.DataLoader(b_dataset, batch_size=batch_size, shuffle=True)

    # Initialize CycleGAN
    if os.path.exists(model_path):
        print('Loading model...')
        model = torch.load(model_path)
    else:
        model = CycleGan()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # move to device
    model.to(device)

    # Loss
    l1_loss = nn.L1Loss().to(device)
    bce_loss = nn.BCELoss().to(device)

    # Training
    bar = tqdm(total = num_epochs)
    for epoch in range(num_epochs):
        for i, (a, b) in enumerate(zip(a_dataloader, b_dataloader)):
            if a.size()[0] != batch_size or b.size()[0] != batch_size:
                continue
            a = a.to(device).to(torch.float32)
            b = b.to(device).to(torch.float32)

            # Train Generator
            optimizer.zero_grad()

            fake_a2b = model.g_a2b(a)
            fake_b2a = model.g_b2a(b)

            fake_a2b2a = model.g_b2a(fake_a2b)
            fake_b2a2b = model.g_a2b(fake_b2a)

            g_loss =  l1_loss(fake_a2b2a, a) + l1_loss(fake_b2a2b, b) 
            g_loss += bce_loss(model.d_a(fake_b2a), torch.ones(batch_size, 1).to(device)) + bce_loss(model.d_b(fake_a2b), torch.ones(batch_size, 1).to(device))
            g_loss += l1_loss(model.g_a2b(b), b) + l1_loss(model.g_b2a(a), a)

            g_loss.backward()
            optimizer.step()

            # Train Discriminator
            optimizer.zero_grad()

            # real
            d_loss =  bce_loss(model.d_a(a), torch.ones(batch_size, 1).to(device)) + bce_loss(model.d_b(b), torch.ones(batch_size, 1).to(device))
            # fake
            d_loss += bce_loss(model.d_a(fake_b2a.detach()), torch.zeros(batch_size, 1).to(device)) + bce_loss(model.d_b(fake_a2b.detach()), torch.zeros(batch_size, 1).to(device))
            # fake (twice translation)
            d_loss += bce_loss(model.d_a(fake_a2b2a.detach()), torch.zeros(batch_size, 1).to(device)) + bce_loss(model.d_b(fake_b2a2b.detach()), torch.zeros(batch_size, 1).to(device))

            d_loss.backward()
            optimizer.step()
            bar.set_description(f'Epoch: {epoch}, d_loss: {d_loss.item()} g_loss: {g_loss.item()}')
            bar.update(0)
            if i % 10 == 0:
                # save to image
                cv2.imwrite('./results/fake_a2b_{}.png'.format(epoch), fake_a2b[0].detach().cpu().numpy().transpose(1,2,0)*255)
                cv2.imwrite('./results/fake_b2a_{}.png'.format(epoch), fake_b2a[0].detach().cpu().numpy().transpose(1,2,0)*255)

        # save model
        bar.set_description(f'Epoch: {epoch}, d_loss: {d_loss.item()} g_loss: {g_loss.item()}')
        bar.update(1)
        torch.save(model, model_path)
        if epoch % 2 == 0:
            torch.save(model, './checkpoints/model_{}.pt'.format(epoch))
        print('Model saved!')

if __name__ == '__main__':
    main()