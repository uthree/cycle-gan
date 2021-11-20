import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from gan import *
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
    batch_size = 1
    model_path = './model.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Load Dataset
    a_dataset = ImageDataset('./vangogh2photo/vangogh2photo/testA/')
    b_dataset = ImageDataset('./vangogh2photo/vangogh2photo/testB/')
    #a_dataset = ImageDataset('./summer2winter_yosemite/testA/')
    #b_dataset = ImageDataset('./summer2winter_yosemite/testB/')

    a_dataloader = torch.utils.data.DataLoader(a_dataset, batch_size=batch_size, shuffle=True)
    b_dataloader = torch.utils.data.DataLoader(b_dataset, batch_size=batch_size, shuffle=True)

    # Initialize CycleGAN
    if os.path.exists(model_path):
        print('Loading model...')
        model = torch.load(model_path, map_location=device)
    else:
        model = CycleGan()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # move to device
    model.to(device)

    # Test
    model.eval()
    for i, (a, b) in enumerate(tqdm(zip(a_dataloader, b_dataloader))):
        a = a.to(device).to(torch.float32)
        b = b.to(device).to(torch.float32)

        a2b = model.g_a2b(a)
        b2a = model.g_b2a(b)

        cv2.imwrite(f'./tests/a2b_{i}_output.png', a2b.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255)
        cv2.imwrite(f'./tests/b2a_{i}_output.png', b2a.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255)
        cv2.imwrite(f'./tests/a2b_{i}_input.png', a.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255)
        cv2.imwrite(f'./tests/b2a_{i}_input.png', b.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255)
if __name__ == '__main__':
    main()