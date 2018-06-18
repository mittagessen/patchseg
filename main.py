#!/usr/bin/env python3
import numpy as np

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from model import PatchNet
from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor, Grayscale, Compose

import click

from skimage.segmentation import slic
from skimage.measure import regionprops

from PIL import Image

@click.group()
def cli():
    pass

@cli.command()
@click.option('-b', '--batch-size', default=32, help='batch size')
@click.option('-e', '--epochs', default=100, help='training time')
@click.option('-l', '--lrate', default=0.001, help='initial learning rate')
@click.option('-w', '--workers', default=0, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-v', '--validation', default='val', help='validation set location')
@click.argument('ground_truth', nargs=1, type=click.Path(exists=True, dir_okay=True))
def train(batch_size, epochs, lrate, workers, device, validation, ground_truth):
    train_set = ImageFolder(ground_truth, transform=Compose([Grayscale(), ToTensor()]))
    val_set = ImageFolder(validation, transform=Compose([Grayscale(), ToTensor()]))

    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=batch_size)

    device = torch.device(device)

    model = PatchNet().to(device)
    criterion = nn.CrossEntropyLoss()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lrate)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    for epoch in range(epochs):
        epoch_loss = 0
        with click.progressbar(train_data_loader, label='epoch {}'.format(epoch)) as bar:
            for sample in bar:
                input, target = sample[0].to(device), sample[1].to(device)
                optimizer.zero_grad()
                o = model(input)
                loss = criterion(o, target)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), 'epoch_{}.ckpt'.format(epoch))
        print("epoch {} complete: avg. loss: {:.4f}".format(epoch, epoch_loss / len(train_data_loader)))
        val_loss = evaluate(model, criterion, device, val_data_loader)
    #    scheduler.step(val_loss)
        print("epoch {} validation loss: {:.4f}".format(epoch, val_loss))


def evaluate(model, criterion, device, data_loader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sample in data_loader:
            input, target = sample[0].to(device), sample[1].to(device)
            o = model(input)
            val_loss += float(criterion(o, target))
    model.train()
    return val_loss / len(data_loader)


@cli.command()
@click.option('-m', '--model', default=None, help='model file')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.argument('images', nargs=-1)
def pred(model, device, images):
    m = PatchNet()
    m.load_state_dict(torch.load(model))
    device = torch.device(device)
    m.to(device)

    transform = ToTensor()

    cmap = {0: (230, 25, 75, 127),
            1: (60, 180, 75, 127),
            2: (255, 225, 25, 127),
            3: (0, 130, 200, 127)}

    for img in images:
        im = Image.open(img)
        gray = im.convert('L')
        gray = gray.resize((im.size[0]//3, im.size[1]//3))
        sp = slic(gray, n_segments=3000)
        props = regionprops(sp)
        cls = np.zeros(sp.shape)
        with click.progressbar(props, label='patches') as bar:
            for prop in bar:
                y = int(prop.centroid[0])
                x = int(prop.centroid[1])
                siz = 14
                patch = gray.crop((x-siz, y-siz, x+siz, y+siz))
                o = m.forward(transform(patch).unsqueeze(0).to(device))
                cls[sp == prop.label] = o.argmax().item()
        # color overlay
        overlay = np.zeros(sp.shape + (4,))
        for idx, val in cmap.items():
            overlay[cls == idx] = val
        gray = gray.convert('RGBA')
        im = Image.alpha_composite(gray.convert('RGBA'), Image.fromarray(overlay.astype('uint8')))
        im.save(os.path.splitext(img)[0] + '_out.png')

if __name__ == '__main__':
    cli()

