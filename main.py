#!/usr/bin/env python3
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from model import PatchNet
from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor, Grayscale, Compose

import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('-b', '--batch-size', default=32, help='batch size')
@click.option('-e', '--epochs', default=100, help='training time')
@click.option('-l', '--lrate', default=0.01, help='learning rate')
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

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate)
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
                #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, idx, len(train_data_loader), loss.item()))
        print("epoch {} complete: avg. loss: {:.4f}".format(epoch, epoch_loss / len(train_data_loader)))
        print("epoch {} validation loss: {:.4f}".format(epoch, evaluate(model, criterion, device, val_data_loader)))


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
def pred():
    pass

if __name__ == '__main__':
    cli()

