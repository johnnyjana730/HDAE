import click
import torch
from parser import parse_args
from Train.get_model import *

def train():
    click.echo(f'Now GPU: {torch.cuda.get_device_name(0)}')
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    config = parse_args()
    train()