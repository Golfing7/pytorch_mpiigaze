#!/usr/bin/env python

import time

import torch
from torch import nn
from fvcore.common.checkpoint import Checkpointer

from gaze_estimation import (create_dataloader, lenet)
from gaze_estimation.utils import (AverageMeter, compute_angle_error,
                                   create_train_output_dir, set_seeds, setup_cudnn)
from gaze_estimation.settings import get_settings


def train(epoch, model, optimizer, scheduler, loss_function, train_loader,
          config):
    print(f'Train {epoch}')

    model.train()

    device = torch.device(config.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    for step, (images, poses, gazes) in enumerate(train_loader):

        images = images.to(device)
        poses = poses.to(device)
        gazes = gazes.to(device)

        optimizer.zero_grad()

        outputs = model(images, poses)
        loss = loss_function(outputs, gazes)
        loss.backward()

        optimizer.step()

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        if step % config.train.log_period == 0:
            print(f'Epoch {epoch} '
                        f'Step {step}/{len(train_loader)} '
                        f'lr {scheduler.get_last_lr()[0]:.6f} '
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'angle error {angle_error_meter.val:.2f} '
                        f'({angle_error_meter.avg:.2f})')

    elapsed = time.time() - start
    print(f'Elapsed {elapsed:.2f}')


def validate(epoch, model, loss_function, val_loader, config):
    print(f'Val {epoch}')

    model.eval()

    device = torch.device(config.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    with torch.no_grad():
        for step, (images, poses, gazes) in enumerate(val_loader):
            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)

            outputs = model(images, poses)
            loss = loss_function(outputs, gazes)

            angle_error = compute_angle_error(outputs, gazes).mean()

            num = images.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(angle_error.item(), num)

    print(f'Epoch {epoch} '
          f'loss {loss_meter.avg:.4f} '
          f'angle error {angle_error_meter.avg:.2f}')

    elapsed = time.time() - start
    print(f'Elapsed {elapsed:.2f}')


def main():
    config = get_settings()

    set_seeds(config.train.seed)
    setup_cudnn(config)

    output_dir = create_train_output_dir(config)
    print(config)

    train_loader, val_loader = create_dataloader(config, is_train=True)
    model = lenet.Model().to(torch.device(config.device))
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD([{
            'params': list(model.parameters()),
            'weight_decay': config.train.weight_decay,
        }], lr=config.train.base_lr, momentum=config.train.momentum, nesterov=config.train.nesterov)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.scheduler.milestones,
            gamma=config.scheduler.lr_decay)
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir.as_posix(),
                                save_to_disk=True)

    if config.train.val_first:
        validate(0, model, loss_function, val_loader, config)

    for epoch in range(1, config.scheduler.epochs + 1):
        train(epoch, model, optimizer, scheduler, loss_function, train_loader,
              config)
        scheduler.step()

        if epoch % config.train.val_period == 0:
            validate(epoch, model, loss_function, val_loader, config)

        if (epoch % config.train.checkpoint_period == 0
                or epoch == config.scheduler.epochs):
            checkpoint_config = {'epoch': epoch, 'config': config.as_dict()}
            checkpointer.save(f'checkpoint_{epoch:04d}', **checkpoint_config)


if __name__ == '__main__':
    main()
