import argparse
import logging
import os
import time

import torch
import copy
from tqdm import tqdm

from data_util import cifar10_dataloader
from model_util import ResNet18
from attack_util import *

logger = logging.getLogger(__name__)

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=160, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument(
        "--eps", type=int, default=8, help="Attack budget: epsilon / 255"
    )
    parser.add_argument(
        "--alpha", type=float, default=2, help="PGD attack step size: alpha / 255"
    )
    parser.add_argument(
        "--attack_step", type=int, default=10, help="Number of PGD iterations"
    )
    parser.add_argument(
        "--confidence", type=float, default=0., help="Confidence tau in C&W loss"
    )
    parser.add_argument(
        '--norm', type=str, default='Linf', choices=['Linf', 'L2', 'L1'], help='Norm to use for attack'
    )
    parser.add_argument(
        "--train_method", type=str, default="fat", choices=['at', 'fat'],
        help="Adversarial Training or Fast Adversarial Training"
    )
    parser.add_argument(
        '--data_dir', default='./data/', type=str, help="Folder to store downloaded dataset"
    )
    parser.add_argument(
        '--save_dir', default='./out/fat/', help='Filepath to the trained model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1024, help='Batch size for attack'
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logfile = os.path.join(args.save_dir, 'train.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    handlers = [logging.FileHandler(logfile, mode='a+'),
                logging.StreamHandler()]
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    logger.info(args)

    train_loader, val_loader, test_loader, dataset_normalization = cifar10_dataloader(args.batch_size, args.data_dir)
    mean, std = dataset_normalization.get_params()

    epsilon = (args.eps / 255.) / std
    alpha = (args.alpha / 255.) / std
    upper_limit = ((1 - mean) / std).cuda()
    lower_limit = ((0 - mean) / std).cuda()
    epsilon = epsilon.view(3, 1, 1).cuda()
    alpha = alpha.view(3, 1, 1).cuda()
    lower_limit = lower_limit.view(3, 1, 1)
    upper_limit = upper_limit.view(3, 1, 1)

    # print(epsilon)

    model = ResNet18().cuda()
    # be lazy...
    opt = torch.optim.SGD(model.parameters(), args.lr, args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    best_adv_acc = 0
    nat_acc_best_adv = 0

    results = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "adv_val_loss": [],
        "adv_val_acc": []
    }

    logger.info("Start training now!")
    start_train_time = time.time()

    for epoch in range(0, args.epochs):
        results['epoch'].append(epoch)
        logger.info(f"Epoch {epoch} starts ...")

        logger.info("Training ...")
        model.train()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(tqdm(train_loader)):
            X = X.cuda()
            y = y.cuda()
            delta = torch.zeros(X.size(0), 3, 32, 32).cuda()
            for j in range(len(epsilon)):
                delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())

            # print("delta.shape:", delta.shape)
            # print("X.shape:", X.shape)
            # print("lower_limit.shape:", lower_limit.shape)
            # print("upper_limit.shape:", upper_limit.shape)

            delta.data = torch.clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True

            if args.train_method == 'fat':
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = torch.clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)

            elif args.train_method == 'at':
                for _ in range(args.attack_step):
                    output = model(X + delta)
                    loss = criterion(output, y)
                    loss.backward()
                    grad = delta.grad.detach()

                    # print(delta.shape)
                    # print(alpha.shape)
                    # print(grad.shape)
                    # print(epsilon.shape)

                    delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = torch.clamp(delta, lower_limit - X, upper_limit - X)
                    delta.grad.zero_()

            delta = delta.detach()

            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()

            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()

        results['train_loss'].append(train_loss / train_n)
        results['train_acc'].append(train_acc / train_n)
        logger.info(f"Epoch: {epoch}")
        logger.info(f"Train Loss: {train_loss / train_n}")
        logger.info(f"Train Acc: {train_acc / train_n}")

        logger.info("Evaluating the standard accuracy ...")
        val_loss = 0
        val_acc = 0
        val_n = 0
        model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(tqdm(val_loader)):
                X, y = X.cuda(), y.cuda()
                output = model(X)
                loss = F.cross_entropy(output, y)
                val_loss += loss.item() * y.size(0)
                val_acc += (output.max(1)[1] == y).sum().item()
                val_n += y.size(0)

        results['val_loss'].append(val_loss / val_n)
        results['val_acc'].append(val_acc / val_n)
        logger.info(f"Epoch: {epoch}")
        logger.info(f"Validation Loss: {val_loss / val_n}")
        logger.info(f"Validation Acc: {val_acc / val_n}")

        logger.info("Evaluating the robust accuracy ...")
        adv_val_loss = 0
        adv_val_acc = 0
        adv_val_n = 0
        model.eval()
        for i, (X, y) in enumerate(tqdm(val_loader)):
            X, y = X.cuda(), y.cuda()
            val_delta = torch.zeros(X.size(0), 3, 32, 32).cuda()
            val_delta.requires_grad = True

            val_output = model(X + val_delta)
            val_loss = F.cross_entropy(val_output, y)
            val_loss.backward()
            val_grad = val_delta.grad.detach()
            val_delta.data = torch.clamp(val_delta + alpha * torch.sign(val_grad), -epsilon, epsilon)
            val_delta.data = torch.clamp(val_delta, lower_limit - X, upper_limit - X)

            with torch.no_grad():
                output = model(X + val_delta)
                loss = F.cross_entropy(output, y)
                adv_val_loss += loss.item() * y.size(0)
                adv_val_acc += (output.max(1)[1] == y).sum().item()
                adv_val_n += y.size(0)

        results['adv_val_loss'].append(adv_val_loss / adv_val_n)
        results['adv_val_acc'].append(adv_val_acc / adv_val_n)
        logger.info(f"Epoch: {epoch}")
        logger.info(f"Adversarial Validation Loss: {adv_val_loss / adv_val_n}")
        logger.info(f"Adversarial Validation Acc: {adv_val_acc / adv_val_n}")

        if adv_val_acc > best_adv_acc:
            best_adv_acc = adv_val_acc
            nat_acc_best_adv = val_acc

            best_ckpt = {}
            best_ckpt['state_dict'] = copy.deepcopy(model.state_dict())
            best_ckpt['opt'] = copy.deepcopy(opt.state_dict())
            best_ckpt['scheduler'] = copy.deepcopy(scheduler.state_dict())
            best_ckpt['epoch'] = epoch
            best_ckpt['adv_acc'] = adv_val_acc
            best_ckpt['nat_acc'] = val_acc

            torch.save(best_ckpt, os.path.join(args.save_dir, 'best_model.pth'))
            logger.info(f"New best record saved at {os.path.join(args.save_dir, 'best_model.pth')}!")

        logger.info(f"Epoch {epoch} done!")


if __name__ == "__main__":
    main()
