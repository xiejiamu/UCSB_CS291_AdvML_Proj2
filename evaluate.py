import os
import csv
import argparse
import random

import numpy as np
import torch
from tqdm import tqdm

import data_util
import model_util
import attack_util
from attack_util import ctx_noparamgrad


def parse_args():
    '''Parse input arguments'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eps", type=int, default=8, help="Attack budget: epsilon / 255"
    )
    parser.add_argument(
        "--alpha", type=float, default=2, help="PGD attack step size: alpha / 255"
    )
    parser.add_argument(
        "--attack_step", type=int, default=50, help="Number of PGD iterations"
    )
    parser.add_argument(
        "--confidence", type=float, default=0., help="Confidence tau in C&W loss"
    )
    parser.add_argument(
        "--attack_method", type=str, default="pgd", choices=['fgsm', 'pgd'], help="Adversarial perturbation generate method"
    )
    parser.add_argument(
        "--loss_type", type=str, default="cw", choices=['ce', 'cw'], help="Loss type for attack"
    )
    parser.add_argument(
        '--data_dir', default='./data/', type=str, help="Folder to store downloaded dataset"
    )
    parser.add_argument(
        '--model_path',
        default='/home/sw99/AdvTrain/out/fat/best_model.pth',
        # default='pgd10_eps8.pth',
        help='Filepath to the trained model'
    )
    parser.add_argument("--targeted", action='store_true', default=False)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--seed", default=1, type=int, choices=[1, 412, 886], help="set the seed to make results reproducable")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # fix seed
    setup_seed(args.seed)
    
    # Load data
    _, _, test_loader, norm_layer = data_util.cifar10_dataloader(data_dir=args.data_dir)
    num_classes = 10
    model = model_util.ResNet18(num_classes=num_classes)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(args.device)
    print("==> Loaded checkpoint at ", args.model_path)

    eps = args.eps / 255
    alpha = args.alpha / 255
    target_label = 1 ## only for targeted attack
    print(f"==> Adversarial hyper-parameters: eps={args.eps}/255, alpha={args.alpha}/255")

    ### Your code here for creating the attacker object
    # Note that FGSM attack is a special case of PGD attack with specific hyper-parameters
    # You can also implement a separate FGSM class if you want
    if args.attack_method == "fgsm":
        print("==> Using FGSM to generate adversarial perturbation!")
        attacker = attack_util.FGSMAttack(
            eps=eps, loss_type=args.loss_type,
            targeted=args.targeted, num_classes=num_classes, device=args.device)
    elif args.attack_method == "pgd":
        print("==> Using PGD to generate adversarial perturbation!")
        attacker = attack_util.PGDAttack(
            attack_step=args.attack_step, eps=eps, alpha=alpha, loss_type=args.loss_type,
            targeted=args.targeted, num_classes=num_classes, confidence=args.confidence,
            target=target_label, device=args.device)
    else:
        print("==> Evaluating the model without adversarial perturbation!")
        attacker = None
    ### Your code ends

    total = 0
    clean_correct_num = 0
    robust_correct_num = 0

    ## Make sure the model is in `eval` mode.
    model.eval()

    for data, labels in tqdm(test_loader):
        data = data.to(args.device)
        labels = labels.to(args.device)
        if args.targeted:
            data_mask = (labels != target_label)
            if data_mask.sum() == 0:
                continue
            data = data[data_mask]
            labels = labels[data_mask]
            attack_labels = torch.ones_like(labels).to(args.device)
        else:
            attack_labels = labels
        attack_labels = attack_labels.to(args.device)
        batch_size = data.size(0)
        total += batch_size
        
        with ctx_noparamgrad(model):
            ### clean accuracy
            predictions = model(data)
            clean_correct_num += torch.sum(torch.argmax(predictions, dim = -1) == labels).item()
            
            ### robust accuracy
            # generate perturbation
            perturbed_data = attacker.perturb(model, data, attack_labels) + data
            # predict
            predictions = model(perturbed_data)
            robust_correct_num += torch.sum(torch.argmax(predictions, dim = -1) == labels).item()

    print(f"Total number of images: {total}\nClean accuracy: {clean_correct_num / total}\nRobust accuracy {robust_correct_num / total}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()