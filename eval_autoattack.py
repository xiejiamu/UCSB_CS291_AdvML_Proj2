import argparse

import torch

import data_util
import model_util


def parse_args():
    '''Parse input arguments'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eps", type=int, default=8, help="Attack budget: epsilon / 255"
    )
    parser.add_argument(
        '--norm', type=str, default='Linf', choices=['Linf', 'L2', 'L1'], help='Norm to use for attack'
    )
    parser.add_argument(
        '--data_dir', default='./data/', type=str, help="Folder to store downloaded dataset"
    )
    parser.add_argument(
        '--model_path', default='resnet_cifar10.pth', help='Filepath to the trained model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1024, help='Batch size for attack'
    )
    parser.add_argument(
        '--log_path', type=str, default='./log_file.txt'
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Load data
    train_loader, val_loader, test_loader, norm_layer = data_util.cifar10_dataloader(data_dir=args.data_dir)
    model = model_util.ResNet18(num_classes=10)
    # model.normalize = norm_layer
    # model.load(args.model_path, args.device)
    # model = model.to(args.device)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(args.device)

    ## Make sure the model is in `eval` mode.
    model.eval()
    
    eps = args.eps / 255
    # load attack 
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=eps, log_path=args.log_path,
        version='standard', device=args.device)
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    
    adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)


if __name__ == "__main__":
    main()