import argparse
import json
from utils import print_process, get_image_data
from cvtrainer import VGGTrainer, VGGALTrainer, ResnetTrainer, ResnetALTrainer
import torch.nn as nn
import pandas as pd


def str_to_module(module_str):

    if module_str == 'relu':
        return nn.ReLU()
    elif module_str == 'sigmoid':
        return nn.Sigmoid()
    elif module_str == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f'Not Implemented: {module_str}')


def get_args():

    parser = argparse.ArgumentParser('Trainer')

    # basic settings
    parser.add_argument("dataset", type=str, help='dataset')
    parser.add_argument("mode", type=str, choices=[
                        "VGG", "VGGAL", "Resnet", "ResnetAL"])
    parser.add_argument("-r", "--repeat", type=int,
                        help='number of repeat training', default=1)

    args = parser.parse_args()

    # detailed settings
    with open('cvconfig.json') as f:
        config = json.load(f)
    for key, value in config.items():
        setattr(args, key, value)

    if type(args.act) == dict:
        for key in args.act:
            args.act[key] = str_to_module(args.act[key])
    else:
        args.act = str_to_module(args.act)
    return args


if __name__ == "__main__":
    import os

    args = get_args()
    train_loader, test_loader, num_class = get_image_data(
        args.dataset, args.train_batch_size, args.test_batch_size, args.augmentation)
    args.num_class = num_class
    print(args)
    all_train_acc, all_test_acc = [], []
    all_train_time = []
    repeat = []
    for i in range(args.repeat):
        if args.mode == "VGG":
            trainer = VGGTrainer(args, train_loader, test_loader)
        elif args.mode == "VGGAL":
            trainer = VGGALTrainer(args, train_loader, test_loader)
        elif args.mode == "Resnet":
            trainer = ResnetTrainer(args, train_loader, test_loader)
        elif args.mode == "ResnetAL":
            trainer = ResnetALTrainer(args, train_loader, test_loader)
        else:
            assert "mode not found, support only vgg and resnet bp/al"
        train_acc, test_acc, train_time = trainer.run()
        all_train_acc.extend(train_acc)
        all_test_acc.extend(test_acc)
        all_train_time.extend(train_time)
        repeat.extend([i+1]*len(train_acc))

    result = {
        "turn": repeat,
        "train_acc": all_train_acc,
        "test_acc": all_test_acc,
        "train_time": all_train_time
    }
    df = pd.DataFrame(result)
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)
    fileName = f"{args.dataset}_{args.mode}.csv"
    df.to_csv(os.path.join(args.outputpath, fileName), index=False)
    print_process(f"Generate {fileName} in {args.outputpath}")
