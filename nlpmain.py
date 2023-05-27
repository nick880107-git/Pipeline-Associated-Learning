import argparse
import json
from utils import get_word_vector, get_nlp_data, print_process
from nlptrainer import TransformerTrainer, TransformerALTrainer, LSTMTrainer, LSTMALTrainer
import torch.nn as nn
import pandas as pd

import nltk
nltk.download('stopwords')

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
    parser.add_argument("max_len", type=int, help='max length of sentences')
    parser.add_argument("mode", type=str,choices=["TransformerAL","Transformer", "LSTM", "LSTMAL"])
    parser.add_argument("-r","--repeat", type=int, help='number of repeat training', default=1)

    args = parser.parse_args()

    # detailed settings
    with open('nlpconfig.json') as f:
        config = json.load(f)
    for key, value in config.items():
        setattr(args, key, value)
    args.act = str_to_module(args.act)
    return args 

if __name__ == "__main__":
    import os
    
    args = get_args()    
    train_loader, test_loader, class_num, vocab = get_nlp_data(args)
    args.pretrained_embedding = get_word_vector(vocab, args.pretrained_model)
    args.class_num = class_num
    print(args)
    all_train_acc, all_test_acc = [], []
    all_train_time = []
    repeat = []
    for i in range(args.repeat):
        if args.mode == "Transformer":
            trainer = TransformerTrainer(args, train_loader, test_loader)
        elif args.mode == "TransformerAL":
            trainer = TransformerALTrainer(args, train_loader, test_loader)
        elif args.mode == "LSTM":
            trainer = LSTMTrainer(args, train_loader, test_loader)
        elif args.mode == "LSTMAL":
            trainer = LSTMALTrainer(args, train_loader, test_loader)
        else:
            assert "mode not found, support only transformer and lstm bp/al"
        train_acc, test_acc, train_time = trainer.run()
        all_train_acc.extend(train_acc)
        all_test_acc.extend(test_acc)
        all_train_time.extend(train_time)
        repeat.extend([i+1]*len(train_acc))

    result = {
        "turn":repeat,
        "train_acc":all_train_acc,
        "test_acc":all_test_acc,
        "train_time":all_train_time
    }
    df = pd.DataFrame(result)
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)
    fileName = f"{args.dataset}_{args.max_len}_{args.mode}.csv"
    df.to_csv(os.path.join(args.outputpath, fileName), index=False)
    print_process(f"Generate {fileName} in {args.outputpath}")
