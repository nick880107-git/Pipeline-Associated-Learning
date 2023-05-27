import math
import re
import string
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import io
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import make_grid
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm

stop_words = set(stopwords.words('english'))


def getParam(model):
    '''
    Use for calculate effective param/ affiliate param for AL model
    '''
    effective, affiliate = 0, 0
    for i in range(len(model.seq)):
        param = model.seq[i].getParam()
        effective += param["f"] + param["h"]
        affiliate += param["g"]
        if i == len(model.seq)-1:
            effective += param["b"]
        else:
            affiliate += param["b"]
    print(f"Effective param: {effective}, Affiliate param: {affiliate}")


def setup_seed(seed):
    if int(seed) == -1:
        return None
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed


def set_device(model, device: list):
    assert len(device) == len(
        model.seq), "Not all layers given to specified device, check the length of device list is equal to the number of model layers."

    model.device = device
    for i in range(len(device)):
        model.seq[i] = model.seq[i].to(device[i])


def print_process(string):
    total = 100
    prefix = 25
    remains = total - prefix - len(f" {string} ")
    res = f"|{'='*prefix} {string} {'='*remains}|"
    print(res)

# # image classification utils


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_batch(dataloader):
    '''Show a sample batch in dataloader'''
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    imshow(make_grid(images))  # Using Torchvision.utils make_grid function
    return images, labels


def get_image_data(dataset, train_bsz, test_bsz, augmentation_type):
    if dataset == "cifar10":
        n_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == "cifar100":
        n_classes = 100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "tinyImageNet":
        n_classes = 200
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    else:
        raise ValueError("Dataset not supported: {}".format(dataset))

    normalize = transforms.Normalize(mean=mean, std=std)
    if dataset == "cifar10" or dataset == "cifar100":

        weak_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        strong_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    if dataset == "tinyImageNet":
        weak_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        strong_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize
        ])

    if augmentation_type == "basic":
        source_transform = weak_transform
        target_transform = None
    elif augmentation_type == "strong":
        source_transform = strong_transform
        target_transform = None
    else:
        raise ValueError(
            "Augmentation type not supported: {}".format(augmentation_type))

    if dataset == "cifar10":
        train_set = datasets.CIFAR10(
            root='./cifar10', transform=source_transform, target_transform=target_transform,  download=True)
        test_set = datasets.CIFAR10(
            root='./cifar10', train=False, transform=test_transform)
    elif dataset == "cifar100":
        train_set = datasets.CIFAR100(
            root='./cifar100', transform=source_transform, target_transform=target_transform, download=True)
        test_set = datasets.CIFAR100(
            root='./cifar100', train=False, transform=test_transform)
    elif dataset == "tinyImageNet":
        train_set = datasets.ImageFolder(
            './tiny-imagenet-200/train', transform=source_transform)
        test_set = datasets.ImageFolder(
            './tiny-imagenet-200/val', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_bsz, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_bsz, shuffle=False, pin_memory=True)

    return train_loader, test_loader, n_classes


def lr_scheduling(optimizer, base_lr, end_lr, step, max_steps):
    q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    lr = base_lr * q + end_lr * (1 - q)
    if optimizer:
        for g in optimizer.param_groups:
            g['lr'] = lr
    return lr


def weight_init(module, init_mode):

    if init_mode == None:
        return

    if isinstance(module, nn.Module):
        if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            for m in module:
                weight_init(m, init_mode)
        else:
            try:
                weight_init(module.seq, init_mode)
            except:
                pass

    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        if init_mode == 'kaiming':
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu")
        elif init_mode == 'xavier':
            nn.init.xavier_normal_(
                module.weight, gain=nn.init.calculate_gain('relu'))
        elif init_mode == 'noraml':
            nn.init.normal_(module.weight, 0, 0.01)
        if module.bias != None:
            nn.init.constant_(module.bias, 0)

    if isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        if module.bias != None:
            nn.init.constant_(module.bias, 0)

# # nlp utils


def get_word_vector(vocab, emb='glove'):

    if emb == 'glove':
        fname = 'embedding/glove.6B.300d.txt'

        with open(fname, 'rt', encoding='utf8') as fi:
            full_content = fi.read().strip().split('\n')

        data = {}
        for i in tqdm(range(len(full_content)), total=len(full_content), desc='loading glove vocabs...'):
            i_word = full_content[i].split(' ')[0]
            if i_word not in vocab.keys():
                continue
            i_embeddings = [float(val)
                            for val in full_content[i].split(' ')[1:]]
            data[i_word] = i_embeddings

    elif emb == 'fasttext':
        fname = 'wiki-news-300d-1M.vec'

        fin = io.open(fname, 'r', encoding='utf-8',
                      newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}

        for line in tqdm(fin, total=1000000, desc='loading fasttext vocabs...'):
            tokens = line.rstrip().split(' ')
            if tokens[0] not in vocab.keys():
                continue
            data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)

    else:
        raise Exception('emb not implemented')

    w = []
    find = 0
    for word in vocab.keys():
        try:
            w.append(torch.tensor(data[word]))
            find += 1
        except:
            w.append(torch.rand(300))

    print('found', find, 'words in', emb)
    return torch.stack(w, dim=0)


def load_raw_data(dataset):
    if dataset != 'imdb':
        train_data = load_dataset(dataset, split='train')
        test_data = load_dataset(dataset, split='test')

        if dataset == 'dbpedia_14':
            tf = 'content'
            class_num = 14
        elif dataset == 'ag_news':
            tf = 'text'
            class_num = 4
        elif dataset == 'banking77':
            tf = 'text'
            class_num = 77
        elif dataset == 'emotion':
            tf = 'text'
            class_num = 6
        elif dataset == 'rotten_tomatoes':
            tf = 'text'
            class_num = 2
        elif dataset == 'yelp_review_full':
            tf = 'text'
            class_num = 5
        elif dataset == 'sst2':
            tf = 'sentence'
            class_num = 2
            test_data = load_dataset(dataset, split='validation')

        train_text = [b[tf] for b in train_data]
        test_text = [b[tf] for b in test_data]
        train_label = [b['label'] for b in train_data]
        test_label = [b['label'] for b in test_data]

    else:
        from sklearn.model_selection import train_test_split
        class_num = 2
        df = pd.read_csv('./IMDB_Dataset.csv')
        text = [t for t in df['review']]

        label = []
        for t in df['sentiment']:
            if t == 'negative':
                label.append(1)
            else:
                label.append(0)
        train_text, test_text, train_label, test_label = train_test_split(
            text, label, test_size=0.2)
    return train_text, train_label, test_text, test_label, class_num


def get_nlp_data(args):

    print_process(f"Load data:{args.dataset}")
    train_text, train_label, test_text, test_label, class_num = load_raw_data(
        args.dataset)

    print_process("Preprocessing")
    clean_train, train_label = clean_data(
        train_text, train_label, args.min_len)
    # clean_test, test_label = clean_data(test_data, test_label, args.min_len)
    clean_test = [data_preprocessing(t, True) for t in test_text]
    vocab = create_vocab(clean_train)

    trainset = Textset(clean_train, train_label, vocab, args.max_len)
    testset = Textset(clean_test, test_label, vocab, args.max_len)
    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, collate_fn=trainset.collate, shuffle=True)
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, collate_fn=testset.collate)

    return train_loader, test_loader, class_num, vocab


def clean_data(text, label, min_len):
    clean_text = []
    clean_label = []
    for index in range(len(text)):
        t = data_preprocessing(text[index], True)
        if len(t.split()) > min_len:
            clean_text.append(t)
            clean_label.append(label[index])
    print(f"Original Data: {len(text)}")
    print(f"Valid Data: {len(clean_text)}")
    return clean_text, clean_label


def data_preprocessing(text, remove_stopword=False):

    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = ''.join([c for c in text if c not in string.punctuation])
    if remove_stopword:
        text = [word for word in text.split() if word not in stop_words]
    else:
        text = [word for word in text.split()]
    text = ' '.join(text)
    return text


def create_vocab(corpus, vocab_size=30000):

    corpus = [t.split() for t in corpus]
    corpus = list(itertools.chain.from_iterable(corpus))
    count_words = Counter(corpus)
    print('total count words', len(count_words))
    sorted_words = count_words.most_common()

    if vocab_size > len(sorted_words):
        v = len(sorted_words)
    else:
        v = vocab_size - 2

    vocab_to_int = {w: i + 2 for i, (w, c) in enumerate(sorted_words[:v])}

    vocab_to_int['<pad>'] = 0
    vocab_to_int['<unk>'] = 1
    print('vocab size', len(vocab_to_int))

    return vocab_to_int


def preview_nlp_dataset(dataset):
    train_text, train_label, test_text, test_label, class_num = load_raw_data(
        dataset)
    print(f"Train:{len(train_text)}、Test:{len(test_text)}")
    train_length = []
    test_length = []
    for text in train_text:
        train_length.append(len(text.split()))

    for text in test_text:
        test_length.append(len(text.split()))

    train_text = Counter(sorted(train_length))
    train_label = Counter(sorted(train_label))
    test_text = Counter(sorted(test_length))
    test_label = Counter(sorted(test_label))

    fig, axes = plt.subplots(2, 2)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    ax1.bar(train_text.keys(), train_text.values())
    ax1.set_title("train_text")
    ax1.set_xlim([0, 100])
    ax2.bar(train_label.keys(), train_label.values())
    ax2.set_title("train_label")
    ax3.bar(test_text.keys(), test_text.values())
    ax3.set_title("test_text")
    ax3.set_xlim([0, 100])
    ax4.bar(test_label.keys(), test_label.values())
    ax4.set_title("test_label")
    fig.tight_layout()
    plt.show()

    print(f"Train text median len:{train_text.most_common()[0][0]}")
    print(f"Train text min len:{train_text.most_common()[-1][0]}")


class Textset(Dataset):
    def __init__(self, texts, label, vocab, max_len):
        super().__init__()

        new_text = []
        for text in texts:
            t = text.split()
            if len(t) > max_len:
                t = t[:max_len]
            new_text.append(' '.join(t))
        self.x = new_text
        self.y = label
        self.vocab = vocab

    def collate(self, batch):

        x = [torch.tensor(x) for x, y in batch]
        y = [y for x, y in batch]
        x_tensor = pad_sequence(x, True)
        y = torch.tensor(y)
        return x_tensor, y

    def convert2id(self, text):
        r = []
        for word in text.split():
            if word in self.vocab.keys():
                r.append(self.vocab[word])
            else:
                r.append(self.vocab['<unk>'])
        return r

    def __getitem__(self, idx):
        text = self.x[idx]
        word_id = self.convert2id(text)
        return word_id, self.y[idx]

    def __len__(self):
        return len(self.x)
