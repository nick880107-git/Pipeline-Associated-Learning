from ALPackage.vgg import VGG, VGGAL
from ALPackage.resnet import Resnet18, Resnet18AL
from utils import set_device, print_process, lr_scheduling, getParam
from tqdm import tqdm
import torch
import torch.nn as nn
import time
import datetime


class ResnetTrainer:

    def __init__(self, args, train_loader, test_loader):
        self.model = Resnet18(args.in_channels, args.num_class, args.act)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), args.base_lr)
        self.loss_fn = nn.CrossEntropyLoss()
        set_device(self.model, args.bp_gpu_list)
        print(
            f"Model params:{sum(p.numel() for p in self.model.parameters())}")

    def run(self):

        train_acc, test_acc = [], []
        train_time = []
        execution_time = 0
        print_process("Training start")
        for i in range(self.args.epochs):
            lr_scheduling(self.optimizer, self.args.base_lr,
                          self.args.end_lr, i, self.args.epochs)
            execution_time += self.train(self.train_loader)
            train_time.append(execution_time)
            train_acc.append(self.eval(self.train_loader))
            test_acc.append(self.eval(self.test_loader))
            print(
                f"\nEpoch {i}: Training acc: {train_acc[i]:.3f}, Testing acc: {test_acc[i]:.3f}")
        print_process(
            f"Spend:{str(datetime.timedelta(seconds=execution_time))}, best test acc:{max(test_acc):.3f}")
        return train_acc, test_acc, train_time

    def train(self, data_loader):

        self.model.train()
        torch.cuda.synchronize()
        start = time.time()
        for inputs, labels in tqdm(data_loader):
            labels = labels.to(self.args.bp_gpu_list[-1])
            pred = self.model(inputs)
            self.loss_fn(pred, labels).backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        torch.cuda.synchronize()
        end = time.time()
        return end - start

    def eval(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for inputs, labels in tqdm(data_loader):
                labels = labels.to(self.args.bp_gpu_list[-1])
                pred = self.model(inputs)
                correct += (pred.argmax(-1) == labels).sum().item()
                total += labels.size(0)
        return correct/total


class ResnetALTrainer:

    def __init__(self, args, train_loader, test_loader):
        self.model = Resnet18AL(args.in_channels, args.num_class,
                                args.enc_dim, args.ae_dim, args.base_lr, args.act)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        set_device(self.model, args.al_gpu_list)
        print(f"Model params:{sum(p.numel() for p in self.model.parameters())}")
        getParam(self.model)

    def run(self):

        train_acc, test_acc = [], []
        train_time = []
        execution_time = 0
        print_process("Training start")
        for i in range(self.args.epochs):
            lr = lr_scheduling(None, self.args.base_lr,
                               self.args.end_lr, i, self.args.epochs)
            self.model.set_lr('enc', lr)
            self.model.set_lr('ae', lr)
            execution_time += self.train_p(self.train_loader)
            train_time.append(execution_time)
            train_acc.append(self.eval(self.train_loader))
            test_acc.append(self.eval(self.test_loader))
            print(
                f"\nEpoch {i}: Training acc: {train_acc[i]:.3f}, Testing acc: {test_acc[i]:.3f}")
        print_process(
            f"Spend:{str(datetime.timedelta(seconds=execution_time))}, best test acc:{max(test_acc):.3f}")
        return train_acc, test_acc, train_time

    def train_p(self, data_loader):
        self.model.train()
        torch.cuda.synchronize()
        start = time.time()
        for inputs, labels in tqdm(data_loader):
            y = torch.nn.functional.one_hot(
                labels, self.args.num_class).float()
            self.model.thread_forward_backward_and_update(inputs, y)
        torch.cuda.synchronize()
        end = time.time()
        return end - start

    def train_m(self, data_loader):
        self.model.train()
        torch.cuda.synchronize()
        start = time.time()
        for inputs, labels in tqdm(data_loader):
            y = torch.nn.functional.one_hot(
                labels, self.args.num_class).float()
            self.model(inputs, y)
            self.model.thread_backward_and_update()
        torch.cuda.synchronize()
        end = time.time()
        return end - start

    def train(self, data_loader):
        self.model.train()
        torch.cuda.synchronize()
        start = time.time()
        for inputs, labels in tqdm(data_loader):
            y = torch.nn.functional.one_hot(
                labels, self.args.num_class).float()
            self.model(inputs, y)
            self.model.backward()
            self.model.update()
        torch.cuda.synchronize()
        end = time.time()
        return end - start

    def eval(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for inputs, labels in tqdm(data_loader):
                labels = labels.to(self.args.al_gpu_list[0])
                pred = self.model.inference(inputs)
                correct += (pred.argmax(-1) == labels).sum().item()
                total += labels.size(0)
        return correct/total


class VGGTrainer:

    def __init__(self, args, train_loader, test_loader):
        self.model = VGG(args.in_channels, args.num_class, args.act)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), args.base_lr)
        self.loss_fn = nn.CrossEntropyLoss()
        set_device(self.model, args.bp_gpu_list)
        print(
            f"Model params:{sum(p.numel() for p in self.model.parameters())}")

    def run(self):

        train_acc, test_acc = [], []
        train_time = []
        execution_time = 0
        print_process("Training start")
        for i in range(self.args.epochs):
            lr_scheduling(self.optimizer, self.args.base_lr,
                          self.args.end_lr, i, self.args.epochs)
            execution_time += self.train(self.train_loader)
            train_time.append(execution_time)
            train_acc.append(self.eval(self.train_loader))
            test_acc.append(self.eval(self.test_loader))
            print(
                f"\nEpoch {i}: Training acc: {train_acc[i]:.3f}, Testing acc: {test_acc[i]:.3f}")
        print_process(
            f"Spend:{str(datetime.timedelta(seconds=execution_time))}, best test acc:{max(test_acc):.3f}")
        return train_acc, test_acc, train_time

    def train(self, data_loader):

        self.model.train()
        torch.cuda.synchronize()
        start = time.time()
        for inputs, labels in tqdm(data_loader):
            labels = labels.to(self.args.bp_gpu_list[-1])
            pred = self.model(inputs)
            self.loss_fn(pred, labels).backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        torch.cuda.synchronize()
        end = time.time()
        return end - start

    def eval(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for inputs, labels in tqdm(data_loader):
                labels = labels.to(self.args.bp_gpu_list[-1])
                pred = self.model(inputs)
                correct += (pred.argmax(-1) == labels).sum().item()
                total += labels.size(0)
        return correct/total


class VGGALTrainer:

    def __init__(self, args, train_loader, test_loader):
        self.model = VGGAL(args.in_channels, args.num_class,
                           args.enc_dim, args.ae_dim, args.act, args.base_lr)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        set_device(self.model, args.al_gpu_list)
        print(f"Model params:{sum(p.numel() for p in self.model.parameters())}")
        getParam(self.model)

    def run(self):

        train_acc, test_acc = [], []
        train_time = []
        execution_time = 0
        print_process("Training start")
        for i in range(self.args.epochs):
            lr = lr_scheduling(None, self.args.base_lr,
                               self.args.end_lr, i, self.args.epochs)
            self.model.set_lr('enc', lr)
            self.model.set_lr('ae', lr)
            execution_time += self.train_p(self.train_loader)
            train_time.append(execution_time)
            train_acc.append(self.eval(self.train_loader))
            test_acc.append(self.eval(self.test_loader))
            print(
                f"\nEpoch {i}: Training acc: {train_acc[i]:.3f}, Testing acc: {test_acc[i]:.3f}")
        print_process(
            f"Spend:{str(datetime.timedelta(seconds=execution_time))}, best test acc:{max(test_acc):.3f}")
        return train_acc, test_acc, train_time

    def train_p(self, data_loader):
        self.model.train()
        torch.cuda.synchronize()
        start = time.time()
        for inputs, labels in tqdm(data_loader):
            y = torch.nn.functional.one_hot(
                labels, self.args.num_class).float()
            self.model.thread_forward_backward_and_update(inputs, y)
        torch.cuda.synchronize()
        end = time.time()
        return end - start

    def train_m(self, data_loader):
        self.model.train()
        torch.cuda.synchronize()
        start = time.time()
        for inputs, labels in tqdm(data_loader):
            y = torch.nn.functional.one_hot(
                labels, self.args.num_class).float()
            self.model(inputs, y)
            self.model.thread_backward_and_update()
        torch.cuda.synchronize()
        end = time.time()
        return end - start

    def train(self, data_loader):
        self.model.train()
        torch.cuda.synchronize()
        start = time.time()
        for inputs, labels in tqdm(data_loader):
            y = torch.nn.functional.one_hot(
                labels, self.args.num_class).float()
            self.model(inputs, y)
            self.model.backward()
            self.model.update()
        torch.cuda.synchronize()
        end = time.time()
        return end - start

    def eval(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for inputs, labels in tqdm(data_loader):
                labels = labels.to(self.args.al_gpu_list[0])
                pred = self.model.inference(inputs)
                correct += (pred.argmax(-1) == labels).sum().item()
                total += labels.size(0)
        return correct/total
