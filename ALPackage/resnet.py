from .base import *
from .vgg import CONV_BN
import threading


class BasicBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, activation: nn.Module, stride=1):
        super().__init__()
        self.seq = nn.Sequential(
            CONV_BN(in_channel, out_channel, activation, stride),
            CONV_BN(out_channel, out_channel, None)
        )
        self.act = activation
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = CONV_BN(in_channel, out_channel, None, stride)

    def forward(self, x):
        out = self.seq(x)
        return self.act(out + self.shortcut(x))


class ResnetLayer(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, strides: list, activation: nn.Module):
        super().__init__()
        layers = []
        prev_channel = in_channel
        for stride in strides:
            layers.append(BasicBlock(
                prev_channel, out_channel, activation, stride))
            prev_channel = out_channel
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class ResnetALLayer(AL):
    def __init__(self, x_in: int, x_out: int, activation: dict or nn.Module, strides: int or list, y_in: int, y_out: int, hid_dim: int, out_image_size: int, lr: float):

        act = {
            "f": nn.ReLU(),
            "b": nn.Sigmoid(),
            "g": nn.Sigmoid(),
            "h": nn.Sigmoid()
        }
        if isinstance(activation, nn.Module):
            act["f"] = activation
        else:
            for key in activation.keys():
                act[key] = activation[key]

        if type(strides) == int:
            f = CONV_BN(x_in, x_out, act["f"], strides)
        else:
            f = ResnetLayer(x_in, x_out, strides, act["f"])
        b_in = x_out * out_image_size * out_image_size
        b = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(b_in), hid_dim),
            act["b"],
            nn.Linear(hid_dim, y_out),
            act["b"]
        )
        g = nn.Sequential(
            nn.Linear(y_in, y_out),
            act["g"]
        )
        h = nn.Sequential(
            nn.Linear(y_out, y_in),
            act["h"]
        )

        enc = ENC(f, b)
        ae = AE(g, h)
        super().__init__(enc, ae, lr)


class Resnet18(nn.Module):
    def __init__(self, in_channels: int, num_class: int, act=nn.LeakyReLU(inplace=True)):
        super().__init__()
        self.seq = nn.ModuleList()
        self.seq.append(CONV_BN(in_channels, 64, act))
        self.seq.append(ResnetLayer(64, 64, [1, 1], act))
        self.seq.append(ResnetLayer(64, 128, [2, 1], act))
        self.seq.append(ResnetLayer(128, 256, [2, 1], act))
        self.seq.append(ResnetLayer(256, 512, [2, 1], act))
        self.seq.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_class)
        ))
        self.device = ["cpu" for i in range(len(self.seq))]

    def forward(self, x):

        for i in range(len(self.seq)):
            x = self.seq[i](x.to(self.device[i]))
        return x


class Resnet18AL(nn.Module):
    def __init__(self, in_channels, num_class, enc_dim, ae_dim, lr, act=nn.LeakyReLU(inplace=True)):
        super().__init__()
        self.seq = nn.ModuleList()
        self.seq.append(ResnetALLayer(in_channels, 64, act, 1,
                        num_class, ae_dim, enc_dim, 32, lr))
        self.seq.append(ResnetALLayer(
            64, 64, act, [1, 1], ae_dim, ae_dim, enc_dim, 32, lr))
        self.seq.append(ResnetALLayer(
            64, 128, act, [2, 1], ae_dim, ae_dim, enc_dim, 16, lr))
        self.seq.append(ResnetALLayer(
            128, 256, act, [2, 1], ae_dim, ae_dim, enc_dim, 8, lr))
        self.seq.append(ResnetALLayer(
            256, 512, act, [2, 1], ae_dim, ae_dim, enc_dim, 4, lr))
        self.device = ["cpu" for i in range(len(self.seq))]

    def forward(self, x, y):
        for i in range(len(self.seq)):
            x = x.to(self.device[i])
            y = y.to(self.device[i])
            x, y = self.seq[i](x, y)

    def backward(self):
        for i in range(len(self.seq)):
            self.seq[i].backward()

    def update(self):
        for i in range(len(self.seq)):
            self.seq[i].update()

    def thread_backward_and_update(self):
        threads = []
        for i in range(len(self.seq)):
            threads.append(threading.Thread(
                target=self.seq[i].backward_and_update))
            threads[i].start()
        for i in range(len(self.seq)):
            threads[i].join()

    def thread_forward_backward_and_update(self, x, y):
        threads = []
        for i in range(len(self.seq)):
            x = x.to(self.device[i])
            y = y.to(self.device[i])
            x, y = self.seq[i](x, y)
            threads.append(threading.Thread(
                target=self.seq[i].backward_and_update))
            threads[i].start()

        for i in range(len(self.seq)):
            threads[i].join()

    def loss(self):
        loss = {
            "enc_loss": [],
            "ae_loss": []
        }
        for i in range(len(self.seq)):
            enc_loss, ae_loss = self.seq[i].loss()
            loss['enc_loss'].append(enc_loss.item())
            loss['ae_loss'].append(ae_loss.item())
        return loss

    def set_lr(self, opt: list, lr: float):
        for i in range(len(self.seq)):
            self.seq[i].set_lr(opt, lr)

    def inference(self, x):
        for i in range(len(self.seq)):
            x = x.to(self.device[i])
            x = self.seq[i].inference(x, 'f')
        x = self.seq[-1].inference(x, 'b')
        for i in range(len(self.seq)-1, -1, -1):
            x = x.to(self.device[i])
            x = self.seq[i].inference(x, 'h')
        return x
