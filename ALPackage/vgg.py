from .base import *
import threading


class CONV_BN(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, activation: nn.Module, stride: int = 1, padding: int = 1, bias: bool = False, isBatchNorm=True):
        super().__init__()
        layers = []
        if isBatchNorm:
            layers.append(nn.Conv2d(in_channel, out_channel,
                          kernel_size=3, stride=stride, padding=padding, bias=bias))
            layers.append(nn.BatchNorm2d(out_channel))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel,
                          kernel_size=3, stride=stride, padding=padding, bias=bias))
        if activation != None:
            layers.append(activation)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class VGGLayer(nn.Module):
    def __init__(self, in_channel: int, out_channel: list, image_size: int, activation: nn.Module, isBatchNorm=True):
        super().__init__()
        layers = []
        self.prev_channel = in_channel
        self.out_image_size = image_size
        for dim in out_channel:
            if dim == 'M':
                layers.append(nn.MaxPool2d(2, 2))
                self.out_image_size /= 2
            else:
                layers.append(CONV_BN(self.prev_channel, dim,
                              activation, isBatchNorm=isBatchNorm))
                self.prev_channel = dim
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class VGGALLayer(AL):
    def __init__(self, x_in: int, x_out: list, activation: dict or nn.Module, y_in: int, y_out: int, hid_dim: int, image_size: int, lr: float, isBatchNorm=True):

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

        f = VGGLayer(x_in, x_out,
                     image_size, act["f"], isBatchNorm=isBatchNorm)
        b_in = f.prev_channel * f.out_image_size * f.out_image_size
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


class VGG(nn.Module):
    def __init__(self, in_channels: int, num_class: int, act=nn.ReLU()):
        super().__init__()
        self.seq = nn.ModuleList()
        self.seq.append(VGGLayer(in_channels, [128, 256, 'M'], 32, act))
        self.seq.append(VGGLayer(256, [256, 512, 'M'], 16, act))
        self.seq.append(VGGLayer(512, [512, 'M'], 8, act))
        self.seq.append(VGGLayer(512, [512, 'M'], 4, act))
        self.seq.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 2500),
            nn.Sigmoid(),
            nn.Linear(2500, num_class)
        ))
        self.device = ["cpu" for i in range(len(self.seq))]

    def forward(self, x):

        for i in range(len(self.seq)):
            x = self.seq[i](x.to(self.device[i]))
        return x


class VGGAL(nn.Module):
    def __init__(self, in_channels, num_class, enc_dim, ae_dim, act, lr, isBatchNorm=True):
        super().__init__()
        self.seq = nn.ModuleList()
        self.seq.append(VGGALLayer(in_channels, [
                        128, 256, 'M'], act, num_class, ae_dim, enc_dim, 32, lr, isBatchNorm))
        self.seq.append(VGGALLayer(
            256, [256, 512, 'M'], act, ae_dim, ae_dim, enc_dim, 16, lr, isBatchNorm))
        self.seq.append(VGGALLayer(
            512, [512, 'M'], act, ae_dim, ae_dim, enc_dim, 8, lr, isBatchNorm))
        self.seq.append(VGGALLayer(
            512, [512, 'M'], act, ae_dim, ae_dim, enc_dim, 4, lr, isBatchNorm))
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
