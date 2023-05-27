import torch
import torch.nn as nn
from torch.profiler import record_function


class ENC(nn.Module):
    r"""The encoder part of AL, which is a nn.Module contains f block and b block.       
        Example:
            f = nn.Sequential(
                nn.Linear(10,10),
                nn.ReLU()
            )
            b = nn.Sequential(
                nn.Linear(10,10),
                nn.ReLU()
            )
            enc = ENC(f, b)
    """

    def __init__(self, f: nn.Module, b: nn.Module, cri=nn.MSELoss()):
        r"""Initialize the encoder, you need to declare what is your f & b block and how they do forward.
            Args:
                f: forward block, stands for original bp model component.
                b: bridge block, use for transforming output of f block to calculate loss with encoding of y.
                cri: the loss func to calculate associated loss, default is nn.MSELoss().        
        """
        super().__init__()
        self.f = f
        self.b = b
        self.cri = cri

    def forward(self, enc_input, ae_output):
        r"""The forward function of AL encoder
        Args:
            enc_input: the encoder's input, depends on which type of input your forward block needs.
            ae_output: the autoencoder's output, use for calculate the associated loss.

        Returns:
            loss: the associated loss, calculated by self.cri(self.b(self.f(enc_input)), ae_output).
            enc_x: the encoder's output.
        """
        enc_x = self.f(enc_input)
        bri_x = self.b(enc_x)
        loss = self.cri(bri_x, ae_output)
        return loss, enc_x


class AE(nn.Module):
    r"""The autoencoder part of AL, which is a nn.Module contains g block and h block.       
        Example:
            g = nn.Sequential(
                nn.Linear(10,10),
                nn.ReLU()
            )
            h = nn.Sequential(
                nn.Linear(10,10),
                nn.ReLU()
            )
            ae = AE(g, h)
    """

    def __init__(self, g: nn.Module, h: nn.Module, cri=nn.MSELoss()):
        r"""Initialize the autoencoder, you need to declare what is your g & h block and how they do forward.
            Args:
                g: encoder block for label or the ground truth.
                h: decoder block for label or the ground truth.
                cri: the loss func to calculate autoencoder loss, default is nn.MSELoss().        
        """
        super().__init__()
        self.g = g
        self.h = h
        self.cri = cri

    def forward(self, ae_input):
        r"""The forward function of AL autoencoder
        Args:
            ae_input: the autoencoder's input, depends on which type of input your encoder block needs

        Returns:
            loss: the autoencoder loss, calculated by self.cri(self.h(self.g(ae_input)), ae_input)
            enc_y: the encoder's output
        """
        enc_y = self.g(ae_input)
        dec_y = self.h(enc_y)
        loss = self.cri(dec_y, ae_input)
        return loss, enc_y


class AL(nn.Module):
    r"""The AL component, you need to declare the ENC and AE first. 
        Example:
                f = nn.Sequential(
                    nn.Linear(10,10),
                    nn.ReLU()
                )
                b = nn.Sequential(
                    nn.Linear(10,10),
                    nn.ReLU()
                )
                enc = ENC(f, b)
                g = nn.Sequential(
                    nn.Linear(10,10),
                    nn.ReLU()
                )
                h = nn.Sequential(
                    nn.Linear(10,10),
                    nn.ReLU()
                )
                ae = AE(g, h)
                al = AL(enc, ae)

        Attributes:
            enc: the encoder part of AL, declared in initialization.
            ae: the autoencoder part of AL, declared in initialization.
            enc_opt: the optimizer for the encoder part of AL, use Adam as default.
            ae_opt: the optimizer for the autoencoder part of AL, use Adam as default.
    """

    def __init__(self, enc: nn.Module, ae: nn.Module, lr: float or dict = 1e-3):
        r"""Initialize AL with ENC and AE module.
        Args:
            enc: The encoder part of AL, which is a nn.Module contains f block and b block.
            ae: The autoencoder part of AL, which is a nn.Module contains g block and h block.
            lr: The learning rate of optimizer, can be dict with keys "enc" and "ae", or float.         
        """
        super().__init__()
        learning_rate = {}
        if type(lr) == float:
            learning_rate["enc"] = lr
            learning_rate["ae"] = lr
        else:
            learning_rate["enc"] = lr["enc"]
            learning_rate["ae"] = lr["ae"]

        self.enc = enc
        self.ae = ae
        self.enc_opt = torch.optim.Adam(
            self.enc.parameters(), lr=learning_rate["enc"])
        self.ae_opt = torch.optim.Adam(
            self.ae.parameters(), lr=learning_rate["ae"])
        self.enc_opt.zero_grad()
        self.ae_opt.zero_grad()
        self.__ae_loss = 0
        self.__enc_loss = 0

    def forward(self, enc_input, ae_input):
        r"""The forward function of AL.
            Args:
                enc_input: input for the encoder of AL.
                ae_input: input for the autoencoder of AL.
            Returns:
                enc_output: encoding of enc_input without gradient flow.
                ae_output: encoding of ae_input without gradient flow.
        """
        self.__ae_loss, ae_output = self.ae(ae_input)
        self.__enc_loss, enc_output = self.enc(
            enc_input, ae_output.clone().detach())
        return enc_output.detach(), ae_output.detach()

    def backward(self):
        r"""Doing loss backward."""
        self.__ae_loss.backward()
        self.__enc_loss.backward()

    def update(self):
        r"""Update parameters."""
        self.ae_opt.step()
        self.ae_opt.zero_grad()
        self.enc_opt.step()
        self.enc_opt.zero_grad()

    def backward_and_update(self):
        r"""Encapsulate backward() and update() for multithread used."""
        self.backward()
        self.update()

    def record_backward_and_update(self):
        with record_function("backward"):
            self.backward()
        with record_function("update"):
            self.update()

    def loss(self):
        r"""Return enc_loss and ae_loss"""
        return self.__enc_loss, self.__ae_loss

    def inference(self, input, mode):
        r"""Return the output of input through specified block"""
        if mode == 'f':
            return self.enc.f(input)
        elif mode == 'b':
            return self.enc.b(input)
        elif mode == 'h':
            return self.ae.h(input)

    def set_lr(self, opt: list, lr: float):
        if 'enc' in opt:
            for g in self.enc_opt.param_groups:
                g['lr'] = lr
        if 'ae' in opt:
            for g in self.ae_opt.param_groups:
                g['lr'] = lr

    def getParam(self):
        param = {}
        param["f"] = sum(p.numel() for p in self.enc.f.parameters())
        param["b"] = sum(p.numel() for p in self.enc.b.parameters())
        param["g"] = sum(p.numel() for p in self.ae.g.parameters())
        param["h"] = sum(p.numel() for p in self.ae.h.parameters())
        return param
