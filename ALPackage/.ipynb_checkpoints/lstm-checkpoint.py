from .base import *
from .emb import EMBALLayer
import threading


class LSTMENC(ENC):
    def forward(self, enc_input, ae_output):
        r"""The forward function of LSTMAL encoder
        Args:
            enc_input: the encoder's input, including x and mask => (x, mask).
                ae_output: the autoencoder's output, use for calculate the associated loss.

            Returns:
                loss: the associated loss, calculated by self.cri(self.b(self.f(enc_input)), ae_output).
                enc_x: the encoder's output.
        """
        x = enc_input[0]
        h = enc_input[1]
        enc_x, (h, c) = self.f(x, h)
        bri_x = h[0] + h[1]
        bri_x = self.b(bri_x)
        loss = self.cri(bri_x, ae_output)
        return loss, (enc_x, (h, c))


class LSTMALLayer(AL):
    def __init__(self, emb_dim, hid_dim, y_in, y_out, act, lr):
        f = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        b = nn.Sequential(
            nn.Linear(hid_dim, y_out),
            act
        )
        g = nn.Sequential(
            nn.Linear(y_in, y_out),
            act
        )
        h = nn.Sequential(
            nn.Linear(y_out, y_in)
        )
        enc = LSTMENC(f, b)
        ae = AE(g, h)
        super().__init__(enc, ae, lr)

    def forward(self, enc_input, ae_input):
        self.__ae_loss, ae_output = self.ae(ae_input)
        self.__enc_loss, enc_output = self.enc(
            enc_input, ae_output.clone().detach())
        (h, c) = enc_output[1]
        h = h.reshape(2, enc_input[0].size(0), -1)
        hidden = (h.detach(), c.detach())
        enc_out = enc_output[0].detach()
        return (enc_out, hidden), ae_output.detach()
    
    def backward(self):
        r"""Doing loss backward."""
        self.__ae_loss.backward()
        self.__enc_loss.backward()

    def inference(self, input, mode):
        if mode == 'f':
            return self.enc.f(input[0], input[1])
        elif mode == 'b':
            x, (h, c) = input
            h = h[0] + h[1]
            return self.enc.b(h)
        elif mode == 'h':
            return self.ae.h(input)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, class_num, pretrained_embedding=None):
        super().__init__()
        self.seq = nn.ModuleList()
        if pretrained_embedding == None:
            self.seq.append(nn.Embedding(vocab_size, emb_dim))
        else:
            self.seq.append(nn.Embedding.from_pretrained(
                pretrained_embedding, freeze=False))
        self.seq.append(nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True))
        self.seq.append(nn.LSTM(hid_dim*2, hid_dim, bidirectional=True, batch_first=True))
        self.seq.append(nn.LSTM(hid_dim*2, hid_dim, bidirectional=True, batch_first=True))
        self.seq.append(nn.Sequential(
            nn.Linear(hid_dim*2, class_num),
            nn.LogSoftmax(dim=1)
        ))
        self.device = ["cpu" for i in range(len(self.seq))]

    def forward(self, x):

        # 1. Feed embedding layer
        x = self.seq[0](x.to(self.device[0]))

        # 2. Feed LSTM
        h = None
        for i in range(1, len(self.seq)-1, 1):
            x = x.to(self.device[i])
            if h != None:
                (h, c) = h
                h = (h.to(self.device[i]), c.to(self.device[i]))
            x, (h, c) = self.seq[i](x, h)
            h = h.reshape(2, x.size(0), -1)
            h = (h, c)

        # 3. Feed fc layer
        x = x[:, -1, :]
        x = x.to(self.device[-1])
        return self.seq[-1](x)


class LSTMAL(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, class_num, y_out, act, lr, pretrained_embedding=None):
        super().__init__()
        self.seq = nn.ModuleList()
        self.seq.append(EMBALLayer(vocab_size, emb_dim, class_num,
                        y_out, act, lr, pretrained_embedding))
        self.seq.append(LSTMALLayer(emb_dim, hid_dim, y_out, y_out, act, lr))
        self.seq.append(LSTMALLayer(hid_dim*2, hid_dim, y_out, y_out, act, lr))
        self.seq.append(LSTMALLayer(hid_dim*2, hid_dim, y_out, y_out, act, lr))
        self.device = ["cpu" for i in range(len(self.seq))]

    def forward(self, x, y):

        # 1. Feed embedding layer
        x, y = self.seq[0](x.to(self.device[0]), y.to(self.device[0]))

        # 2. Feed LSTMAL
        for i in range(1, len(self.seq), 1):
            y = y.to(self.device[i])
            if i == 1:
                inputs = (x.to(self.device[i]), None)
            else:
                (enc_x, (h, c)) = x
                inputs = (enc_x.to(self.device[i]), (h.to(
                    self.device[i]), c.to(self.device[i])))
            x, y = self.seq[i](inputs, y)

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

        # 1. Feed embedding layer
        x, y = self.seq[0](x.to(self.device[0]), y.to(self.device[0]))
        threads.append(threading.Thread(
            target=self.seq[0].backward_and_update))
        threads[0].start()

        # 2. Feed LSTMAL
        for i in range(1, len(self.seq), 1):
            y = y.to(self.device[i])
            if i == 1:
                inputs = (x.to(self.device[i]), None)
            else:
                (enc_x, (h, c)) = x
                inputs = (enc_x.to(self.device[i]), (h.to(
                    self.device[i]), c.to(self.device[i])))
            x, y = self.seq[i](inputs, y)
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

    def inference(self, x):
        x = self.seq[0].inference(x.to(self.device[0]), 'f')
        for i in range(1, len(self.seq), 1):
            if i == 1:
                inputs = (x.to(self.device[i]), None)
            else:
                (enc_x, (h, c)) = x
                inputs = (enc_x.to(self.device[i]), (h.to(
                    self.device[i]), c.to(self.device[i])))
            x = self.seq[i].inference(inputs, 'f')
        x = self.seq[-1].inference(x, 'b')
        for i in range(len(self.seq)-1, -1, -1):
            x = x.to(self.device[i])
            x = self.seq[i].inference(x, 'h')
        return x
