from .base import *
from .emb import *
from .encoder import TransformerEncoder
import threading


class TransformerEncoderENC(ENC):
    def forward(self, enc_input, ae_output):
        r"""The forward function of TransformerEncoderAL encoder
        Args:
            enc_input: the encoder's input, including x and mask => (x, mask).
                ae_output: the autoencoder's output, use for calculate the associated loss.

            Returns:
                loss: the associated loss, calculated by self.cri(self.b(self.f(enc_input)), ae_output).
                enc_x: the encoder's output.
        """
        x = enc_input[0]
        mask = enc_input[1]
        enc_x = self.f(x, mask)
        bri_x = torch.sum(enc_x * mask.unsqueeze(-1), dim=1) / \
            torch.sum(mask, -1, keepdim=True)
        bri_x = self.b(bri_x)
        loss = self.cri(bri_x, ae_output)
        return loss, enc_x


class TransformerEncoderALLayer(AL):
    def __init__(self, emb_dim, hid_dim, y_in, y_out, act, lr, nheads=6, nlayers=1, dropout=0):
        f = TransformerEncoder(emb_dim, hid_dim, nheads, nlayers, dropout)
        b = nn.Sequential(
            nn.Linear(emb_dim, y_out),
            act
        )
        g = nn.Sequential(
            nn.Linear(y_in, y_out),
            act
        )
        h = nn.Sequential(
            nn.Linear(y_out, y_in)
        )
        enc = TransformerEncoderENC(f, b)
        ae = AE(g, h)
        super().__init__(enc, ae, lr)

    def inference(self, input, mode):
        if mode == 'f':
            return self.enc.f(input[0], input[1])
        elif mode == 'b':
            denom = torch.sum(input[1], -1, keepdim=True)
            feat = torch.sum(input[0] * input[1].unsqueeze(-1), dim=1) / denom
            return self.enc.b(feat)
        elif mode == 'h':
            return self.ae.h(input)


class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, class_num, nheads=6, nlayers=1, dropout=0, pretrained_embedding=None):
        super().__init__()
        self.seq = nn.ModuleList()
        if pretrained_embedding == None:
            self.seq.append(nn.Embedding(vocab_size, emb_dim))
        else:
            self.seq.append(nn.Embedding.from_pretrained(
                pretrained_embedding, freeze=False))
        self.seq.append(TransformerEncoder(
            emb_dim, hid_dim, nheads, nlayers, dropout))
        self.seq.append(TransformerEncoder(
            emb_dim, hid_dim, nheads, nlayers, dropout))
        self.seq.append(TransformerEncoder(
            emb_dim, hid_dim, nheads, nlayers, dropout))
        self.seq.append(nn.Sequential(
            nn.Linear(emb_dim, class_num),
            nn.LogSoftmax(dim=1)
        ))
        self.device = ["cpu" for i in range(len(self.seq))]

    def forward(self, x):
        mask = self.get_mask(x)

        # 1. Feed embedding layer
        x = self.seq[0](x.to(self.device[0]))

        # 2. Feed Transformer
        for i in range(1, len(self.seq)-1, 1):
            x = self.seq[i](x.to(self.device[i]), mask.to(self.device[i]))

        # 3. Feed fc layer
        x = x.to(self.device[-1])
        mask = mask.to(self.device[-1])
        feat = torch.sum(x*mask.unsqueeze(-1), dim=1) / \
            torch.sum(mask, -1, keepdim=True)
        return self.seq[-1](feat)

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask


class TransformerAL(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, class_num, y_out, act, lr, nheads=6, nlayers=1, dropout=0, pretrained_embedding=None):
        super().__init__()
        self.seq = nn.ModuleList()
        self.seq.append(EMBALLayer(vocab_size, emb_dim, class_num,
                        y_out, act, lr, pretrained_embedding))
        self.seq.append(TransformerEncoderALLayer(
            emb_dim, hid_dim, y_out, y_out, act, lr, nheads, nlayers, dropout))
        self.seq.append(TransformerEncoderALLayer(
            emb_dim, hid_dim, y_out, y_out, act, lr, nheads, nlayers, dropout))
        self.seq.append(TransformerEncoderALLayer(
            emb_dim, hid_dim, y_out, y_out, act, lr, nheads, nlayers, dropout))
        self.device = ["cpu" for i in range(len(self.seq))]

    def forward(self, x, y):
        mask = self.get_mask(x)

        # 1. Feed embedding layer
        x, y = self.seq[0](x.to(self.device[0]), y.to(self.device[0]))

        # 2. Feed Transformer
        for i in range(1, len(self.seq), 1):
            inputs = (x.to(self.device[i]), mask.to(self.device[i]))
            y = y.to(self.device[i])
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
        mask = self.get_mask(x)

        # 1. Feed embedding layer
        x, y = self.seq[0](x.to(self.device[0]), y.to(self.device[0]))
        threads.append(threading.Thread(
            target=self.seq[0].backward_and_update))
        threads[0].start()

        # 2. Feed Transformer
        for i in range(1, len(self.seq), 1):
            inputs = (x.to(self.device[i]), mask.to(self.device[i]))
            y = y.to(self.device[i])
            x, y = self.seq[i](inputs, y)
            threads.append(threading.Thread(
                target=self.seq[i].backward_and_update))
            threads[i].start()

        for i in range(len(self.seq)):
            threads[i].join()

    def record_thread_backward_and_update(self):
        threads = []
        for i in range(len(self.seq)):
            threads.append(threading.Thread(
                target=self.seq[i].record_backward_and_update))
            threads[i].start()
        for i in range(len(self.seq)):
            threads[i].join()

    def record_thread_forward_backward_and_update(self, x, y):
        threads = []
        mask = self.get_mask(x)

        # 1. Feed embedding layer
        with record_function("forward"):
            x, y = self.seq[0](x.to(self.device[0]), y.to(self.device[0]))
        threads.append(threading.Thread(
            target=self.seq[0].record_backward_and_update))
        threads[0].start()

        # 2. Feed Transformer
        for i in range(1, len(self.seq), 1):
            with record_function("forward"):
                inputs = (x.to(self.device[i]), mask.to(self.device[i]))
                y = y.to(self.device[i])
                x, y = self.seq[i](inputs, y)
            threads.append(threading.Thread(
                target=self.seq[i].record_backward_and_update))
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
        mask = self.get_mask(x)
        x = self.seq[0].inference(x.to(self.device[0]), 'f')
        for i in range(1, len(self.seq), 1):
            inputs = (x.to(self.device[i]), mask.to(self.device[i]))
            x = self.seq[i].inference(inputs, 'f')
        inputs = (x, mask.to(self.device[i]))
        x = self.seq[-1].inference(inputs, 'b')
        for i in range(len(self.seq)-1, -1, -1):
            x = x.to(self.device[i])
            x = self.seq[i].inference(x, 'h')
        return x

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask
