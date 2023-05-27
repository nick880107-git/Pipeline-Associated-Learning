from .base import *

class EMBENC(ENC):
    def forward(self, x, enc_y):
        enc_x = self.f(x.long())
        bri_x = self.b(enc_x.mean(1))
        loss = self.cri(bri_x, enc_y)
        return loss, enc_x

class EMBAE(AE):
    def forward(self, y):
        enc_y = self.g(y)
        dec_y = self.h(enc_y)
        loss = self.cri(dec_y, y.argmax(1))
        return loss, enc_y

class EMBALLayer(AL):
    def __init__(self, vocab_size, emb_dim, class_num, y_out, act, lr, pretrained_embedding=None):
        if pretrained_embedding == None:
            f = nn.Embedding(vocab_size, emb_dim)
        else:
            f = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        b = nn.Sequential(
            nn.Linear(emb_dim, y_out),
            act
        )
        g = nn.Sequential(
            nn.Linear(class_num, y_out),
            act
        )
        h = nn.Sequential(
            nn.Linear(y_out, class_num),
            act
        )
        enc = EMBENC(f, b)
        ae = EMBAE(g, h, nn.CrossEntropyLoss())
        super().__init__(enc, ae, lr)