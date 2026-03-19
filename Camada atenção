class Atencao(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, q, k, v, mask=None):
        out = atencao(self.wq(q), self.wk(k), self.wv(v), mask)
        return self.wo(out)
