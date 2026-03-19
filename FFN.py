class FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.l1 = nn.Linear(d, d_ff)
        self.l2 = nn.Linear(d_ff, d)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


class AddNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x, sub):
        return self.norm(x + sub)
