def pos_encoding(seq_len, d):
    pe = torch.zeros(seq_len, d)
    for pos in range(seq_len):
        for i in range(0, d, 2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/d)))
            if i+1 < d:
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2*i)/d)))
    return pe
