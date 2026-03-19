D = 64
D_FF = 256
N = 2
VOCAB = 10
START = 1
EOS = 2

vocab_map = {
    0: '<PAD>', 1: '<START>', 2: '<EOS>',
    3: 'Redes', 4: 'neurais',
    5: 'são', 6: 'fortes',
    7: '.', 8: 'muito', 9: '!'
}

torch.manual_seed(7)

enc = Encoder(N, D, D_FF)
dec = Decoder(VOCAB, N, D, D_FF)

enc.eval()
dec.eval()

embeddings = torch.rand(VOCAB, D)
entrada_tokens = torch.tensor([[3, 4]])
entrada = embeddings[entrada_tokens]

pe = pos_encoding(entrada.size(1), D)
entrada = entrada + pe.unsqueeze(0)

with torch.no_grad():
    z = enc(entrada)

print('Z shape:', z.shape)

with torch.no_grad():
    z = enc(entrada)

print('Z shape:', z.shape)
