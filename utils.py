def mascara_causal(tam, device='cpu'):
    mask = torch.tril(torch.ones(tam, tam, device=device))
    return mask.unsqueeze(0)
