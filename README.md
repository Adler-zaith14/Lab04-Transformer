# Lab P1-04 — Transformer from Scratch

**Disciplina:** Tópicos em Inteligência Artificial — 2026.1  
**Professor:** Dimmy Magalhães  
**Instituição:** ICEV — Instituto de Ensino Superior  
**Autor:** Adler Castro Alves  

---

Este projeto foi desenvolvido como parte de um laboratório com o objetivo de implementar a arquitetura Transformer do zero, sem o uso de implementações prontas.

A proposta é compreender, na prática, como os principais componentes do modelo funcionam e como ocorre o fluxo de dados entre encoder e decoder.

---

## Implementação

Foram implementados os seguintes componentes:

- Scaled Dot-Product Attention
- Feed-Forward Network (FFN)
- Add & Norm (conexão residual com normalização de camada)
- Encoder empilhado
- Decoder com Masked Self-Attention e Cross-Attention
- Máscara causal
- Positional Encoding
- Geração auto-regressiva

---

## Funcionamento

### Encoder

O encoder recebe a sequência de entrada (por exemplo, "Thinking Machines") e gera uma representação contextualizada **Z**.

**Fluxo:**

```
Self-Attention → Add & Norm → FFN → Add & Norm
```

### Decoder

O decoder gera a sequência de saída com base no que já foi gerado e na saída do encoder.

**Fluxo:**

```
Masked Self-Attention → Add & Norm
→ Cross-Attention → Add & Norm
→ FFN → Add & Norm → Linear → Softmax
```

---

## Inferência

- O processo inicia com o token `<START>`
- A cada passo, o modelo prevê o próximo token
- O token previsto é concatenado à entrada do decoder
- O processo continua até a geração de `<EOS>` ou até atingir um limite máximo

---

## Observação

O modelo não foi treinado, portanto os pesos estão inicializados aleatoriamente.

Dessa forma, as sequências geradas não possuem significado semântico. O objetivo do projeto é validar a arquitetura e o fluxo de execução do Transformer.

---

## Estrutura do projeto

```
attention.py
encoder.py
decoder.py
ffn.py
utils.py
input_processing.py
main.py
README.md
```

---

## Tecnologias utilizadas

- Python
- PyTorch

---

## Objetivo

- Compreender o funcionamento interno do Transformer
- Implementar manualmente os principais componentes do modelo
- Analisar o fluxo encoder-decoder em tarefas sequenciais

---
**Anexo Google Colab:**
[https://colab.research.google.com/drive/18xaIvYa7SQkIF_1nKpSd5JJU06xOSLHv?usp=sharing]


**Referência:**  
* GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. Deep Learning. [S. l.]: MIT Press, 2016..
 * JURAFSKY, Daniel; MARTIN, James H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3. ed. draft. [S. l.]: Stanford University/University of Colorado at Boulder, 2026..
 * RASCHKA, Sebastian. Build a Large Language Model (From Scratch). 1. ed. [S. l.]: Manning (MEAP), 2021..
 * UNIVERSIDADE FEDERAL DO PIAUÍ. Estágio Curricular Supervisionado - Fábrica de Software I: normas para o estágio supervisionado. Teresina: UFPI, 2026..
 * VASWANI, Ashish et al. Atenção é tudo o que você precisa. Tradução de Machine Translated by Google. [S. l.]: Google Brain/Google Research, 2017..
