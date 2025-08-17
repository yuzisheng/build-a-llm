import torch

from model_utils import create_dataloader_v1


def test_dataloader(tmp_path):
    with open("data/corpus.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    raw_text = raw_text[:10000]

    vocab_size = 50257
    output_dim = 256
    context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    batch_size = 8
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length
    )

    for batch in dataloader:
        x, y = batch
        token_embeddings = token_embedding_layer(x)  # shape: (batch_size, num_tokens, output_dim)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))  # shape: (num_tokens, output_dim)
        input_embeddings = token_embeddings + pos_embeddings  # broadcast
        assert input_embeddings.shape == torch.Size([8, 4, 256])
        break
