import pytest
import tiktoken
import torch

from model import GPTModel, GPTModelFast
from model_utils import generate_text_simple

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}


@pytest.mark.parametrize("ModelClass", [GPTModel, GPTModelFast])
@pytest.mark.parametrize("generate_fn", [generate_text_simple])
def test_model_without_kvcache(ModelClass, generate_fn):
    torch.manual_seed(123)
    model = ModelClass(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_fn(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    expect = torch.tensor([
        [15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267,
         49706, 43231, 47062, 34657]
    ])
    assert torch.equal(expect, out), "Generated output does not match expected output"


from model_gpt2 import GPTModel as GPTModelKV
from model_utils import generate_text_simple_with_kvcache


@pytest.mark.parametrize("ModelClass", [GPTModelKV])
@pytest.mark.parametrize("generate_fn", [generate_text_simple_with_kvcache])
def test_model_with_kvcache(ModelClass, generate_fn):
    torch.manual_seed(123)
    model = ModelClass(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_fn(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    expect = torch.tensor([
        [15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267,
         49706, 43231, 47062, 34657]
    ])
    assert torch.equal(expect, out), "Generated output does not match expected output"
