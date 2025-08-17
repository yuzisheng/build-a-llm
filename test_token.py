import torch
import tiktoken

from model_utils import text_to_token_ids, token_ids_to_text


def test_token(tmp_path):
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "你好，世界"

    encoded = text_to_token_ids(text, tokenizer)
    expect_encoded = torch.tensor([
        [19526, 254, 25001, 121, 171, 120, 234, 10310, 244, 45911, 234]
    ])
    assert torch.equal(expect_encoded, encoded)

    decoded = token_ids_to_text(encoded, tokenizer)
    assert text == decoded
