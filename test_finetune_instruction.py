import json
from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

from finetune_instruction import (
    InstructionDataset, format_input, custom_collate_fn
)
from model import GPTModel
from model_train import train_model_simple


def test_finetune_instruction(tmp_path):
    #######################################
    # prepare dataset
    #######################################
    file_path = "data/instruction.json"
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # Use very small subset for testing purposes
    train_data = train_data[:15]
    val_data = val_data[:15]
    test_data = test_data[:15]

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=100)

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # Load pretrained model
    #######################################

    # Small GPT model for testing purposes
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 120,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "emb_dim": 12,
        "n_layers": 1,
        "n_heads": 2
    }
    model = GPTModel(BASE_CONFIG)
    model.eval()
    device = "cpu"
    CHOOSE_MODEL = "Small test model"

    print("Loaded model:", CHOOSE_MODEL)
    print(50 * "-")

    #######################################
    # Finetuning the model
    #######################################

    num_epochs = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    assert round(train_losses[0], 1) == 10.9
    assert round(val_losses[0], 1) == 10.9
    assert train_losses[-1] < train_losses[0]
