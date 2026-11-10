import unittest

import torch
from torch import nn

from tiny_gpt_model import TinyGPT


class TinyGPTModelTests(unittest.TestCase):
    def test_future_tokens_do_not_change_earlier_logits(self):
        torch.manual_seed(0)
        model = TinyGPT(
            vocab_size=16,
            embedding_dim=16,
            num_heads=4,
            block_size=8,
            num_layers=2,
            dropout=0.2,
        )
        model.eval()

        original = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        changed = original.clone()
        changed[:, 4:] = torch.tensor([[10, 11, 12, 13]])

        with torch.no_grad():
            original_logits = model(original)
            changed_logits = model(changed)

        torch.testing.assert_close(
            original_logits[:, :4],
            changed_logits[:, :4],
        )

    def test_model_can_overfit_a_tiny_batch(self):
        torch.manual_seed(0)
        model = TinyGPT(
            vocab_size=8,
            embedding_dim=16,
            num_heads=4,
            block_size=8,
            num_layers=2,
            dropout=0.0,
        )
        x = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0],
            [2, 3, 4, 5, 6, 7, 0, 1],
            [3, 4, 5, 6, 7, 0, 1, 2],
        ])
        y = torch.roll(x, shifts=-1, dims=1)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        model.eval()
        with torch.no_grad():
            initial_loss = loss_fn(
                model(x).reshape(-1, 8),
                y.reshape(-1),
            ).item()

        model.train()
        for _ in range(200):
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, 8), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_loss = loss_fn(
                model(x).reshape(-1, 8),
                y.reshape(-1),
            ).item()

        self.assertLess(final_loss, initial_loss * 0.1)


if __name__ == "__main__":
    unittest.main()
