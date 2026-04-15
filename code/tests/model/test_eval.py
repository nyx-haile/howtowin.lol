import torch
from model.eval import top5_by_minute


def test_top5_by_minute_shape():
    B, L, C = 2, 20, 11
    logits = torch.randn(B, L, C)
    labels = torch.zeros(B, L, C)
    labels[:, :, 0] = 1.0  # all label = class 0
    mask = torch.ones(B, L)
    timestamps = torch.arange(0, L * 60000, 60000, dtype=torch.float32).unsqueeze(0).expand(B, L)
    result = top5_by_minute(logits, labels, mask, timestamps)
    assert "overall" in result
    assert "by_minute" in result
    assert isinstance(result["by_minute"], dict)


def test_top5_perfect_when_class_0_is_argmax():
    B, L, C = 1, 5, 11
    logits = torch.zeros(B, L, C)
    logits[..., 0] = 10.0  # class 0 easily in top-5
    labels = torch.zeros(B, L, C)
    labels[..., 0] = 1.0
    mask = torch.ones(B, L)
    timestamps = torch.zeros(B, L)
    result = top5_by_minute(logits, labels, mask, timestamps)
    assert result["overall"] == 1.0
