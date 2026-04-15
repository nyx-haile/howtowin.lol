"""Evaluation helpers beyond per-batch top-5."""
import torch


def top5_by_minute(logits, labels, mask, timestamps, k=5):
    """Return dict with 'overall' top-k accuracy and 'by_minute' breakdown.

    logits: (B, L, C)
    labels: (B, L, C) multi-hot
    mask:   (B, L)
    timestamps: (B, L) in ms
    """
    B, L, C = logits.shape
    topk = logits.topk(k=min(k, C), dim=-1).indices
    top_true = labels.argmax(dim=-1)
    pos_mask = (labels.sum(dim=-1) > 0) & mask.bool()
    hits = (topk == top_true.unsqueeze(-1)).any(dim=-1) & pos_mask

    minutes = (timestamps / 60000.0).long()

    result = {"overall": 0.0, "by_minute": {}}
    flat_hits = hits[pos_mask]
    if flat_hits.numel() > 0:
        result["overall"] = flat_hits.float().mean().item()

    unique_minutes = torch.unique(minutes[pos_mask])
    for m in unique_minutes.tolist():
        m_mask = (minutes == m) & pos_mask
        if m_mask.any():
            acc = hits[m_mask].float().mean().item()
            result["by_minute"][m] = acc

    return result
