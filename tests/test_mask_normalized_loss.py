import torch

from rna_head_alignment.alignment.attention_losses import mask_normalized_mse


def test_mask_normalized_loss_ignores_padding():
    teacher = torch.tensor([[[[0.7, 0.3], [0.2, 0.8]]]], dtype=torch.float32)
    student = torch.tensor([[[[0.6, 0.4], [0.1, 0.9]]]], dtype=torch.float32)
    mask = torch.tensor([[1, 1]], dtype=torch.bool)
    base = mask_normalized_mse(student, teacher, mask)

    teacher_pad = torch.zeros((1, 1, 4, 4))
    student_pad = torch.zeros((1, 1, 4, 4))
    teacher_pad[:, :, :2, :2] = teacher
    student_pad[:, :, :2, :2] = student
    mask_pad = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    padded = mask_normalized_mse(student_pad, teacher_pad, mask_pad)
    assert torch.isfinite(padded)
    assert torch.allclose(base, padded)
