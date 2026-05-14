# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models import gemma4_mtp


def test_gemma4_mtp_full_lm_head_weight_cache_reuses_tp_all_gather(
    monkeypatch,
) -> None:
    model = gemma4_mtp.Gemma4MTP.__new__(gemma4_mtp.Gemma4MTP)
    model.lm_head = SimpleNamespace(
        weight=torch.arange(12, dtype=torch.float32).reshape(6, 2)
    )
    model.masked_embedding = SimpleNamespace(vocab_size=10)
    model._full_lm_head_weight_cache = None

    gather_calls = 0

    def fake_all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        nonlocal gather_calls
        gather_calls += 1
        return torch.cat([tensor, tensor], dim=dim)

    monkeypatch.setattr(gemma4_mtp, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(gemma4_mtp, "tensor_model_parallel_all_gather", fake_all_gather)

    first = model._get_full_lm_head_weight()
    second = model._get_full_lm_head_weight()

    assert gather_calls == 1
    assert first.data_ptr() == second.data_ptr()
    assert first.shape == (10, 2)

    model._invalidate_full_lm_head_weight_cache()
    third = model._get_full_lm_head_weight()

    assert gather_calls == 2
    assert third.data_ptr() != first.data_ptr()
