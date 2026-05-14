# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.gemma4 import _should_use_kv_sharing_fast_prefill


def test_kv_sharing_fast_prefill_requires_reduced_cross_decoder_batch() -> None:
    logits_indices = torch.arange(8)

    assert not _should_use_kv_sharing_fast_prefill(None, None, batch_size=32)
    assert not _should_use_kv_sharing_fast_prefill(
        logits_indices, num_logits_indices=32, batch_size=32
    )
    assert not _should_use_kv_sharing_fast_prefill(
        torch.arange(32), num_logits_indices=4, batch_size=32
    )
    assert _should_use_kv_sharing_fast_prefill(
        logits_indices, num_logits_indices=4, batch_size=32
    )
