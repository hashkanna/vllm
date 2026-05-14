# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.models import gemma4_mtp


def test_gemma4_mtp_full_attention_uses_global_kv_heads_for_k_eq_v() -> None:
    config = SimpleNamespace(
        attention_k_eq_v=True,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
    )

    assert gemma4_mtp._get_mtp_num_kv_heads(config, is_full_attention=True) == 2
    assert gemma4_mtp._get_mtp_num_kv_heads(config, is_full_attention=False) == 8
