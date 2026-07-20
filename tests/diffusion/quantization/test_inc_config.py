# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for INC/AutoRound quantization via the unified framework."""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def test_build_quant_config_autoround():
    """build_quant_config("auto-round", ...) should produce an INCConfig."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        "auto-round",
        bits=4,
        group_size=128,
        sym=True,
        packing_format="auto_round:auto_gptq",
    )
    assert config is not None
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4
    assert config.group_size == 128


def test_build_quant_config_inc():
    """build_quant_config("inc", ...) should also produce an INCConfig."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("inc", bits=4, group_size=128)
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4


def test_build_quant_config_autoround_mxfp4_preserves_schema():
    """Qwen3-Omni MXFP4 metadata must reach vLLM's INCConfig unchanged."""
    from vllm_omni.quantization import build_quant_config

    extra_config = {
        ".*self_attn.*": {
            "bits": 16,
            "data_type": "float",
            "act_bits": 16,
            "act_data_type": "float",
        }
    }
    config = build_quant_config(
        "auto-round",
        bits=4,
        act_bits=4,
        data_type="mx_fp",
        act_data_type="mx_fp",
        group_size=32,
        act_group_size=32,
        sym=True,
        packing_format="auto_round:llm_compressor",
        block_name_to_quantize="thinker.model.layers",
        extra_config=extra_config,
    )

    assert config.weight_bits == 4
    assert config.group_size == 32
    assert config.data_type == "mx_fp"
    assert config.packing_format == "auto_round:llm_compressor"
    assert config.block_name_to_quantize == ["thinker.model.layers"]
    assert config.extra_config == extra_config


def test_build_quant_config_autoround_dict():
    """Dict-style config with method=auto-round should work."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        {
            "method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "packing_format": "auto_round:auto_gptq",
        }
    )
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4


def test_build_quant_config_autoround_filters_metadata():
    """Checkpoint metadata keys (autoround_version, batch_size, iters)
    should be silently filtered out instead of causing TypeError."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        "auto-round",
        bits=4,
        group_size=128,
        sym=True,
        packing_format="auto_round:auto_gptq",
        block_name_to_quantize="transformer_blocks,single_transformer_blocks",
        autoround_version="0.12.0",  # metadata — must be filtered
        batch_size=1,  # metadata — must be filtered
        iters=0,  # metadata — must be filtered
    )
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4
    assert config.group_size == 128


def test_build_quant_config_bits_to_weight_bits_mapping():
    """The 'bits' key from checkpoints should be mapped to 'weight_bits'."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    # If weight_bits is already provided, bits should be ignored
    config = build_quant_config("auto-round", weight_bits=4, group_size=128)
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4


def test_autoround_in_supported_methods():
    """auto-round and inc should appear in SUPPORTED_QUANTIZATION_METHODS."""
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert "auto-round" in SUPPORTED_QUANTIZATION_METHODS
    assert "inc" in SUPPORTED_QUANTIZATION_METHODS


def test_integration_autoround_via_omni_diffusion_config():
    """OmniDiffusionConfig with auto-round quantization dict should resolve."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization_config={
            "method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": True,
        },
    )
    assert isinstance(config.quantization_config, INCConfig)
    assert config.quantization_config.weight_bits == 4


def test_integration_autodetect_from_transformer_config():
    """When TransformerConfig has quant_config, OmniDiffusionConfig should
    auto-detect it even without explicit quantization_config."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig

    tf_config = TransformerConfig.from_dict(
        {
            "quantization_config": {
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "packing_format": "auto_round:auto_gptq",
                "autoround_version": "0.12.0",
                "batch_size": 1,
                "iters": 0,
            }
        }
    )
    assert tf_config.quant_method == "auto-round"
    assert isinstance(tf_config.quant_config, INCConfig)

    od_config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
    assert isinstance(od_config.quantization_config, INCConfig)
    assert od_config.quantization_config.weight_bits == 4
