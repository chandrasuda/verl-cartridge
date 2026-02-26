# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for CartridgePPOActor and CartridgeActorWorker.

Tests that require the ``cartridges`` package will be skipped if it is not installed.
Run: pytest tests/workers/actor/test_cartridge_actor.py -v
"""

import importlib

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

_has_cartridges = importlib.util.find_spec("cartridges.cache") is not None
requires_cartridges = pytest.mark.skipif(not _has_cartridges, reason="cartridges package not installed")


# ---------------------------------------------------------------------------
# Tests that work WITHOUT the cartridges package
# ---------------------------------------------------------------------------


class TestCartridgePPOActorImport:
    """Tests for CartridgePPOActor that run without cartridges installed."""

    def test_has_cartridges_flag(self):
        """HAS_CARTRIDGES flag should be a bool (True/False depending on install)."""
        from verl.workers.actor.cartridge_actor.dp_cartridge_actor import HAS_CARTRIDGES

        assert isinstance(HAS_CARTRIDGES, bool)

    def test_import_module_does_not_crash(self):
        """Importing the module should never crash, even without cartridges."""
        import verl.workers.actor.cartridge_actor  # noqa: F401

    def test_lazy_getattr(self):
        """Accessing CartridgePPOActor via __init__ should succeed (class defined even without cartridges)."""
        from verl.workers.actor.cartridge_actor.dp_cartridge_actor import CartridgePPOActor

        assert CartridgePPOActor is not None


class TestCartridgeActorWorker:
    """Test CartridgeActorWorker config detection (does NOT require cartridges)."""

    def test_import(self):
        """CartridgeActorWorker should be importable when full veRL deps are available."""
        try:
            from verl.workers.actor.cartridge_actor.cartridge_worker import CartridgeActorWorker  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Missing veRL transitive dependency: {e}")

    def test_worker_config_detection(self):
        """Worker correctly reads use_cartridge / cartridge_path / num_frozen_tokens from config."""
        from omegaconf import OmegaConf

        try:
            # This may fail if veRL's full dependency tree isn't installed
            with patch("verl.workers.actor.cartridge_actor.cartridge_worker.ActorRolloutRefWorker.__init__"):
                from verl.workers.actor.cartridge_actor.cartridge_worker import CartridgeActorWorker

                config = OmegaConf.create({
                    "actor": {
                        "use_cartridge": True,
                        "cartridge_path": "/tmp/test.pt",
                        "cartridge_num_frozen_tokens": 2,
                    }
                })

                worker = CartridgeActorWorker(config, role="actor")

                assert worker.use_cartridge is True
                assert worker.cartridge_path == "/tmp/test.pt"
                assert worker.cartridge_num_frozen_tokens == 2
        except ImportError as e:
            pytest.skip(f"Missing veRL transitive dependency: {e}")


# ---------------------------------------------------------------------------
# Tests that REQUIRE the cartridges package
# ---------------------------------------------------------------------------


@requires_cartridges
class TestTrainableCache:
    """Tests for the Cartridges TrainableCache integration."""

    def test_creation(self):
        """Create a TrainableCache and verify shape."""
        from cartridges.cache import AttnConfig, TrainableCache

        config = AttnConfig(n_layers=2, n_heads=4, head_dim=64)
        init_keys = [torch.randn(1, 4, 16, 64) * 0.01 for _ in range(2)]
        init_values = [torch.randn(1, 4, 16, 64) * 0.01 for _ in range(2)]

        cache = TrainableCache(
            config=config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=1,
        )

        assert cache._num_trainable_tokens == 15
        assert cache._num_frozen_tokens == 1
        assert len(cache.trainable_keys) == 2
        assert len(cache.frozen_keys) == 2

    def test_parameters_require_grad(self):
        """Trainable params have grad, frozen params do not."""
        from cartridges.cache import AttnConfig, TrainableCache

        config = AttnConfig(n_layers=2, n_heads=4, head_dim=64)
        init_keys = [torch.randn(1, 4, 16, 64) * 0.01 for _ in range(2)]
        init_values = [torch.randn(1, 4, 16, 64) * 0.01 for _ in range(2)]

        cache = TrainableCache(
            config=config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=1,
        )

        for param in cache.trainable_keys:
            assert param.requires_grad
        for param in cache.trainable_values:
            assert param.requires_grad
        for param in cache.frozen_keys:
            assert not param.requires_grad
        for param in cache.frozen_values:
            assert not param.requires_grad


@requires_cartridges
class TestCacheAndModel:
    """Test CacheAndModel wrapping."""

    def test_creation(self):
        """Wrap a mock model with CacheAndModel."""
        from cartridges.cache import AttnConfig, TrainableCache
        from cartridges.train import CacheAndModel

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MagicMock()
                self.config.num_hidden_layers = 2
                self.config.num_key_value_heads = 4
                self.config.head_dim = 64

            def forward(self, input_ids, seq_ids, position_ids, use_cache, past_key_values):
                return MagicMock(logits=torch.randn(1, 10, 1000))

        model = MockModel()
        config = AttnConfig(n_layers=2, n_heads=4, head_dim=64)
        init_keys = [torch.randn(1, 4, 16, 64) * 0.01 for _ in range(2)]
        init_values = [torch.randn(1, 4, 16, 64) * 0.01 for _ in range(2)]

        cache = TrainableCache(
            config=config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=1,
        )

        wrapped = CacheAndModel(cache=cache, model=model)
        assert wrapped.cache is cache
        assert wrapped.model is model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
