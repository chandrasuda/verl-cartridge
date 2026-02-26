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
DataParallel Cartridge PPO Actor.

This actor wraps a HuggingFace model with CacheAndModel from the Cartridges repo,
enabling gradient flow into the TrainableCache during backprop.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

# Cartridges imports
try:
    from cartridges.cache import AttnConfig, TrainableCache
    from cartridges.models.attention import create_block_mask_w_cache, flex_attention_forward
    from cartridges.train import CacheAndModel
    HAS_CARTRIDGES = True
except ImportError:
    HAS_CARTRIDGES = False

__all__ = ["CartridgePPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CartridgePPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor with Cartridge support.
    
    This actor wraps the model with CacheAndModel, enabling gradient flow
    into the TrainableCache while keeping the base model frozen.
    
    Args:
        config: Actor configuration
        actor_module: The base HuggingFace model (will be frozen)
        actor_optimizer: Optimizer (will only optimize cache parameters)
        cartridge_path: Path to the saved TrainableCache checkpoint
        num_frozen_tokens: Number of tokens to freeze at start of cache
    """

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        cartridge_path: Optional[str] = None,
        num_frozen_tokens: int = 1,
    ):
        if not HAS_CARTRIDGES:
            raise ImportError(
                "Cartridges package not found. Please install from cartridges/ directory: "
                "cd cartridges && pip install -e ."
            )
        
        super().__init__(config)
        self.config = config
        self.base_model = actor_module
        self.cartridge_path = cartridge_path
        self.num_frozen_tokens = num_frozen_tokens
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Load or create TrainableCache
        self.cache = self._load_or_create_cache()
        
        # Wrap with CacheAndModel
        self.wrapped_model = CacheAndModel(cache=self.cache, model=self.base_model)
        
        # FSDP wrap the combined model (only cache params are trainable)
        self.actor_module = self._fsdp_wrap_model(self.wrapped_model)
        
        # Create optimizer for cache parameters only
        if actor_optimizer is None:
            self.actor_optimizer = self._create_optimizer()
        else:
            self.actor_optimizer = actor_optimizer
        
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        
        # Set up gradient scaler for fp16
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None
        
        # Compile entropy computation
        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits
        
        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else entropy_from_logits
        )

    def _load_or_create_cache(self) -> TrainableCache:
        """Load TrainableCache from checkpoint or create a new one."""
        device = get_device_id()
        
        if self.cartridge_path and os.path.exists(self.cartridge_path):
            logger.info(f"Loading TrainableCache from {self.cartridge_path}")
            cache = TrainableCache.from_pretrained(self.cartridge_path, device=device)
        else:
            # Create a new cache with random initialization
            logger.info("Creating new TrainableCache")
            config = self._get_attn_config()
            # Initialize with small random values
            init_keys = [
                torch.randn(1, config.n_heads, 128, config.head_dim, device=device) * 0.01
                for _ in range(config.n_layers)
            ]
            init_values = [
                torch.randn(1, config.n_heads, 128, config.head_dim, device=device) * 0.01
                for _ in range(config.n_layers)
            ]
            cache = TrainableCache(
                config=config,
                init_keys=init_keys,
                init_values=init_values,
                num_frozen_tokens=self.num_frozen_tokens,
            )
        
        return cache.to(device)

    def _get_attn_config(self) -> AttnConfig:
        """Extract attention config from the base model."""
        model_config = self.base_model.config
        return AttnConfig(
            n_layers=model_config.num_hidden_layers,
            n_heads=model_config.num_key_value_heads,
            head_dim=(
                model_config.head_dim
                if hasattr(model_config, "head_dim")
                else model_config.hidden_size // model_config.num_attention_heads
            ),
        )

    def _fsdp_wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with FSDP, only sharding the cache parameters."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import MixedPrecision
        
        mp_policy = MixedPrecision(
            param_dtype=self.param_dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=self.param_dtype,
        )
        
        # Only wrap the cache parameters with FSDP
        # The base model is already frozen and doesn't need gradients
        return FSDP(
            model,
            mixed_precision=mp_policy,
            use_orig_params=True,
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for cache parameters only."""
        from verl.workers.config.optimizer import build_optimizer
        
        # Only optimize cache parameters
        return build_optimizer(
            self.config.optim,
            self.cache.parameters(),
        )

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with cartridge cache.
        
        Returns:
            dict with log_probs and optionally entropys
        """
        response_length = micro_batch["responses"].size(-1)
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        
        # Create seq_ids for flex_attention (all same sequence for training)
        seq_ids = torch.zeros(seqlen, dtype=torch.long, device=input_ids.device)
        
        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            # Clear cache before forward
            self.cache.clear()
            
            # Forward through CacheAndModel
            outputs = self.actor_module(
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
            )
            
            logits = outputs.logits
            logits.div_(temperature)
            
            # Extract response logits
            response_logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            
            # Compute log probs
            log_probs = logprobs_from_logits(response_logits, micro_batch["responses"])
            
            # Compute entropy if requested
            entropy = None
            if calculate_entropy:
                if not self.config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(response_logits)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(
                        verl_F.entropy_from_logits, response_logits
                    )
        
        result = {"log_probs": log_probs}
        if calculate_entropy:
            result["entropys"] = entropy
        
        return result

    @GPUMemoryLogger(role="cartridge actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute log probabilities with cartridge cache.
        
        This is called during the actor update phase and computes gradients
        that flow into the TrainableCache.
        """
        self.actor_module.eval()
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        data = data.select(batch_keys=select_keys)
        
        micro_batches = data.split(micro_batch_size)
        
        log_probs_lst = []
        entropy_lst = []
        
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, "pad_token_id": pad_token_id}
            
            # During actor update, we need gradients
            outputs = self._forward_micro_batch(
                model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
            )
            
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
        
        log_probs = torch.concat(log_probs_lst, dim=0)
        
        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = torch.concat(entropy_lst, dim=0)
        
        return outputs

    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        assert self.config.grad_clip is not None
        
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(
                self.actor_module.parameters(),
                max_norm=self.config.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(),
                max_norm=self.config.grad_clip
            )
        
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                logger.warning(f"grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        
        return grad_norm

    @GPUMemoryLogger(role="cartridge actor", logger=logger)
    def update_policy(self, data: DataProto):
        """Update the cartridge cache using PPO/GRPO loss.
        
        The base model remains frozen; only the cache parameters are updated.
        """
        self.actor_module.train()
        
        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        
        data = data.select(batch_keys=select_keys)
        
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        
        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                self.gradient_accumulation = (
                    self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                
                self.actor_optimizer.zero_grad()
                
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]
                    
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    
                    loss_scale_factor = 1 / self.gradient_accumulation
                    
                    # Forward pass (gradients flow to cache)
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
                    
                    log_prob = outputs["log_probs"]
                    entropy = outputs.get("entropys")
                    
                    # Use detached old_log_prob for PPO
                    old_log_prob = model_inputs["old_log_probs"]
                    
                    # Compute policy loss
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )
                    
                    policy_loss = pg_loss
                    
                    # Add entropy bonus
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff
                    
                    # Add KL penalty if using reference policy
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                    
                    # Scale and backward
                    loss = policy_loss * loss_scale_factor
                    
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, pg_metrics)
                
                # Optimizer step
                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        
        self.actor_optimizer.zero_grad()
        return metrics

    def save_cartridge(self, path: str):
        """Save the current cartridge cache to disk."""
        self.cache.save(path)
        logger.info(f"Saved cartridge to {path}")
