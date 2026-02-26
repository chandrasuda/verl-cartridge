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
Cartridge Actor Worker for veRL.

This worker extends the standard ActorRolloutRefWorker to use CartridgePPOActor,
enabling on-policy training of Cartridge caches.
"""

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import fsdp_version
from verl.workers.actor.cartridge_actor.dp_cartridge_actor import CartridgePPOActor
from verl.workers.config import ActorConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CartridgeActorWorker(ActorRolloutRefWorker):
    """Worker that creates a Cartridge-aware actor for on-policy cartridge training.
    
    This worker:
    1. Loads/creates a TrainableCache
    2. Wraps the model with CacheAndModel
    3. Freezes base model parameters
    4. Only optimizes the cartridge cache
    
    Config requirements:
        actor_rollout_ref:
          actor:
            use_cartridge: true
            cartridge_path: /path/to/cache.pt  # Optional
            cartridge_num_frozen_tokens: 1  # Default
    """

    def __init__(self, config: DictConfig, role: str = "actor", **kwargs):
        # Check if cartridge is enabled
        self.use_cartridge = config.actor.get("use_cartridge", False)
        self.cartridge_path = config.actor.get("cartridge_path", None)
        self.cartridge_num_frozen_tokens = config.actor.get("cartridge_num_frozen_tokens", 1)
        
        if self.use_cartridge:
            logger.info(f"Initializing CartridgeActorWorker with cartridge_path={self.cartridge_path}")
        
        # Call parent init
        super().__init__(config=config, role=role, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model with cartridge support."""
        if not self.use_cartridge:
            # Fall back to standard initialization
            return super().init_model()
        
        # Import here to avoid circular dependencies
        from verl.workers.actor import DataParallelPPOActor
        from verl.utils.import_utils import import_external_libs
        
        import_external_libs(self.config.model.get("external_lib", None))
        
        # Initialize QAT config
        self._init_qat_config()
        
        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        
        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = None
            
            local_path = self._copy_to_local(self.config.model.path, use_shm=use_shm)
            
            # Build model (this creates the base HuggingFace model)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get(
                    "enable_gradient_checkpointing", False
                ),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get(
                    "enable_activation_offload", False
                ),
            )
            
            # Get unwrapped module for cartridge wrapping
            if fsdp_version(self.actor_module_fsdp) == 1:
                base_model = self.actor_module_fsdp._fsdp_wrapped_module
            else:
                base_model = self.actor_module_fsdp
            
            if self._is_offload_param:
                from verl.utils.fsdp_utils import load_fsdp_model_to_gpu
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            
            if self._is_offload_optimizer:
                from verl.utils.fsdp_utils import offload_fsdp_optimizer
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        
        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            
            # Create CartridgePPOActor instead of standard DataParallelPPOActor
            self.actor = CartridgePPOActor(
                config=actor_cfg,
                actor_module=base_model,
                actor_optimizer=self.actor_optimizer,
                cartridge_path=self.cartridge_path,
                num_frozen_tokens=self.cartridge_num_frozen_tokens,
            )
            
            logger.info("Created CartridgePPOActor")
            
            # Update FSDP reference to use the wrapped model
            self.actor_module_fsdp = self.actor.actor_module
        
        if self._is_rollout:
            self._build_rollout(
                trust_remote_code=self.config.model.get("trust_remote_code", False)
            )
        
        if self._is_ref:
            # Reference policy doesn't need cartridge (it's frozen anyway)
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)
            
            if self.rank == 0:
                print("reference model:", ref_model_path)
            
            local_path = self._copy_to_local(ref_model_path, use_shm=use_shm)
            
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            
            # Use standard actor for reference (no cartridge needed)
            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.ref_module_fsdp,
            )
        
        if self._is_actor:
            from verl.utils.flops_counter import FlopsCounter
            from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
            
            self.flops_counter = FlopsCounter(self.actor_model_config)
            
            # Checkpoint manager for cartridge
            self.checkpoint_manager = CartridgeCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                cartridge=self.actor.cache,
            )

    def _copy_to_local(self, path: str, use_shm: bool = False) -> str:
        """Helper to copy model to local storage."""
        from verl.utils.fs import copy_to_local
        return copy_to_local(path, use_shm=use_shm)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_cartridge(self, path: Optional[str] = None):
        """Save the current cartridge cache.
        
        Args:
            path: Path to save cartridge. If None, uses default path.
        """
        if not self.use_cartridge or not hasattr(self, "actor"):
            logger.warning("Cannot save cartridge: not using cartridge actor")
            return
        
        if path is None:
            path = os.path.join(self.config.actor.checkpoint.checkpoint_dir, "cartridge_latest.pt")
        
        self.actor.save_cartridge(path)
        
        if dist.get_rank() == 0:
            logger.info(f"Saved cartridge to {path}")


class CartridgeCheckpointManager:
    """Checkpoint manager that also saves/loads the cartridge cache."""
    
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        processing_class,
        checkpoint_config,
        trust_remote_code: bool = False,
        cartridge=None,
    ):
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
        
        self.base_manager = FSDPCheckpointManager(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
            trust_remote_code=trust_remote_code,
        )
        self.cartridge = cartridge
    
    def save_checkpoint(self, step: int, path: str = None):
        """Save model, optimizer, and cartridge."""
        # Save base checkpoint
        self.base_manager.save_checkpoint(step=step, path=path)
        
        # Save cartridge
        if self.cartridge is not None:
            cartridge_path = os.path.join(path, "cartridge.pt") if path else f"cartridge_step_{step}.pt"
            self.cartridge.save(cartridge_path)
            
            if dist.get_rank() == 0:
                logger.info(f"Saved cartridge checkpoint to {cartridge_path}")
    
    def load_checkpoint(self, step: int, path: str = None):
        """Load model, optimizer, and cartridge."""
        # Load base checkpoint
        self.base_manager.load_checkpoint(step=step, path=path)
        
        # Load cartridge if exists
        if self.cartridge is not None and path:
            cartridge_path = os.path.join(path, "cartridge.pt")
            if os.path.exists(cartridge_path):
                # Note: Cartridge is already loaded in CartridgePPOActor.__init__
                # This is for loading during resume
                logger.info(f"Found cartridge checkpoint at {cartridge_path}")
