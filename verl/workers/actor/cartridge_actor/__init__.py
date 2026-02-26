# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Cartridge-aware Actor for on-policy cartridge training with veRL.

Requires the ``cartridges`` package to be installed.
"""


__all__ = ["CartridgePPOActor"]


def __getattr__(name: str):
    """Lazy import to avoid crashing when cartridges package is not installed."""
    if name == "CartridgePPOActor":
        from verl.workers.actor.cartridge_actor.dp_cartridge_actor import CartridgePPOActor

        return CartridgePPOActor
    # Let Python handle submodule lookups normally
    raise AttributeError(name)
