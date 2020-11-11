# Copyright 2019 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utilities for using Sonnet with TensorFlow Distribution Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet.src.distribute.batch_norm import CrossReplicaBatchNorm
from sonnet.src.distribute.replicator import create_variables_eagerly
from sonnet.src.distribute.replicator import Replicator
from sonnet.src.distribute.replicator import TpuReplicator

__all__ = (
    "create_variables_eagerly",
    "Replicator",
    "TpuReplicator",
    "CrossReplicaBatchNorm",
)
