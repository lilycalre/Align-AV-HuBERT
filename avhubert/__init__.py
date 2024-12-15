# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .hubert_asr import *  # noqa
from .hubert_pretraining import *


#add alignment
from .utils import Compose
from .hubert_dataset_plusalignment import *
from .hubert_pretraining_plusalignment import *
from .hubert_plusalignment import *