# Copyright 2022 The mT5 Authors.
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

"""Tests for mT5 tasks."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import multilingual_t5.tasks  # pylint:disable=unused-import
import seqio
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()
tf.enable_eager_execution()

MixtureRegistry = seqio.MixtureRegistry
TaskRegistry = seqio.TaskRegistry
_SEQUENCE_LENGTH = {'inputs': 128, 'targets': 128}

_TASKS = [
    'mc4.en',
    'mt5_wiki.en',
    'mt5_super_glue_boolq_v102',
    'mt5_super_glue_cb_v102',
    'mt5_super_glue_copa_v102',
    'mt5_super_glue_multirc_v102',
    'mt5_super_glue_record_v102',
    'mt5_super_glue_rte_v102',
    'mt5_super_glue_wic_v102',
    'mt5_super_glue_wsc_v102_simple_eval',
    'mt5_super_glue_wsc_v102_simple_train',
    'mt5_wmt15_enfr_v003',
    'mt5_wmt16_enro_v003',
    'mt5_wmt_t2t_ende_v003',
    'mt5_gem_xsum',
    'mt5_xnli_translate_train.ar',
]

_MIXTURES = [
    'mt5_xnli_zeroshot',
    'mt5_pawsx_zeroshot',
    'mt5_pawsx_translate_train',
    'mt5_tydiqa',
    'mt5_tydiqa_zeroshot',
    'mt5_tydiqa_translate_train',
    'mt5_xquad_zeroshot',
    'mt5_xquad_translate_train',
    'mt5_mlqa_zeroshot',
    'mt5_mlqa_translate_train',
    'mt5_super_glue_v102_proportional',
]


class TasksTest(parameterized.TestCase):

  @parameterized.parameters(((name,) for name in _TASKS))
  def test_task(self, name):
    task = TaskRegistry.get(name)
    split = 'train' if 'train' in task.splits else 'validation'
    logging.info('task=%s, split=%s', name, split)
    ds = task.get_dataset(_SEQUENCE_LENGTH, split)
    for d in ds:
      logging.info(d)
      break

  @parameterized.parameters(((name,) for name in _MIXTURES))
  def test_mixture(self, name):
    mixture = MixtureRegistry.get(name)
    logging.info('mixture=%s', name)
    ds = mixture.get_dataset(_SEQUENCE_LENGTH, 'train')
    for d in ds:
      logging.info(d)
      break


if __name__ == '__main__':
  absltest.main()
