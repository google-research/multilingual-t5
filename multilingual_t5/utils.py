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

"""Utilities for Multilingual T5 data loading and processing.
"""

import tensorflow_datasets as tfds


# Languages present in XQuAD train and dev splits.
XQUAD_LANGS_TRAIN_DEV = ["ar", "de", "el", "es", "hi", "ru", "th", "tr", "vi",
                         "zh"]
# Languages present XQuAD test split.
XQUAD_LANGS_TEST = XQUAD_LANGS_TRAIN_DEV + ["en"]

PAWSX_LANGS = ["de", "en", "es", "fr", "ja", "ko", "zh"]


def _merge_langs_dataset_fn(split, shuffle_files, tfds_name, langs, seed):
  """Creates a single dataset containing all languages in a tfds dataset.

  Loads individual language datasets for the specified languages and
  concatenates them into a single dataset. This is done so that we can have a
  single test task which contains all the languages in order to compute a single
  overall performance metric across all languages.

  Args:
    split: string, which split to use.
    shuffle_files: boolean, whether to shuffle the input files.
    tfds_name: string, name of the TFDS dataset to load.
    langs: list of strings, the languages to load.

  Returns:
    A tf.data.Dataset.
  """
  ds = []
  for lang in langs:
    ds.append(tfds.load("{}/{}".format(tfds_name, lang),
                        split=split,
                        shuffle_files=shuffle_files,
                        read_config=tfds.ReadConfig(
                          shuffle_seed=seed)
                        ))
  all_langs_data = ds[0]
  for lang_data in ds[1:]:
    all_langs_data = all_langs_data.concatenate(lang_data)

  return all_langs_data


def xquad_all_langs_dataset_fn(split="test", shuffle_files=False, seed=None):
  """Creates a single dataset for XQuAD with all its constituent languages.

  split and shuffle_files are necessary arguments for dataset_fns.
  Args:
    split: string, which split to use.
    shuffle_files: boolean, whether to shuffle the input files.
  Returns:
    A tf.data.Dataset.
  """
  return _merge_langs_dataset_fn(split=split,
                                 shuffle_files=shuffle_files,
                                 tfds_name="xquad",
                                 langs=XQUAD_LANGS_TEST,
                                 seed=seed)


def pawsx_all_langs_dataset_fn(split="test", shuffle_files=False, seed=None):
  """Creates a single dataset for PAWS-X with all its constituent languages.

  split and shuffle_files are necessary arguments for dataset_fns.
  Args:
    split: string, which split to use.
    shuffle_files: boolean, whether to shuffle the input files.
  Returns:
    A tf.data.Dataset.
  """
  return _merge_langs_dataset_fn(split=split,
                                 shuffle_files=shuffle_files,
                                 tfds_name="paws_x_wiki",
                                 langs=PAWSX_LANGS,
                                 seed=seed)

