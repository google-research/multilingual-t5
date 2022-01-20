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

"""Tests for multilingual_t5.data.preprocessors."""

from absl.testing import absltest

from multilingual_t5 import preprocessors
import t5.data
import tensorflow.compat.v2 as tf


class PreprocessorsTest(absltest.TestCase):

  def test_process_xquad(self):
    dataset = tf.data.Dataset.from_tensors({
        'id': '123',
        'context': 'Some context.',
        'question': 'Whose portrait by François Clouet was included' +
                    ' in the Jones bequest of 1882?',
        'answers': {
            'text': ['The answer.', 'Another answer.'],
        }
    })

    dataset = preprocessors.xquad(dataset)
    t5.data.assert_dataset(
        dataset, {
            'id': '123',
            'inputs': 'question: Whose portrait by François Clouet was' +
                      ' included in the Jones bequest of 1882 ? context: Some'
                      ' context . ',
            'targets': 'The answer . ',
            'context': 'Some context . ',
            'question': 'Whose portrait by François Clouet was included' +
                        ' in the Jones bequest of 1882 ? ',
            'answers': ['The answer . ', 'Another answer . '],
        })

  def test_process_wikiann(self):
    dataset = tf.data.Dataset.from_tensors({
        'tokens': ['rick', 'and', 'morty', 'are', 'cool', '.'],
        'tags': ['B-PER', 'O', 'B-PER', 'O', 'O', 'O'],
        'langs': ['en', 'en', 'en', 'en', 'en', 'en'],
        'spans': ['PER: rick', 'PER: morty']
    })

    dataset = preprocessors.wikiann(dataset)
    t5.data.assert_dataset(
        dataset, {
            'inputs': 'tag: rick and morty are cool .',
            'targets': 'PER: rick $$ PER: morty',
            'tokens': ['rick', 'and', 'morty', 'are', 'cool', '.'],
            'tags': ['B-PER', 'O', 'B-PER', 'O', 'O', 'O'],
            'langs': ['en', 'en', 'en', 'en', 'en', 'en'],
            'spans': ['PER: rick', 'PER: morty']
        })

  def test_process_mnli(self):
    dataset = tf.data.Dataset.from_tensors({
        'hypothesis': 'hypothesis1',
        'label': 1,
        'premise': 'premise1'
        })

    dataset = preprocessors.process_mnli(dataset)
    t5.data.assert_dataset(
        dataset, {
            'inputs': 'xnli: premise: premise1 hypothesis: hypothesis1',
            'targets': '1'
        })

  def test_process_xnli_single_lang(self):
    dataset = tf.data.Dataset.from_tensors({
        'hypothesis': {
            'language': ['lang1', 'lang2', 'lang3'],
            'translation': ['translation1', 'translation2', 'translation3'],
        },
        'label': 1,
        'premise': {
            'lang1': 'premise1',
            'lang2': 'premise2',
            'lang3': 'premise3'
        }
    })

    dataset = preprocessors.process_xnli(dataset, target_languages=['lang1'])
    t5.data.assert_dataset(
        dataset, {
            'inputs': 'xnli: premise: premise1 hypothesis: translation1',
            'targets': '1'
        })

  def test_process_xnl_multiple_langs(self):
    dataset = tf.data.Dataset.from_tensors({
        'hypothesis': {
            'language': ['lang1', 'lang2', 'lang3'],
            'translation': ['translation1', 'translation2', 'translation3'],
        },
        'label': 1,
        'premise': {
            'lang1': 'premise1',
            'lang2': 'premise2',
            'lang3': 'premise3'
        }
    })

    dataset = preprocessors.process_xnli(dataset,
                                         target_languages=['lang1',
                                                           'lang2',
                                                           'lang3'])
    t5.data.assert_dataset(
        dataset, [
            {'inputs': 'xnli: premise: premise1 hypothesis: translation1',
             'targets': '1'
            },
            {'inputs': 'xnli: premise: premise2 hypothesis: translation2',
             'targets': '1'
            },
            {'inputs': 'xnli: premise: premise3 hypothesis: translation3',
             'targets': '1'
            }
        ])


if __name__ == '__main__':
  absltest.main()
