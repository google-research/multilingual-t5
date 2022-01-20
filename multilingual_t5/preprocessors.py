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

"""Preprocessors for Multilingual T5."""

import tensorflow.compat.v2 as tf


def _string_join(lst):
  # Join on space, but collapse consecutive spaces.
  out = tf.strings.join(lst, separator=' ')
  return tf.strings.regex_replace(out, r'\s+', ' ')


def _pad_punctuation(text):
  r"""Adds spaces around punctuation.

  Pads punctuation in the input text. This function replicates what
  preprocessors.squad() does with a few key differences. preprocessors.squad
  does the following substitution r'(\W)' -> r' \1 '. While this pads
  punctuation as expected, it has the unexpected effected of padding certain
  unicode characters with accents, with spaces as well. For instance:
  "François" becomes "Fran ç ois".

  Args:
    text: tf.string.

  Returns:
    tf.string
  """
  # Pad everything except for: underscores (_), whitespace (\s),
  # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
  text = tf.strings.regex_replace(text, r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ')
  # Collapse consecutive whitespace into one space.
  text = tf.strings.regex_replace(text, r'\s+', ' ')
  return text


def xnli_map_hypothesis_premise(dataset, target_language):
  """Generates XNLI dataset with the hypothesis restricted to a target language.

  The XNLI dataset (https://www.tensorflow.org/datasets/catalog/xnli) contains
  the hypothesis in a TranslationVariableLanguages feature. The hypothesis looks
  like:
    hypothesis: {
      'language': ['ar', 'bg', 'en', ...],
      'translation': ['t1', 't2', 't3', ...]
    }
  This function processes this hypothesis to return a dataset of the form:
    {
      'language': 'ar',
      'translation': 't1',
      'label': '1'
    }
  The label is also extracted along with the hypothesis.

  Args:
    dataset: tf.data.Dataset to process.
    target_language: string, the target language to restrict the hypothesis in
      the dataset to.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def _process(x):
    languages = x['hypothesis']['language']
    translations = x['hypothesis']['translation']

    # Create a tensor of the same length as languages so that everything can be
    # unbatched into examples for each language later.
    label = tf.fill(tf.shape(languages), x['label'])
    premise = tf.fill(tf.shape(languages), x['premise'][target_language])

    return {
        'language': languages,
        'translation': translations,
        'label': label,
        'premise': premise
    }

  dataset = dataset.map(
      _process, num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()
  dataset = dataset.filter(
      lambda x: tf.math.equal(x['language'], target_language))
  return dataset


def wikiann(dataset):
  """Convert WikiAnn NER dataset into a text2text format.

  Wikiann produces examples with IOB2 tagging form:
  {
    'tokens': ["rick", "and", "morty", "are", "cool", "."],
    'tags': ["B-PER", "O" , "B-PER", "O", "O", "O"],
    'langs': ["en", "en", "en", "en", "en", "en"],
    'spans': ["PER: rick", "PER: morty"]
  }

  This function will return examples of the format:
  {
    'inputs': 'tag: rick and morty are cool .',
    'targets': 'PER: rick $$ PER: morty',
    'tokens': ["rick", "and", "morty", "are", "cool", "."],
    'tags': ["B-PER", "O" , "B-PER", "O", "O", "O"],
    'langs': ["en", "en", "en", "en", "en", "en"]
  }
  
  Args:
    dataset: tf.data.Dataset to process.

  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """

  # "$$" is a delimiter used to separate the tags. It was used because it
  # doesn't occur in the dataset otherwise and hence doesn't interfere with
  # eval.
  def _process(x, delimiter=' $$ '):
    """Create WikiAnn text2text example."""
    inputs = 'tag: ' + tf.strings.reduce_join(x['tokens'], separator=' ')
    targets = tf.strings.reduce_join(x['spans'], separator=delimiter)

    return {
        'inputs': inputs,
        'targets': targets,
        'tokens': x['tokens'],
        'tags': x['tags'],
        'langs': x['langs'],
        'spans': x['spans']
    }

  return dataset.map(_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def xquad(dataset, include_context=True):
  """Convert SQuAD-like examples to a text2text pair.

  This function is very similar to the t5 preprocessors.squad function except
  that it doesn't pad punctuation with spaces since that causes issues in i18n
  languages.

  SQuAD produces examples with this form:
    {'id': <id>, context': <article>, 'question': <question>,
     'answers': { 'text': [<n answers>] }}
  This function will return examples of the format:
    {'inputs': 'question: <question> context: <article>',
     'targets': '<answer_0>',
     'id': <id>,
     'question': <question>,
     'context': <context>,
     'answers': [<n answers>]},

  Args:
    dataset: a tf.data.Dataset to process.
    include_context: a boolean
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def my_fn(x):
    """Create squad example."""

    a = _pad_punctuation(x['answers']['text'])
    q = _pad_punctuation(x['question'])
    c = _pad_punctuation(x['context'])
    if include_context:
      inputs = _string_join(['question:', q, 'context:', c])
    else:
      inputs = _string_join(['squad trivia question:', q])
    return {
        'inputs': inputs,
        'targets': a[0],
        'id': x['id'],
        'context': c,
        'question': q,
        'answers': a
    }

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def process_mnli(dataset):
  """Convert MNLI dataset into a text2text format.

  This function will return examples of the form:
  {
     'inputs': 'xnli: premise: <premise> hypothesis: <hypothesis>',
     'targets': '<target>'
  }

  Args:
    dataset: tf.data.Dataset to process.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def _process(x):
    return {
        'inputs': tf.strings.join(['xnli: premise: ', x['premise'],
                                   ' hypothesis: ', x['hypothesis']]),
        'targets': tf.strings.as_string(x['label'])
    }

  return dataset.map(_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def process_xnli(dataset, target_languages):
  """Processes the XNLI dataset into examples by language in a text2text format.

  The XNLI dataset contains examples of the form:
  {
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
    }

  This function processes the XNLI dataset and returns examples of the form:
  {
    'inputs': 'xnli: premise: <premise> hypothesis: <hypothesis>',
    'targets': <target>
  }
  for each language in the list of input target languages.
  Args:
    dataset: tf.data.Dataset to process.
    target_languages: list of strings, the target languages.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def _process(x):
    return {
        'inputs': tf.strings.join(['xnli: premise: ', x['premise'],
                                   ' hypothesis: ', x['translation']]),
        'targets': tf.strings.as_string(x['label'])
    }

  output = []
  for language in target_languages:
    examples = xnli_map_hypothesis_premise(dataset, target_language=language)
    d = examples.map(_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    output.append(d)

  output_dataset = output[0]
  for lang_dataset in output[1:]:
    output_dataset = output_dataset.concatenate(lang_dataset)
  return output_dataset


def filter_tydiqa_by_language(dataset, lang):
  """Filters TyDi QA train dataset to keep examples of a single language."""

  def function_matches_lang(x):
    # TyDi QA ids start with the language.
    # Example : finnish--7633091408814529542-0
    return tf.strings.split(x['id'], sep='-')[0] == lang

  return dataset.filter(function_matches_lang)
