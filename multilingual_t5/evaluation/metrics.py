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

"""Implementation of various metrics to be used with the T5 library.
"""


from __future__ import print_function

import collections
import re
import string
import sys
import unicodedata

from t5.evaluation import qa_utils


def normalize_mlqa(s, lang, punct):
  """Lower text and remove punctuation, articles and extra whitespace.

  Based on third_party/py/xtreme/third_party/evaluate_mlqa.py
  Args:
    s: string, typically the answer span predicted by a QA model.
    lang: ISO code of language.
    punct: set of punctuation characters.

  Returns:
    string, after applying normalization rules.
  """

  whitespace_langs = ['en', 'es', 'hi', 'vi', 'de', 'ar']
  mixed_segmentation_langs = ['zh']

  def whitespace_tokenize(text):
    return text.split()

  def mixed_segmentation(text):
    segs_out = []
    temp_str = ''
    for char in text:
      if re.search(r'[\u4e00-\u9fa5]', char) or char in punct:
        if temp_str != '':
          ss = whitespace_tokenize(temp_str)
          segs_out.extend(ss)
          temp_str = ''
        segs_out.append(char)
      else:
        temp_str += char
    if temp_str != '':
      ss = whitespace_tokenize(temp_str)
      segs_out.extend(ss)
    return segs_out

  def drop_articles(text, lang):
    if lang == 'en':
      return re.sub(r'\b(a|an|the)\b', ' ', text)
    elif lang == 'es':
      return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
    elif lang == 'hi':
      return text
    elif lang == 'vi':
      return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
    elif lang == 'de':
      return re.sub(
          r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b',
          ' ', text)
    elif lang == 'ar':
      return re.sub('\sال^|ال', ' ', text)
    elif lang == 'zh':
      return text

  def white_space_fix(text, lang):
    if lang in whitespace_langs:
      tokens = whitespace_tokenize(text)
    elif lang in mixed_segmentation_langs:
      tokens = mixed_segmentation(text)
    return ' '.join([t for t in tokens if t.strip()])

  def drop_punc(text):
    return ''.join(c for c in text if c not in punct)

  s = s.lower()
  s = drop_punc(s)
  s = drop_articles(s, lang)
  s = white_space_fix(s, lang)
  return s


def mlqa(targets, predictions, lang=None):
  """Computes MLQA metrics, maximizing over answers per question.

  Args:
    targets: list of lists of strings
    predictions: list of strings
    lang: ISO code of language

  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  assert lang is not None
  punct = {
      chr(i)
      for i in range(sys.maxunicode)
      if unicodedata.category(chr(i)).startswith('P')
  }.union(string.punctuation)
  targets = [[normalize_mlqa(t, lang, punct) for t in u] for u in targets]
  predictions = [normalize_mlqa(p, lang, punct) for p in predictions]
  return qa_utils.qa_metrics(targets, predictions)


def span_f1(targets, predictions):
  """Computes Span based F1 score.

  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings

  Returns:
    span f1 across all targets and predictions (Based on CoNLL script)
  """
  true_positives = collections.defaultdict(int)
  false_positives = collections.defaultdict(int)
  false_negatives = collections.defaultdict(int)

  def tags_to_spans(tag_sequence, delimiter=' $$ '):
    """Extract spans from IOB1 or BIO tags."""
    tag_sequence_split = [x.strip() for x in tag_sequence.split(delimiter)]
    tags_entities = []
    for tag_entity in tag_sequence_split:
      tag_entity_split = tag_entity.split(':')
      if len(tag_entity_split) != 2:
        continue
      tag = tag_entity_split[0].strip()
      entity = tag_entity_split[1].strip()
      tags_entities.append((tag, entity))
    return tags_entities

  def compute_f1_metrics(true_positives, false_positives, false_negatives):
    precision = float(true_positives) / float(true_positives + false_positives +
                                              1e-13)
    recall = float(true_positives) / float(true_positives + false_negatives +
                                           1e-13)
    f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
    return precision, recall, f1_measure

  for target, pred in zip(targets, predictions):
    gold_spans = tags_to_spans(target)
    predicted_spans = tags_to_spans(pred)

    for span in predicted_spans:
      if span in gold_spans:
        true_positives[span[0]] += 1
        gold_spans.remove(span)
      else:
        false_positives[span[0]] += 1
    # These spans weren't predicted.
    for span in gold_spans:
      false_negatives[span[0]] += 1

  _, _, f1_measure = compute_f1_metrics(
      sum(true_positives.values()), sum(false_positives.values()),
      sum(false_negatives.values()))

  return {'span_f1': f1_measure}
