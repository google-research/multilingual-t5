# Copyright 2021 The mT5 Authors.
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

import re
import string
import sys
import unicodedata

from t5.evaluation import qa_utils


def normalize_mlqa(s, lang):
  """Lower text and remove punctuation, articles and extra whitespace.

  Based on third_party/py/xtreme/third_party/evaluate_mlqa.py
  Args:
    s: string, typically the answer span predicted by a QA model.
    lang: ISO code of language.

  Returns:
    string, after applying normalization rules.
  """

  punct = {
      chr(i)
      for i in range(sys.maxunicode)
      if unicodedata.category(chr(i)).startswith('P')
  }.union(string.punctuation)
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


def mlqa(targets, predictions, lang):
  """Computes MLQA metrics, maximizing over answers per question.

  Args:
    targets: list of lists of strings
    predictions: list of strings
    lang: ISO code of language

  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  targets = [[normalize_mlqa(t, lang) for t in u] for u in targets]
  predictions = [normalize_mlqa(p, lang) for p in predictions]
  return qa_utils.qa_metrics(targets, predictions)
