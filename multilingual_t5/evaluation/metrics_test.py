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

"""Tests for t5.evaluation.metrics."""

from absl.testing import absltest

from multilingual_t5.evaluation import metrics
from t5.evaluation import test_utils


class MetricsTest(test_utils.BaseMetricsTest):

  def test_same_mlqa(self):
    ref = "this is a string"
    self.assertDictClose(
        metrics.mlqa([["", ref], [ref, ref]], [ref, ref], lang="en"), {
            "em": 100,
            "f1": 100,
        })

  def test_different_mlqa(self):
    ref = "this is a string"
    self.assertDictClose(
        metrics.mlqa([[ref, ref], [ref, ref]], ["", ""], lang="en"), {
            "em": 0,
            "f1": 0
        })

  def test_article_drop_mlqa(self):
    ref = "this unas a string"
    pred = "this a string"
    self.assertDictClose(
        metrics.mlqa([[ref]], [pred], lang="es"), {
            "em": 100,
            "f1": 100,
        })

  def test_mlqa_small(self):
    self.assertDictClose(
        metrics.mlqa([["abc abd", "$$$$"]], ["abd"], lang="en"),
        {"f1": 100 * 2.0 / 3.0, "em": 0.},
    )


if __name__ == "__main__":
  absltest.main()
