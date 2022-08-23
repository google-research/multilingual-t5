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

"""Add Tasks to registry."""
import functools

from multilingual_t5 import preprocessors
from multilingual_t5 import utils
from multilingual_t5 import vocab
from multilingual_t5.evaluation import metrics as mt5_metrics

import seqio
import t5.data
import t5.data.tasks
from t5.evaluation import metrics
import tensorflow_datasets as tfds

# TODO: remove this after users have switched to
# using references from vocab.py
DEFAULT_VOCAB = vocab.DEFAULT_VOCAB


DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)

DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        t5.data.Feature(
            vocabulary=vocab.DEFAULT_VOCAB, add_eos=True, required=False),
    "targets":
        t5.data.Feature(vocabulary=vocab.DEFAULT_VOCAB, add_eos=True)
}

MC4_LANGS = tfds.text.c4.MC4_LANGUAGES

# Multilingual BERT was trained on 104 languages. We include 103 of these
# languages, as tfds.wikipedia doesn't distinguish between simplified and
# traditional Chinese, and only contains "zh" (which is a mix of simplified
# and traditional).
# https://github.com/google-research/bert/blob/master/multilingual.md
WIKI_LANGS = [
    "af", "an", "ar", "ast", "az", "azb", "ba", "bar", "be", "bg", "bn", "bpy",
    "br", "bs", "ca", "ce", "ceb", "cs", "cv", "cy", "da", "de", "el", "en",
    "es", "et", "eu", "fa", "fi", "fr", "fy", "ga", "gl", "gu", "he", "hi",
    "hr", "ht", "hu", "hy", "id", "io", "is", "it", "ja", "jv", "ka", "kk",
    "kn", "ko", "ky", "la", "lb", "lmo", "lt", "lv", "mg", "min", "mk", "ml",
    "mn", "mr", "ms", "my", "nds-nl", "ne", "new", "nl", "nn", "no", "oc", "pa",
    "pl", "pms", "pnb", "pt", "ro", "ru", "scn", "sco", "sh", "sk", "sl", "sq",
    "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tl", "tr", "tt", "uk",
    "ur", "uz", "vi", "vo", "war", "yo", "zh"
]

# =========================== Pretraining Tasks/Mixtures =======================
# mC4
for lang in MC4_LANGS:
  seqio.TaskRegistry.add(
      "mc4.{}".format(lang.replace("-", "_")),
      source=seqio.TfdsDataSource(
          tfds_name="c4/multilingual:3.0.1",
          splits={
              "train": lang,
              "validation": f"{lang}-validation"
          }),
      preprocessors=[
          functools.partial(
              t5.data.preprocessors.rekey,
              key_map={
                  "inputs": None,
                  "targets": "text"
              }),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          t5.data.preprocessors.span_corruption,
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])

mc4 = ["mc4.{}".format(lang.replace("-", "_")) for lang in MC4_LANGS]
seqio.MixtureRegistry.add("mc4", mc4, default_rate=DEFAULT_MIX_RATE)

# Wikipedia
for lang in WIKI_LANGS:
  seqio.TaskRegistry.add(
      "mt5_wiki.{}".format(lang.replace("-", "_")),
      source=seqio.TfdsDataSource(
          tfds_name="wikipedia/20200301.{}:1.0.0".format(lang)),
      preprocessors=[
          functools.partial(
              t5.data.preprocessors.rekey,
              key_map={
                  "inputs": None,
                  "targets": "text"
              }),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          t5.data.preprocessors.span_corruption,
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])

wiki = ["mt5_wiki.{}".format(lang.replace("-", "_")) for lang in WIKI_LANGS]
seqio.MixtureRegistry.add("mt5_wiki", wiki, default_rate=DEFAULT_MIX_RATE)

# Mixture of mC4 and WIKI
seqio.MixtureRegistry.add("mc4_wiki", mc4 + wiki, default_rate=DEFAULT_MIX_RATE)

# =========================== Fine-tuning Tasks/Mixtures =======================
# ----- XNLI -----
# XNLI zero-shot task. This fine-tunes on English MNLI training data and then
# evaluates on multilingual XNLI dev/test data.

XNLI_LANGS = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr",
    "ur", "vi", "zh"
]


def create_xnli_tasks_and_mixtures(task_prefix, task_suffix, output_features):
  """Helper function to create XNLI tasks and mixtures."""
  if task_suffix:
    task_suffix = "_" + task_suffix
  seqio.TaskRegistry.add(
      f"{task_prefix}xnli_train{task_suffix}",
      source=seqio.TfdsDataSource(
          tfds_name="multi_nli:1.1.0", splits=["train"]),
      preprocessors=[
          preprocessors.process_mnli,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=output_features,
      metric_fns=[metrics.accuracy])
  for xnli_lang in XNLI_LANGS:
    seqio.TaskRegistry.add(
        f"{task_prefix}xnli_dev_test{task_suffix}.{xnli_lang}",
        source=seqio.TfdsDataSource(
            tfds_name="xnli:1.1.0", splits=["validation", "test"]),
        preprocessors=[
            functools.partial(
                preprocessors.process_xnli, target_languages=[xnli_lang]),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=output_features,
        metric_fns=[metrics.accuracy])
    if xnli_lang == "en":
      continue
    seqio.TaskRegistry.add(
        f"{task_prefix}xnli_translate_train{task_suffix}.{xnli_lang}",
        source=seqio.TfdsDataSource(
            tfds_name="xtreme_xnli:1.1.0", splits=["train"]),
        preprocessors=[
            functools.partial(
                preprocessors.process_xnli, target_languages=[xnli_lang]),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=output_features,
        metric_fns=[metrics.accuracy])
  seqio.TaskRegistry.add(
      f"{task_prefix}xnli_dev_test{task_suffix}.all_langs",
      source=seqio.TfdsDataSource(
          tfds_name="xnli:1.1.0", splits=["validation", "test"]),
      preprocessors=[
          functools.partial(
              preprocessors.process_xnli, target_languages=XNLI_LANGS),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=output_features,
      metric_fns=[metrics.accuracy])
  xnli_zeroshot = ([
      f"{task_prefix}xnli_train{task_suffix}",
      f"{task_prefix}xnli_dev_test{task_suffix}.all_langs"
  ] + [
      f"{task_prefix}xnli_dev_test{task_suffix}.{lang}" for lang in XNLI_LANGS
  ])
  seqio.MixtureRegistry.add(
      f"{task_prefix}xnli_zeroshot{task_suffix}",
      xnli_zeroshot,
      default_rate=1.0)
  xnli_translate_train = xnli_zeroshot + [
      f"{task_prefix}xnli_translate_train{task_suffix}.{lang}"
      for lang in XNLI_LANGS
      if lang != "en"
  ]
  seqio.MixtureRegistry.add(
      f"{task_prefix}xnli_translate_train{task_suffix}",
      xnli_translate_train,
      default_rate=1.0)


create_xnli_tasks_and_mixtures(
    task_prefix="mt5_", task_suffix="", output_features=DEFAULT_OUTPUT_FEATURES)

# ----- PAWS -----
label_names = ["different_meaning", "paraphrase"]
text_preprocessor = functools.partial(
    t5.data.preprocessors.glue,
    benchmark_name="paws",
    label_names=label_names,
    feature_names=["sentence1", "sentence2"],
    id_key=None)

postprocess_fn = functools.partial(
    t5.data.postprocessors.string_label_to_class_id, label_classes=label_names)

seqio.TaskRegistry.add(
    "mt5_paws",
    source=seqio.TfdsDataSource(
        tfds_name="paws_x_wiki/en:1.0.0", splits=["train"]),
    preprocessors=[
        text_preprocessor,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=postprocess_fn,
    metric_fns=[metrics.accuracy])

for lang in utils.PAWSX_LANGS:
  seqio.TaskRegistry.add(
      "mt5_pawsx_dev_test.{}".format(lang),
      source=seqio.TfdsDataSource(
          tfds_name="paws_x_wiki/{}:1.0.0".format(lang),
          splits=["validation", "test"]),
      preprocessors=[
          text_preprocessor,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=postprocess_fn,
      metric_fns=[metrics.accuracy])

  # This uses machine translations provided by the PAWS-X paper.
  seqio.TaskRegistry.add(
      "mt5_pawsx_translate_train_original.{}".format(lang),
      source=seqio.TfdsDataSource(
          tfds_name="paws_x_wiki/{}:1.0.0".format(lang), splits=["train"]),
      preprocessors=[
          text_preprocessor,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=postprocess_fn,
      metric_fns=[metrics.accuracy])

  if lang != "en":
    # This uses machine translations provided by the XTREME paper.
    seqio.TaskRegistry.add(
        "mt5_pawsx_translate_train.{}".format(lang),
        source=seqio.TfdsDataSource(
            tfds_name="xtreme_pawsx/{}:1.0.0".format(lang), splits=["train"]),
        preprocessors=[
            text_preprocessor,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        postprocess_fn=postprocess_fn,
        metric_fns=[metrics.accuracy])

seqio.TaskRegistry.add(
    "mt5_pawsx_dev_test.all_langs",
    source=seqio.FunctionDataSource(
        dataset_fn=utils.pawsx_all_langs_dataset_fn,
        splits=["validation", "test"]),
    preprocessors=[
        text_preprocessor,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=postprocess_fn,
    metric_fns=[metrics.accuracy])

# PAWSX Zero-Shot
pawsx_eval = [
    "mt5_pawsx_dev_test.{}".format(lang) for lang in utils.PAWSX_LANGS
] + ["mt5_pawsx_dev_test.all_langs"]
pawsx = ["mt5_paws"] + pawsx_eval
seqio.MixtureRegistry.add("mt5_pawsx_zeroshot", pawsx, default_rate=1.0)

pawsx_translate_train = ["mt5_paws"] + [
    "mt5_pawsx_translate_train.{}".format(lang)
    for lang in utils.PAWSX_LANGS
    if lang != "en"
] + pawsx_eval
seqio.MixtureRegistry.add(
    "mt5_pawsx_translate_train", pawsx_translate_train, default_rate=1.0)

pawsx_translate_train_original = [
    "mt5_pawsx_translate_train_original.{}".format(lang)
    for lang in utils.PAWSX_LANGS
] + pawsx_eval
seqio.MixtureRegistry.add(
    "mt5_pawsx_translate_train_original",
    pawsx_translate_train,
    default_rate=1.0)

# ----- TyDiQA GoldP-----
# The "validation" split contains all the validation examples for all the
# individual languages together.
TYDIQA_LANGS = ["ar", "bn", "en", "fi", "id", "ko", "ru", "sw", "te"]

seqio.TaskRegistry.add(
    "mt5_tydiqa_train_dev",
    source=seqio.TfdsDataSource(
        tfds_name="tydi_qa/goldp:2.1.0", splits=["train", "validation"]),
    preprocessors=[
        preprocessors.xquad,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad])

for lang in TYDIQA_LANGS:
  seqio.TaskRegistry.add(
      "mt5_tydiqa_dev.{}".format(lang),
      source=seqio.TfdsDataSource(
          tfds_name="tydi_qa/goldp:2.1.0",
          splits={"validation": "validation-{}".format(lang)}),
      preprocessors=[
          preprocessors.xquad,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[metrics.squad])

tydiqa = (["mt5_tydiqa_train_dev"] +
          ["mt5_tydiqa_dev.{}".format(lang) for lang in TYDIQA_LANGS])
seqio.MixtureRegistry.add("mt5_tydiqa", tydiqa, default_rate=1.0)

# ----- TyDiQA GoldP Zero-Shot-----
# This Zero-Shot setting matches the XTREME setup, where training is done on
# the English data of TyDiQA. In the TyDiQA paper, fine-tuning was done on
# SQuAD for zero-shot evaluation.
TYDIQA_LANGS = ["ar", "bn", "en", "fi", "id", "ko", "ru", "sw", "te"]
seqio.TaskRegistry.add(
    "mt5_tydiqa_train.en",
    source=seqio.TfdsDataSource(
        tfds_name="tydi_qa/goldp:2.1.0", splits=["train"]),
    preprocessors=[
        preprocessors.xquad,
        functools.partial(
            preprocessors.filter_tydiqa_by_language, lang="english"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad])

tydiqa_zeroshot = (["mt5_tydiqa_train.en"] +
                   ["mt5_tydiqa_dev.{}".format(lang) for lang in TYDIQA_LANGS])
seqio.MixtureRegistry.add(
    "mt5_tydiqa_zeroshot", tydiqa_zeroshot, default_rate=1.0)

# Defining translate-train tasks.
for lang in TYDIQA_LANGS:
  # Skipping English, since translate-train is not available.
  if lang == "en":
    continue
  seqio.TaskRegistry.add(
      "mt5_tydiqa_translate_train.{}".format(lang),
      source=seqio.TfdsDataSource(
          tfds_name="tydi_qa/goldp:2.1.0",
          splits={"train": "translate-train-{}".format(lang)}),
      preprocessors=[
          preprocessors.xquad,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[metrics.squad])

tydiqa_translate_train = (
    ["mt5_tydiqa_train.en"]
    + [f"mt5_tydiqa_translate_train.{lang}"
       for lang in TYDIQA_LANGS if lang != "en"]
    + [f"mt5_tydiqa_dev.{lang}" for lang in TYDIQA_LANGS])
seqio.MixtureRegistry.add(
    "mt5_tydiqa_translate_train", tydiqa_translate_train, default_rate=1.0)


# ----- XQuAD -----
def create_xquad_tasks_and_mixtures(task_prefix, task_suffix, output_features):
  """Helper function for XQuad tasks and mixtures."""
  # ----- English SQUAD -----
  seqio.TaskRegistry.add(
      f"{task_prefix}squad_train_dev{task_suffix}",
      source=seqio.TfdsDataSource(
          tfds_name="squad/v1.1:3.0.0", splits=["train", "validation"]),
      preprocessors=[
          preprocessors.xquad,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=output_features,
      metric_fns=[metrics.squad])
  for xquad_lang in utils.XQUAD_LANGS_TRAIN_DEV:
    seqio.TaskRegistry.add(
        f"{task_prefix}xquad_translate_train_dev{task_suffix}.{xquad_lang}",
        source=seqio.TfdsDataSource(
            tfds_name="xquad/{}:3.0.0".format(xquad_lang),
            splits={
                "train": "translate-train",
                "validation": "translate-dev"
            }),
        preprocessors=[
            preprocessors.xquad,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=t5.data.postprocessors.qa,
        output_features=output_features,
        metric_fns=[metrics.squad])

  for xquad_lang in utils.XQUAD_LANGS_TEST:
    seqio.TaskRegistry.add(
        f"{task_prefix}xquad_test{task_suffix}.{xquad_lang}",
        source=seqio.TfdsDataSource(
            tfds_name=f"xquad/{xquad_lang}:3.0.0", splits=["test"]),
        preprocessors=[
            preprocessors.xquad,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=t5.data.postprocessors.qa,
        output_features=output_features,
        metric_fns=[metrics.squad])

  # Additional test task containing all the languages.
  seqio.TaskRegistry.add(
      f"{task_prefix}xquad_test{task_suffix}.all_langs",
      source=seqio.FunctionDataSource(
          dataset_fn=utils.xquad_all_langs_dataset_fn, splits=["test"]),
      preprocessors=[
          preprocessors.xquad,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=output_features,
      metric_fns=[metrics.squad])

  # XQuAD Zero-Shot (SQuAD train, SQuAD dev, XQuAD test).
  xquad_test = ([
      f"{task_prefix}xquad_test{task_suffix}.{xquad_lang}"
      for xquad_lang in utils.XQUAD_LANGS_TEST
  ])
  xquad_zeroshot = ([
      f"{task_prefix}squad_train_dev{task_suffix}",
      f"{task_prefix}xquad_test{task_suffix}.all_langs"
  ] + xquad_test)
  seqio.MixtureRegistry.add(
      f"{task_prefix}xquad_zeroshot{task_suffix}",
      xquad_zeroshot,
      default_rate=1.0)

  # XQuAD Translate-Train (English SQuAD, XQuAD translate-train,
  # XQuAD translate-dev, XQuAD test)
  # Note that the QA translate-train baselines from Hu et al (XTREME)
  # do not include the English data. However, Fang et al (FILTER) do include
  # English data.
  xquad_translate_train = [
      f"{task_prefix}xquad_translate_train_dev{task_suffix}.{xquad_lang}"
      for xquad_lang in utils.XQUAD_LANGS_TRAIN_DEV
  ] + [f"{task_prefix}squad_train_dev{task_suffix}"
      ] + [f"{task_prefix}xquad_test{task_suffix}.all_langs"] + xquad_test
  seqio.MixtureRegistry.add(
      f"{task_prefix}xquad_translate_train{task_suffix}",
      xquad_translate_train,
      default_rate=1.0)


create_xquad_tasks_and_mixtures(
    task_prefix="mt5_", task_suffix="", output_features=DEFAULT_OUTPUT_FEATURES)

# ----- MLQA -----

MLQA_LANGS = ["ar", "de", "en", "es", "hi", "vi", "zh"]

for lang in MLQA_LANGS:
  seqio.TaskRegistry.add(
      "mt5_mlqa_dev_test.{}".format(lang),
      source=seqio.TfdsDataSource(
          tfds_name="mlqa/{}:1.0.0".format(lang), splits=["validation",
                                                          "test"]),
      preprocessors=[
          preprocessors.xquad,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[functools.partial(mt5_metrics.mlqa, lang=lang)])

# MLQA Zero-Shot
mlqa_dev_test = [f"mt5_mlqa_dev_test.{lang}" for lang in MLQA_LANGS]
mlqa_zeroshot = ["mt5_squad_train_dev"] + mlqa_dev_test
seqio.MixtureRegistry.add("mt5_mlqa_zeroshot", mlqa_zeroshot, default_rate=1.0)

# MLQA Translate-Train
mlqa_translate_train = [
    "mt5_xquad_translate_train_dev.{}".format(lang)
    for lang in MLQA_LANGS
    if lang != "en"
] + ["mt5_squad_train_dev"] + mlqa_dev_test

seqio.MixtureRegistry.add(
    "mt5_mlqa_translate_train", mlqa_translate_train, default_rate=1.0)

# ----- WikiAnn NER -----

NER_LANGS = [
    "af", "ar", "bg", "bn", "de", "el", "en", "es", "et", "eu", "fa", "fi",
    "fr", "he", "hi", "hu", "id", "it", "ja", "jv", "ka", "kk", "ko", "ml",
    "mr", "ms", "my", "nl", "pt", "ru", "sw", "ta", "te", "th", "tl", "tr",
    "ur", "vi", "yo", "zh"
]

for lang in NER_LANGS:
  seqio.TaskRegistry.add(
      "mt5_ner_train.{}".format(lang),
      source=seqio.TfdsDataSource(
          tfds_name="wikiann/{}:1.0.0".format(lang), splits=["train"]),
      preprocessors=[
          preprocessors.wikiann,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[mt5_metrics.span_f1])

  seqio.TaskRegistry.add(
      "mt5_ner_eval.{}".format(lang),
      source=seqio.TfdsDataSource(
          tfds_name="wikiann/{}:1.0.0".format(lang),
          splits=["validation", "test"]),
      preprocessors=[
          preprocessors.wikiann,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[mt5_metrics.span_f1])

# NER zero-shot
seqio.MixtureRegistry.add(
    "mt5_ner_zeroshot", ["mt5_ner_train.{}".format("en")] +
    ["mt5_ner_eval.{}".format(lang) for lang in NER_LANGS],
    default_rate=1.0)

# NER multilingual
seqio.MixtureRegistry.add(
    "mt5_ner_multilingual",
    ["mt5_ner_train.{}".format(lang) for lang in NER_LANGS] +
    ["mt5_ner_eval.{}".format(lang) for lang in NER_LANGS],
    default_rate=1.0)

# ----- GLUE -----
# The glue tasks are already present in the task registry from importing the
# t5.data.tasks file. However, that task and mixture use the t5 sentence piece
# vocabulary. This task and mixture use the mt5 vocabulary.
for b in tfds.text.glue.Glue.builder_configs.values():
  seqio.TaskRegistry.add(
      "mt5_glue_%s_v002" % b.name,
      source=seqio.TfdsDataSource(
          tfds_name="glue/%s:1.0.0" % b.name,
          splits=["test"] if b.name == "ax" else None),
      preprocessors=[
          t5.data.glue_utils.get_glue_text_preprocessor(b),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=t5.data.glue_utils.get_glue_metric(b.name),
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=t5.data.glue_utils.get_glue_postprocess_fn(b))

seqio.MixtureRegistry.add(
    "mt5_glue_v002_proportional",
    list({
        f"mt5_{k}": v
        for k, v in t5.data.glue_utils.get_glue_weight_mapping().items()
    }.items()))

# ============== Summarization ==============
# ----- CNN/DailyMail -----
seqio.TaskRegistry.add(
    "mt5_cnn_dailymail_v002",
    source=seqio.TfdsDataSource(tfds_name="cnn_dailymail:3.1.0"),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.summarize,
            article_key="article",
            summary_key="highlights"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

# ----- GEM-XSum -----
_rouge_fn = functools.partial(
    metrics.rouge, score_keys=["rouge1", "rouge2", "rougeL", "rougeLsum"])

seqio.TaskRegistry.add(
    "mt5_gem_xsum",
    source=seqio.TfdsDataSource(tfds_name="gem/xsum:1.0.1"),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.summarize,
            article_key="document",
            summary_key="target"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.bleu, _rouge_fn],
    output_features=DEFAULT_OUTPUT_FEATURES,
)

# ----- SuperGLUE -----
# The superglue tasks are already present in the task registry from importing
# the t5.data.tasks file. However, that task and mixture use the t5 sentence
# piece vocabulary. This task and mixture use the mt5 vocabulary.
for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
  # We use a simplified version of WSC, defined below
  if "wsc" in b.name:
    continue
  if b.name == "axb":
    text_preprocessor = [
        functools.partial(
            t5.data.preprocessors.rekey,
            key_map={
                "premise": "sentence1",
                "hypothesis": "sentence2",
                "label": "label",
                "idx": "idx",
            }),
        t5.data.glue_utils.get_glue_text_preprocessor(b),
    ]
  else:
    text_preprocessor = [t5.data.glue_utils.get_glue_text_preprocessor(b)]

  seqio.TaskRegistry.add(
      "mt5_super_glue_%s_v102" % b.name,
      source=seqio.TfdsDataSource(
          tfds_name="super_glue/%s:1.0.2" % b.name,
          splits=["test"] if b.name in ["axb", "axg"] else None),
      preprocessors=text_preprocessor + [
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=t5.data.glue_utils.get_super_glue_metric(b.name),
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=t5.data.glue_utils.get_glue_postprocess_fn(b))

# ----- DPR -----
seqio.TaskRegistry.add(
    "mt5_dpr_v001_simple",
    source=seqio.TfdsDataSource(tfds_name="definite_pronoun_resolution:1.1.0"),
    preprocessors=[
        t5.data.preprocessors.definite_pronoun_resolution_simple,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

# ------ WSC ----------
seqio.TaskRegistry.add(
    "mt5_super_glue_wsc_v102_simple_train",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/wsc.fixed:1.0.2", splits=["train"]),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.wsc_simple, correct_referent_only=True),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[],
    output_features=DEFAULT_OUTPUT_FEATURES)

seqio.TaskRegistry.add(
    "mt5_super_glue_wsc_v102_simple_eval",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/wsc.fixed:1.0.2", splits=["validation", "test"]),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.wsc_simple, correct_referent_only=False),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5.data.postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

_mt5_super_glue_tasks = {}
for task, value in t5.data.get_super_glue_weight_mapping().items():
  _mt5_super_glue_tasks["mt5_" + task] = value

seqio.MixtureRegistry.add("mt5_super_glue_v102_proportional",
                          list(_mt5_super_glue_tasks.items()))
