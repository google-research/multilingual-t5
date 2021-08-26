# mT5: Multilingual T5

Multilingual T5 (mT5) is a massively multilingual pretrained text-to-text
transformer model, trained following a similar recipe as
[T5](https://github.com/google-research/text-to-text-transfer-transformer).
This repo can be used to reproduce the experiments in the [mT5 paper][paper].

## Table of Contents

* [Languages covered](#languages-covered)
* [Results](#results)
* [Usage](#usage)
  * [Training](#training)
  * [Fine-Tuning](#fine-tuning)
* [Released Model Checkpoints](#released-model-checkpoints)
* [How to Cite](#how-to-cite)

## Languages covered

mT5 is pretrained on the [mC4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual_nights_stay) corpus, covering 101 languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Azerbaijani, Basque,
Belarusian, Bengali, Bulgarian, Burmese, Catalan, Cebuano, Chichewa, Chinese,
Corsican, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino,
Finnish, French, Galician, Georgian, German, Greek, Gujarati, Haitian Creole,
Hausa, Hawaiian, Hebrew, Hindi, Hmong, Hungarian, Icelandic, Igbo, Indonesian,
Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish,
Kyrgyz, Lao, Latin, Latvian, Lithuanian, Luxembourgish, Macedonian, Malagasy,
Malay, Malayalam, Maltese, Maori, Marathi, Mongolian, Nepali, Norwegian,
Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Samoan,
Scottish Gaelic, Serbian, Shona, Sindhi, Sinhala, Slovak, Slovenian, Somali,
Sotho, Spanish, Sundanese, Swahili, Swedish, Tajik, Tamil, Telugu, Thai,
Turkish, Ukrainian, Urdu, Uzbek, Vietnamese, Welsh, West Frisian, Xhosa,
Yiddish, Yoruba, Zulu.

## Results

mT5 achieves state-of-the-art performance on many cross-lingual NLP tasks, as
of November 2020. For example, on
[XTREME](https://github.com/google-research/xtreme) zero-shot classification,
structured prediction and QA tasks (showing F1 scores):

| Model | XNLI | PAWS-X | WikiAnn-NER | XQuAD | MLQA | TyDiQA-GoldP |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| mBERT | 65.4 | 81.9 | 62.2 | 64.5 | 61.4 | 59.7 |
| XLM | 69.1 | 80.9 | 61.2 | 59.8 | 48.5 | 43.6 |
| InfoXLM | 81.4 | - | - | - | 73.6 | - |
| X-STILTs | 80.4 | 87.7 | 64.7 | 77.2 | 72.3 | 76.0 |
| XLM-R | 79.2 | 86.4 | 65.4 | 76.6 | 71.6 | 65.1 |
| VECO | 79.9 | 88.7 | 65.7 | 77.3 | 71.7 | 67.6 |
| RemBERT | 80.8 | 87.5 | **70.1** | 79.6 | 73.1 | 77.0 |
| mT5-Small | 67.5 | 82.4 | 50.5 | 58.1 | 54.6 | 36.4 |
| mT5-Base | 75.4 | 86.4 | 55.7 | 67.0 | 64.6 | 59.1 |
| mT5-Large | 81.1 | 88.9 | 58.5 | 77.8 | 71.2 | 68.4 |
| mT5-XL | 82.9 | 89.6 | 65.5 | 79.5 | 73.5 | 77.8 |
| mT5-XXL | **85.0** | **90.0** | 69.2 | **82.5** | **76.0** | **82.0** |

## Usage

### Training

To run this code, you need to install the [t5
library](https://pypi.org/project/t5/). General instructions for training,
fine-tuning, evaluation, and exporting models for inference can be found in the
[t5
repo](https://github.com/google-research/text-to-text-transfer-transformer). In
order to use the additional mT5 tasks provided in this library with the
`t5_mesh_transformer` command, run from this directory and add the flag
`--module_import="multilingual_t5.tasks"`. There is also support for [mT5 in
HuggingFace](https://huggingface.co/transformers/model_doc/mt5.html); see
instructions in the T5 repo
[here](https://github.com/google-research/text-to-text-transfer-transformer#t5models).

To train an `mT5-Large` model on the
[mc4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual_nights_stay)
task from scratch as described in the paper:

```
export PROJECT=yourproject
export ZONE=yourzone
export BUCKET=yourbucket
export TPU=yourtpu

ctpu up --name=$TPU --project=$PROJECT --zone=$ZONE --tpu-size=v3-256 --tpu-only --noconf

TASK=mc4
MODEL_DIR="${BUCKET}${TASK}"

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="models/t5.1.1.large.gin" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 1024, 'targets': 256}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 1048576)" \
  --gin_param="utils.run.learning_rate_schedule=@learning_rate_schedules.rsqrt_no_ramp_down" \
  --gin_param="run.train_steps = 1000000" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-256'" \
  --eval_mode="perplexity_eval" \
  --eval_gin_param="mesh_eval_dataset_fn.num_eval_examples = 10000" \
  --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
  --module_import="multilingual_t5.tasks"
```

### Fine-Tuning

The example below shows how to finetune the `mT5-Large` model on the XNLI
zeroshot task. See `finetune_mt5_tasks.sh` for hyperparameter settings for
other tasks.

```
export PROJECT=yourproject
export ZONE=yourzone
export BUCKET=yourbucket
export TPU=yourtpu

ctpu up --name=$TPU --project=$PROJECT --zone=$ZONE --tpu-size=v3-256 --tpu-only --noconf

TASK=mt5_xnli_zeroshot
SEQUENCE_LENGTH_GIN=xnli
PRETRAINED_DIR=gs://t5-data/pretrained_models/mt5/large
PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000
MODEL_DIR="${BUCKET}${TASK}"

# Run fine-tuning
python -m t5.models.mesh_transformer_main \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
  --gin_file="sequence_lengths/${SEQUENCE_LENGTH_GIN}.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-256'" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS+FINETUNE_STEPS))" \
  --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
  --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
  --module_import="multilingual_t5.tasks" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 1048576)" \
  --gin_location_prefix="multilingual_t5/gin/"
```

The remaining experiments are shown in the [tasks.py](multilingual_t5/tasks.py) file.

## Released Model Checkpoints

We have released the following checkpoints for pre-trained models described in our [paper][paper]:

* **mT5-Small** (300 million parameters): [gs://t5-data/pretrained_models/mt5/small](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/mt5/small/)
* **mT5-Base** (580 million parameters): [gs://t5-data/pretrained_models/mt5/base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/mt5/base/)
* **mT5-Large** (1.2 billion parameters): [gs://t5-data/pretrained_models/mt5/large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/mt5/large/)
* **mT5-XL** (3.7 billion parameters): [gs://t5-data/pretrained_models/mt5/xl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/mt5/xl/)
* **mT5-XXL** (13 billion parameters): [gs://t5-data/pretrained_models/mt5/xxl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/mt5/xxl/)

# How to Cite

If you extend or use this work, please cite the [paper][paper] where it was
introduced:

```
@inproceedings{xue-etal-2021-mt5,
    title = "m{T}5: A Massively Multilingual Pre-trained Text-to-Text Transformer",
    author = "Xue, Linting  and
      Constant, Noah  and
      Roberts, Adam  and
      Kale, Mihir  and
      Al-Rfou, Rami  and
      Siddhant, Aditya  and
      Barua, Aditya  and
      Raffel, Colin",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.41",
    doi = "10.18653/v1/2021.naacl-main.41",
    pages = "483--498"
}
```

[paper]: https://aclanthology.org/2021.naacl-main.41/
