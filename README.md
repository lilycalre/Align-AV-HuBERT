# Align-AV-HuBERT
[Align-AV-HuBERT: AV-HuBERT with Audio-Visual Temporal Alignment](https://ieeexplore.ieee.org/document/11210211)
```
@inproceedings{li2025align,
  title={Align-AV-HuBERT: AV-HuBERT with Audio-Visual Temporal Alignment},
  author={Li, Cancan and Su, Fei and Liu, Juan},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```
## Introduction
Align-AV-HuBERT, which eneable the model for simultaneous audio-visual temporal alignment and audio-visual speech representation.


## Pre-trained and fine-tuned models

Please find the checkpoints in avhubert/checkpoint/


## Train Align-AV-HuBERT

Traning code will be released soon.
### Data preparation

Follow the steps in [`preparation`](https://github.com/facebookresearch/av_hubert/avhubert/preparation/) to pre-process:
- LRS3 and VoxCeleb2 datasets

Follow the steps in [`clustering`](https://github.com/facebookresearch/av_hubert/avhubert/clustering/) (pre-train only) to create:
- `{train,valid}.km` frame-aligned pseudo label files.
The `label_rate` is the same as the feature frame rate used for clustering,
which is 100Hz for MFCC features and 25Hz for AV-HuBERT features by default.

### Pre-train an Align-AV-HuBERT model

Suppose `{train,valid}.tsv` are saved at `/path/to/data`, `{train,valid}.km`
are saved at `/path/to/labels`, the configuration file is saved at `/path/to/conf/conf-name`, and the label rate is 100Hz.

To train a model, run:
```sh
$ cd avhubert
$ fairseq-hydra-train --config-dir conf/alignment/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  model.label_rate=25 hydra.run.dir=/path/to/experiment/pretrain/ \
  common.user_dir=`pwd`
```
We use label created by 5-th iteration of AV-HuBERT base.
conf-name can be selected from {base_lrs3_plusalignment.yaml, base_vox_plusalignment.yaml}

### Finetune an Align-AV-HuBERT model with Seq2Seq
Suppose `{train,valid}.tsv` are saved at `/path/to/data`, `{train,valid}.wrd`
are saved at `/path/to/labels`, the configuration file is saved at `/path/to/conf/conf-name`.

To fine-tune a pre-trained model at `/path/to/checkpoint`, run:
```sh
$ cd avhubert
$ fairseq-hydra-train --config-dir conf/finetune --config-name conf-name \      
  task.data=/path/to/data task.label_dir=/path/to/label \
  task.tokenizer_bpe_model=/path/to/tokenizer model.w2v_path=/path/to/checkpoint \
  hydra.run.dir=/path/to/experiment/finetune/ common.user_dir=`pwd`
```

### Test an Align-AV-HuBERT model
Suppose the `test.tsv` and `test.wrd` are the video list and transcripts of
the split to be decoded, saved at `/path/to/data`, and the fine-tuned model is
saved at `/path/to/checkpoint`.

Note that in Fairseq, all gpus are used for inference and data are splited according to number of gpus. So it is suggested to specify a gpu for inference.

#### Seq2Seq decoding on fiexed shift number from (-25, 25) frames

In terms of Fairseq, to test on fiexed shift number from (-25, 25) frames, please use "from .hubert_dataset_fixedshift import AVHubertDataset" and "from .sequence_generator_plusalignment import SequenceGenerator" in the hubert_pretraining.py.


```sh
$ cd avhubert
$ python -B infer_s2s_align.py --config-dir ./conf/inference --config-name s2s_decode.yaml \
  dataset.gen_subset=test common_eval.path=/path/to/checkpoint \
  common_eval.results_path=/path/to/experiment/decode/s2s/test \
  override.modalities=['video'] common.user_dir=`pwd` \
  +override.data=/path/to/data/30h_data +override.label_dir=/path/to/data/30h_data
```

#### Testing accuracy for predicting shift number

In terms of Fairseq, to test accuracy for predicting shift number, please change "from .hubert_dataset_fixedshift import AVHubertDataset" to "from .hubert_dataset_alignment import AVHubertDataset" and change "from .sequence_generator_plusalignment import SequenceGenerator" to "from .sequence_generator_alignment import SequenceGenerator" in the hubert_pretraining.py.

```sh
$ cd avhubert
$ python -B infer_alignment.py --config-dir ./conf/inference --config-name infer_alignment.yaml \
  dataset.gen_subset=test common_eval.path=/path/to/checkpoint \
  common_eval.results_path=/path/to/experiment/decode/accuracy/test \
  override.modalities=['video'] common.user_dir=`pwd` \
  +override.data=/path/to/data/30h_data +override.label_dir=/path/to/data/30h_data
```


