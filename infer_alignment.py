# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
from itertools import chain
import logging
import math
import os
import sys
import json
import hashlib
import editdistance
from argparse import Namespace

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.models import FairseqLanguageModel
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    GenerationConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"

@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: [""], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})

@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for recognition!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(cfg.common_eval.results_path, "decode.log")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos, generator.pad}

def _main(cfg, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("hybrid.speech_recognize")
    if output_file is not sys.stdout:  # also print to stdout
        logger.addHandler(logging.StreamHandler(sys.stdout))

    utils.import_user_module(cfg.common)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path])
    models = [model.eval().cuda() for model in models]
    
    task = tasks.setup_task(saved_cfg.task)

    task.build_tokenizer(saved_cfg.tokenizer)
    task.build_bpe(saved_cfg.bpe)

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available()


    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
    task.cfg.labels=['wrd']
    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    if cfg.generation.match_source_len:
        logger.warning(
            "The option match_source_len is not applicable to speech recognition. Ignoring it."
        )
    gen_timer = StopwatchMeter()
    extra_gen_cls_kwargs = {
        "lm_model": lms[0],
        "lm_weight": cfg.generation.lm_weight,
    }
    cfg.generation.score_reference = False  #
    save_attention_plot = cfg.generation.print_alignment is not None
    cfg.generation.print_alignment = None  #
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    def decode_fn(x):
        symbols_ignore = get_symbols_to_strip_from_output(generator)
        symbols_ignore.add(dictionary.pad())
        if hasattr(task.datasets[cfg.dataset.gen_subset].label_processors[0], 'decode'):
            return task.datasets[cfg.dataset.gen_subset].label_processors[0].decode(x, symbols_ignore)
        chars = dictionary.string(x, extra_symbols_to_ignore=symbols_ignore)
        words = " ".join("".join(chars.split()).replace('|', ' ').split())
        return words

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    result_dict = {'utt_id': [], 'ref': [], 'hypo': []}
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]


        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )


        for i in range(len(sample["id"])):
            result_dict['utt_id'].append(sample['utt_id'][i])
            ref_sent = sample['net_input']['source']['shift_num'][i]
            result_dict['ref'].append(ref_sent)
            result_dict['hypo'].append(hypos[i])
            logger.info(f"\nREF:{ref_sent}\nHYP:{hypos[i]}\n")

    res0=0 #prediction equall shift num 
    res1=0 #one frame error
    reslarge=0 #more than 5 frames error
    for r,h in zip(result_dict['ref'],result_dict['hypo']):
        if abs(r-h)==0:
            res0+=1
        if abs(r-h)==1:
            res1+=1
        if abs(r-h)>5:
            reslarge+=1
    res0 = 100 * res0/len(result_dict['ref'])
    res1 = 100 * res1/len(result_dict['ref'])
    reslarge = 100 * reslarge/len(result_dict['ref'])
    logger.info(f"ACC0: {res0}%, ACC1: {res1}%, ACC5+: {reslarge}%")
    return


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    wer = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))
    return


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
