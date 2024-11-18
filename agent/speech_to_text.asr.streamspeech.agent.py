##########################################
# Streaming ASR Agent for StreamSpeech
#
# StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning (ACL 2024)
##########################################

from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from pathlib import Path
from typing import Any, Dict, Optional, Union
from fairseq.data.audio.audio_utils import convert_waveform
from examples.speech_to_text.data_utils import extract_fbank_features
import ast
import math
import os
import json
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from fairseq import checkpoint_utils, tasks, utils, options
from fairseq.file_io import PathManager
from fairseq import search
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform

from examples.speech_to_speech.asr_bleu.utils import retrieve_asr_config, ASRGenerator


SHIFT_SIZE = 10
WINDOW_SIZE = 25
ORG_SAMPLE_RATE = 48000
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"
DEFAULT_EOS = 2


class OnlineFeatureExtractor:
    """
    Extract speech feature on the fly.
    """

    def __init__(self, args, cfg):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.sample_rate = args.sample_rate
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.previous_residual_samples = []
        self.global_cmvn = args.global_cmvn
        self.device = "cuda" if args.device == "gpu" else "cpu"
        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            {"feature_transforms": ["utterance_cmvn"]}
        )

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples, sr=ORG_SAMPLE_RATE):
        samples = new_samples

        # # num_frames is the number of frames from the new segment
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
            / self.num_samples_per_shift
        )

        # # the number of frames used for feature extraction
        # # including some part of thte previous segment
        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )
        samples = samples[:effective_num_samples]
        waveform, sample_rate = convert_waveform(
            torch.tensor([samples]), sr, to_mono=True, to_sample_rate=16000
        )
        output = extract_fbank_features(waveform, 16000)
        output = self.transform(output)
        return torch.tensor(output, device=self.device)

    def transform(self, input):
        if self.global_cmvn is None:
            return input

        mean = self.global_cmvn["mean"]
        std = self.global_cmvn["std"]

        x = np.subtract(input, mean)
        x = np.divide(x, std)
        return x


@entrypoint
class StreamSpeechASRAgent(SpeechToTextAgent):
    """
    Incrementally feed text to this offline Fastspeech2 TTS model,
    with a minimum numbers of phonemes every chunk.
    """

    def __init__(self, args):
        super().__init__(args)
        self.eos = DEFAULT_EOS

        self.gpu = self.args.device == "gpu"
        self.device = "cuda" if args.device == "gpu" else "cpu"

        self.args = args

        self.load_model_vocab(args)

        self.max_len = args.max_len

        self.force_finish = args.force_finish

        torch.set_grad_enabled(False)

        tgt_dict_mt = self.dict[f"{self.models[0].mt_task_name}"]
        tgt_dict = self.dict["tgt"]
        tgt_dict_asr = self.dict["source_unigram"]
        tgt_dict_st = self.dict["ctc_target_unigram"]
        args.user_dir=args.agent_dir
        utils.import_user_module(args)
        from agent.sequence_generator import SequenceGenerator
        from agent.ctc_generator import CTCSequenceGenerator
        from agent.ctc_decoder import CTCDecoder
        from agent.tts.vocoder import CodeHiFiGANVocoderWithDur

        self.ctc_generator = CTCSequenceGenerator(
            tgt_dict, self.models, use_incremental_states=True
        )

        self.asr_ctc_generator = CTCDecoder(tgt_dict_asr, self.models)
        self.st_ctc_generator = CTCDecoder(tgt_dict_st, self.models)

        self.generator = SequenceGenerator(
            self.models,
            tgt_dict,
            beam_size=1,
            max_len_a=1,
            max_len_b=200,
            max_len=0,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=search.BeamSearch(tgt_dict),
            eos=tgt_dict.eos(),
            symbols_to_strip_from_output=None,
        )

        self.generator_mt = SequenceGenerator(
            self.models,
            tgt_dict_mt,
            beam_size=1,
            max_len_a=1,
            max_len_b=200,
            max_len=0,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=search.BeamSearch(tgt_dict_mt),
            eos=tgt_dict_mt.eos(),
            symbols_to_strip_from_output=None,
            use_incremental_states=True,
        )
        self.lagging_k1 = args.lagging_k1
        self.lagging_k2 = args.lagging_k2
        self.segment_size = args.segment_size
        self.stride_n = args.stride_n
        self.unit_per_subword = args.unit_per_subword
        self.stride_n2 = args.stride_n2
        if args.extra_output_dir is not None:
            self.asr_file = Path(args.extra_output_dir + "/asr.txt")
            self.st_file = Path(args.extra_output_dir + "/st.txt")
            self.unit_file = Path(args.extra_output_dir + "/unit.txt")
            #     pass
            self.quiet = False
        else:
            self.quiet = True

        self.reset()

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="path to your pretrained model.",
        )
        parser.add_argument(
            "--data-bin", type=str, required=True, help="Path of data binary"
        )
        parser.add_argument(
            "--config-yaml", type=str, default=None, help="Path to config yaml file"
        )
        parser.add_argument(
            "--multitask-config-yaml",
            type=str,
            default=None,
            help="Path to config yaml file",
        )
        parser.add_argument(
            "--global-stats",
            type=str,
            default=None,
            help="Path to json file containing cmvn stats",
        )
        parser.add_argument(
            "--tgt-splitter-type",
            type=str,
            default="SentencePiece",
            help="Subword splitter type for target text",
        )
        parser.add_argument(
            "--tgt-splitter-path",
            type=str,
            default=None,
            help="Subword splitter model path for target text",
        )
        parser.add_argument(
            "--user-dir",
            type=str,
            default="researches/ctc_unity",
            help="User directory for model",
        )
        parser.add_argument(
            "--agent-dir",
            type=str,
            default="agent",
            help="User directory for agents",
        )
        parser.add_argument(
            "--max-len", type=int, default=200, help="Max length of translation"
        )
        parser.add_argument(
            "--force-finish",
            default=False,
            action="store_true",
            help="Force the model to finish the hypothsis if the source is not finished",
        )
        parser.add_argument(
            "--shift-size",
            type=int,
            default=SHIFT_SIZE,
            help="Shift size of feature extraction window.",
        )
        parser.add_argument(
            "--window-size",
            type=int,
            default=WINDOW_SIZE,
            help="Window size of feature extraction window.",
        )
        parser.add_argument(
            "--sample-rate", type=int, default=ORG_SAMPLE_RATE, help="Sample rate"
        )
        parser.add_argument(
            "--feature-dim",
            type=int,
            default=FEATURE_DIM,
            help="Acoustic feature dimension.",
        )

        parser.add_argument("--lagging-k1", type=int, default=0, help="lagging number")
        parser.add_argument("--lagging-k2", type=int, default=0, help="lagging number")
        parser.add_argument(
            "--segment-size", type=int, default=320, help="segment-size"
        )
        parser.add_argument("--stride-n", type=int, default=1, help="lagging number")
        parser.add_argument("--stride-n2", type=int, default=1, help="lagging number")
        parser.add_argument(
            "--unit-per-subword", type=int, default=15, help="lagging number"
        )
        parser.add_argument(
            "--extra-output-dir", type=str, default=None, help="extra output dir"
        )

    def reset(self):
        self.src_seg_num = 0
        self.tgt_subwords_indices = None
        self.src_ctc_indices = None
        self.src_ctc_prefix_length = 0
        self.tgt_ctc_prefix_length = 0
        self.tgt_units_indices = None
        self.prev_output_tokens_mt = None
        self.tgt_text = ""
        self.asr_text = ""
        self.mt_decoder_out = None
        self.unit = None
        self.wav = []
        self.post_transcription = ""
        self.unfinished_wav = None
        self.states.reset()
        try:
            self.generator_mt.reset_incremental_states()
            self.ctc_generator.reset_incremental_states()
        except:
            pass

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    def load_model_vocab(self, args):
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)
        state["cfg"].common['user_dir']=args.user_dir
        utils.import_user_module(state["cfg"].common)

        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        args.global_cmvn = None
        if args.config_yaml is not None:
            task_args.config_yaml = args.config_yaml
            with open(os.path.join(args.data_bin, args.config_yaml), "r") as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)

            if "global_cmvn" in config:
                args.global_cmvn = np.load(config["global_cmvn"]["stats_npz_path"])

        self.feature_extractor = OnlineFeatureExtractor(args, config)

        if args.multitask_config_yaml is not None:
            task_args.multitask_config_yaml = args.multitask_config_yaml

        task = tasks.setup_task(task_args)
        self.task = task
        overrides = ast.literal_eval(state["cfg"].common_eval.model_overrides)

        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(filename),
            arg_overrides=overrides,
            task=task,
            suffix=state["cfg"].checkpoint.checkpoint_suffix,
            strict=(state["cfg"].checkpoint.checkpoint_shard_count == 1),
            num_shards=state["cfg"].checkpoint.checkpoint_shard_count,
        )

        chunk_size = args.source_segment_size // 40

        self.models = models

        for model in self.models:
            model.eval()
            model.share_memory()
            if self.gpu:
                model.cuda()
            model.encoder.chunk_size = chunk_size
            chunk_size = min(chunk_size, 16)
            for conv in model.encoder.subsample.conv_layers:
                conv.chunk_size = chunk_size
            for layer in model.encoder.conformer_layers:
                layer.conv_module.depthwise_conv.chunk_size = chunk_size

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary

        for k, v in task.multitask_tasks.items():
            self.dict[k] = v.tgt_dict

    @torch.inference_mode()
    def policy(self):

        feature = self.feature_extractor(self.states.source)
        if feature.size(0) == 0 and not self.states.source_finished:
            return ReadAction()

        src_indices = feature.unsqueeze(0)
        src_lengths = torch.tensor([feature.size(0)], device=self.device).long()

        self.encoder_outs = self.generator.model.forward_encoder(
            {"src_tokens": src_indices, "src_lengths": src_lengths}
        )

        finalized_asr = self.asr_ctc_generator.generate(
            self.encoder_outs[0], aux_task_name="source_unigram"
        )
        asr_probs = torch.exp(finalized_asr[0][0]["lprobs"])

        for i, hypo in enumerate(finalized_asr):
            i_beam = 0
            tmp = hypo[i_beam]["tokens"].int()  # hyp + eos
            src_ctc_indices = tmp
            src_ctc_index = hypo[i_beam]["index"]
            tokens = [self.dict["source_unigram"][c] for c in tmp]
            text = "".join([self.dict["source_unigram"][c] for c in tmp])
            text = text.replace("_", " ")
            text = text.replace("▁", " ")
            text = text.replace("<unk>", " ")
            text = text.replace("<s>", "")
            text = text.replace("</s>", "")
            if len(text) > 0 and text[0] == " ":
                text = text[1:]
            if self.states.source_finished and not self.quiet:
                with open(self.asr_file, "a") as file:
                    print(text, file=file)

        text = " ".join(tokens)
        new_text = text[len(self.asr_text) :]

        self.asr_text = text

        if self.states.source_finished:
            self.states.target_finished = True
            self.reset()

        return WriteAction(
            new_text,
            finished=self.states.target_finished,
        )
