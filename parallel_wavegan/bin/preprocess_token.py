#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
# Copyright 2024 Yuxun Tang
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""

import argparse
import logging
import os

import librosa
import numpy as np
import resampy
import soundfile as sf
import yaml
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d

from parallel_wavegan.datasets import AudioDataset, AudioSCPDataset
from parallel_wavegan.utils import write_hdf5


def _convert_to_continuous_f0(f0: np.array) -> np.array:
    if (f0 == 0).all():
        logging.warning("All frames seems to be unvoiced.")
        return f0

    # padding start and end of f0 sequence
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nonzero_idxs = np.where(f0 != 0)[0]

    # perform linear interpolation
    interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs])
    f0 = interp_fn(np.arange(0, f0.shape[0]))

    return f0

def f0_dio(
    audio,
    sampling_rate,
    hop_size=160,
    pitch_min=80,
    pitch_max=10000,
    use_log_f0=True,
    use_continuous_f0=True,
):
    """Compute F0 with pyworld.dio

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        hop_size (int): Hop size.
        pitch_min (int): Minimum pitch in pitch extraction.
        pitch_max (int): Maximum pitch in pitch extraction.

    Returns:
        ndarray: f0 feature (#frames, ).

    Note:
        Unvoiced frame has value = 0.

    """
    if torch.is_tensor(audio):
        x = audio.cpu().numpy().astype(np.double)
    else:
        x = audio.astype(np.double)
    frame_period = 1000 * hop_size / sampling_rate
    import pyworld
    f0, timeaxis = pyworld.dio(
        x,
        sampling_rate,
        f0_floor=pitch_min,
        f0_ceil=pitch_max,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(x, f0, timeaxis, sampling_rate)
    if use_continuous_f0:
        f0 = _convert_to_continuous_f0(f0)
    if use_log_f0:
        nonzero_idxs = np.where(f0 != 0)[0]
        f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
    return f0


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess audio and then extract features (See detail in"
            " parallel_wavegan/bin/preprocess.py)."
        )
    )
    parser.add_argument(
        "--wav-scp",
        "--scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file. you need to specify either scp or rootdir.",
    )
    parser.add_argument(
        "--segments",
        default=None,
        type=str,
        help=(
            "kaldi-style segments file. if use, you must to specify both scp and"
            " segments."
        ),
    )
    parser.add_argument(
        "--text",
        default=None,
        type=str,
        help="kaldi-style text format hubert embedding index.",
    )
    parser.add_argument(
        "--utt2spk",
        default=None,
        type=str,
        help=(
            "kaldi-style utt2spk file. If you want to add global conditionning with "
            "speaker id, you need to specify this argument."
        ),
    )
    parser.add_argument(
        "--spk2idx",
        default=None,
        type=str,
        help=(
            "kaldi-style spk2idx file. If you want to add global conditionning with "
            "speaker id, you need to specify this argument."
        ),
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        help=(
            "directory including wav files. you need to specify either scp or rootdir."
        ),
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--use-f0",
        default=False,
        action="store_true",
        help="whether to use f0 sequence.",
    )
    parser.add_argument(
        "--use-embedding-feats",
        default=False,
        action="store_true",
        help="whether to use pretrain model to get feature.",
    )
    parser.add_argument(
        "--use-multi-layer",
        default=False,
        action="store_true",
        help="whether to use multi layer feature.",
    )
    parser.add_argument(
        "--use-multi-resolution-token",
        default=False,
        action="store_true",
        help="whether to input multi resolution token.",
    )
    parser.add_argument(
        "--feat-layer",
        type=int,
        default=1,
        help="all layer numbert or specific layer",
    )
    parser.add_argument(
        "--multi-token-files",
        type=str,
        default="",
        help="list of token files in multi token pattern",
    )
    # parser.add_argument(
    #     "--multi-token-mix-type",
    #     type=str,
    #     default="sequence",
    #     help="token mix type in multi token pattern",
    #     choices=["sequence", "frame"],
    # )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="facebook/hubert-base-ls960",
        help="pretrained model for embedding feature",
    )
    parser.add_argument(
        "--skip_existed_file",
        default=False,
        action="store_true"
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if (args.wav_scp is not None and args.rootdir is not None) or (
        args.wav_scp is None and args.rootdir is None
    ):
        raise ValueError("Please specify either --rootdir or --wav-scp.")

    # get dataset
    if args.rootdir is not None:
        dataset = AudioDataset(
            args.rootdir,
            "*.wav",
            audio_load_fn=sf.read,
            return_utt_id=True,
        )
    else:
        dataset = AudioSCPDataset(
            args.wav_scp,
            segments=args.segments,
            return_utt_id=True,
            return_sampling_rate=True,
        )

    # Only for discrete token, load discrete files
    if args.use_embedding_feats is False:
        logging.info(f'path: {args.text}')
        if not args.multi_token_files:
            # NOTE(Yuxun): load single layer token file, `args.text` will path of the token file
            with open(args.text) as f:
                lines = [line.strip() for line in f.readlines()]
            text = {
                line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1].split() for line in lines
            }
        else: 
            # NOTE(Yuxun): load multi layer token file, `args.text` will path of files directory
            text = {}
            token_files = args.multi_token_files.split(" ")
            for fname in token_files:
                fpath = os.path.join(args.text, fname)
                if not os.path.exists(fpath):
                    raise FileExistsError(f"{fpath} does not exist.")
                with open(fpath, 'r', encoding="utf-8") as f:
                    for line in f:
                        utt_name, tokens = line.strip().split(maxsplit=1) # name, list
                        tokens = tokens.split() 
                        if text.get(utt_name) is None:
                            text[utt_name] = []
                        # combine in sequence way, [L, T]
                        text[utt_name].append(tokens)

    # load spk2utt file
    if args.utt2spk is not None:
        with open(args.utt2spk) as f:
            lines = [l.replace("\n", "") for l in f.readlines()]
        utt2spk = {l.split()[0]: l.split()[1] for l in lines}
        with open(args.spk2idx) as f:
            lines = [l.replace("\n", "") for l in f.readlines()]
        spk2idx = {l.split()[0]: int(l.split()[1]) for l in lines}

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        if args.skip_existed_file and os.path.exists(os.path.join(args.dumpdir, f"{utt_id}.h5")):
            logging.info(f'{utt_id} skip')
            continue
        logging.info(f'{utt_id} run')
        
        # check
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."

        # downsample
        if fs != config["sampling_rate"]:
            audio = resampy.resample(audio, fs, config["sampling_rate"], axis=0)

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )
        
        if args.use_embedding_feats:
            # Case: intermediate feature will be continout embedding features (for teacher-forcing)
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
            pretrained_model = args.pretrained_model
            logging.info(f'loading preatrined model: {pretrained_model}')
            model = AutoModel.from_pretrained(pretrained_model)
            processor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model) 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            inputs = processor(audio, sampling_rate=fs, return_tensors="pt")
            def move_to_device(dict, device):
                for key in dict:
                    dict[key] = dict[key].to(device)
                return dict
            inputs = move_to_device(inputs, device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            features = outputs.hidden_states

            # Whether to use multi layer
            if args.use_multi_layer:
                # In multi layer, `emb_layer` will specify as number of layers used.
                mel = torch.stack(features[: args.emb_layer + 1], 0)
                # (L, 1, T, C) -> (T, L, C)
                mel = mel.squeeze(1).transpose(0, 1).cpu().detach().numpy()
            else:
                # In single layer, `emb_layer` will specify as the layer id used.
                mel = features[args.emb_layer].squeeze(0).cpu().detach().numpy()
                # Output mel as (T, C) 
        else:
            # Case: intermediate feature will be discrete token. Index of discrete dict is used as mel.
            if args.use_multi_resolution_token:
                # multi resolution for SingOMD, source resolution is the finest grained.
                resolution = config['generator_params']['resolution']
                resolution = sorted(resolution)
                logging.info(f'Origin resolution of feature is {resolution[0]}')
                rs_token = text[utt_id]
                if not isinstance(text[utt_id][0], list):
                    rs_token = [rs_token]
                rs_token = sorted(rs_token, key=len, reverse=True)
                mel = np.array(rs_token[0]).astype(np.int64)
            else:
                # normal discrete token
                mel = np.array(text[utt_id]).astype(np.int64) # [L, T] in 'sequence' / [T]
                if args.use_multi_layer: 
                    # multi layer, transfer mix type from 'sequence' to 'frame. [L, T] -> [T, L]
                    mel = mel.transpose(1, 0)
                    mel = mel.reshape(-1, args.feat_layer)
                else:
                    # single layer, [T] -> [T, 1]
                    mel = mel.reshape(-1, 1)
                # output mel as [T, L]
                
        logging.info(f'mel({mel.shape})')
        # logging.info(f'mel: {mel}')
        
        if args.spk2idx is not None:
            if args.use_multi_resolution_token:
                logging.warning("It doesn't support speaker embedding now when using multi reoslution features.")
            spk = utt2spk[utt_id]
            if spk in spk2idx:
                idx = spk2idx[spk]
            else:
                logging.warning(f"{spk} is unknown speaker.")
                max_idx = max(spk2idx.values()) + 1
                idx = max_idx

            # concatenate with mel
            idx = np.repeat(np.array(idx).reshape(1, 1), len(mel), axis=0)
            mel = np.concatenate([mel, idx], axis=1)

        # make sure the audio length and feature length are matched
        logging.info(f"Mod: {len(audio) - len(mel) * config['hop_size']}")
        if len(mel) * config["hop_size"] <= len(audio):
            logging.warning(
                "len(mel) * config['hop_size'] <= len(audio), may be errors"
            )
        # logging.info(f'audio: {len(audio)}')
        # import math
        # token_len = math.ceil(len(audio) / config["hop_size"])
        # assert len(mel) == token_len
        mel = mel[: len(audio) // config["hop_size"]]
        audio = audio[: len(mel) * config["hop_size"]]
        assert len(mel) * config["hop_size"] == len(audio)

        # if args.multi_token_mix_type == "frame":
        #     mel = mel
        # elif args.mult_token_mix_tpe == "sequence":
        #     mel = mel.transpose(1, 0)
        
        # use f0
        if args.use_f0:
            f0 = f0_dio(
                audio,
                sampling_rate=config["sampling_rate"],
                hop_size=config["hop_size"],
            ) # (#frames,) 
            # logging.info(f'f0({f0.shape}): {f0}') 
            if len(f0) > len(mel):
                f0 = f0[: len(mel)]
            else:
                f0 = np.pad(f0, (0, len(mel) - len(f0)), mode="edge")

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio *= config["global_gain_scale"]
        if np.abs(audio).max() >= 1.0:
            logging.warning(
                f"{utt_id} causes clipping. "
                "it is better to re-consider global gain scale."
            )
            continue

        # save
        if config["format"] == "hdf5":
            # logging.info(f'mel: {mel.shape} f0: {f0.shape}')
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "wave",
                audio.astype(np.float32),
            )
            if not args.use_multi_resolution_token:
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"),
                    "feats",
                    mel.astype(np.float32),
                )
            else:
                for rs, feat in zip(resolution, rs_token):
                    mel = np.array(feat).astype(np.float32)
                    mel = mel.reshape(-1, 1)
                    logging.info(f'{rs}: {mel.shape}')
                    logging.info(f'mel: {mel.shape}')
                    write_hdf5(
                        os.path.join(args.dumpdir, f"{utt_id}.h5"),
                        f"feats-{rs}",
                        mel,
                    )
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"),
                    f"resolution",
                    np.array(resolution).astype(np.int32),
                )
            if args.use_f0:
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"),
                    "f0",
                    f0.astype(np.float32),
                )
                
        elif config["format"] == "npy":
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
                audio.astype(np.float32),
                allow_pickle=False,
            )
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
                mel.astype(np.float32),
                allow_pickle=False,
            )
            if args.use_f0:
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-f0.npy"),
                    f0.astype(np.float32),
                    allow_pickle=False,
                )
                
        else:
            raise ValueError("support only hdf5 or npy format.")


if __name__ == "__main__":
    main()
