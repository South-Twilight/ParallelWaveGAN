#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained Parallel WaveGAN Generator."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from parallel_wavegan.datasets import (
    AudioDataset,
    AudioSCPDataset,
    MelDataset,
    MRTokenDataset,
    MelF0ExcitationDataset,
    MelSCPDataset,
    MelF0Dataset,
)
from parallel_wavegan.utils import load_model, read_hdf5, write_hdf5


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode dumped features with trained Parallel WaveGAN Generator "
            "(See detail in parallel_wavegan/bin/decode.py)."
        )
    )
    parser.add_argument(
        "--scp",
        default=None,
        type=str,
        help=(
            "kaldi-style feats.scp file. "
            "you need to specify either feats-scp or dumpdir."
        ),
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        help=(
            "directory including feature files. "
            "you need to specify either feats-scp or dumpdir."
        ),
    )
    parser.add_argument(
        "--segments",
        default=None,
        type=str,
        help="kaldi-style segments file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help=(
            "yaml format configuration file. if not explicitly provided, "
            "it will be searched in the checkpoint directory. (default=None)"
        ),
    )
    parser.add_argument(
        "--normalize-before",
        default=False,
        action="store_true",
        help=(
            "whether to perform feature normalization before input to the model. if"
            " true, it assumes that the feature is de-normalized. this is useful when"
            " text2mel model and vocoder use different feature statistics."
        ),
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
        "--store-feature",
        default=False,
        action="store_true",
        help="whether to store feature.",
    )
    parser.add_argument(
        "--storedir",
        default=None,
        type=str,
        help=(
            "directory to store feature."
        ),
    )
    parser.add_argument(
        "--use-multi-resolution-token",
        default=False,
        action="store_true",
        help="whether to use multi resolution_token.",
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

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if (args.scp is not None and args.dumpdir is not None) or (
        args.scp is None and args.dumpdir is None
    ):
        raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = load_model(args.checkpoint, config)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    if args.normalize_before:
        assert hasattr(model, "mean"), "Feature stats are not registered."
        assert hasattr(model, "scale"), "Feature stats are not registered."
    model.remove_weight_norm()
    model = model.eval().to(device)
    model.to(device)

    # check model type
    generator_type = config.get("generator_type", "ParallelWaveGANGenerator")
    use_aux_input = "VQVAE" not in generator_type
    use_global_condition = config.get("use_global_condition", False)
    use_local_condition = config.get("use_local_condition", False)
    use_f0_and_excitation = generator_type == "UHiFiGANGenerator"

    if use_aux_input:
        ############################
        #       MEL2WAV CASE       #
        ############################
        # setup dataset
        if args.dumpdir is not None:
            if config["format"] == "hdf5":
                mel_query = "*.h5"
                mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
                if args.use_f0:
                    f0_query = "*.h5"
                    f0_load_fn = lambda x: read_hdf5(x, "f0")  # NOQA
                if use_f0_and_excitation:
                    f0_query = "*.h5"
                    f0_load_fn = lambda x: read_hdf5(x, "f0")  # NOQA
                    excitation_query = "*.h5"
                    excitation_load_fn = lambda x: read_hdf5(x, "excitation")  # NOQA
                if args.use_multi_resolution_token:
                    resolution_query = "*.h5"
                    resolution_load_fn = lambda x: read_hdf5(x, "resolution")
            elif config["format"] == "npy":
                mel_query = "*-feats.npy"
                mel_load_fn = np.load
                if args.use_f0:
                    f0_query = "*-f0.npy"
                    f0_load_fn = np.load
                if use_f0_and_excitation:
                    f0_query = "*-f0.npy"
                    f0_load_fn = np.load
                    excitation_query = "*-excitation.npy"
                    excitation_load_fn = np.load
            else:
                raise ValueError("Support only hdf5 or npy format.")

            if args.use_f0:
                dataset = MelF0Dataset(
                    root_dir=args.dumpdir,
                    mel_query=mel_query,
                    f0_query=f0_query,
                    mel_load_fn=mel_load_fn,
                    f0_load_fn=f0_load_fn,
                    return_utt_id=True,
                )
            elif not use_f0_and_excitation:
                if args.use_multi_resolution_token:
                    dataset = MRTokenDataset(
                        args.dumpdir,
                        mel_query=mel_query,
                        resolution_query=resolution_query,
                        resolution_load_fn=resolution_load_fn,
                        return_utt_id=True,
                    )
                else:
                    dataset = MelDataset(
                        args.dumpdir,
                        mel_query=mel_query,
                        mel_load_fn=mel_load_fn,
                        return_utt_id=True,
                    )
            else:
                dataset = MelF0ExcitationDataset(
                    root_dir=args.dumpdir,
                    mel_query=mel_query,
                    f0_query=f0_query,
                    excitation_query=excitation_query,
                    mel_load_fn=mel_load_fn,
                    f0_load_fn=f0_load_fn,
                    excitation_load_fn=excitation_load_fn,
                    return_utt_id=True,
                )
        else:
            if use_f0_and_excitation:
                raise NotImplementedError(
                    "SCP format is not supported for f0 and excitation."
                )
            dataset = MelSCPDataset(
                feats_scp=args.scp,
                return_utt_id=True,
            )
        logging.info(f"The number of features to be decoded = {len(dataset)}.")

        # start generation
        total_rtf = 0.0
        with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
            for idx, items in enumerate(pbar, 1):
                if args.use_f0:
                    utt_id, c, f0 = items
                    excitation = None
                elif not use_f0_and_excitation:
                    utt_id, c = items
                    f0, excitation = None, None
                else:
                    utt_id, c, f0, excitation = items
                batch = dict(normalize_before=args.normalize_before)
                if c is not None:
                    if isinstance(c, dict):
                        for key, value in c.items():
                            c[key] = torch.tensor(value, dtype=torch.float).to(device)
                    else:
                        c = torch.tensor(c, dtype=torch.float).to(device)
                    batch.update(c=c)
                if f0 is not None:
                    f0 = torch.tensor(f0, dtype=torch.float).to(device)
                    batch.update(f0=f0)
                if excitation is not None:
                    excitation = torch.tensor(excitation, dtype=torch.float).to(device)
                    batch.update(excitation=excitation)
                if args.store_feature:
                    batch.update(store_feature=True)
                    y = model.inference(**batch)
                    if config["format"] == "hdf5":
                        # Tensor for single resolution, dict for multi resolution
                        if isinstance(y, torch.Tensor):
                            # y input as [B, C, T]
                            y = y.squeeze(0).cpu().numpy().astype(np.float32)
                            # item in y should input as [T, C]
                            write_hdf5(
                                os.path.join(args.storedir, f"{utt_id}.h5"),
                                "feats",
                                y,
                            )
                        elif isinstance(y, dict):
                            # value of y input as [B, T, C]
                            rs = []
                            for key, value in y.items():
                                feat = value.squeeze(0).cpu().numpy().astype(np.float32)
                                rs.append(key)
                                # logging.info(f'{key}: {feat.shape}') 
                                # item in y should input as [T, C]
                                write_hdf5(
                                    os.path.join(args.storedir, f"{utt_id}.h5"),
                                    f"feats-{key}",
                                    feat,
                                )
                            rs = np.array(rs)
                            write_hdf5(
                                os.path.join(args.storedir, f"{utt_id}.h5"),
                                f"resolution",
                                rs.astype(np.int32),
                            )
                                
                else:
                    start = time.time()
                    y = model.inference(**batch).view(-1)
                    rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
                    pbar.set_postfix({"RTF": rtf})
                    total_rtf += rtf

                    # save as PCM 16 bit wav file
                    sf.write(
                        os.path.join(config["outdir"], f"{utt_id}_gen.wav"),
                        y.cpu().numpy(),
                        config["sampling_rate"],
                        "PCM_16",
                    )

        # report average RTF
        logging.info(
            f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
        )
    else:
        ############################
        #      VQ-WAV2WAV CASE     #
        ############################
        # setup dataset
        if args.dumpdir is not None:
            local_query = None
            local_load_fn = None
            global_query = None
            global_load_fn = None
            if config["format"] == "hdf5":
                audio_query = "*.h5"
                audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
                if use_local_condition:
                    local_query = "*.h5"
                    local_load_fn = lambda x: read_hdf5(x, "local")  # NOQA
                if use_global_condition:
                    global_query = "*.h5"
                    global_load_fn = lambda x: read_hdf5(x, "global")  # NOQA
            elif config["format"] == "npy":
                audio_query = "*-wave.npy"
                audio_load_fn = np.load
                if use_local_condition:
                    local_query = "*-local.npy"
                    local_load_fn = np.load
                if use_global_condition:
                    global_query = "*-global.npy"
                    global_load_fn = np.load
            else:
                raise ValueError("support only hdf5 or npy format.")
            dataset = AudioDataset(
                args.dumpdir,
                audio_query=audio_query,
                audio_load_fn=audio_load_fn,
                local_query=local_query,
                local_load_fn=local_load_fn,
                global_query=global_query,
                global_load_fn=global_load_fn,
                return_utt_id=True,
            )
        else:
            if use_local_condition:
                raise NotImplementedError("Not supported.")
            if use_global_condition:
                raise NotImplementedError("Not supported.")
            dataset = AudioSCPDataset(
                args.scp,
                segments=args.segments,
                return_utt_id=True,
            )
        logging.info(f"The number of features to be decoded = {len(dataset)}.")

        # start generation
        total_rtf = 0.0
        text = os.path.join(config["outdir"], "text")
        with torch.no_grad(), open(text, "w") as f, tqdm(
            dataset, desc="[decode]"
        ) as pbar:
            for idx, items in enumerate(pbar, 1):
                # setup input
                if use_local_condition and use_global_condition:
                    utt_id, x, l_, g = items
                    l_ = (
                        torch.from_numpy(l_)
                        .float()
                        .unsqueeze(0)
                        .transpose(1, 2)
                        .to(device)
                    )
                    g = torch.from_numpy(g).long().view(1).to(device)
                elif use_local_condition:
                    utt_id, x, l_ = items
                    l_ = (
                        torch.from_numpy(l_)
                        .float()
                        .unsqueeze(0)
                        .transpose(1, 2)
                        .to(device)
                    )
                    g = None
                elif use_global_condition:
                    utt_id, x, g = items
                    g = torch.from_numpy(g).long().view(1).to(device)
                    l_ = None
                else:
                    utt_id, x = items
                    l_, g = None, None
                x = torch.from_numpy(x).float().view(1, 1, -1).to(device)

                # generate
                start = time.time()
                if config["generator_params"]["out_channels"] == 1:
                    z = model.encode(x)
                    y = model.decode(z, l_, g).view(-1).cpu().numpy()
                else:
                    z = model.encode(model.pqmf.analysis(x))
                    y_ = model.decode(z, l_, g)
                    y = model.pqmf.synthesis(y_).view(-1).cpu().numpy()
                rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
                pbar.set_postfix({"RTF": rtf})
                total_rtf += rtf

                # save as PCM 16 bit wav file
                sf.write(
                    os.path.join(config["outdir"], f"{utt_id}_gen.wav"),
                    y,
                    config["sampling_rate"],
                    "PCM_16",
                )

                # save encode discrete symbols
                symbols = " ".join([str(z) for z in z.view(-1).cpu().numpy()])
                f.write(f"{utt_id} {symbols}\n")

        # report average RTF
        logging.info(
            f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
        )


if __name__ == "__main__":
    main()
