import argparse
import os

import librosa
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Opencpop Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("--dbname", type=str, help="database name")
    parser.add_argument("--tgt_dir", type=str, default="data/database")
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directory (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.tgt_dir, args.dbname)):
        os.makedirs(os.path.join(args.tgt_dir, args.dbname))
    print(f'db_dir: {os.path.join(args.tgt_dir, args.dbname)}')
    wavscp = open(
        os.path.join(args.tgt_dir, args.dbname, "wav.scp"), "w", encoding="utf-8"
    )
    utt2spk = open(
        os.path.join(args.tgt_dir, args.dbname, "utt2spk"), "w", encoding="utf-8"
    )
    utt2num = open(
        os.path.join(args.tgt_dir, args.dbname, "utt2num_samples"), "w", encoding="utf-8"
    )
    
    for rt, dirs, files in os.walk(args.src_data):
        for file in files:
            if not file.endswith('.wav'):
                continue
            uid = file[:-4]
            cmd = (
                f"sox {os.path.join(rt, uid)}.wav -c 1 -t wavpcm -b 16 -r"
                f" {args.sr} {os.path.join(args.wav_dumpdir, uid)}.wav"
            )
            os.system(cmd)
            wavscp.write("{} {}.wav\n".format(uid, os.path.join(args.wav_dumpdir, uid)))
            utt2spk.write("{} {}\n".format(uid, args.dbname))
            y, sr = librosa.load(f'{os.path.join(args.wav_dumpdir, uid)}.wav', sr=args.sr)
            utt2num.write("{} {}\n".format(uid, len(y)))

