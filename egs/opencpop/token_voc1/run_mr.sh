#!/bin/bash

# Copyright 2024 Yuxun Tang
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=8      # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/hifigan_token_16k_nodp_f0.v1.yaml

# directory path setting
db_root=/data3/tyx/dataset/opencpop # direcotry including wavfiles (MODIFY BY YOURSELF)
                          # each wav filename in the directory should be unique
                          # e.g.
                          # /path/to/database
                          # ├── utt_1.wav
                          # ├── utt_2.wav
                          # │   ...
                          # └── utt_N.wav
dumpdir=dump           # directory to dump features
dumpdir_token=dump_token

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related setting
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

train_set="train"       # name of training data directory
dev_set="dev"           # name of development data direcotry
eval_set="test"         # name of evaluation data direcotry

token_text=""
use_f0=false                    # whether to add f0 
use_embedding_feats=true      # whether to use pretrain feature as input
pretrained_model="facebook/hubert-base-ls960"      # pre-trained model (confirm it on Huggingface)
use_multi_layer=true
emb_layer=6

fs=16000
subexp="exp"

store_feature=false             # store model intermediate representation
storedir=feat_store             # directory

use_cluster_token=true
num_threads=20      # number of cpu threads in learn_kmeans
nclusters=1024
cluster_dir=dump_cluster

use_multi_resolution=false
rs_list=

skip_score=true

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    if [ ! -e "${db_root}" ]; then
        echo "ERROR: Opencpop data does not exist."
    	echo "ERROR: Please download https://wenet.org.cn/opencpop/download/ and locate it at ${download_dir}"
    	exit 1
    fi
    echo "Please make sure fs=${fs} is right sample rate for model."
    mkdir -p wav_dump
    python local/data_prep.py ${db_root} \
        --wav_dumpdir wav_dump \
        --sr ${fs} \

    sort -o data/train/wav.scp data/train/wav.scp

    dev_num=50
    train_num=$(( $(wc -l < data/train/wav.scp) - dev_num ))

    mkdir -p data/${dev_set}
    head -n $train_num data/${train_set}/wav.scp > data/${train_set}/wav.scp.tmp
    tail -n $dev_num data/${train_set}/wav.scp > data/${dev_set}/wav.scp.tmp
    mv data/${dev_set}/wav.scp.tmp data/${dev_set}/wav.scp
    mv data/${train_set}/wav.scp.tmp data/${train_set}/wav.scp
    
    head -n $train_num data/${train_set}/utt2spk > data/${train_set}/utt2spk.tmp
    tail -n $dev_num data/${train_set}/utt2spk > data/${dev_set}/utt2spk.tmp
    mv data/${dev_set}/utt2spk.tmp data/${dev_set}/utt2spk
    mv data/${train_set}/utt2spk.tmp data/${train_set}/utt2spk

    head -n $train_num data/${train_set}/utt2num_samples > data/${train_set}/utt2num_samples.tmp
    tail -n $dev_num data/${train_set}/utt2num_samples > data/${dev_set}/utt2num_samples.tmp
    mv data/${dev_set}/utt2num_samples.tmp data/${dev_set}/utt2num_samples
    mv data/${train_set}/utt2num_samples.tmp data/${train_set}/utt2spk
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Embedding Feature extraction"
    # extract raw features
    pids=()
    # for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    for name in "${train_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"

        _opts=
        if [ ${use_f0} == true ]; then
            _opts+="--use-f0 "
        fi
        if [ ${use_multi_layer} == "true" ]; then
            _opts+="--use-multi-layer "
        fi
        if [ ${use_embedding_feats} == "true" ]; then
            _opts+="--use-embedding-feats "
            _opts+="--pretrained-model ${pretrained_model} "
            _opts+="--emb-layer ${emb_layer} "
        fi

        # preprocess embedding feature instead of token
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            local/preprocess_token.py \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --verbose "${verbose}" ${_opts}
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."
fi

if [ -z "${tag}" ]; then
    expdir="${subexp}/${train_set}_opencpop_$(basename "${conf}" .yaml)"
else
    expdir="${subexp}/${train_set}_opencpop_$(basename "${conf}" .yaml)_${tag}"
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training / Adatper training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m parallel_wavegan.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="parallel-wavegan-train"
    fi
    _opts=
    if [ ${use_f0} == true ]; then
        _opts+="--use-f0 "
    fi
    # shellcheck disable=SC2012
    resume="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/raw" \
            --dev-dumpdir "${dumpdir}/${dev_set}/raw" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}" ${_opts}
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding / Embedding feature extracting"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${train_set}"; do
    # for name in "${dev_set}" "${eval_set}" "${train_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        _opts=
        if [ ${use_f0} == true ]; then
            _opts+="--use-f0 "
        fi
        if [ ${store_feature} == true ]; then
            _opts+="--store-feature "
            _opts+="--storedir ${storedir}/${name} "
        fi
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${dumpdir}/${name}/raw" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}" ${_opts}      
        echo "Successfully finished decoding of ${name} set."
    ) & 
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ] && [ ${use_cluster_token} = true ]; then
    echo "Stage 4: Cluster embedding feature to obtain token feature (K-means)"
    sub_stage=1 
    sub_stop_stage=3
    feat_dir=$dumpdir

    if [ $use_multi_resolution == true ]; then
        #NOTE(Yuxun): Please check conf to get right resolution
        echo "NOTE: Please check conf file to get right resolution."
        src_rs=$(yq .generator_params.src_rs $conf)
        add_rs=$(yq .generator_params.add_rs $conf | jq -r '.[]' | paste -sd " ")
        IFS=" " read -r -a elements <<< "$add_rs"
        rs_list=$src_rs
        for element in "${elements[@]}"; do
            rs_list=("${rs_list[@]}" "$element")
        done
        echo "There are resolution (${rs_list[@]})"
    fi

    if [ ${sub_stage} -le 1 ] && [ ${sub_stop_stage} -ge 1 ]; then
        echo "sub-stage 1: Dump embedding feature"
        mkdir -p ${cluster_dir}/data

        if [ ${store_feature} == true ]; then
            feat_dir=$storedir
        fi

        for name in "${dev_set}" "${eval_set}" "${train_set}"; do
            mkdir -p ${cluster_dir}/data/${name}
            > "${cluster_dir}/data/${name}/feats.scp"
            > "${cluster_dir}/data/${name}/wav.scp"
            find $feat_dir/${name} -type f -name "*.h5" -exec sh -c '
                cluster_dir="$1"
                name="$2"
                shift 2 # file will include incoming paramters
                for file; do
                    filename=$(basename "$file" .h5)
                    echo "${filename} ${file}:feats" >> ${cluster_dir}/data/$name/feats.scp
                    echo "${filename} wav_dump/${filename}.wav" >> ${cluster_dir}/data/$name/wav.scp
                done
            ' sh "${cluster_dir}" "${name}" {} +

            if [ $use_multi_resolution == true ]; then
                for rs in "${rs_list[@]}"; do
                    awk -v rs="$rs" '{sub(":feats", ":feats-" rs)}1'  ${cluster_dir}/data/$name/feats.scp >  "${cluster_dir}/data/$name/feats_${rs}.scp"
                    echo "process $name/resolution($rs) done."
                done
            fi
        done
    fi

    skip_train=false
    km_dir="${cluster_dir}/km_dir/${nclusters}_store${store_feature}"
    portion=0.1

    if [ ${sub_stage} -le 2 ] && [ ${sub_stop_stage} -ge 2 ] && [ "$skip_train" == false ];then
        echo "sub-stage 2: Learn K-means with embedding feature based on scikit-learn"
        if [ $use_multi_resolution == true ]; then
            for rs in "${rs_list[@]}"; do
                echo "Learning K-means on resolution($rs)"
                rs_km_dir="${km_dir}_${rs}"
                mkdir -p $rs_km_dir
                _portion=${portion}
                _dset="${train_set}"
                if (( $(echo "${_portion} >= 1.0" | bc -l) )); then
                    cp ${cluster_dir}/data/${_dset}/feats_${rs}.scp $rs_km_dir/train.scp
                else
                    nutt=$(<"${cluster_dir}/data/${_dset}"/feats_${rs}.scp wc -l)
                    portion_nutt=$(echo ${nutt} ${_portion} | awk '{print(int($1 * $2)+1)}')

                    utils/subset_scp.pl \
                        ${portion_nutt} ${cluster_dir}/data/${_dset}/feats_${rs}.scp \
                        > "${rs_km_dir}/train.scp" || exit 1;
                    echo "Subsampling ${portion_nutt} utterances for Kmeans training."
                fi
                if [ -e "${rs_km_dir}/km_${nclusters}.mdl" ]; then
                    echo "Km model exists. Skip Training."
                else
                    # It typically requires 120GB RAM to run kmeans steps.
                    ${train_cmd} --num_threads ${num_threads} ${rs_km_dir}/log/learn_kmeans.log \
                    python3 utils/py_utils/learn_kmeans.py \
                        --km_path ${rs_km_dir}/km_${nclusters}.mdl \
                        --n_clusters ${nclusters} \
                        --percent -1 \
                        --in_filetype hdf5 \
                        "scp:${rs_km_dir}/train.scp" || exit 1;
                fi
            done
        else
            mkdir -p $km_dir
            _portion=${portion}
            _dset="${train_set}"
            if (( $(echo "${_portion} >= 1.0" | bc -l) )); then
                cp ${cluster_dir}/data/${_dset}/feats.scp $km_dir/train.scp
            else
                nutt=$(<"${cluster_dir}/data/${_dset}"/feats.scp wc -l)
                portion_nutt=$(echo ${nutt} ${_portion} | awk '{print(int($1 * $2)+1)}')

                utils/subset_scp.pl \
                    ${portion_nutt} ${cluster_dir}/data/${_dset}/feats.scp \
                    > "${km_dir}/train.scp" || exit 1;
                echo "Subsampling ${portion_nutt} utterances for Kmeans training."
            fi
            # It typically requires 120GB RAM to run kmeans steps.
            ${train_cmd} --num_threads ${num_threads} ${km_dir}/log/learn_kmeans.log \
            python3 utils/py_utils/learn_kmeans.py \
                --km_path ${km_dir}/km_${nclusters}.mdl \
                --n_clusters ${nclusters} \
                --percent -1 \
                --in_filetype hdf5 \
                "scp:${km_dir}/train.scp" || exit 1;
        fi
    fi

    mkdir -p ${cluster_dir}/rs_token
    tgt_text="${cluster_dir}/rs_token/pseudo_labels_km${nclusters}.txt"

    if [ ${sub_stage} -le 3 ] && [ ${sub_stop_stage} -ge 3 ]; then
        echo "sub-stage 3: Generate K-means pseudo-labels"
        use_gpu=true
        if ${use_gpu}; then
            _cmd="${cuda_cmd} --gpu 1"
        else
            _cmd="${cpu_cmd}"
        fi

        if [ $use_multi_resolution == true ]; then
            for rs in "${rs_list[@]}"; do
                echo "Generating lables on resolution($rs)"
                rs_km_dir="${km_dir}_${rs}"
                rs_tgt_text="${tgt_text}_${rs}"
                > "${rs_tgt_text}"
                for dset in "${dev_set}" "${eval_set}" "${train_set}"; do
                # for dset in "${dev_set}"; do
                    echo "Extract labels to ${cluster_dir}/data/$dset"
                    _dump_dir=${cluster_dir}/data/$dset

                    _opts=
                    _opts+="--in_filetype hdf5 "
                    mkdir -p "${_dump_dir}"/logdir

                    key="feats_${rs}.scp"
                    _nj=4
                    nutt=$(<"${_dump_dir}"/${key} wc -l)
                    _nj=$((_nj<nutt?_nj:nutt))

                    key_file="${_dump_dir}"/${key}
                    split_scps=""
                    for n in $(seq ${_nj}); do
                        split_scps+=" ${_dump_dir}/logdir/inference_kmeans_${rs}.${n}.scp"
                    done
                    utils/split_scp.pl "${key_file}" ${split_scps}
                    for n in $(seq ${_nj}); do
                        awk '(FILENAME==ARGV[1]){utt2num[$1]=$2} (FILENAME==ARGV[2]){print($1, utt2num[$1])}' \
                            data/${dset}/utt2num_samples ${_dump_dir}/logdir/inference_kmeans_${rs}.${n}.scp \
                            > ${_dump_dir}/logdir/utt2num_samples.${n}
                    done

                    ${_cmd} JOB=1:${_nj} "${_dump_dir}"/logdir/inference_pseudo_labels_km${nclusters}_${rs}.JOB.log \
                        python3 utils/py_utils/dump_km_label.py \
                            ${_opts} \
                            --km_path "${rs_km_dir}/km_${nclusters}.mdl" \
                            --out_filetype "mat" \
                            --use_gpu ${use_gpu} \
                            --utt2num_samples "${_dump_dir}/logdir/utt2num_samples.JOB" \
                            "scp:${_dump_dir}/logdir/inference_kmeans_${rs}.JOB.scp" \
                            "ark,t:${_dump_dir}/logdir/pseudo_labels_km${nclusters}_${rs}.JOB.txt" || exit 1;

                    for n in $(seq ${_nj}); do
                        cat "${_dump_dir}"/logdir/pseudo_labels_km${nclusters}_${rs}.${n}.txt || exit 1;
                    done | sed 's/ \[ \| \]//g' | sort -u > "${_dump_dir}"/pseudo_labels_km${nclusters}_${rs}.txt || exit 1;
                    cat "${_dump_dir}"/pseudo_labels_km${nclusters}_${rs}.txt >> "${rs_tgt_text}"
                done
            done
        else
            > $tgt_text
            for dset in "${dev_set}"; do
            # for dset in "${dev_set}" "${eval_set}" "${train_set}"; do
                echo "Extract labels to ${cluster_dir}/data/$dset"
                _dump_dir=${cluster_dir}/data/$dset

                _opts=
                _opts+="--in_filetype hdf5 "
                mkdir -p "${_dump_dir}"/logdir

                key="feats.scp"
                _nj=4
                nutt=$(<"${_dump_dir}"/${key} wc -l)
                _nj=$((_nj<nutt?_nj:nutt))

                key_file="${_dump_dir}"/${key}
                split_scps=""
                for n in $(seq ${_nj}); do
                    split_scps+=" ${_dump_dir}/logdir/inference_kmeans.${n}.scp"
                done
                utils/split_scp.pl "${key_file}" ${split_scps}
                for n in $(seq ${_nj}); do
                    awk '(FILENAME==ARGV[1]){utt2num[$1]=$2} (FILENAME==ARGV[2]){print($1, utt2num[$1])}' \
                        data/${dset}/utt2num_samples ${_dump_dir}/logdir/inference_kmeans.${n}.scp \
                        > ${_dump_dir}/logdir/utt2num_samples.${n}
                done

                ${_cmd} JOB=1:${_nj} "${_dump_dir}"/logdir/inference_pseudo_labels_km${nclusters}.JOB.log \
                    python3 utils/py_utils/dump_km_label.py \
                        ${_opts} \
                        --km_path "${km_dir}/km_${nclusters}.mdl" \
                        --out_filetype "mat" \
                        --use_gpu ${use_gpu} \
                        --utt2num_samples "${_dump_dir}/logdir/utt2num_samples.JOB" \
                        "scp:${_dump_dir}/logdir/inference_kmeans.JOB.scp" \
                        "ark,t:${_dump_dir}/logdir/pseudo_labels_km${nclusters}.JOB.txt" || exit 1;
            done
            
            for dset in "${dev_set}" "${eval_set}" "${train_set}"; do
                for n in $(seq ${_nj}); do
                    cat "${_dump_dir}"/logdir/pseudo_labels_km${nclusters}.${n}.txt || exit 1;
                done | sed 's/ \[ \| \]//g' | sort -u > "${_dump_dir}"/pseudo_labels_km${nclusters}.txt || exit 1;
                cat "${_dump_dir}"/pseudo_labels_km${nclusters}.txt >> "${cluster_dir}/pseudo_labels_km${nclusters}.txt"
            done
        fi
    fi
    echo "finished  training and preprocessing for Adapter. Genereated token at ${cluster_dir}/pseudo_labels_km${nclusters}.txt"
else
    echo "Skip stage for clustering token"
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ] && [ $skip_score == false ]; then
    echo "Stage 5: Scoring Model / Adapter"
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    for dset in ${eval_set}; do
        _data="data/${dset}"
        _gt_wavscp="${_data}/wav.scp"
        _dir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
        _gen_wavdir="${_dir}/${dset}"

        # Objective Evaluation - MCD
        echo "Begin Scoring for MCD metrics on ${dset}, results are written under ${_dir}/MCD_res"

        mkdir -p "${_dir}/MCD_res"
        python utils/py_utils/evaluate_mcd.py \
            ${_gen_wavdir} \
            ${_gt_wavscp} \
            --outdir "${_dir}/MCD_res"

        # Objective Evaluation - log-F0 RMSE
        echo "Begin Scoring for F0 related metrics on ${dset}, results are written under ${_dir}/F0_res"

        mkdir -p "${_dir}/F0_res"
        python utils/py_utils/evaluate_f0.py \
            ${_gen_wavdir} \
            ${_gt_wavscp} \
            --outdir "${_dir}/F0_res"

        # # Objective Evaluation - semitone ACC
        # echo "Begin Scoring for SEMITONE related metrics on ${dset}, results are written under ${_dir}/SEMITONE_res"

        # mkdir -p "${_dir}/SEMITONE_res"
        # python utils/py_utils/evaluate_semitone.py \
        #     ${_gen_wavdir} \
        #     ${_gt_wavscp} \
        #     --outdir "${_dir}/SEMITONE_res"

        #     # Objective Evaluation - VUV error
        # echo "Begin Scoring for VUV related metrics on ${dset}, results are written under ${_dir}/VUV_res"

        # mkdir -p "${_dir}/VUV_res"
        # python utils/py_utils/evaluate_vuv.py \
        #     ${_gen_wavdir} \
        #     ${_gt_wavscp} \
        #     --outdir "${_dir}/VUV_res"

    done
else
    echo "Skip the evaluation stages"
fi

echo "Finished"
