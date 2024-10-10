#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

sub_stage=1 
sub_stop_stage=1

use_multi_resolution=false
skip_train=false


conf=conf/train.yaml

featdir=dump
feature_type=emb
cluster_dir=dump_cluster/emb
nclusters=1024
km_dir=${cluster_dir}/km_dir

portion=1.0
njobs=4
num_threads=20      # number of cpu threads in learn_kmeans

train_set="train"
dev_set="dev"
eval_set="test"

log "$0 $*"
. utils/parse_options.sh

echo $portion

cluster_dir=dump_cluster/${feature_type}
km_dir="${cluster_dir}/km_dir/${nclusters}_${portion}"

# dump_cluster
# |-- model
#       |-- data
#       |-- km_dir
#       |-- rs_token

if [ $use_multi_resolution == true ]; then
    #NOTE(Yuxun): Please check conf to get right resolution
    log "NOTE: Please check conf file to get right resolution."
    src_rs=$(yq .generator_params.src_rs $conf)
    add_rs=$(yq .generator_params.add_rs $conf | jq -r '.[]' | paste -sd " ")
    IFS=" " read -r -a elements <<< "$add_rs"
    rs_list=$src_rs
    for element in "${elements[@]}"; do
        rs_list=("${rs_list[@]}" "$element")
    done
    log "There are resolution (${rs_list[@]})"
fi

if [ ${sub_stage} -le 1 ] && [ ${sub_stop_stage} -ge 1 ]; then
    log "sub-stage 1: Dump embedding feature from ${featdir}"
    
    mkdir -p ${cluster_dir}/data

    for name in "${dev_set}" "${eval_set}" "${train_set}"; do
        mkdir -p ${cluster_dir}/data/${name}
        > "${cluster_dir}/data/${name}/feats.scp"
        > "${cluster_dir}/data/${name}/wav.scp"

        find "$featdir/${name}" -type f -name "*.h5" -exec sh -c '
            cluster_dir="$1"
            name="$2"
            shift 2 # file will include 2 paramters inputted
            for file; do
                filename=$(basename "$file" .h5)
                echo "${filename} ${file}:feats" >> ${cluster_dir}/data/$name/feats.scp
                echo "${filename} wav_dump/${filename}.wav" >> ${cluster_dir}/data/$name/wav.scp
            done
        ' sh "${cluster_dir}" "${name}" {} +

        if [ $use_multi_resolution == true ]; then
            for rs in "${rs_list[@]}"; do
                awk -v rs="$rs" '{sub(":feats", ":feats-" rs)}1'  ${cluster_dir}/data/$name/feats.scp >  "${cluster_dir}/data/$name/feats_${rs}.scp"
                log "process $name/resolution($rs) done."
            done
        fi
    done
fi


if [ ${sub_stage} -le 2 ] && [ ${sub_stop_stage} -ge 2 ] && [ "$skip_train" == false ];then
    log "sub-stage 2: Learn K-means with embedding feature based on scikit-learn"

    if [ $use_multi_resolution == true ]; then

        for rs in "${rs_list[@]}"; do
            log "Learning K-means on resolution($rs)"

            rs_km_dir="${km_dir}_${rs}"
            mkdir -p $rs_km_dir
            
            _logdir="${rs_km_dir}/logdir"
            mkdir -p ${_logdir}

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
                log "Subsampling ${portion_nutt} utterances for Kmeans training."
            fi
            if [ -e "${rs_km_dir}/km_${nclusters}.mdl" ]; then
                log "Km model exists. Skip Training."
            else
                # It typically requires 120GB RAM to run kmeans steps.
                ${train_cmd} --num_threads ${num_threads} ${_logdir}/learn_kmeans.log \
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
        
        _logdir="${km_dir}/logdir"
        mkdir -p ${_logdir}

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
            log "Subsampling ${portion_nutt} utterances for Kmeans training."
        fi

        # It typically requires 120GB RAM to run kmeans steps.
        ${train_cmd} --num_threads ${num_threads} ${_logdir}/learn_kmeans.log \
        python3 utils/py_utils/learn_kmeans.py \
            --km_path ${km_dir}/km_${nclusters}.mdl \
            --n_clusters ${nclusters} \
            --percent -1 \
            --in_filetype hdf5 \
            "scp:${km_dir}/train.scp" || exit 1;
    fi
fi

mkdir -p ${cluster_dir}/rs_token
tgt_text="${cluster_dir}/rs_token/pseudo_labels_km${nclusters}_${feature_type}.txt"

if [ ${sub_stage} -le 3 ] && [ ${sub_stop_stage} -ge 3 ]; then
    log "sub-stage 3: Generate K-means pseudo-labels"

    use_gpu=true
    if ${use_gpu}; then
        _cmd="${cuda_cmd} --gpu 1"
    else
        _cmd="${cpu_cmd}"
    fi

    if [ $use_multi_resolution == true ]; then
        for rs in "${rs_list[@]}"; do
            log "Generating lables on resolution($rs)"

            rs_km_dir="${km_dir}_${rs}"
            rs_tgt_text="${tgt_text}_${rs}"

            > "${rs_tgt_text}"

            for dset in "${dev_set}" "${eval_set}" "${train_set}"; do
            # for dset in "${dev_set}"; do
                echo "Extract labels to ${cluster_dir}/data/$dset"
                _dump_dir=${cluster_dir}/data/$dset

                _opts=

                _opts+="--in_filetype hdf5 "
                key="feats_${rs}.scp"
                nutt=$(<"${_dump_dir}"/${key} wc -l)
                _nj=$((njobs<nutt?njobs:nutt))
                
                mkdir -p "${_dump_dir}"/logdir

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

        # for dset in "${dev_set}"; do
        for dset in "${dev_set}" "${eval_set}" "${train_set}"; do
            log "Extract labels to ${cluster_dir}/data/$dset"

            _dump_dir=${cluster_dir}/data/$dset

            _opts=

            _opts+="--in_filetype hdf5 "
            key="feats.scp"
            nutt=$(<"${_dump_dir}"/${key} wc -l)
            _nj=$((njobs<nutt?njobs:nutt))
            
            mkdir -p "${_dump_dir}"/logdir

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

            for n in $(seq ${_nj}); do
                cat "${_dump_dir}"/logdir/pseudo_labels_km${nclusters}.${n}.txt || exit 1;
            done | sed 's/ \[ \| \]//g' | sort -u > "${_dump_dir}"/pseudo_labels_km${nclusters}.txt || exit 1;
            cat "${_dump_dir}"/pseudo_labels_km${nclusters}.txt >> "${tgt_text}"
        done
    fi

    echo "finished  training and preprocessing for Adapter. Genereated token at $tgt_text."
fi

