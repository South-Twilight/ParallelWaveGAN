#!/bin/bash

# 设置变量
MAX_TRIES=1000  # 最大尝试次数

# 循环执行直到正常退出或达到最大尝试次数
tries=0
while true; do
    echo "运行第 $((tries+1)) 次..."
    CUDA_VISIBLE_DEVICES=6 ./run.sh --stage 1 --stop_stage 1 --dumpdir dump_embedding/dump_hubert-base_embedding_weightsum_nof0 --tag hifigan_hubert-base_embedding_weightsum_nof0 --conf conf/tuning/embedding_sum_16k_nodp.yaml --n_gpus 1  --n_jobs 8 --subexp exp/hubert-base --use_f0 false --use_multi_layer true --emb_layer 12 --use_embedding_feats true --pretrained_model /data3/tyx/.cache/huggingface/hub/models--facebook--hubert-base-ls960/snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a
    exit_code=$?  # 获取退出码

    if [ $exit_code -eq 0 ]; then
        echo "程序正常退出"
        break
    elif [ $exit_code -eq 2 ]; then
        echo "CUDA out of memory，重试中..."
    fi

    tries=$((tries+1))
    if [ $tries -eq $MAX_TRIES ]; then
        echo "达到最大尝试次数，退出"
        exit 1
    fi
done
