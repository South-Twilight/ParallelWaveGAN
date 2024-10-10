# SingOMD: Singing Oriented Multi-resolution Discrete Representation Construction from Speech Models

### How to Run?

In `prep_singomd.sh`, we will firstly train the adapter model (called `am1` for convenience) using embedding feature from `pretrained_model`.

Then, we will extract multi-resolution embedding features from `am1` model and conduct a cluster over these multi_resoltuion embedding to get singing-oriented multi-resolution discrete (**SingOMD**) token.

Finally, the extracted SingOMD tokens will be tested.

More details about it can be found in our paper https://arxiv.org/abs/2406.08905

Details of each stage in `run.sh` are shown below.

- Stage 0: prepare combined dataset

- Stage 1: extract embediding features from combined dataset and store them

- Stage 2: train adapter model 'am1`

- Stage 3: extract multi-resoltuion embedding from `am1`

- Stage 4: Cluster over multi-resolution embedding feature to obtain **SingOMD** token.

- Stage 5: test over `run.sh` (Please check the setting)

## Run Command

To get multi resolution discrete tokens, you should first train a model to generate multi resolution embedding features.

```sh
# This is an example to generate features in [20, 40, 80] with hubert-base whose is 20ms.
./prep_singomd.sh \
--stage 4 --stop_stage 4 \
--dumpdir dump_embedding \
--conf conf/hifigan_embedding_16k_nodp_mr.v1.yaml \
--n_gpus 1  --n_jobs 8 \
--subexp exp \
--use_f0 false \
--use_embedding_feats true \
--use_multi_layer true --emb_layer 12 \
--use_cluster_token true --nclusters 1024 --portion 1.0 --feature_type mr3 \
--use_multi_resolution true \
--store_feature true --storedir store/mr3 \
--pretrained_model facebook/hubert-base-ls960
```

After this process, we will get generated multi resolution discrete token at `dump_cluster/mr3/rs_token/*`. The file `pseudo_labels_km1024_emb.txt_20` represents that resolution of tokens in this file is 20ms.

Then, just run the unit hifigan to synthesis singing waveform from discrets tokens.

```sh
./run.sh \
--stage 1 \
--dump dump \
--token_file dump_cluster/mr3/rs_token \
--conf conf/hifigan_token_16k_nodp_mr.v1.yaml \
--n_gpus 1  --n_jobs 8 \
--subexp exp \
--use_f0 false \
--use_embedding_feats false \
--use_multi_resolution true 
```