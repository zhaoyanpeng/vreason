#!/usr/bin/bash

#export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=$2

run_type=$1
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })

port=$(expr $RANDOM + 1000)
ngpu=${#gpu_list[@]}

mode="ddp"
autocast=false
num_proc=1
seed=8765 #8765 #7879 #6768 #5657 #4546 #2324 #1213 #3435 #22 #9876

echo "GPUs: "$CUDA_VISIBLE_DEVICES "#"$ngpu "PORT: "$port "NPROC: "$num_proc

root=/data/cl/user/bailinw/vreason # /mnt3

expid=7
alias_root="$root/exp/ipcfg-run$expid"
model_root="$root/exp/ipcfg-run$expid"

text_fold=CAPTION_4.0_25_captions
data_root=/mnt/yann/data/CLEVR_text/$text_fold
more_root=/mnt/yann/data/CLEVR_v1.0_320x320_bbox

# pre-trained VQGAN 
vq_root=/mnt/yann/model/vqgan_tuned/logs

# 256 x 16384 
vq_time=2022-09-10T00-25-13
vq_name=clevr_vqgan_128e_16384
vq_file=last.ckpt
vocab_size=16384
max_vis_len=256
gumbel=False

# 256 x 1024 
vq_time=2022-09-10T05-13-14
vq_name=clevr_vqgan_128e_1024
vq_file=last.ckpt
vocab_size=1024
max_vis_len=256
gumbel=False

# use pre-encoded data
model_sign=(${vq_file//./ }[0])
# data_root=/mnt/yann/data/dallemini/$vq_time"_"$vq_name/$model_sign/$text_fold
data_root=$root/exp/$vq_time"_"$vq_name/$model_sign/$text_fold

data_name="train/\{001..007\}.parquet"
eval_name="val/\{001..002\}.parquet"

num_nt=6
num_t=3

model_name=ipcfg_test_oneset
logname=$model_name
logroot="$root/exp/ipcfg-run/log"

mtask="verbose=True monitor=DalleMonitor worker=IPCFG alias_name=$model_name check_unused_param=false

model.vq.model_root=$vq_root model.vq.model_time=$vq_time model.vq.model_name=$vq_name model.vq.load=false
model.vq.model_file=$vq_file model.vq.vocab_size=$vocab_size model.vq.max_vis_len=$max_vis_len model.vq.gumbel=$gumbel
data.data_root=$data_root data.more_root=$more_root

data.data_name=$data_name data.eval_name=$eval_name
+data.use_preencoded_pandas=true +data.require_txt=false +data.rot90_image=0e0 
data.txt_vocab_name=txt_vocab_is_null

data.max_obj_num=1
data.min_train_vid=0 data.max_train_vid=9999
data.min_eval_vid=0 data.max_eval_vid=999

model.decoder.name=IPCFG2DDecHead
model.decoder.nt_la=false
model/encoder=dummy +model.encoder.z_dim=0 
model.decoder.NT=$num_nt model.decoder.T=$num_t 
model.decoder.n_set=2
model.decoder.s_dim=256

data.vis_vocab_size=3 
+data.vis_fake_hw=[6,6] +data.obj_fake_hw=[[4,1],[1,3]] +data.obj_delt_hw=[] model.decoder.grid_size=[6,6]

model.loss.mini_1d_ll=false model.loss.mini_1d_2d=false

model.decoder.drop_1d=false model.decoder.beta_1d=0.1 model.decoder.rate_1d=5.
model.decoder.drop_2d=false model.decoder.beta_2d=0.1 model.decoder.rate_2d=3.

model.loss.bh_beta=0 model.loss.sh_beta=1 model.loss.th_beta=0

model.loss.kl_cycle_steps=2.5e2 model.loss.kl_cycle=false model.loss.kl_max_beta=1e0

optimizer.warmup_steps=2.5e2 optimizer.decay_step=1e3 optimizer.decay_rate=1e0
optimizer.lr=3e-3 optimizer.warmup_fn=linear optimizer.betas=[0.9,0.75] optimizer.max_gnorm=3 optimizer.weight_decay=1e-7

running.v_peep_time=1
running.v_peep_topk=-1
running.topk_best=3

+data.txt_special_token={bos:\"<TXT|BOS>\",pad:\"<PAD>\",unk:\"<UNK>\"}
+data.vis_special_token={bos:\"<VIS|BOS>\",unk:\"<UNK>\"}

running.epochs=60 running.batch_size=64 running.peep_rate=32 running.save_rate=1e10 running.save_epoch=True
data.train_samples=[0,1024] data.eval_samples=[0,512] data.test_samples=null
"

extra="$mtask "
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup 
python train.py port=$port num_gpus=$ngpu eval=False mode=$mode num_proc=$num_proc seed=$seed \
    autocast=$autocast alias_root=$alias_root model_root=$model_root \
    +model/vq=default \
    +model/embedder=dummy \
    +model/encoder=latent_none \
    +model/decoder=ipcfg \
    +model/loss=ipcfg \
    +optimizer=ipcfg \
    +data=clevr_dalle \
    +running=$run_type $extra 
#> ./log/$logname 2>&1 &
