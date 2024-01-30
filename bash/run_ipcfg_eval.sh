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
seed=1213 # 4546 # 2324 #

echo "GPUs: "$CUDA_VISIBLE_DEVICES "#"$ngpu "PORT: "$port "NPROC: "$num_proc

#TAMING=/net/nfs2.mosaic/yann/code/taming-transformers
#export PYTHONPATH=$TAMING:$PYTHONPATH
#echo $PYTHONPATH

root=/mnt # /mnt3

alias_root="$root/yann/model/ipcfg"
model_root="$root/yann/model/ipcfg"

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
data_root=/mnt/yann/data/dallemini/$vq_time"_"$vq_name/$model_sign/$text_fold

data_name="train/\{001..007\}.parquet"
eval_name="val/\{001..002\}.parquet"

num_t=32
num_nt=24

model_name=o1to1_20e_08b_lr1e2_5e3w_2e4d_1e0r_kl1e0_5e3w_bh0e0_sh0e0
model_file=00025000.pth
#  

model_name=o1to1_20e_08b_lr1e2_5e3w_2e4d_1e0r_kl1e0_5e3w_bh0e0_sh0e0_labrmecu
model_file=0000075x.pth
#  

alias_name=ipcfg_eval
logname=$model_name
logroot="$root/yann/model/log"

mtask="verbose=True monitor=DalleMonitor worker=IPCFG check_unused_param=false

alias_name=$alias_name model_name=$model_name model_file=$model_file 

model.vq.model_root=$vq_root model.vq.model_time=$vq_time model.vq.model_name=$vq_name model.vq.load=false
model.vq.model_file=$vq_file model.vq.vocab_size=$vocab_size model.vq.max_vis_len=$max_vis_len model.vq.gumbel=$gumbel
data.data_root=$data_root data.more_root=$more_root

data.data_name=$data_name data.eval_name=$eval_name
+data.use_preencoded_pandas=true +data.require_txt=false
data.txt_vocab_name=txt_vocab_is_null

alias_odir=rnd_cky_xxx
data.min_obj_num=1 data.max_obj_num=1

model.decoder.NT=$num_nt model.decoder.T=$num_t

model/encoder=dummy +model.encoder.z_dim=0 
model.decoder.NT=4 model.decoder.T=2 model.decoder.s_dim=128 data.vis_vocab_size=2 +data.vis_fake_hw=2

running.infer_mode=true
running.v_peep_time=1
running.v_peep_topk=-1
running.topk_best=5

+data.txt_special_token={bos:\"<TXT|BOS>\",pad:\"<PAD>\",unk:\"<UNK>\"}
+data.vis_special_token={bos:\"<VIS|BOS>\",unk:\"<UNK>\"}

data.min_train_vid=0 data.max_train_vid=7
data.min_eval_vid=1000 data.max_eval_vid=1499

running.epochs=1 running.batch_size=8 running.peep_rate=1 running.save_rate=1e10 running.save_last=True running.save_epoch=True
data.train_samples=null data.eval_samples=[0,64] data.test_samples=null
"


#+data.rot90_image=0e0
#+data.object_name=\"(large\sbrown\smetal\scube)\"

#model/encoder=latent_slot model.encoder.num_slot=0

#running.epochs=1 running.batch_size=1 running.peep_rate=1 running.save_rate=1e10 running.save_last=True running.save_epoch=True
#data.train_samples=[0,16] data.eval_samples=[0,2] data.test_samples=null

#running.epochs=20 running.batch_size=8 running.peep_rate=250 running.save_rate=1e10 running.save_epoch=True
#data.train_samples=null data.eval_samples=null data.test_samples=null

#optimizer.warmup_steps=5e3 optimizer.decay_step=1.5e4 optimizer.decay_rate=2.5e-1

#running.epochs=1000 running.batch_size=256 running.peep_rate=100 running.save_rate=200 running.save_last=True
#data.train_samples=null data.eval_samples=null data.test_samples=null


#echo "exit..."
#exit 0

# config
extra="$mtask "
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup 
python train.py port=$port num_gpus=$ngpu eval=true mode=$mode num_proc=$num_proc seed=$seed \
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
