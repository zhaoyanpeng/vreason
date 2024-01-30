#!/usr/bin/bash

#export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=$2

run_type=$1
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })

port=$(expr $RANDOM + 1000)
ngpu=${#gpu_list[@]}

mode="dp"
num_proc=0
seed=2324 #1213

echo "GPUs: "$CUDA_VISIBLE_DEVICES "#"$ngpu "PORT: "$port

#TAMING=/net/nfs2.mosaic/yann/code/taming-transformers
#export PYTHONPATH=$TAMING:$PYTHONPATH
#echo $PYTHONPATH


alias_root="/mnt/yann/model/dalle-clevr"
model_root="/mnt/yann/model/dalle-clevr"

data_root=/mnt/yann/data/CLEVR_text/CAPTION_100_captions
more_root=/mnt/yann/data/CLEVR_bbox_320x320

vqgan_root=/mnt/yann/model/vqgan_tuned/logs

# pre-trained GumbelVQ
vqgan_time=2022-07-05T02-53-25
vqgan_name=clevr_gumbel_1024x8192
vqgan_file=epoch168.ckpt
vocab_size=8192
max_vis_len=1024
gumbel=True

# pre-trained VQGAN
vqgan_time=2022-06-23T07-38-52
vqgan_name=clevr_vqgan_1024
vqgan_file=epoch299.ckpt
vocab_size=1024
max_vis_len=256
gumbel=False

model_name=igpt_test
logname=$model_name

mtask="verbose=True monitor=DalleMonitor worker=IPCFG alias_name=$model_name

model.vq.model_root=$vqgan_root model.vq.model_time=$vqgan_time model.vq.model_name=$vqgan_name 
model.vq.model_file=$vqgan_file model.vq.vocab_size=$vocab_size model.vq.max_vis_len=$max_vis_len model.vq.gumbel=$gumbel
data.data_root=$data_root data.more_root=$more_root
data.vis_only=true data.max_txt_len=1 data.num_txt_sample=0

optimizer.warmup=True optimizer.warmup_steps=5e3 optimizer.batch_sch=True
optimizer.lr=1e-3 optimizer.decay_step=1e4 optimizer.decay_rate=2.5e-1
optimizer.betas=[0.9,0.75] optimizer.max_gnorm=3 optimizer.weight_decay=1e-6

optimizer.warmup=False optimizer.scheduler=[]

+data.txt_special_token={bos:\"<TXT|BOS>\",pad:\"<PAD>\",unk:\"<UNK>\"}
+data.vis_special_token={bos:\"<VIS|BOS>\",unk:\"<UNK>\"}

running.epochs=1 running.batch_size=2 running.peep_rate=1 running.save_rate=1e10 running.save_epoch=True
data.train_samples=[0,2] data.eval_samples=[5000,5004] data.test_samples=null
data.txt_vocab_name=txt_vocab_is_null
"

#data.data_name=\"caption_val.all.json\"

#echo "exit..."
#exit 0

# config
extra="$mtask "
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup 
python train.py port=$port num_gpus=$ngpu eval=False mode=$mode num_proc=$num_proc seed=$seed \
    alias_root=$alias_root model_root=$model_root \
    +model/vq=default \
    +model/embedder=dummy \
    +model/encoder=dummy \
    +model/decoder=ipcfg \
    +model/loss=ipcfg \
    +optimizer=dalle \
    +data=clevr_dalle \
    +running=$run_type $extra 
#> ./log/$logname 2>&1 &
