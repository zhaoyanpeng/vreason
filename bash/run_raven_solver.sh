#!/usr/bin/bash

#export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=$2

run_type=$1
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })

port=$(expr $RANDOM + 1000)
ngpu=${#gpu_list[@]}

mode="dp"
num_proc=4
seed=1213

echo "GPUs: "$CUDA_VISIBLE_DEVICES "#"$ngpu "PORT: "$port

TAMING=/net/nfs2.mosaic/yann/code/taming-transformers
export PYTHONPATH=$TAMING:$PYTHONPATH
echo $PYTHONPATH


alias_root="/home/yanpengz/model/raven"
model_root="/home/yanpengz/model/vqgan/logs"

data_root=/home/yanpengz/data/raven

vqgan_time=2022-02-16T14-39-50
vqgan_name=raven_all
vqgan_file=last.ckpt

vqgan_time=2022-03-14T05-00-45
vqgan_name=raven_distribute_nine
vqgan_file=epoch161.ckpt

data_name=$vqgan_time"_"$vqgan_name


tasks="[center_single,distribute_four,distribute_nine,in_center_single_out_center_single,in_distribute_four_out_center_single,left_center_single_right_center_single,up_center_single_down_center_single]"
tasks="[distribute_nine]"

model_name=first_all

model_name=dup_dist_nine_old
model_name=dup_dist_nine_new
model_name=test
logname=$model_name

#mtask="verbose=True monitor=RavenSolverMonitor worker=RavenSolver alias_name=$model_name 
mtask="verbose=True monitor=RavenSolverMonitor worker=NeoRavenSolver alias_name=$model_name 
model.vqgan.model_root=$model_root model.vqgan.model_time=$vqgan_time model.vqgan.model_name=$vqgan_name 
model.vqgan.model_file=$vqgan_file data.name=$data_name data.data_root=$data_root data.tasks=$tasks


model.embedder.mode=enc_dec
model.embedder.aug_prob=0.5
model.embedder.name=NeoTokenEncHead
model.decoder.name=NeoTorchTFDecHead 
model.encoder.num_layer=2 
model.decoder.num_layer=2


optimizer.warmup=True optimizer.warmup_steps=1000 optimizer.warmup_times=20 
optimizer.lr=1e-4 optimizer.scheduler=[MultiStepLR,{milestones:[15,36,45,50,100],gamma:0.5}]

running.epochs=1000 running.batch_size=32 running.peep_rate=50 running.save_rate=1e9 running.save_epoch=True 
data.train_samples=1e9 data.eval_samples=1e9 data.test_samples=0

optimizer.warmup_steps=16 running.batch_size=4 running.epochs=1 running.peep_rate=1 data.train_samples=64 data.eval_samples=0 data.test_samples=0

"

#optimizer.warmup_steps=16 running.batch_size=4 running.epochs=1 running.peep_rate=1 data.train_samples=128 data.eval_samples=32 data.test_samples=32

#echo "exit..."
#exit 0

# config
extra="$mtask "
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup 
python train.py port=$port num_gpus=$ngpu eval=False mode=$mode num_proc=$num_proc seed=$seed \
    alias_root=$alias_root model_root=$model_root \
    +model/vqgan=default \
    +model/embedder=enc_dec \
    +model/encoder=torch_tf \
    +model/decoder=torch_tf \
    +model/loss=ce_lm \
    +optimizer=raven \
    +data=raven \
    +running=$run_type $extra 
#> ./log/$logname 2>&1 &
