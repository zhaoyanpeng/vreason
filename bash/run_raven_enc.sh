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


alias_root="/net/nfs2.mosaic/yann/model/vqgan"
model_root="/home/yanpengz/model/vqgan/logs"

data_root=/home/yanpengz/data/raven

model_time=2022-03-14T05-00-45
model_name=raven_distribute_nine
model_file=epoch161.ckpt

data_name=$vqgan_time"_"$vqgan_name

tasks="[center_single,distribute_four,in_center_single_out_center_single,in_distribute_four_out_center_single,left_center_single_right_center_single,up_center_single_down_center_single]"
tasks="[distribute_nine]"

splits="[val]"
splits="[train,val,test]"
logname="all_$model_name"

mtask="verbose=True monitor=RavenVQEncMonitor worker=VQGANEncoder 
data.data_root=$data_root data.splits=$splits data.tasks=$tasks

data.save_root=$data_root/$data_name

running.peep_rate=100 data.eval_samples=1e9 data.test_samples=1e9
"

# config
extra="$mtask "
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup 
python train.py port=$port num_gpus=$ngpu eval=True mode=$mode num_proc=$num_proc seed=$seed \
    alias_root=$alias_root model_root=$model_root model_name=$model_name model_time=$model_time model_file=$model_file \
    +data=raven \
    +running=$run_type $extra 
#> ./log/$logname 2>&1 &
