#!/usr/bin/bash

#export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=$2

run_type=$1
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })

port=$(expr $RANDOM + 1000)
ngpu=${#gpu_list[@]}

mode="dp"
num_proc=4
seed=2324 #1213

echo "GPUs: "$CUDA_VISIBLE_DEVICES "#"$ngpu "PORT: "$port

#TAMING=/net/nfs2.mosaic/yann/code/taming-transformers
#export PYTHONPATH=$TAMING:$PYTHONPATH
#echo $PYTHONPATH


alias_root="/home/s1847450/model/slotattn"
model_root="/home/s1847450/model/slotattn"

data_root="/home/s1847450/data/AbstractScenes_v1.1"

model_name=test
model_name=scene_test
model_name=scene_test_standard
model_name=scene_standard_seed-$seed
logname=$model_name

mtask="verbose=True monitor=SlotLearnerMonitor worker=SlotLearner alias_name=$model_name

model.encoder.num_slot=7

data.data_root=$data_root data.max_num_obj=7

optimizer.max_gnorm=null optimizer.weight_decay=0
optimizer.warmup=False optimizer.batch_sch=True optimizer.sch_name=SlotattnLR

running.epochs=5000 running.batch_size=32 running.peep_rate=50 running.save_rate=1e10 running.save_epoch=True
"

#optimizer.warmup=True optimizer.warmup_steps=2e3 optimizer.batch_sch=True
#optimizer.max_gnorm=null optimizer.decay_step=2e4 optimizer.weight_decay=0


#optimizer.lr=4e-4 optimizer.scheduler=[ExpDecayLR,{step_size:5,gamma:0.5}]

#running.epochs=3 running.batch_size=32 running.peep_rate=1 running.save_rate=1e10 running.save_epoch=True
#data.data_name=[0,8] data.eval_name=[8,16] data.test_name=null


#echo "exit..."
#exit 0

# config
extra="$mtask "
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup 
python train.py port=$port num_gpus=$ngpu eval=False mode=$mode num_proc=$num_proc seed=$seed \
    alias_root=$alias_root model_root=$model_root \
    +model/encoder=slotattn \
    +model/decoder=slotattn \
    +model/loss=mse \
    +optimizer=slotattn \
    +data=abscene \
    +running=$run_type $extra 
#> ./log/$logname 2>&1 &
