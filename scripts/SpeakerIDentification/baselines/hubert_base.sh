CUDA_VISIBLE_DEVICES=0,1,2,3 \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29502 run_downstream.py \
-m train \
-u hubert \
-d voxceleb1 \
-n hubert_sid_lr1e3 \
-o config.optimizer.lr=0.001,,config.runner.gradient_accumulate_steps=1

