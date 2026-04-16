CUDA_VISIBLE_DEVICES=0,4 \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29502 run_downstream.py \
-m train \
-u hubert \
-d voxceleb1 \
-n HuBert_Base_SID_lr1e3 \
-o config.optimizer.lr=0.001,,config.runner.gradient_accumulate_steps=1

CUDA_VISIBLE_DEVICES=0,4 \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29602 run_downstream.py \
-m train \
-u hubert \
-d voxceleb1 \
-n HuBert_Base_SID

CUDA_VISIBLE_DEVICES=4,5,6,7 \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29605 run_downstream.py \
-m train \
-u hubert \
-d voxceleb1 \
-n HuBert_Base_SID_lr1e3_grad_acmu_1 \
-o config.optimizer.lr=1.0e-3,,config.runner.gradient_accumulate_steps=1