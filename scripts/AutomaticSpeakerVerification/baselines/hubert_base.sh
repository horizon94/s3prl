CUDA_VISIBLE_DEVICES=0,4 \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29600 run_downstream.py \
-m train \
-u hubert \
-d sv_voxceleb1 \
-n HuBERT_Base__ASV \


CUDA_VISIBLE_DEVICES=0,4 \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29511 run_downstream.py \
-m train \
-u hubert \
-d sv_voxceleb1 \
-n HuBERT_Base_ASV