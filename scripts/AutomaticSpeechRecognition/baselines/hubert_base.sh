CUDA_VISIBLE_DEVICES=0,1,2,3 \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=4  --master_port=29500 run_downstream.py \
-m train \
-u hubert \
-d asr \
-n HuBERT_Base__ASR \

