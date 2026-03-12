CUDA_VISIBLE_DEVICES=4,5,6,7 \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=4  --master_port=29500 run_downstream.py \
-m train \
-u hubert \
-n HuBERT_Base__PR \
-d ctc \
-c downstream/ctc/libriphone.yaml
