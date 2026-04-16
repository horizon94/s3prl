CUDA_VISIBLE_DEVICES=0,4 \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 run_downstream.py \
-m train \
-u hubert \
-d diarization \
-n HuBert_Base_SD