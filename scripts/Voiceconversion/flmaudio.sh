CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29502 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-615000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d a2o-vc-vcc2020 \
-c downstream/a2o-vc-vcc2020/config.yaml \
-n flmaudio_vc_lr1e4 \
-o config.downstream_expert.trgspk=TEF1