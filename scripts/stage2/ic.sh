CUDA_VISIBLE_DEVICES=0 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=1  --master_port=29500 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d fluent_commands \
-n IC_FLM_600k_base


CUDA_VISIBLE_DEVICES=1 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=1  --master_port=29501 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d fluent_commands \
-n IC_FLM_600k_dropout80


CUDA_VISIBLE_DEVICES=2 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=1  --master_port=29502 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d fluent_commands \
-n IC_FLM_600k_dropout50


CUDA_VISIBLE_DEVICES=3 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=1  --master_port=29503 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d fluent_commands \
-n IC_FLM_600k_dropout30


CUDA_VISIBLE_DEVICES=4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=1  --master_port=29504 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d fluent_commands \
-n IC_FLM_600k_dropout90



CUDA_VISIBLE_DEVICES=5 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=1  --master_port=29505 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-100000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d fluent_commands \
-n IC_FLM_100k_dropout50

CUDA_VISIBLE_DEVICES=6 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=1  --master_port=29506 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-200000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d fluent_commands \
-n IC_FLM_200k_dropout50


CUDA_VISIBLE_DEVICES=7 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=1  --master_port=29507 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-400000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d fluent_commands \
-n IC_FLM_400k_dropout50