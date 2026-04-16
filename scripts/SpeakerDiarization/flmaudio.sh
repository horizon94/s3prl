### PROJECT_PATH=/share/project/jiangxin/projects/s3prl

CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29502 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d diarization \
-o config.downstream_expert.modelrc.dropout=0.3 \
-n SD_FLM_600k_lr1e4_dropout30



CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29503 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d diarization \
-o config.downstream_expert.modelrc.dropout=0.3 \
-n SD_FLM_600k_lr1e4_dropout30_8layers \
--upstream_feature_selection hidden_state_8


CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29504 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d diarization \
-o config.optimizer.lr=5.0e-5 \
-n SD_FLM_600k_lr5e5_8layers \
--upstream_feature_selection hidden_state_8

CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29503 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d diarization \
-n SD_FLM_600k_lr1e4_12layers_dropout50 \
--upstream_feature_selection hidden_state_12


CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29504 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d diarization \
-n SD_FLM_600k_lr1e4_12layers_dropout80 \
--upstream_feature_selection hidden_state_12


CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29504 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d diarization \
-n SD_FLM_600k_lr1e4_12layers_dropout30 \
--upstream_feature_selection hidden_state_12

CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29505 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d diarization \
-n SD_FLM_600k_lr1e4_8layers_dropout40 \
--upstream_feature_selection hidden_state_8

CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29504 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d diarization \
-o config.optimizer.lr=5.0e-5 \
-n SD_FLM_600k_lr5e5_16layers_dropout30 \
--upstream_feature_selection hidden_state_16

CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29506 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-300000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d diarization \
-o config.optimizer.lr=5.0e-5 \
-n SD_FLM_300k_lr5e5_16layers_dropout30 \
--upstream_feature_selection hidden_state_16


CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29506 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-300000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d diarization \
-o config.optimizer.lr=5.0e-5,,config.runner.log_step=50,,config.runner.eval_step=50,,config.runner.save_step=50 \
-n SD_FLM_300k_lr5e5_16layers_dropout30_step_50 \
--upstream_feature_selection hidden_state_16

