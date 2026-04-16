### PROJECT_PATH=/share/project/jiangxin/projects/s3prl


CUDA_VISIBLE_DEVICES=4,5,6,7 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=4  --master_port=29702 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d ctc \
-c downstream/ctc/snips.yaml \
-n SF_FLM_60k_lr1e4_16layers \
--upstream_feature_selection hidden_state_16



CUDA_VISIBLE_DEVICES=4,5,6,7 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=4  --master_port=29703 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d ctc \
-c downstream/ctc/snips.yaml \
-n SF_FLM_60k_lr1e4_12layers \
--upstream_feature_selection hidden_state_12



CUDA_VISIBLE_DEVICES=4,5,6,7 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=4  --master_port=29704 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d ctc \
-c downstream/ctc/snips.yaml \
-n SF_FLM_60k_lr1e4_8layers \
--upstream_feature_selection hidden_state_8