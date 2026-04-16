# PROJECT_PATH=/share/project/jiangxin/projects/s3prl

CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29509 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d sv_voxceleb1 \
-o config.optimizer.lr=1.0e-4 \
-n ASV_FLM_600k_lr1e4_16layers \
--upstream_feature_selection hidden_state_16


CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29506 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d sv_voxceleb1 \
-o config.optimizer.lr=1.0e-4 \
-n ASV_FLM_600k_lr1e4_16layers_drop30 \
--upstream_feature_selection hidden_state_16

CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29506 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d sv_voxceleb1 \
-o config.optimizer.lr=1.0e-4 \
-n ASV_FLM_600k_lr1e4_16layers_drop50 \
--upstream_feature_selection hidden_state_16


CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29522 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d sv_voxceleb1 \
-o config.optimizer.lr=1.0e-4 \
-n ASV_FLM_600k_lr1e4_8layers \
--upstream_feature_selection hidden_state_8


CUDA_VISIBLE_DEVICES=0,4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=2  --master_port=29506 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d sv_voxceleb1 \
-o config.optimizer.lr=1.0e-3 \
-n ASV_FLM_600k_lr1e3_16layers \
--upstream_feature_selection hidden_state_16


——————————————————————————————————————————————————————————
#eval_a_sample
CUDA_VISIBLE_DEVICES=4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python3 run_downstream.py -m evaluate -e /share/project/intern/qt/s3prl/s3prl/result/downstream/ASV_FLM_600k_lr1e4_16layers_drop30/states-20000.ckpt

#eval_dir
CUDA_VISIBLE_DEVICES=4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
./downstream/sv_voxceleb1/test_expdir.sh \
/share/project/intern/qt/s3prl/s3prl/result/downstream/flmaudio_asv_lr1e4 \
/share/project/jiangxin/data/afm_data/benchmark/VoxCeleb1

#ASV
CUDA_VISIBLE_DEVICES=0 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
./downstream/sv_voxceleb1/test_expdir.sh \
/share/project/intern/qt/s3prl/s3prl/result/downstream/ASV_FLM_600k_lr1e4_16layers \
/share/project/jiangxin/data/afm_data/benchmark/VoxCeleb1

CUDA_VISIBLE_DEVICES=4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
./downstream/sv_voxceleb1/test_expdir.sh \
/share/project/intern/qt/s3prl/s3prl/result/downstream/ASV_FLM_600k_lr1e4_16layers_drop30 \
/share/project/jiangxin/data/afm_data/benchmark/VoxCeleb1

CUDA_VISIBLE_DEVICES=4 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
./downstream/sv_voxceleb1/test_expdir.sh \
/share/project/intern/qt/s3prl/s3prl/result/downstream/ASV_FLM_600k_lr1e4_16layers_drop50 \
/share/project/jiangxin/data/afm_data/benchmark/VoxCeleb1

