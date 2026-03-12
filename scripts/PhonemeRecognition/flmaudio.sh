CUDA_VISIBLE_DEVICES=4,5,6,7 \
FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python -m torch.distributed.launch --nproc_per_node=4  --master_port=29506 run_downstream.py \
-m train \
-u flmaudio \
-k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-150000 \
-g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
-d ctc \
-c downstream/ctc/libriphone.yaml \
-n flmaudio__PR_ddp_c150000_lr1e3 \
-o config.optimizer.lr=0.001,,config.downstream_expert.corpus.batch_size=8,,config.runner.gradient_accumulate_steps=1
