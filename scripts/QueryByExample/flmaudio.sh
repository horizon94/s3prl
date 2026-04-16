##PROJECT_PATH=/share/project/jiangxin/projects/s3prl

layer=0;
dist_fn=cosine;
PROJECT_PATH=/share/project/intern/qt/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python3 run_downstream.py -m evaluate -t "dev" -u hubert -l ${layer} \
    -d quesst14_dtw -n Hubert_Base_QBE_${layer}_dev \
    -o config.downstream_expert.dtwrc.dist_method=$dist_fn

-------------------------------------------------------
#对query 和搜索库做 DTW（动态时间规整），把结果写到result/downstream/...
dist_fn=cosine
PROJECT_PATH=/share/project/intern/qt/s3prl
export PYTHONPATH=$PROJECT_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
for layer in 0 3 7 11 15; do
  echo "Running layer ${layer} ..."
  python3 run_downstream.py -m evaluate -t dev -u flmaudio -l ${layer} \
      -d quesst14_dtw -n QbE_flmaudio_60k_${layer}_dev \
      -k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
      -g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
      -o config.downstream_expert.dtwrc.dist_method=$dist_fn
done


#给dev打分，去选最好的层
export S3PRL_DIR=/share/project/intern/qt/s3prl/s3prl
cd /share/project/jiangxin/data/afm_data/benchmark/qbe//quesst14Database/scoring

for layer in 0 3 7 11 15; do
  echo "===== Scoring layer ${layer} ====="
  ./score-TWV-Cnxe.sh $S3PRL_DIR/result/downstream/QbE_flmaudio_60k_${layer}_dev \
      groundtruth_quesst14_dev -10
done



#发现第11层的表现最好，TWV=0.006064893
-------------------------------------------------------
#用最佳层跑test


#对query 和搜索库做 DTW（动态时间规整），把结果写到result/downstream/...
dist_fn=cosine
PROJECT_PATH=/share/project/intern/qt/s3prl
export PYTHONPATH=$PROJECT_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export FLMAUDIO_MIMI_CKPT=/share/project/jiangxin/models/pretrained_models/mimi_model \
for layer in 11; do
  echo "Running layer ${layer} ..."
  python3 run_downstream.py -m evaluate -t "test" -u fbank -l ${layer} \
      -d quesst14_dtw -n QbE_flmaudio_60k_${layer}_test \
      -k /share/project/lx/projects/NativeAudio-trainer-ddp/outputs/ckpts/audio_model_multi_node_mix/checkpoint-step-600000 \
      -g /share/project/lx/projects/NativeAudio-trainer-ddp/audio_models/config_mini.json \
      -o config.downstream_expert.dtwrc.dist_method=$dist_fn
done


#对test打分
export S3PRL_DIR=/share/project/intern/qt/s3prl/s3prl
cd /share/project/jiangxin/data/afm_data/benchmark/qbe//quesst14Database/scoring

for layer in 11; do
  echo "===== Scoring layer ${layer} ====="
  ./score-TWV-Cnxe.sh $S3PRL_DIR/result/downstream/QbE_flmaudio_60k_${layer}_test \
      groundtruth_quesst14_dev -10
done


