layer=0;
dist_fn=cosine;
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python3 run_downstream.py -m evaluate -t "dev" -u hubert -l ${layer} \
    -d quesst14_dtw -n Hubert_Base_QBE_${layer}_dev \
    -o config.downstream_expert.dtwrc.dist_method=$dist_fn
layer=12;
dist_fn=cosine;
PROJECT_PATH=/share/project/jiangxin/projects/s3prl \
PYTHONPATH=$PROJECT_PATH \
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
python3 run_downstream.py -m evaluate -t "dev" -u hubert -l ${layer} \
    -d quesst14_dtw -n Hubert_Base_QBE_${layer}_dev \
    -o config.downstream_expert.dtwrc.dist_method=$dist_fn
