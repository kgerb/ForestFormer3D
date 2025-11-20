#!/bin/bash
docker run --shm-size=128g --rm -it \
    --gpus device=1 \
    -v $PWD:/workspace \
    -v $PWD/data:/workspace/data \
    -v $PWD/segmentator:/workspace/segmentator \
    --name forestformer3d-container \
    forestformer3d-3dt \
    bash -c "cp replace_mmdetection_files/loops.py /opt/conda/lib/python3.10/site-packages/mmengine/runner/ && \
    cp replace_mmdetection_files/base_model.py /opt/conda/lib/python3.10/site-packages/mmengine/model/base_model/ && \
    cp replace_mmdetection_files/transforms_3d.py /opt/conda/lib/python3.10/site-packages/mmdet3d/datasets/transforms/ && \
    cp replace_mmdetection_files/loading.py /opt/conda/lib/python3.10/site-packages/mmdet3d/datasets/pipelines/loading.py && \
    cp replace_mmdetection_files/forainetv2_dataset.py /opt/conda/lib/python3.10/site-packages/mmdet3d/datasets/forainetv2_dataset.py && \
    python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py work_dirs/model_file/epoch_3000_fix.pth"