# MAL-inference-deepsort
This repo accomplishes object tracking by combining the object detections from [Multiple Anchor Learning(MAL)](https://github.com/DeLightCMU/MAL) with [DeepSORT](https://github.com/ZQPei/deep_sort_pytorch). The tracker is tested on VisDrone-MOT videos.

For more details of MAL and DeepSORT, please refer to:
- [Multiple Anchor Learning for Visual Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ke_Multiple_Anchor_Learning_for_Visual_Object_Detection_CVPR_2020_paper.pdf)  and the [original MAL repo](https://github.com/DeLightCMU/MAL).
- [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402) and [Pytorch Implementation](https://github.com/ZQPei/deep_sort_pytorch).


## Installation Instructions

Follow from [MAL-inference](https://github.com/DeLightCMU/MAL-inference).



## Steps to visualize inference 

Use the infer_track subcommand.

1. Run on a video from VisDrone dataset. For eg: `uav0000086_00000_v`.
```bash
CUDA_VISIBLE_DEVICES=0 retinanet infer_track --config configs/MAL_R-50-FPN_e2e.yaml --images ../data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v --batch 1
```
2. Unless otherwise specified, the outputs are saved in newly created directory `uav0000086_00000_v`. The output includes:
  a. Per-frame bounding boxes detection along with tracking id, class id. People are depicted using green boxes, vehicles using red boxes, while all other classes are ignored.
  b. Output detection file `uav0000086_00000_v.txt` for computing either CLEAR-MOT metrics or AP metrics.
  c. Output detections in JSON format.

3. To generate a video from the output images, use [get_video.sh](get_video.sh).
```bash
ffmpeg -framerate 25 -i %d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
```

4. To compute AP metrics, run
```bash
python compute_map.py <groundtruth_annotations> <detections_file>
```

5. For computing CLEAR-MOT metrics, use the ground truth annotations and detections file with [py-motmetrics](https://github.com/cheind/py-motmetrics).
```bash
python -m motmetrics.apps.eval_motchallenge <groundtruth_annotations_folder> <detections_folder>
```

## Results

Sample low-resolution video
![uav0000086_00000_v](images/uav0000086_00000_v.gif)


| Sequence Number | Video (Download Recommended) |
| :-------------: | :----: |
| uav0000086_00000_v | [Link](https://drive.google.com/file/d/1GUjdH0PBHpmB8kipowqxWBFKyNfcGDjq/view?usp=sharing)|
| uav0000117_02622_v | [Link](https://drive.google.com/file/d/1i-b2hg3Fg9Gl_zOyC39O052gZhlCgYiF/view?usp=sharing)|


| Sequence Number | AP | AP@0.25 | AP@0.5 | AP@0.75 | AP(people) | AP(vehicle) | MOTA |
| :-------------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| uav0000086_00000_v | 0.248 | 0.408 | 0.264 | 0.073 | 0.431 | 0.065 | 26.3 |
| uav0000117_02622_v | 0.327 | 0.507 | 0.369 | 0.105 | 0.234 | 0.421 | -29.3 |


## Ablations

- Varying input image size

| Image Size | AP | AP for Seq1 | AP for Seq2 | FW Time | Tracking Time |
| :---------: | :---------: | :---------: | :---------: |:---------: |:---------: |
| 800 | 0.287 | 0.248 | 0.327 | 0.480s | 1.214s |
| 1024 | 0.287 | 0.248 | 0.327 | 0.487s | 1.050s |
| 400 | 0.2 | 0.178 | 0.222 | 0.174s | 1.261s |

- Varying NMS threshold

| Image Size | AP | AP for Seq1 | AP for Seq2 | FW Time | Tracking Time |
| :---------: | :---------: | :---------: | :---------: |:---------: |:---------: |
| 0.5 | 0.287 | 0.248 | 0.327 | 0.480s | 1.214s |
| 0.75 | 0.268 | 0.235 | 0.302 | 0.486s | 1.016s |
| 0.25 | 0.288 | 0.247 | 0.329 | 0.498s | 1.580s |

- Varying number of detections

| Number of detections | AP | AP for Seq1 | AP for Seq2 | FW Time | Tracking Time |
| :---------: | :---------: | :---------: | :---------: |:---------: |:---------: |
| 100 | 0.287 | 0.248 | 0.327 | 0.480s | 1.214s |
| 200 | 0.285 | 0.242 | 0.329 | 0.552s | 1.651s |
| 50 | 0.268 | 0.235 | 0.302 | 0.483s | 0.625s |