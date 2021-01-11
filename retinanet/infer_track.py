import os
import json
import tempfile
from contextlib import redirect_stdout
import torch
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from .data import DataIterator
from .dali import DaliDataIterator
from .model import Model
from .utils import Profiler
from .utils import show_detections

import cv2

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    # """
    # palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    # color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    # return tuple(color)
    if label == 1:
        return (0, 255, 0)
    elif label == 2:
        return (0, 0, 255)
    else:
        return (255, 255, 255)

def draw_boxes(img, bbox, identities=None, pclasses=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        pclass = pclasses[i]
        color = compute_color_for_labels(pclass)
        label = '{}{:d},{}'.format("", id, pclass)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 3), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 3), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)
    return img

def convert_labels_into_visdrone(oldlabels):
    newlabels = torch.zeros_like(oldlabels)
    newlabels[oldlabels == 0.] = 1
    newlabels[(oldlabels <= 8.) & (oldlabels >= 1.)] = 2
    return newlabels.int()

def infer(model, path, detections_path, detections_file, resize, max_size, batch_size, mixed_precision=False, is_master=True, world=0, annotations=None, use_dali=True, is_validation=False, verbose=True):
    'Run inference on images from path'

    print('model',model)
    backend = 'pytorch' if isinstance(model, Model) or isinstance(model, DDP) else 'tensorrt'

    #print("backend",backend)
    stride = model.module.stride if isinstance(model, DDP) else model.stride
    #print('!!!!!!!!model.stride:', model.stride)
    # Create annotations if none was provided
    if not annotations:
        annotations = tempfile.mktemp('.json')
        images = [{ 'id': i, 'file_name': f} for i, f in enumerate(os.listdir(path))]
        json.dump({ 'images': images }, open(annotations, 'w'))

    # TensorRT only supports fixed input sizes, so override input size accordingly
    if backend == 'tensorrt': max_size = max(model.input_size)

    # Prepare dataset
    if verbose: print('Preparing dataset...')
    data_iterator = (DaliDataIterator if use_dali else DataIterator)(
        path, resize, max_size, batch_size, stride,
        world, annotations, training=False)
    if verbose: print(data_iterator)

    # Prepare model
    if backend is 'pytorch':
        # If we are doing validation during training,
        # no need to register model with AMP again
        if not is_validation:
            if torch.cuda.is_available(): model = model.cuda()
            model = amp.initialize(model, None,
                               opt_level = 'O2' if mixed_precision else 'O0',
                               keep_batchnorm_fp32 = True,
                               verbosity = 0)

        model.eval()

        cfg = get_config()
        cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    if verbose:
        print('   backend: {}'.format(backend))
        print('    device: {} {}'.format(
            world, 'cpu' if not torch.cuda.is_available() else 'gpu' if world == 1 else 'gpus'))
        print('     batch: {}, precision: {}'.format(batch_size,
            'unknown' if backend is 'tensorrt' else 'mixed' if mixed_precision else 'full'))
        print('Running inference...')

    assert(batch_size == 1) #only batchsize 1 supported
    imcount = 0
    detections_file_txt = os.path.join(detections_path, path.split(os.sep)[-1] + '.txt')


    results = []
    profiler = Profiler(['infer', 'fw', 'track'])
    with torch.no_grad():
        for i, (data, ids, ratios, imgs) in enumerate(data_iterator):

            frame_idx = ids[0].item()
            imcount += 1
            # if imcount > 50:
            #     break

            # Forward pass
            #print('start  profiler')
            profiler.start('fw')
            #print("data:",data)
            scores, boxes, classes = model(data)
            profiler.stop('fw')

            #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            results.append([scores, boxes, classes, ids, ratios])
            
            scores = scores.squeeze()
            boxes = boxes.squeeze()
            classes = classes.squeeze()
            ratios = ratios.squeeze()

            # Convert classes to VisDrone format
            classes = convert_labels_into_visdrone(classes)
            # print(classes.size(), classes > 0, ratios.size(), scores.size())
            scores = scores[classes > 0]
            boxes = boxes[classes > 0]
            classes = classes[classes > 0]

            bbox_xywh = []
            confs = []
            # Adapt detections to deep sort input format
            for bi in range(scores.size(0)):
                x_c, y_c, bbox_w, bbox_h = bbox_rel(*boxes[bi])
                obj = [x_c, y_c, bbox_w, bbox_h]
                bbox_xywh.append(obj)
                confs.append([scores[bi].item()])

            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            profiler.start('track')
            # Pass detections to deepsort
            outputs_deepsort = deepsort.update(xywhs, confss, classes, imgs[0])
            profile.stop('track')

            if len(outputs_deepsort) > 0:
                bbox_xyxy = outputs_deepsort[:, :4]
                identities = outputs_deepsort[:, -3]
                pclasses = outputs_deepsort[:, -2]
                draw_boxes(imgs[0], bbox_xyxy, identities, pclasses)
                cv2.imwrite(os.path.join(detections_path, "{}.jpg".format(frame_idx)), imgs[0])

            # Write MOT compliant results to file
            if len(outputs_deepsort) > 0:
                for j, output in enumerate(outputs_deepsort):
                    # print(ratios.item())
                    bbox_left = output[0] / ratios.item()
                    bbox_top = output[1] / ratios.item()
                    bbox_w = (output[2] - output[0]) / ratios.item()
                    bbox_h = (output[3] - output[1]) / ratios.item()
                    identity = output[-3]
                    pclass = output[-2]
                    conf = output[-1] * 1e-6
                    with open(detections_file_txt, 'a') as f:
                        f.write(('%g,' * 9 + '%g\n') % (frame_idx, identity, bbox_left,
                                                       bbox_top, bbox_w, bbox_h, conf, pclass, -1, -1))  # label format

            profiler.bump('infer')
            if verbose and (profiler.totals['infer'] > 60 or i == len(data_iterator) - 1):
                size = len(data_iterator.ids)
                msg  = '[{:{len}}/{}]'.format(min((i + 1) * batch_size,
                    size), size, len=len(str(size)))
                msg += ' {:.3f}s/{}-batch'.format(profiler.means['infer'], batch_size)
                msg += ' (fw: {:.3f}s, track: {:.3f}s)'.format(profiler.means['fw'], profiler.means['track'])
                msg += ', {:.1f} im/s'.format(batch_size / profiler.means['infer'])
                print(msg, flush=True)

                profiler.reset()

    # Gather results from all devices
    if verbose: print('Gathering results...')
    results = [torch.cat(r, dim=0) for r in zip(*results)]
    if world > 1:
        for r, result in enumerate(results):
            all_result = [torch.ones_like(result, device=result.device) for _ in range(world)]
            torch.distributed.all_gather(list(all_result), result)
            results[r] = torch.cat(all_result, dim=0)

    if is_master:
        # Copy buffers back to host
        results = [r.cpu() for r in results]

        # Collect detections
        detections = []
        processed_ids = set()
        for scores, boxes, classes, image_id, ratios in zip(*results):
            image_id = image_id.item()
            if image_id in processed_ids:
                continue
            processed_ids.add(image_id)
              
            keep = (scores > 0).nonzero()
            scores = scores[keep].view(-1)
            boxes = boxes[keep, :].view(-1, 4) / ratios
            classes = classes[keep].view(-1).int()
            #print('classes', classes)
            for score, box, cat in zip(scores, boxes, classes):
                x1, y1, x2, y2 = box.data.tolist()
                cat = cat.item()

                detections.append({
                    'image_id': image_id,
                    'score': score.item(),
                    'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                    'category_id': cat
                })
                #show_detections(detections)

        if detections:
            # Save detections
            if detections_file and verbose: print('Writing {}...'.format(detections_file))
            detections = { 'annotations': detections }
            detections['images'] = data_iterator.dataset.images
            if detections_file:
                json.dump(detections, open(os.path.join(detections_path, detections_file), 'w'), indent=4)
        else:
            print('No detections!')
