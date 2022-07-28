import argparse
import os.path as osp
from tqdm import tqdm
from pathlib import Path 

import random
import colorsys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pycocotools.mask import decode

import mmcv
from mmcv import Config, DictAction
from mmcv.image import tensor2imgs
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.core.visualization import get_palette, palette_val

def parse_args():
    parser = argparse.ArgumentParser(
        description='Viz results from mmdet results picklet file')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('res', help='Results pickle file from mmdet')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--randomcolor',
        action='store_true'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    return args

EPS = 1e-2

def get_random_rgb():
    h = random.random()
    rgb = colorsys.hsv_to_rgb(h, 0.5, 0.95)
    rgb_255 = np.array(rgb)*255
    return rgb_255.astype(np.uint8)

def color_divide_255(rgbs):
    return tuple([c/255 for c in rgbs])

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      label_filter=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=13,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None,
                      pasted=None,
                      pasted_color=None,
                      mask_alpha=0.5,
                      random_color=False,
                      instances_palette=None,
                      ):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.
        pasted (list[int], optional): a list of flags of whether instance is pasted or not (for copy paste augmentation)
        pasted_color (list[tuple] | tuple | str | None, optional): Colors of
           pasted instances. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        mask_alpha (float): between 0. to 1., how much "transparency" masks should appear as. 
        random_color (bool): flag to turn on random colors
        instances_palette (list[tuple], optional): Provide list of tuple of RGBs for each instance to appear as one of the colours given

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)
    orig_img = img.copy()

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    max_label = int(max(labels)) if labels.shape[0] > 0 else -1
    bbox_color = palette_val(get_palette(bbox_color, max_label + 1))
    text_color = palette_val(get_palette(text_color, max_label + 1))
    if pasted:
        pasted_color = np.array(get_palette(pasted_color,1)[0], dtype=np.uint8)
    mask_color = get_palette(mask_color, max_label + 1)
    mask_color = np.array(mask_color, dtype=np.uint8)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        if label_filter is not None:
            if label not in label_filter:
                continue 

        if instances_palette is not None:
            color_final = np.array(instances_palette[i%(len(instances_palette)-1)], dtype=np.uint8)
        elif random_color:
            random_rgb = get_random_rgb()
        if segms is not None:
            if instances_palette is not None:
                color_mask = color_final
            elif random_color: 
                color_mask = random_rgb
            elif pasted and pasted[i]:
                color_mask = pasted_color
            else:
                color_mask = mask_color[labels[i]]
            mask = segms[i].astype(bool)

            img[mask] = img[mask] * (1-mask_alpha) + color_mask * mask_alpha
        
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        
        if instances_palette is not None:
            this_text_color = color_divide_255(color_final)
            color.append( this_text_color )
        elif random_color: 
            rand_color_mpl = color_divide_255(random_rgb)
            color.append( rand_color_mpl )
            this_text_color = rand_color_mpl
        elif segms is not None and pasted and pasted[i]:
            color_mask_mpl = tuple([c/255 for c in color_mask])
            color.append(color_mask_mpl)
            this_text_color = color_mask_mpl 
        else:
            color.append(bbox_color[label])
            this_text_color = text_color[label]
        
        if label >= len(class_names):
            continue
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        ax.text(
            bbox_int[0],
            bbox_int[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=this_text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')

    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    try:
        buffer = np.frombuffer(stream, dtype='uint8')
        if len(buffer)/width%4 == 0:
            img_rgba = buffer.reshape(-1, width, 4)
        elif len(buffer)/height%4 == 0:
            img_rgba = buffer.reshape(height, -1, 4)
        else:
            # this does not work anw, it results in slanted image
            target_len = 4 * height * width
            assert len(buffer) < target_len
            buffer = np.pad(buffer, (0, target_len-len(buffer)))
            if out_file is not None:
                out_file = Path(out_file)
                out_file = out_file.parent / 'pad' / f'{out_file.name}'
                print(f'PADDED: {out_file}')
            img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        img = rgb.astype('uint8')
        img = mmcv.rgb2bgr(img)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
    except ValueError as e:
        print('Unable to write due to', e)
        import pdb; pdb.set_trace()

    plt.close()

    if out_file:
        out_file_path = Path(out_file)
        orig_img_outfile = out_file_path.parent / f'{out_file_path.stem}_base.jpg'
        mmcv.imwrite(orig_img, orig_img_outfile)

    return img

def parse_result(result, im_wh=None, decode_segm=True):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if decode_segm:
            segm_result = decode(segm_result[0])
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    if decode_segm and im_wh is not None:
        img_w, img_h = im_wh
        segms = segms.T.reshape(-1, img_h, img_w)
    return bboxes, labels, segms

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)


    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    results = mmcv.load(args.res)

    batch_size = len(results)
    for i, data in enumerate(tqdm(data_loader)):
        if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
            img_tensor = data['img'][0]
        else:
            img_tensor = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        result = results[i]

        for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            if args.show_dir:
                out_file = osp.join(args.show_dir, img_meta['ori_filename'])
            else:
                out_file = None
            bboxes, labels, segms = parse_result(result, (ori_w,ori_h))

            imshow_det_bboxes(
                img_show,
                bboxes,
                labels, 
                segms, 
                class_names=dataset.CLASSES,
                bbox_color=None,
                text_color=None,
                mask_color=None,
                out_file=out_file,
                score_thr=args.show_score_thr,
                show=False,
                random_color=True,
                instances_palette=None,
                # instances_palette=[(0, 217, 255),(255, 82, 241),(0, 255, 42),(0, 12, 184)], #cyan, pink, green, blue
                font_size=5,
                )


if __name__=='__main__':
    main()