# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import Sequence
from pathlib import Path

import numpy as np
import cv2
import mmcv
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
# from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset
from mmdet.apis import init_random_seed, set_random_seed

from viz_utils import imshow_det_bboxes


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--write-pastees',
        action='store_true',
        help='Write out constituents of images and instances too'
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
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options):

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    # while 'dataset' in train_data_cfg and train_data_cfg[
        # 'type'] != 'MultiImageMixDataset':
        # 'type'] != 'MultiDatasetsMixDataset':
    while 'dataset' in train_data_cfg and not (train_data_cfg['type'] =='MultiImageMixDataset' or train_data_cfg['type']=='MultiDatasetsMixDataset'):
        train_data_cfg = train_data_cfg['dataset']

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    # set random seeds
    seed = init_random_seed(cfg.seed)
    print(f'Set random seed to {seed}')
    set_random_seed(seed)
    cfg.seed = seed

    print(cfg.data.train)
    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))

    out_dir = Path(args.output_dir)

    paste_out_dir = out_dir / 'pastees'
    paste_out_dir.mkdir(exist_ok=True, parents=True)

    for item in dataset:
        item_filename = Path(item['filename'])
        filename = out_dir / f'{item_filename.stem}_final.jpg'
      
        if len(dataset.CLASSES)==1:
            colors = {
                'bbox':(0, 128, 255),
                'text':(0, 128, 255),
                'mask':[(0, 128, 255)],
                'paste':(255, 125, 199)
            }
        else:
            colors = {
                'bbox': None,
                'text': None,
                'mask': None,
                'paste': (255, 125, 199)
            }

        gt_masks = item.get('gt_masks', None)
        if gt_masks is not None:
            gt_masks = mask2ndarray(gt_masks)
        
        imshow_det_bboxes(
            item['img'],
            item['gt_bboxes'],
            item['gt_labels'],
            gt_masks,
            class_names=dataset.CLASSES,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=filename,
            thickness=1,
            bbox_color=colors['bbox'],
            text_color=colors['text'],
            mask_color=colors['mask'],
            pasted_color=colors['paste'],
            pasted= (item['pasted_flags'] if 'pasted_flags' in item else None),
            mask_alpha=0.35
            )

        if args.write_pastees and item.get('originals') is not None:
            orig_results, pasted = item['originals']
            orig_filename = out_dir / f'{item_filename.stem}_orig.jpg'
            for i, paste in enumerate(pasted): 
                instance_block, instance_mask, mix_img = paste
                pasted_orig_filename = paste_out_dir / f'{item_filename.stem}_pasted{i}_orig.jpg'
                cv2.imwrite(str(pasted_orig_filename), instance_block)

                pasted_orig_img = paste_out_dir / f'{item_filename.stem}_pasted{i}_orig_fullimg.jpg'
                cv2.imwrite(str(pasted_orig_img), mix_img)

                instance_mask = instance_mask[:,:, None] * np.ones(3)[None, None,:]
                out_paste = instance_block * instance_mask
                pasted_filename = paste_out_dir / f'{item_filename.stem}_pasted{i}.jpg'
                cv2.imwrite(str(pasted_filename), out_paste)

            orig_gt_masks = orig_results.get('gt_masks', None)
            if orig_gt_masks is not None:
                orig_gt_masks = mask2ndarray(orig_gt_masks)

            imshow_det_bboxes(
                orig_results['img'],
                orig_results['gt_bboxes'],
                orig_results['gt_labels'],
                orig_gt_masks,
                class_names=dataset.CLASSES,
                show=not args.not_show,
                wait_time=args.show_interval,
                out_file=orig_filename,
                thickness=1,
                bbox_color=colors['bbox'],
                text_color=colors['text'],
                mask_color=colors['mask'],
                pasted_color=colors['paste'],
                mask_alpha=0.35
                )

        progress_bar.update()


if __name__ == '__main__':
    main()
