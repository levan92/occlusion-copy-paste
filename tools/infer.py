
import argparse
from pathlib import Path

import cv2
import mmcv

from mmdet.apis import inference_detector, init_detector

from utils.viz_results_pkl import parse_result, imshow_det_bboxes

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--outframes', type=str, help='Output folder to store result frames')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    parser.add_argument('--save', help='Results pickle file for saving')
    parser.add_argument('--load', help='Results pickle file for loading')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show or args.outframes, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(args.config, args.checkpoint, device=args.device)

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))
    if args.save:
        results = []
    elif args.load: 
        results = mmcv.load(args.load)
        print(f'Loaded pre-ran model results from {args.load}. Not running model.')

    if args.outframes: 
        outframe_dir = Path(args.outframes)
        outframe_dir.mkdir(exist_ok=True, parents=True)

    frame_idx = 0
    for frame in mmcv.track_iter_progress(video_reader):
        if args.load: 
            result = results[frame_idx]
        else:
            result = inference_detector(model, frame)

        bboxes, labels, segms = parse_result(result, decode_segm=False)

        img_show = frame.copy()
        img_show = imshow_det_bboxes(
            img_show,
            bboxes,
            labels, 
            segms, 
            class_names=['person',],
            label_filter=[0,],
            score_thr=args.score_thr,
            show=False,
            random_color=True,
            font_size=10,
            )
        # frame = model.show_result(frame, result, score_thr=args.score_thr)
        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(img_show, 'video', args.wait_time)
        if args.out:
            video_writer.write(img_show)

        if args.outframes: 
            out_path = outframe_dir / f'frame_{frame_idx+1}.jpg'
            cv2.imwrite(str(out_path), img_show)

        if args.save:
            # decoding mask RLE takes too long. does not justify even having the loading functionality. rather have a big pkl file. 
            # if isinstance(result, tuple):
            #     result = (result[0], encode_mask_results(result[1]))
            results.append(result)
        frame_idx += 1

    if args.save:
        mmcv.dump(results, f'{args.save}')
        print(f'Dumped model results to {args.save}')

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
