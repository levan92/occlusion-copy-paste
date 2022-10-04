import logging
import numpy as np
random = np.random.default_rng()

import mmcv
from mmcv.utils import print_log

from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES

def geom_jitter(
        img, 
        mask,
        hflip_p = 0.5,
        scale_range=(0.8, 1.2),
        theta_range=(-10, 10),
    ):
    '''
    Args:
        img: input image
        mask: GT mask of image
        hflip_p: probability of horizon flip 
        scale_range: (min scale, max scale)
        theta_range: rotation in degree
    '''
    if hflip_p >= 1.0 or (hflip_p > 0.0 and random.random() <= hflip_p):
        img = img[:, ::-1, :]  # horizontal flip
        mask = mask[:, ::-1]  

    scale = random.uniform(*scale_range)
    angle = random.integers(*theta_range)

    img = mmcv.imrotate(img, angle, scale=scale, auto_bound=True)
    mask = mmcv.imrotate(mask, angle, scale=scale, auto_bound=True)

    return img, mask

def colour_jitter(image):
    image = mmcv.adjust_color(image, alpha=random.integers(8, 12)/10.)
    image = mmcv.adjust_brightness(image, factor=random.integers(5, 20)/10.)
    image = mmcv.adjust_contrast(image, factor=random.integers(5, 20)/10.)
    image = mmcv.adjust_sharpness(image, factor=random.integers(0, 15)/10.)
    return image  

def image_jitter(img, color_jitter_p=0.5):
    if random.random() <= color_jitter_p:
        img = colour_jitter(img)
    return img

def bbox_from_mask(mask):
    x_any = mask.any(axis=0)
    y_any = mask.any(axis=1)
    x = np.where(x_any)[0]
    y = np.where(y_any)[0]
    if len(x) > 0 and len(y) > 0:
    # use +1 for x_max and y_max so that the right and bottom
    # boundary of instance masks are fully included by the box
        bbox = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1],
                                dtype=np.float32)
    else:
        bbox = np.array([0,0,0,0])
    return bbox

def get_targeted_bounds(mask, buffer, img_w, img_h):
    target_bb = bbox_from_mask(mask)
    tx_min, ty_min, tx_max, ty_max = target_bb
    tw = tx_max - tx_min
    th = ty_max - ty_min
    tx_min = max(0, tx_min - tw*buffer)
    tx_max = min(img_w-1, tx_max + tw*buffer)
    ty_min = max(0, ty_min - th*buffer)
    ty_max = min(img_h-1, ty_max + th*buffer)
    return [[int(tx_min),int(tx_max)],[int(ty_min),int(ty_max)]]

def side_length(s1,s2):
    return (s1*s2) ** 0.5

@PIPELINES.register_module()
class OccCopyPaste:
    """CopyPaste augmentation.

    Copy & Paste is an augmentation that samples random instances + masks from the entire dataset and paste them into your current image. Particularly useful for crowded and occluded cases where instances occlude each other.

    First we sample random images in the dataset, then sample instances from within this basket of images. Then we paste it onto the current image and handle the overlapping mask regions (by default we remove the non-visible area of the mask that got overlapped).

    This implementation provides various add-ons/features as arguments:
        1. minimum pasting size control
        2. targeted pasting (to maximise occlusion in training)
        3. augmented pasting 
        4. the other realisms that did not work (as discussed in paper) is not provided in this code base, please request if interested

    Args:
        prob (float): Overall probability of OCP happening
        basket_size (int): Number of images from which instances are sampled from
        paste_num (List[int]): Range of number of instances to be pasted on one image. Randomly chosen in range: [min, max].
        min_size_paste (float): Minimum size (as a proportion of equalized image side length) for which to consider a valid pasting instance. Equalized image side length is the square root of the area of the image. 
        min_size_occ (float): Minimum size (as a proportion of equalized image side length) for which to still consider as a GT instance. GT instances's bboxes may drop below min_size after pasting other instance in due to occlusion. This will not be imposed on existing instances that does not intersect with any new pasted instances. 
        blending (bool): To do blended pasting of instances or not. 
        blend_float_prob (float): Between 0.0 to 1.0. Probability of doing do gaussian blurring on masks that are floats (as oppose to int). Float blurring leads to "fading" effect. Note that float blending is 3x slower. 
        gaussian_kernel (List | int): Range of sizes of gaussian blur kernel (have to be odd): [min, max]. Kernel size chosen random between given range. If int is given, size is deterministic. Only relevant if blending is True. 
        gaussian_sd (List | int): Range of sizes of gaussian kernel's standard deviation(sigma): [min, max]. Gaussian kernel s.d. chosen randomly between given range. If int is given, value is deterministic. Only relevant if blending is True.
        targeted_paste_prob (float): Between 0.0 to 1.0. Probability of applying targeted pasting. Targeted pasting tries to maximise occlusion of pasted instances with existing instance. Defaults to 0, meaning no targeted pasting done. 
        targeted_paste_buffer (float): Ratio of target bounding box size. Amount of buffer around target bounding box, buffer + bb + buffer will be the bounds that a new instance will be pasted in during targeted paste mode. Can be negative. 
        scale_aware (bool): Flag True to turn on scaling of pasted instance according to the size of existing instances in the image. Defaults to False.
        context_paste (bool): Flag True to turn on pasting based on context. If context_paste is True, but image has no context info, it will paste near human vicinity (same as targeted paste). Defaults to False.
        self_paste (bool): Flag True to turn on self-pasting.
        aug_paste_geom_jitter (bool): Flag True to turn on geometric augmentation (flip/scale/rotation) on pasted instances
        aug_paste_img_jitter: Flag True to turn on color jittering augmentation on pasted instances
        viz (bool): Flag True to store additional results on which instances are pasted on synthetically. Stored in results['pasted_flags']. Typically used for visualisation purpose. 
    """

    def __init__(self,
                 prob=0.8,
                 basket_size=1,
                 paste_num=[1,3],
                 min_size_paste=0.1,
                 min_size_occ=0.01,
                 targeted_paste_prob=0.8,
                 targeted_paste_buffer=0.4,
                 aug_paste_geom_jitter=False,
                 aug_paste_img_jitter=False,
                 viz=False,
                 ):
        self.prob = prob
        self.basket_size = basket_size
        self.paste_num = paste_num
        assert isinstance(self.paste_num, list) 
        assert isinstance(self.paste_num[0], int) 
        self.min_size_paste = min_size_paste
        self.min_size_occ = min_size_occ
        self.targeted_paste_prob = targeted_paste_prob
        self.targeted_paste_buffer = targeted_paste_buffer
        self.aug_paste_geom_jitter = aug_paste_geom_jitter
        self.aug_paste_img_jitter = aug_paste_img_jitter
        self.viz = viz
        self.zero_candidates_count = 0
        self.zero_candidates_warnlimit = 20

    def __call__(self, results):
        """Call function returns image augmented with other instances.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with instances pasted in.
        """
        if random.uniform(0, 1) > self.prob:
            return results

        results = self._copy_paste(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """
        
        indexes = random.choice(range(len(dataset)), size=self.basket_size, replace=False)
        return indexes

    def _get_paste_bounds(self, results, exist_masks, img_w, img_h):
        targeted_paste = False if self.targeted_paste_prob <= 0.0 else (random.random() < self.targeted_paste_prob) # throw dice only if probability > 0 

        if not targeted_paste:
            return [[0,img_w-1],[0,img_h-1]] # image bounds

        if len(exist_masks) > 0: 
            return get_targeted_bounds(random.choice(exist_masks), self.targeted_paste_buffer, img_w, img_h)
        else: 
            return [[0,img_w-1],[0,img_h-1]] # image bounds

    def _copy_paste(self, results):
        """Copy Paste transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        
        out_img = results['img'].copy()
        img_h, img_w, img_c = out_img.shape
        img_size = side_length(img_h, img_w)

        all_masks = []
        gt_bboxes = None
        ## IMPORTANT: WHEN USING OCP, LoadAnnotations must have poly2mask=True
        for basket_idx, target in enumerate(results['mix_results']):
            mask_bboxes = target['gt_masks'].get_bboxes()
            assert len(target['gt_masks']) == len(mask_bboxes)
            assert len(target['gt_masks']) == len(target['gt_labels'])
            for mask, mask_bb, label in zip(target['gt_masks'], mask_bboxes, target['gt_labels']):
                mx1, my1, mx2, my2 = mask_bb
                if (mx2-mx1)/img_size > self.min_size_paste and (my2-my1)/img_size > self.min_size_paste:
                    all_masks.append((mask, mask_bb, basket_idx, label))
        num_candidate = len(all_masks)

        if num_candidate == 0: 
            self.zero_candidates_count+=1
        else: 
            self.zero_candidates_count=0

        if self.zero_candidates_count >= self.zero_candidates_warnlimit: 
            # self.logger.warning(
            #     f'OCP is activated, but there has been no copy paste candidates consecutively for more than {self.zero_candidates_warnlimit} iterations. Is something wrong?')
            print_log(f'OCP is activated, but there has been no copy paste candidates consecutively for more than {self.zero_candidates_warnlimit} iterations. Is something wrong?',
            level=logging.WARN
            )


        num_aug_pax = min(num_candidate, random.integers(*self.paste_num))
        chosen_mask_idxes = random.choice(range(num_candidate), size=num_aug_pax, replace=False)

        exist_masks = results['gt_masks'].to_ndarray()        
        labels = results['gt_labels']
        if self.viz:
            pasted_flags = [ 0 for _ in range(len(labels)) ] 
        for i, idx in enumerate(chosen_mask_idxes):
            mask, mask_bb, basket_idx, mask_label = all_masks[idx]
            mask_xmin, mask_ymin, mask_xmax, mask_ymax = [int(x) for x in mask_bb]
            mask_w = mask_xmax - mask_xmin
            mask_h = mask_ymax - mask_ymin

            paste_result = results['mix_results'][basket_idx]
            # crop out instance block from instance's original image
            instance_block = paste_result['img'][mask_ymin:mask_ymax, mask_xmin:mask_xmax]
            instance_mask = mask[mask_ymin:mask_ymax, mask_xmin:mask_xmax]
            
            # various augmentation on pastee
            if self.aug_paste_img_jitter:
                instance_block = image_jitter(instance_block)
                
            if self.aug_paste_geom_jitter:
                instance_block, instance_mask = geom_jitter(instance_block, instance_mask)
                mask_h, mask_w = instance_mask.shape

            # determining where to paste           
            bounds = self._get_paste_bounds(results, exist_masks, img_w, img_h)

            cx = random.integers( *bounds[0] )
            cy = random.integers( *bounds[1] )
            ymin = cy - mask_h // 2
            ymax = ymin + mask_h
            xmin = cx - mask_w // 2
            xmax = xmin + mask_w

            # crop instance mask to clip within original image size
            left_crop = max(0, -xmin)
            right_crop = mask_w - max(0, xmax-img_w+1)
            top_crop = max(0, -ymin)
            bot_crop = mask_h - max(0, ymax-img_h+1)
            
            instance_block = instance_block[top_crop:bot_crop,left_crop:right_crop]
            instance_mask = instance_mask[top_crop:bot_crop,left_crop:right_crop]
            mask_h, mask_w = instance_mask.shape
            if instance_mask.sum() <= 0 or mask_h <=0 or mask_w <=0: 
                continue 

            # new coord after clipping
            xmin = max(0, xmin)
            xmax = min(img_w-1, xmax)
            ymin = max(0, ymin)
            ymax = min(img_h-1, ymax)

            # pasting instance onto base image
            mask_full = np.zeros((img_h, img_w), dtype=out_img.dtype)
            mask_full[ymin:ymax,xmin:xmax] = instance_mask
            block_full = np.zeros((img_h, img_w, img_c), dtype=out_img.dtype)
            block_full[ymin:ymax,xmin:xmax] = instance_block

            # boolean pasting for int mask
            mask_bool = (mask_full == 1)
            out_img[mask_bool] = block_full[mask_bool] 

            # modify existing GT masks due to pasted instance possibly occluding them
            new_masks = []
            new_labels = []
            new_pasted_flag = []
            for j, orig_mask_label in enumerate(zip(exist_masks, labels)):
                orig_mask, orig_label = orig_mask_label
                intersection = orig_mask * mask_full
                did_intersect = intersection.sum() > 0
                new_mask = orig_mask - intersection if did_intersect else orig_mask
                if did_intersect: # then check new mask size
                    bxmin, bymin, bxmax, bymax = bbox_from_mask(new_mask)
                    bw = bxmax - bxmin
                    bh = bymax - bymin
                    too_small = bw/img_size <= self.min_size_occ or bh/img_size <= self.min_size_occ
                if not did_intersect or not too_small:
                    new_masks.append(new_mask)
                    new_labels.append(orig_label)
                    if self.viz: 
                        new_pasted_flag.append(pasted_flags[j])
            if len(new_masks):
                exist_masks = np.concatenate((new_masks, np.expand_dims(mask_full, 0)))
            else: 
                exist_masks = np.expand_dims(mask_full, 0)
            labels = np.append(new_labels, int(mask_label))
            if self.viz: 
                pasted_flags = new_pasted_flag + [1]

        results['gt_labels'] = np.array(labels, dtype=np.long)
        results['gt_masks'] = BitmapMasks(exist_masks, img_h, img_w)
        results['gt_bboxes'] = results['gt_masks'].get_bboxes()
        results['img'] = out_img
        if self.viz: 
            results['pasted_flags'] = pasted_flags

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'basket_size={self.basket_size}, '
        repr_str += f'paste_num={self.paste_num}, '
        repr_str += f'min_size_paste={self.min_size_paste},'
        repr_str += f'min_size_occ={self.min_size_occ},'
        repr_str += f'targeted_paste_prob={self.targeted_paste_prob},'
        repr_str += f'targeted_paste_buffer={self.targeted_paste_buffer},'
        repr_str += f'aug_paste_geom_jitter={self.aug_paste_geom_jitter},'
        repr_str += f'aug_paste_img_jitter={self.aug_paste_img_jitter},'

        repr_str += f'viz={self.viz})'
        return repr_str

