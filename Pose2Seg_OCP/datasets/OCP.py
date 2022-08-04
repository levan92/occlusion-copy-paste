import numpy as np 
import mmcv

from .CocoDatasetInfo import CocoDatasetInfo, annToMask
from .utils import imrotate

random = np.random.default_rng()

def check_kpts_oob(kpts, width, height):

    condition = np.concatenate(
        (kpts[:2]<0,
         (kpts[0]>width)[None,:],
         (kpts[1]>height)[None,:])
    )

    kpts[2] = np.where(
            np.logical_and(
                np.logical_or.reduce(condition), 
                kpts[2]==2
                ), 
            1, 
            kpts[2]
            )
    
    return kpts

def translate_kpts(kpts, translate_vector, width, height):
    '''
    kpts : nd.array (3, 17)
    translate_vector : (x,y)
    '''
    translate_vector = np.array(translate_vector)
    kpts[:2] = kpts[:2] - translate_vector[:,None]
    kpts = check_kpts_oob(kpts, width, height)
    return kpts

def intersect_kpts(orig_kpts, intersection): 
    '''
    orig_kpts : (3, 17)
    intersection : (h, w)
    '''
    x, y = orig_kpts[:, orig_kpts[2]==2][:2]
    orig_kpts[2, orig_kpts[2]==2] = np.where(intersection[y.astype(int), x.astype(int)], 1, orig_kpts[2, orig_kpts[2]==2])
    return orig_kpts

def geom_jitter(
        img, 
        mask,
        kpt,
        hflip_p = 0.5,
        scale_range=(0.8, 1.2),
        theta_range=(-10, 10),
    ):
    '''
    Args:
        img: input image
        mask: GT mask of image
        kpt: KP (3,17)
        hflip_p: probability of horizon flip 
        scale_range: (min scale, max scale)
        theta_range: rotation in degree
    '''
    _, w = mask.shape
    if hflip_p >= 1.0 or (hflip_p > 0.0 and random.random() <= hflip_p):
        img = img[:, ::-1, :]  # horizontal flip
        mask = mask[:, ::-1]  
        kpt[0] = w - kpt[0]

    scale = random.uniform(*scale_range)
    angle = random.integers(*theta_range)

    img, _= imrotate(img, angle, scale=scale, auto_bound=True)
    mask, matrix = imrotate(mask, angle, scale=scale, auto_bound=True)
    h, w = mask.shape
    
    points = np.vstack((kpt[:2], np.ones((1,17))))
    
    kpt[:2] = matrix.dot(points)
    kpt = check_kpts_oob(kpt, w, h)

    return img, mask, kpt

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


class OCP_Dataset():
    def __init__(self, 
            imageroot = './data/coco2017/train2017',
            annofile = './data/coco2017/annotations/person_keypoints_train2017_pose2seg.json',
            prob=0.8,
            basket_size=1,
            paste_num=[1,3],
            min_size_paste=0.1,
            min_size_occ=0.01,
            targeted_paste_prob=0.8,
            targeted_paste_buffer=0.4,
            aug_paste_geom_jitter=True,
            aug_paste_img_jitter=True,
            ):
        self.datainfos = CocoDatasetInfo(imageroot, annofile, onlyperson=True, loadimg=True)
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
    
    def __len__(self):
        return len(self.datainfos)
    
    def _get_paste_bounds(self, gt_masks, img_w, img_h):
        targeted_paste = False if self.targeted_paste_prob <= 0.0 else (random.random() < self.targeted_paste_prob) # throw dice only if probability > 0 

        if not targeted_paste:
            return [[0,img_w-1],[0,img_h-1]] # image bounds

        if len(gt_masks) > 0: 
            return get_targeted_bounds(random.choice(gt_masks), self.targeted_paste_buffer, img_w, img_h)
        else: 
            return [[0,img_w-1],[0,img_h-1]] # image bounds

    def _copy_paste(self, img, gt_masks, gt_kpts):
        """Copy Paste transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        chosen_basket_idxes = random.choice(range(len(self)), size=self.basket_size, replace=False)
        basket = [ self._extract(self.datainfos[idx]) for idx in chosen_basket_idxes ]
        #src_img, src_gt_masks, src_gt_kpts

        out_img = img.copy()
        img_h, img_w, img_c = out_img.shape
        img_size = side_length(img_h, img_w)

        all_masks = []
        for basket_idx, src in enumerate(basket):
            for mask, kpt in zip(src[1],src[2]):
                mask_bb = bbox_from_mask(mask)
                mx1, my1, mx2, my2 = mask_bb
                if (mx2-mx1)/img_size > self.min_size_paste and (my2-my1)/img_size > self.min_size_paste:
                    all_masks.append((mask, mask_bb, basket_idx, kpt))
        num_candidate = len(all_masks)

        num_aug_pax = min(num_candidate, random.integers(*self.paste_num))
        chosen_mask_idxes = random.choice(range(num_candidate), size=num_aug_pax, replace=False)

        for idx in chosen_mask_idxes:
            mask, mask_bb, basket_idx, src_kpts = all_masks[idx]
            mask_xmin, mask_ymin, mask_xmax, mask_ymax = [int(x) for x in mask_bb]
            mask_w = mask_xmax - mask_xmin
            mask_h = mask_ymax - mask_ymin

            paste_result = basket[basket_idx]
            # crop out instance block from instance's original image
            instance_block = paste_result[0][mask_ymin:mask_ymax, mask_xmin:mask_xmax]
            instance_mask = mask[mask_ymin:mask_ymax, mask_xmin:mask_xmax]
            instance_height, instance_width = instance_mask.shape
            src_kpts = translate_kpts(src_kpts, (mask_xmin, mask_ymin), instance_width, instance_height)
            
            # various augmentation on pastee
            if self.aug_paste_img_jitter:
                instance_block = image_jitter(instance_block)
                
            if self.aug_paste_geom_jitter:
                instance_block, instance_mask, src_kpts = geom_jitter(instance_block, instance_mask, src_kpts)
                mask_h, mask_w = instance_mask.shape

            # determining where to paste           
            bounds = self._get_paste_bounds(gt_masks, img_w, img_h)

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
            src_kpts = translate_kpts(src_kpts, (left_crop, top_crop), mask_w, mask_h)

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

            src_kpts = translate_kpts(src_kpts, (-xmin, -ymin), img_w, img_h)

            # boolean pasting for int mask
            mask_bool = (mask_full == 1)
            out_img[mask_bool] = block_full[mask_bool] 

            # modify existing GT masks due to pasted instance possibly occluding them
            new_masks = []
            new_kpts = []
            for orig_mask, orig_kpts in zip(gt_masks, gt_kpts):
                intersection = orig_mask * mask_full
                did_intersect = intersection.sum() > 0
                new_mask = orig_mask - intersection if did_intersect else orig_mask
                new_kpt = intersect_kpts(orig_kpts, intersection)
                if did_intersect: # then check new mask size
                    bxmin, bymin, bxmax, bymax = bbox_from_mask(new_mask)
                    bw = bxmax - bxmin
                    bh = bymax - bymin
                    too_small = bw/img_size <= self.min_size_occ or bh/img_size <= self.min_size_occ
                if not did_intersect or not too_small:
                    new_masks.append(new_mask)
                    new_kpts.append(new_kpt)
            if len(new_masks):
                gt_masks = np.concatenate((new_masks, np.expand_dims(mask_full, 0)))
                gt_kpts = np.concatenate((new_kpts, np.expand_dims(src_kpts, 0)))
            else: 
                gt_masks = np.expand_dims(mask_full, 0)
                gt_kpts = np.expand_dims(src_kpts, 0)

        return out_img, gt_masks, gt_kpts


    def _extract(self, rawdata):
        img = rawdata['data']
        height, width = img.shape[0:2]
        gt_kpts = np.float32(rawdata['gt_keypoints']) # (N, 3, 17)
        gt_segms = rawdata['segms']
        gt_masks = np.array([annToMask(segm, height, width) for segm in gt_segms])
        return img, gt_masks, gt_kpts

    def __getitem__(self, idx):
        rawdata = self.datainfos[idx]
        img, gt_masks, gt_kpts = self._extract(rawdata)
        img, gt_masks, gt_kpts = self._copy_paste(img, gt_masks, gt_kpts)
        trans_gt_kpts = gt_kpts.transpose(0, 2, 1) # (N, 17, 3)
        return {'img': img, 'kpts': trans_gt_kpts, 'masks': gt_masks}
        
    def collate_fn(self, batch):
        batchimgs = [data['img'] for data in batch]
        batchkpts = [data['kpts'] for data in batch]
        batchmasks = [data['masks'] for data in batch]
        return {'batchimgs': batchimgs, 'batchkpts': batchkpts, 'batchmasks':batchmasks}
