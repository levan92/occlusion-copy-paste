import json 
from pathlib import Path
import numpy as np 

aedet_json='/Users/levan/Workspace/Pose2Seg/pose_pred/ochval_refine_dt.json'
ochuman_json='/Users/levan/Data/OCHuman/ochuman_coco_format_val_range_0.00_1.00.json'
# ochuman_json='/Users/levan/Data/OCHuman/ochuman_coco_format_test_range_0.00_1.00.json'

och_path = Path(ochuman_json)
det_path = Path(aedet_json)

out_path = och_path.parent / f'{och_path.stem}-{det_path.stem}.json'

with det_path.open('r') as rf:
    det_list = json.load(rf)

with och_path.open('r') as rf: 
    och_dict = json.load(rf)

img_ids = [img['id'] for img in och_dict['images']]

new_annots = []
for det in det_list: 
    kpts = np.array(det['keypoints'])
    assert np.equal(kpts[2::3],1).all()
    kpts[2::3] = 2

    annot = {'image_id': det['image_id'], 
            'area': None, 
            'num_keypoints': 0, 
            'iscrowd': 0, 
            'id': det['id'], 
            'category_id': 1, 
            'keypoints': list(kpts), 
            'segmentation': [[]], 
            'bbox': []
            }

    assert annot['image_id'] in img_ids

    new_annots.append(annot)

och_dict['annotations'] = new_annots

with out_path.open('w') as wf: 
    json.dump(och_dict, wf)

print('Written to', out_path)