import json 
from pathlib import Path
from collections import defaultdict
from pprint import pprint 
import pandas as pd

# rootdir='/datadisk/levan/improved-instance-segm/models'
rootdir='/Users/levan/Workspace/improved-instance-segm/outputs'
rootdir_path = Path(rootdir)

MAX_REPS=5
TESTDIR_NAME="test-best"
EVAL_JSON_NAME="eval.json"

# STATS=True
STATS=False

wanted_basenames = [
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP-5e',
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-SCP-5e',
]

wanted_metrics = [
    "0_bbox_mAP",
    "0_segm_mAP",
    "1_bbox_mAP",
    "1_segm_mAP",
    "2_bbox_mAP",
    "2_segm_mAP",
]

results = { k : { metric:[] for metric in wanted_metrics } for k in wanted_basenames }

for basename, expt_res_dict in results.items():
    expts = [ basename ] + [ f'{basename}-{i}' for i in range(2, MAX_REPS+1) ]
    for expt in expts:
        expt_dir = rootdir_path/expt 
        if expt_dir.is_dir():
            exp_json =expt_dir / TESTDIR_NAME / EVAL_JSON_NAME
            assert exp_json.is_file(),f'{exp_json} does not exist!' 
            print(expt)
            with exp_json.open('r') as jf:
                eval_dict=json.load(jf)['metric']
            for metric, res_list in expt_res_dict.items():
                res_list.append(eval_dict[metric])

pprint(results)

if STATS:
    detailed_results = {}
    for expt, metrics_dict in results.items():
        detailed_results[expt]={}
        for metric, values in metrics_dict.items():
            stats = pd.DataFrame(values).describe(percentiles=[.5,]).to_dict()[0]
            stats['median'] = stats["50%"]
            del stats['50%']
            detailed_results[expt][metric] = stats

    pprint(detailed_results)