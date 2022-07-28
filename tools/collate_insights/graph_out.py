import json 
from pathlib import Path
from pprint import pprint
import pandas as pd

# rootdir='/datadisk/levan/improved-instance-segm/models'
rootdir='/Users/levan/Workspace/improved-instance-segm/outputs'
rootdir_path = Path(rootdir)

MAX_REPS=5
TESTDIR_NAME="test-best"
EVAL_JSON_NAME="eval.json"
OUT_DIR='/Users/levan/Workspace/improved-instance-segm/graphs/'

# STATS=True
STATS=False
# TITLE="Segm AP on COCO (Orig Val & Occ-Person)"
# TITLE="Segm AP on COCO"
TITLE="Segm AP on COCO (all)"
# TITLE="Segm AP on Test sets (all)"
# TITLE="Segm AP on OCHuman"
# TITLE="Segm AP on OCHuman (all)"

wanted_basenames = {
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-reduceLR': 'FT',
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-SCP-5e': 'SCP',
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP-5e': 'OCP',
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP_aug-5e': 'OCP-Aug',
    # 'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP-5e-nodice': 'OCP-NoDice',
}

wanted_metrics = {
    # "0_bbox_mAP",
    # "0_segm_mAP": "COCO-Val",
    # "1_bbox_mAP",
    "1_segm_mAP": "OCH-Val",
    # "2_bbox_mAP",
    "2_segm_mAP": "OCH-Test",
    # "3_segm_mAP": "COCO-Val-OccPerson",
}

# YMIN, YMAX = None, None
# YMIN, YMAX = 50, 60
# YMIN, YMAX = 48, 53
# YMIN, YMAX = 25, 60
YMIN, YMAX = 25, 30

results = []
for basename, nickname in wanted_basenames.items():
    expts = [ basename ] + [ f'{basename}-{i}' for i in range(2, MAX_REPS+1) ]
    for expt in expts:
        expt_dir = rootdir_path/expt 
        if expt_dir.is_dir():
            exp_json =expt_dir / TESTDIR_NAME / EVAL_JSON_NAME
            assert exp_json.is_file(),f'{exp_json} does not exist!' 
            print(expt)
            with exp_json.open('r') as jf:
                eval_dict=json.load(jf)['metric']
            
            for metric, metric_nick in wanted_metrics.items():
                results.append([nickname, metric_nick, eval_dict[metric]*100])

results_df = pd.DataFrame(results, columns=['Runs', 'Set', 'Scores'])

print(results_df)

import matplotlib.pyplot as plt
import seaborn as sns 

import numpy as np

# Set the figure size
plt.figure(figsize=(14, 8))

sns.set_palette(sns.color_palette("pastel", 8))

# plot a bar chart
ax = sns.barplot(x="Runs", y="Scores", hue="Set", data=results_df, estimator=np.median, errorbar=("pi", 95), capsize=.2)

ax.set(title=TITLE)
ax.set(ylim=(YMIN, YMAX))

plt.grid()
plt.tight_layout()
plt.show()

out_dir = Path(OUT_DIR)
out = out_dir / f"{TITLE}.jpg"
ax.figure.savefig(out) 


