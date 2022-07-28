from pathlib import Path
import json 

rootdir='/Users/levan/Workspace/improved-instance-segm/outputs'
rootdir_path = Path(rootdir)

MAX_REPS = 3
DEST_TESTDIR_NAME = "test-best"
# EXTRA_TESTDIR_NAME = "test_coco_occ_person"
EXTRA_TESTDIR_NAME = "test_coco_val_person"
EVAL_JSON_NAME = "eval.json"

# SET_INDEX = 3
SET_INDEX = 4

wanted_basenames = [
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP_aug-5e',
    "mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP_aug-5e-SchB",
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP_aug-15e',
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP-5e',
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP-5e-higherLR',
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP-5e-nodice',
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP-5e-SchB',
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-reduceLR',
    'mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-SCP-5e',
]
    
results = []
for basename in wanted_basenames:
    expts = [ basename ] + [ f'{basename}-{i}' for i in range(2, MAX_REPS+1) ]
    for expt in expts:
        expt_dir = rootdir_path/expt 
        if expt_dir.is_dir():
            exp_json =expt_dir / EXTRA_TESTDIR_NAME / EVAL_JSON_NAME
            assert exp_json.is_file(),f'{exp_json} does not exist!' 
            print(expt)
            with exp_json.open('r') as jf:
                eval_dict=json.load(jf)['metric']
            new_dict = {}
            for k, v in eval_dict.items():
                new_dict[f"{SET_INDEX}_{k}"] = v
            # print(new_dict)

            dst_exp_json =expt_dir / DEST_TESTDIR_NAME / EVAL_JSON_NAME
            orig_exp_json = dst_exp_json.parent / f"{dst_exp_json.stem}_orig{SET_INDEX}.json"
            assert dst_exp_json.is_file(),f'{dst_exp_json} does not exist!' 
            with dst_exp_json.open('r') as jf:
                dst_eval_dict_full=json.load(jf)

            # copy original
            with orig_exp_json.open('w') as wf:
                json.dump(dst_eval_dict_full, wf)

            dst_eval_dict_full['metric'].update(new_dict)
            print(dst_eval_dict_full)

            # overwrite updated dict
            with dst_exp_json.open('w') as wf:
                json.dump(dst_eval_dict_full, wf)
