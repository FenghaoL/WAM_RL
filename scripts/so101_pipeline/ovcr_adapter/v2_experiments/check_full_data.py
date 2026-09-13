"""Read real HUMAN boundary samples and verify strict video eligibility."""
import json
from pathlib import Path
from omegaconf import OmegaConf
from ahawam.runtime import build_datasets

root=Path(__file__).resolve().parents[4]
cfg=OmegaConf.load(root/'runs/ovcr_adapter_padapter_v2/configs/p1_full_balanced_video.yaml')
train,_=build_datasets(cfg.data,cfg.model)
human=train.correction_dataset
base=human.lerobot_dataset
checks=[]
for predicate in (lambda n:n>=49,lambda n:n<49):
    segment_id=next(i for i,s in enumerate(base.human_segments) if predicate(s.length))
    segment=base.human_segments[segment_id]
    start=int(base.episode_data_index['from'][segment_id])
    offsets=sorted({0,max(0,segment.length-49),max(0,segment.length-48),segment.length-1})
    for offset in offsets:
        sample=human[start+offset]
        expected=segment.length-offset>=49
        actual=bool(sample['video_supervision_mask'])
        assert actual==expected,(segment.length,offset,actual,expected)
        assert int(sample['data_source'])==1
        checks.append(dict(segment_length=segment.length,offset=offset,remaining=segment.length-offset,video_supervised=actual,padded_images=int(sample['image_is_pad'].sum())))
report=dict(demo_length=train.demo_length,human_length=train.correction_length,boundary_checks=checks)
(root/'runs/ovcr_adapter_padapter_v2/full_video_data_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
