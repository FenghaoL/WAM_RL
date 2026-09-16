#!/usr/bin/env python3
"""Read-only audit of recorded metrics; does not launch inference or modify runs."""
import json, math, statistics, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser()
parser.add_argument('--task',choices=['cylinder','padapter'],default='cylinder')
TASK=parser.parse_args().task
BASE=ROOT/('runs/ovcr_adapter_cylinder_hil_v2/video_preview' if TASK=='cylinder' else 'runs/ovcr_adapter_padapter_hil_v3/video_preview')
RUNS={'S':'sft_step_006572','H':'cyl_v2_p1_step_003285','HV':'cyl_v2_vidext_step_003285'}
SR={'S':1/13,'H':5/13,'HV':6/13}
if TASK=='padapter':
 RUNS={'S':'pad_v2_sft_step_005328','H':'pad_v3_p1_step_002664','HV':'pad_v3_vidext_step_002664'}
 SR={'S':10/20,'H':13/20,'HV':11/20}
manifest=json.loads((BASE/'samples_hil_val/manifest.json').read_text())
meta={x['tag']:x for x in manifest['samples']}
rows={k:{p.parent.name:json.loads(p.read_text()) for p in (BASE/'val'/v).glob('*/metrics.json')} for k,v in RUNS.items()}
assert all(set(v)==set(meta) for v in rows.values())
def pearson(a,b):
 a=[x-statistics.mean(a) for x in a];b=[x-statistics.mean(b) for x in b]
 return sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
out={'unit':'checkpoint-level association, not matched trial-level correlation','validation_windows':len(meta),'source_episodes':len({m['source_episode'] for m in meta.values()}),'padding':[{'sample':k,'frozen_frames':v['gt_frozen_video_frames']} for k,v in meta.items() if v['gt_frozen_video_frames']],'models':{},'paired':{},'associations':{}}
for k,v in rows.items():
 summary=json.loads((BASE/'val'/RUNS[k]/'summary.json').read_text())
 out['models'][k]={'weights':summary['weights'],'success_rate':SR[k],'means':{key:statistics.mean(x[key] for x in v.values()) for key in ['online_future_rgb_psnr','training_future_rgb_psnr','none_future_rgb_psnr','fixed_rgb_psnr','wrist_rgb_psnr','future_rgb_mse']},'by_segment_length':{}}
 for label,lo,hi in [('short',0,20),('medium',20,40),('long',40,99999)]:
  values=[v[t]['future_rgb_psnr'] for t,m in meta.items() if lo<=m['segment_frames']<hi]
  out['models'][k]['by_segment_length'][label]={'n':len(values),'online_psnr':statistics.mean(values)}
for mode in ['online','training','none']:
 out['paired'][mode]={}
 for a,b in [('H','S'),('HV','S'),('HV','H')]:
  vals=[rows[a][t][mode+'_future_rgb_psnr']-rows[b][t][mode+'_future_rgb_psnr'] for t in sorted(meta)]
  out['paired'][mode][a+'-'+b]={'wins':sum(x>0 for x in vals),'n':len(vals),'mean':statistics.mean(vals),'median':statistics.median(vals),'min':min(vals),'max':max(vals),'deltas':vals}
 means=[out['models'][k]['means'][mode+'_future_rgb_psnr'] for k in RUNS]
 out['associations'][mode]={'pearson':pearson(means,list(SR.values())),'spearman':pearson([sorted(means).index(x) for x in means],[sorted(SR.values()).index(x) for x in SR.values()]),'n_checkpoints':3,'inferential_claim':False}
output=Path(__file__).with_name('video_results_audit.json' if TASK=='cylinder' else 'padapter_video_results_audit.json');output.write_text(json.dumps(out,indent=2)+'\n');print(output)
print(json.dumps(out['associations'],indent=2))
