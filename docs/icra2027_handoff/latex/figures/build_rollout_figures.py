#!/usr/bin/env python3
"""Horizontal, provenance-preserving LaTeX layouts of original rollout photographs."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT.parent/'figures/rollouts'
SEL={'cylinder_sft_fail':[76,85,89],'cylinder_dagger_success':[165,171,184],
     'power_adapter_sft_fail':[619,629,634],'power_adapter_dagger_success':[656,665,681]}
TASKS=[('Cylinder insertion','cylinder_sft_fail','cylinder_dagger_success'),
       ('Power-adapter insertion','power_adapter_sft_fail','power_adapter_dagger_success')]
manifest={'layout':'Each rollout is one row, increasing RPC left to right; SFT above DAgger for each task.','compact_view':'Original pixel rectangle (0,0,320,256), global camera only; raw files unchanged.',
 'selection':'Three manually selected frames per rollout. Power SFT uses RPC 634 instead of 636 as requested. These are qualitative examples, not synchronized or paired trial samples.',
 'outcomes':'User-provided directory labels; exact checkpoint mapping not supplied.','files':[]}
for p in sorted(RAW.glob('*/*.png')):
 manifest['files'].append({'path':p.relative_to(RAW).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'in_paper':int(p.stem.split('_')[1]) in SEL[p.parent.name]})
(ROOT/'figures/rollout_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for compact in (True,False):
 cell=48 if compact else 40
 cols=3 if compact else 9
 width=22+cols*(cell+2); y=5; body=[]
 for ti,(title,sft,dag) in enumerate(TASKS):
  body += [rf'\fill[panel,rounded corners=3pt] (1,-{y}) rectangle ({width-1},-{y+8});',rf'\node[anchor=west,font=\sffamily\bfseries\small,text=ink] at (4,-{y+4}) {{{chr(97+ti)}\quad {title}}};',rf'\node[anchor=east,font=\sffamily\scriptsize,text=ink] at ({width-4},-{y+4}) {{Time $\longrightarrow$}};']
  y+=11
  for folder,label,outcome,col in [(sft,'SFT','Failure','sft'),(dag,'DAgger','Success','dag')]:
   paths=sorted((RAW/folder).glob('*.png'))
   if compact:paths=[p for p in paths if int(p.stem.split('_')[1]) in SEL[folder]]
   ih=cell*(.8 if compact else 1.2)
   body += [rf'\node[align=center,text width=16mm,text={col},font=\sffamily\small] at (10,-{y+ih/2}) '+r'{\textbf{'+label+r'}\\{\scriptsize '+outcome+r'}};']
   for i,p in enumerate(paths):
    x=21+i*(cell+2); opts=f'width={cell}mm'+(',trim=0 128bp 0 0,clip' if compact else '')
    rel=p.relative_to(ROOT.parent).as_posix()
    body += [rf'\node[anchor=north west,inner sep=0] at ({x},-{y}) {{\includegraphics[{opts}]{{../../{rel}}}}};',rf'\node[anchor=north west,inner sep=0,font=\sffamily\fontsize{{6}}{{7}}\selectfont,text=ink] at ({x},-{y+ih+1}) {{RPC {int(p.stem.split("_")[1]):06d}}};']
   y+=ih+6
  y+=3
 height=y+6
 pre=rf'''\documentclass{{article}}
\usepackage[paperwidth={width}mm,paperheight={height}mm,margin=0mm]{{geometry}}
\usepackage{{graphicx,tikz}}
\definecolor{{ink}}{{HTML}}{{26364A}}
\definecolor{{panel}}{{HTML}}{{EEF2F6}}
\definecolor{{sft}}{{HTML}}{{98634E}}
\definecolor{{dag}}{{HTML}}{{187B75}}
\pagestyle{{empty}}\setlength{{\parindent}}{{0pt}}\setlength{{\topskip}}{{0pt}}
\begin{{document}}\noindent
\begin{{tikzpicture}}[x=1mm,y=1mm]
\path[use as bounding box] (0,0) rectangle ({width},-{height});
'''
 name='rollout_comparison' if compact else 'rollout_comparison_all_frames'
 (ROOT/'figures'/f'{name}.tex').write_text(pre+'\n'.join(body)+'\n\\end{tikzpicture}\n\\end{document}\n')
print('Wrote horizontal figure sources; compact power-SFT frame is RPC 634.')
