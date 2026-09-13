"""B: restart task SFT from AHA pretrained, natural demo/HUMAN mix, four epochs."""
import json
import math
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT/'runs/ovcr_adapter_padapter_v2'
NAME = 'b_full_pretrained_natural_4ep'

def main():
    path = RUN/'configs'/f'{NAME}.yaml'
    if path.exists(): raise FileExistsError(path)
    manifest = json.loads((RUN/'data_manifest.json').read_text())
    cfg = OmegaConf.load(RUN/'configs/p1_full_balanced_video.yaml')
    cfg.init_checkpoint = 'checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt'
    if not (ROOT/cfg.init_checkpoint).is_file(): raise FileNotFoundError(cfg.init_checkpoint)
    cfg.model.checkpoint_shape_adapt = True  # same initialization adaptation as original SO101 SFT
    cfg.data.train.sampling_strategy = 'natural'
    cfg.num_epochs = 4
    steps = math.ceil((manifest['demo_train_frames']+manifest['train']['human_frames'])/12)
    cfg.max_steps = 4*steps
    cfg.save_every = steps
    cfg.eval_every = math.ceil(steps/4)
    cfg.experiment.name = NAME
    cfg.experiment.initialization = 'AHA pretrained, not task step_005328'
    cfg.experiment.steps_per_epoch = steps
    cfg.output_dir = f'runs/ovcr_adapter_padapter_v2/{NAME}/model'
    OmegaConf.save(cfg,path,resolve=True)
    print(OmegaConf.to_yaml(cfg.experiment))
    print(f'epochs=4 steps_per_epoch={steps} max_steps={cfg.max_steps} init={cfg.init_checkpoint}')

if __name__ == '__main__': main()
