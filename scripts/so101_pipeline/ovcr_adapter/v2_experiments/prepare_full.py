"""Build the full-parameter mixed-v2 baseline from the frozen experiment split."""
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT/'runs/ovcr_adapter_padapter_v2'

def main():
    destination = RUN/'configs/p1_full_balanced_video.yaml'
    if destination.exists():
        raise FileExistsError(f'Refusing to replace existing configuration: {destination}')
    cfg = OmegaConf.load(RUN/'configs/p1_v32_balanced.yaml')
    cfg.model.ovcr_adapter = None
    cfg.model.ovcr_anchor_weight = 0.0
    cfg.model.loss.lambda_video = 1.0
    cfg.freeze_video_dit = False
    cfg.learning_rate = 1e-5
    cfg.experiment.name = 'p1_full_balanced_video'
    cfg.experiment.training_scope = 'full'
    cfg.output_dir = 'runs/ovcr_adapter_padapter_v2/p1_full_balanced_video/model'
    cfg.data.train.video_supervision_require_full_window = True
    cfg.correction_val.video_supervision_require_full_window = True
    OmegaConf.save(cfg, destination, resolve=True)
    print(destination)

if __name__ == '__main__': main()
