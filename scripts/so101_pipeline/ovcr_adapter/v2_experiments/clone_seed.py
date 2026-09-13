"""Create independent seed repeats without changing a running configuration."""
import argparse
from pathlib import Path
from omegaconf import OmegaConf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', required=True)
p.add_argument('--seeds', nargs='+', type=int, default=[142,242])
a = p.parse_args()
source = Path(a.config)
for seed in a.seeds:
    cfg = OmegaConf.load(source)
    name = f'{cfg.experiment.name}_seed{seed}'
    destination = source.parent/f'{name}.yaml'
    if destination.exists(): raise FileExistsError(destination)
    cfg.seed = seed
    cfg.experiment.name = name
    cfg.output_dir = str(Path(cfg.output_dir).parent.parent/name/'model')
    OmegaConf.save(cfg,destination,resolve=True)
    print(destination)
