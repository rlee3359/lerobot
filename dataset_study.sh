#!/bin/bash
# Define experiment name and time for directory organization
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT="dataset_size_study_${TIMESTAMP}"
# Run all configurations
echo "Starting dataset size study at ${TIMESTAMP}"

# Create timestamp in Hydra format (YYYY-MM-DD/HH-MM-SS)
HYDRA_TIME=$(date "+%Y-%m-%d/%H-%M-%S")

# Full dataset (baseline)
echo "Starting 100% dataset training..."
python lerobot/scripts/train.py policy=diffusion env=pusht '+dataset.split="train"' "hydra.run.dir=outputs/train/${HYDRA_TIME}_pusht_diffusion_100pct" wandb.enable=true 'hydra.job.name=pusht_diffusion_100pct'

# 75% episodes
echo "Starting 75% dataset training..."
python lerobot/scripts/train.py policy=diffusion env=pusht '+dataset.split="train[:18942]"' "hydra.run.dir=outputs/train/${HYDRA_TIME}_pusht_diffusion_75pct" wandb.enable=true 'hydra.job.name=pusht_diffusion_75pct'

# 50% episodes
echo "Starting 50% dataset training..."
python lerobot/scripts/train.py policy=diffusion env=pusht '+dataset.split="train[:12167]"' "hydra.run.dir=outputs/train/${HYDRA_TIME}_pusht_diffusion_50pct" wandb.enable=true 'hydra.job.name=pusht_diffusion_50pct'

# 25% episodes
echo "Starting 25% dataset training..."
python lerobot/scripts/train.py policy=diffusion env=pusht '+dataset.split="train[:6304]"' "hydra.run.dir=outputs/train/${HYDRA_TIME}_pusht_diffusion_25pct" wandb.enable=true 'hydra.job.name=pusht_diffusion_25pct'

# 10% episodes
echo "Starting 10% dataset training..."
python lerobot/scripts/train.py policy=diffusion env=pusht '+dataset.split="train[:2598]"' "hydra.run.dir=outputs/train/${HYDRA_TIME}_pusht_diffusion_10pct" wandb.enable=true 'hydra.job.name=pusht_diffusion_10pct'

echo "All training runs completed!"