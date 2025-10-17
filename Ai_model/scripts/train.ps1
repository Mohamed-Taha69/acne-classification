param(
    [string]$ConfigPath = "configs/default.yaml"
)

Write-Host "Starting training with config: $ConfigPath"
python -m src.training.train --config $ConfigPath

