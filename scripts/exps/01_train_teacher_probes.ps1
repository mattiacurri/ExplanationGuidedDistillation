param(
  [string]$QwenModel = "Qwen/Qwen2.5-VL-3B-Instruct",
  [string]$PreprocessedDatasetDir = "./runs/mini_imagenet_preprocessed_392",
  [int]$Epochs = 10,
  [int]$BatchSize = 16,
  [double]$LearningRate = 3e-4,
  [double]$WeightDecay = 0.01,
  [double]$WarmupRatio = 0.1,
  [int]$NumWorkers = 2,
  [int]$Seed = 42,
  [string]$TorchDtype = "float16",
  [string[]]$Only = @(),
  [switch]$NoWandb
)

. "$PSScriptRoot/common.ps1"

Assert-PathExists -Path $PreprocessedDatasetDir -Description "Dataset preprocessato"

$variants = Get-TeacherProbeMatrix
if ($Only.Count -gt 0) {
  $variants = @($variants | Where-Object { $Only -contains $_.Id -or $Only -contains $_.Pooling })
}
if ($variants.Count -eq 0) {
  throw "Nessun teacher probe selezionato. Usa ID come T-attn-full o pooling come attention."
}

foreach ($variant in $variants) {
  $wandbArgs = @()
  if (-not $NoWandb) {
    $wandbArgs = @(
      "--wandb",
      "--wandb-project", "compvis-teacher-probes",
      "--wandb-run-name", "qwen25vl3b-$($variant.Pooling)-probe-full"
    )
  }

  Invoke-ExpStep "Teacher probe $($variant.Id) ($($variant.Pooling))" {
    Invoke-Uv run compvis-finetune-vlm-vit-mini-imagenet `
      --vlm-model $QwenModel `
      --preprocessed-dataset-dir $PreprocessedDatasetDir `
      --output-dir $variant.OutputDir `
      --pooling $variant.Pooling `
      --epochs $Epochs `
      --batch-size $BatchSize `
      --lr $LearningRate `
      --weight-decay $WeightDecay `
      --warmup-ratio $WarmupRatio `
      --num-workers $NumWorkers `
      --eval-every 1 `
      --test-every 1 `
      --torch-dtype $TorchDtype `
      --seed $Seed `
      @wandbArgs
  }
}
