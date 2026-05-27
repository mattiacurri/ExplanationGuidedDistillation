param(
  [string]$QwenModel = "Qwen/Qwen2.5-VL-3B-Instruct",
  [string]$PreprocessedDatasetDir = "./runs/mini_imagenet_preprocessed_392",
  [string]$FineTuneDir = "./runs/vlm_qwen25vl3b_premerge_finetuned_392_c300",
  [string]$PrecomputedDir = "./runs/expl_distill_qwen_embeddings_392_c300",
  [string]$StudentDir = "./runs/expl_student_qwen_embeddings_392_c300_tokenmse",
  [string]$ReportPath = "./runs/qwen_student_prompt_report_392_c300.json",
  [int]$ExamplesPerClass = 300,
  [int]$ReportPerClass = 5,
  [int]$FineTuneEpochs = 10,
  [int]$DistillEpochs = 50,
  [int]$BatchSize = 16,
  [int]$PrecomputeBatchSize = 4,
  [int]$NumWorkers = 2,
  [int]$MaxNewTokens = 128,
  [int]$Seed = 42,
  [string]$TorchDtype = "float16",
  [switch]$NoWandb,
  [switch]$SkipFineTune,
  [switch]$SkipDistillation,
  [switch]$SkipReport,
  [switch]$OverwritePrecompute
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
  param(
    [string]$Name,
    [scriptblock]$Body
  )
  Write-Host ""
  Write-Host "==> $Name" -ForegroundColor Cyan
  & $Body
}

$fineTuneBest = Join-Path $FineTuneDir "best"
$studentBest = Join-Path $StudentDir "best"
$wandbFineTune = @()
$wandbDistill = @()
if (-not $NoWandb) {
  $wandbFineTune = @(
    "--wandb",
    "--wandb-project", "compvis-finetuning",
    "--wandb-run-name", "qwen25vl3b-premerge-finetune-392-c$ExamplesPerClass"
  )
  $wandbDistill = @(
    "--wandb",
    "--wandb-project", "compvis-distillation",
    "--wandb-run-name", "qwen25vl3b-qwen-embeddings-392-distill-c$ExamplesPerClass-tokenmse"
  )
}

if (-not $SkipFineTune) {
  Invoke-Step "Fine-tuning Qwen visual su $ExamplesPerClass esempi/classe" {
    uv run compvis-finetune-vlm-vit-mini-imagenet `
      --vlm-model $QwenModel `
      --preprocessed-dataset-dir $PreprocessedDatasetDir `
      --output-dir $FineTuneDir `
      --pooling attention `
      --subset-per-class $ExamplesPerClass `
      --epochs $FineTuneEpochs `
      --batch-size $BatchSize `
      --lr 3e-4 `
      --weight-decay 0.01 `
      --warmup-ratio 0.1 `
      --num-workers $NumWorkers `
      --eval-every 1 `
      --eval-subset-per-class 100 `
      --seed $Seed `
      --torch-dtype $TorchDtype `
      @wandbFineTune
  }
}

if (-not (Test-Path -LiteralPath $fineTuneBest)) {
  throw "Checkpoint fine-tuning non trovato: $fineTuneBest"
}

if (-not $SkipDistillation) {
  $overwriteArgs = @()
  if ($OverwritePrecompute) {
    $overwriteArgs = @("--overwrite-precompute")
  }

  Invoke-Step "Distillazione verso Qwen image embeddings" {
    uv run compvis-run-expl-vit-distillation `
      --teacher-model-dir $fineTuneBest `
      --preprocessed-dataset-dir $PreprocessedDatasetDir `
      --precomputed-dir $PrecomputedDir `
      --student-output-dir $StudentDir `
      --subset-per-class $ExamplesPerClass `
      --precompute-batch-size $PrecomputeBatchSize `
      --train-batch-size $BatchSize `
      --num-workers $NumWorkers `
      --epochs $DistillEpochs `
      --lr 3e-4 `
      --seed $Seed `
      --torch-dtype $TorchDtype `
      --distillation-target qwen_image_embeddings `
      --lambda-global 1.0 `
      --lambda-expl 0.5 `
      --lambda-token-mse 2.0 `
      --pretrained-student `
      @overwriteArgs `
      @wandbDistill
  }
}

if (-not (Test-Path -LiteralPath $studentBest)) {
  throw "Checkpoint student non trovato: $studentBest"
}

if (-not $SkipReport) {
  Invoke-Step "Report prompt su test set: $ReportPerClass immagini/classe" {
    uv run compvis-qwen-student-prompt-report `
      --qwen-model $QwenModel `
      --preprocessed-dataset-dir $PreprocessedDatasetDir `
      --student-checkpoint-dir $studentBest `
      --report-path $ReportPath `
      --samples-per-class $ReportPerClass `
      --split test `
      --max-new-tokens $MaxNewTokens `
      --torch-dtype $TorchDtype `
      --seed $Seed
  }
}

Write-Host ""
Write-Host "Pipeline completata." -ForegroundColor Green
Write-Host "Fine-tuned teacher: $fineTuneBest"
Write-Host "Student: $studentBest"
Write-Host "Report: $ReportPath"
