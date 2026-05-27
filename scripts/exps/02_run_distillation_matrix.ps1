param(
  [string]$QwenModel = "Qwen/Qwen2.5-VL-3B-Instruct",
  [string]$PreprocessedDatasetDir = "./runs/mini_imagenet_preprocessed_392",
  [string]$RunsRoot = "./runs/exps",
  [string]$StudentModel = "vit_base_patch16_224",
  [int]$Epochs = 20,
  [int]$TrainBatchSize = 16,
  [int]$PrecomputeBatchSize = 4,
  [int]$NumWorkers = 4,
  [double]$LearningRate = 3e-4,
  [int]$EarlyStoppingPatience = 5,
  [string]$WandbRunSuffix = "valtest-es",
  [int]$Seed = 42,
  [string]$TorchDtype = "float16",
  [string[]]$TeacherProbes = @("T-attn-full"),
  [string[]]$Losses = @("D-mse", "D-expl-only", "D-mse-expl"),
  [switch]$OverwritePrecompute,
  [switch]$NoWandb
)

. "$PSScriptRoot/common.ps1"

Assert-PathExists -Path $PreprocessedDatasetDir -Description "Dataset preprocessato"

$teacherMatrix = Get-TeacherProbeMatrix
$lossMatrix = Get-DistillationLossMatrix
$selectedTeachers = @($teacherMatrix | Where-Object { $TeacherProbes -contains $_.Id -or $TeacherProbes -contains $_.Pooling })
$selectedLosses = @($lossMatrix | Where-Object { $Losses -contains $_.Id -or $Losses -contains $_.Suffix })

if ($selectedTeachers.Count -eq 0 -and ($selectedLosses | Where-Object { $_.Id -ne "D-mse" }).Count -gt 0) {
  throw "Nessun teacher probe selezionato."
}
if ($selectedLosses.Count -eq 0) {
  throw "Nessuna loss selezionata."
}

foreach ($loss in $selectedLosses) {
  $teachersForLoss = $selectedTeachers
  if ($loss.Id -eq "D-mse") {
    $teachersForLoss = @(Get-BaseTeacher)
  }

  foreach ($teacher in $teachersForLoss) {
    $teacherSlug = Get-TeacherProbeSlug -TeacherProbeId $teacher.Id
    $precomputedDir = Join-Path $RunsRoot "precompute_${teacherSlug}_vitbase_$($loss.Suffix)"
    $studentDir = Join-Path $RunsRoot "student_${teacherSlug}_vitbase_$($loss.Suffix)"

    $teacherArgs = @("--teacher-model-id", $QwenModel, "--attention-source", "uniform")
    if (-not $teacher.IsBase) {
      $teacherBest = Join-Path $teacher.OutputDir "best"
      Assert-PathExists -Path $teacherBest -Description "Checkpoint teacher probe $($teacher.Id)"
      $teacherArgs = @("--teacher-model-dir", $teacherBest, "--attention-source", "gradcam")
    }

    $overwriteArgs = @()
    if ($OverwritePrecompute) {
      $overwriteArgs = @("--overwrite-precompute")
    }

    $wandbArgs = @()
    if (-not $NoWandb) {
      $wandbRunName = "student-${teacherSlug}-vitbase-$($loss.Suffix)-$WandbRunSuffix"
      $wandbArgs = @(
        "--wandb",
        "--wandb-project", "compvis-distillation-matrix",
        "--wandb-run-name", $wandbRunName
      )
    }

    Invoke-ExpStep "Distill $($teacher.Id) x $($loss.Id)" {
      Invoke-Uv run compvis-run-expl-vit-distillation `
        @teacherArgs `
        --preprocessed-dataset-dir $PreprocessedDatasetDir `
        --precomputed-dir $precomputedDir `
        --student-output-dir $studentDir `
        --student-model $StudentModel `
        --precompute-batch-size $PrecomputeBatchSize `
        --train-batch-size $TrainBatchSize `
        --num-workers $NumWorkers `
        --epochs $Epochs `
        --lr $LearningRate `
        --early-stopping-patience $EarlyStoppingPatience `
        --eval-every 1 `
        --seed $Seed `
        --torch-dtype $TorchDtype `
        --distillation-target qwen_image_embeddings `
        --lambda-global $loss.LambdaGlobal `
        --lambda-expl $loss.LambdaExpl `
        --lambda-token-mse $loss.LambdaTokenMse `
        --pretrained-student `
        @overwriteArgs `
        @wandbArgs
    }
  }
}
