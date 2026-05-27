param(
  [string]$QwenModel = "Qwen/Qwen2.5-VL-3B-Instruct",
  [string]$PreprocessedDatasetDir = "./runs/mini_imagenet_preprocessed_392",
  [string]$RunsRoot = "./runs/exps",
  [string]$ReportRoot = "./runs/exps/reports",
  [int]$SamplesPerClass = 1,
  [string]$Split = "test",
  [int]$MaxNewTokens = 128,
  [int]$Seed = 42,
  [string]$TorchDtype = "float16",
  [string[]]$Baselines = @("no_image", "mismatched"),
  [switch]$SkipBaselines,
  [string[]]$TeacherProbes = @("T-attn-full"),
  [string[]]$Losses = @("D-mse", "D-expl-only", "D-mse-expl"),
  [switch]$NoWandb,
  [string]$WandbProject = "compvis-generation-reports",
  [string]$WandbEntity = "",
  [string]$WandbMode = ""
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

New-Item -ItemType Directory -Force -Path $ReportRoot | Out-Null

foreach ($loss in $selectedLosses) {
  $teachersForLoss = $selectedTeachers
  if ($loss.Id -eq "D-mse") {
    $teachersForLoss = @(Get-BaseTeacher)
  }

  foreach ($teacher in $teachersForLoss) {
    $teacherSlug = Get-TeacherProbeSlug -TeacherProbeId $teacher.Id
    $studentBest = Join-Path (Join-Path $RunsRoot "student_${teacherSlug}_vitbase_$($loss.Suffix)") "best"
    Assert-PathExists -Path $studentBest -Description "Checkpoint student $($teacher.Id) x $($loss.Id)"

    $reportPath = Join-Path $ReportRoot "qwen_report_${teacherSlug}_vitbase_$($loss.Suffix).json"
    Invoke-ExpStep "Generate report $($teacher.Id) x $($loss.Id)" {
      Invoke-Uv run compvis-qwen-student-prompt-report `
        --qwen-model $QwenModel `
        --preprocessed-dataset-dir $PreprocessedDatasetDir `
        --student-checkpoint-dir $studentBest `
        --report-path $reportPath `
        --samples-per-class $SamplesPerClass `
        --split $Split `
        --max-new-tokens $MaxNewTokens `
        --torch-dtype $TorchDtype `
        --seed $Seed
    }

    if (-not $SkipBaselines -and $Baselines.Count -gt 0) {
      Invoke-ExpStep "Add negative baselines $($teacher.Id) x $($loss.Id)" {
        Invoke-Uv run compvis-augment-qwen-report-baselines `
          --report-path $reportPath `
          --in-place `
          --baselines $Baselines
      }
    }

    if (-not $NoWandb) {
      $metadata = @{
        stage = "generation_report"
        teacher_probe_id = $teacher.Id
        teacher_probe_pooling = $teacher.Pooling
        teacher_probe_dir = $teacher.OutputDir
        teacher_probe_slug = $teacherSlug
        distillation_loss_id = $loss.Id
        distillation_loss_suffix = $loss.Suffix
        lambda_global = $loss.LambdaGlobal
        lambda_expl = $loss.LambdaExpl
        lambda_token_mse = $loss.LambdaTokenMse
        student_model = "vit_base_patch16_224"
        student_checkpoint_dir = $studentBest
        qwen_model = $QwenModel
        preprocessed_dataset_dir = $PreprocessedDatasetDir
        split = $Split
        samples_per_class = $SamplesPerClass
        max_new_tokens = $MaxNewTokens
        seed = $Seed
        torch_dtype = $TorchDtype
        local_path = $reportPath
      }
      $artifactName = "qwen-report-${teacherSlug}-vitbase-$($loss.Suffix)"
      Publish-WandbArtifact `
        -Path $reportPath `
        -Project $WandbProject `
        -RunName "upload-$artifactName" `
        -ArtifactName $artifactName `
        -ArtifactType "generation-report" `
        -Metadata $metadata `
        -Entity $WandbEntity `
        -Mode $WandbMode
    }
  }
}
