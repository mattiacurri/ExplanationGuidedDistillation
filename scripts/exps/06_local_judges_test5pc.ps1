param(
  [string]$QwenModel = "Qwen/Qwen2.5-VL-3B-Instruct",
  [string]$PreprocessedDatasetDir = "./runs/mini_imagenet_preprocessed_392",
  [string]$RunsRoot = "./runs/exps",
  [string]$ReportRoot = "./runs/exps/reports_test_5pc",
  [string]$ScoredRoot = "./runs/exps/scored_reports_test_5pc",
  [int]$SamplesPerClass = 5,
  [string]$Split = "test",
  [int]$MaxNewTokens = 128,
  [int]$Seed = 42,
  [string]$TorchDtype = "float16",
  [string[]]$TeacherProbes = @("T-attn-full", "T-cross-full", "T-mean-full"),
  [string[]]$Losses = @("D-mse", "D-expl-only", "D-mse-expl"),
  [string]$LmStudioBaseUrl = "http://127.0.0.1:1234/v1",
  [int]$LmStudioPort = 1234,
  [string]$LmStudioBind = "127.0.0.1",
  [int]$LmStudioContextLength = 32768,
  [int]$LmStudioParallel = 1,
  [int]$LmStudioTimeoutSeconds = 180,
  [int]$JudgeLimit = 0,
  [switch]$SkipReportGeneration,
  [switch]$SkipTextJudges,
  [switch]$SkipVisionJudges,
  [switch]$SkipAggregation,
  [switch]$NoForce
)

. "$PSScriptRoot/common.ps1"

Assert-PathExists -Path $PreprocessedDatasetDir -Description "Dataset preprocessato"

$localJudges = @(
  [pscustomobject]@{
    Id = "qwen35_9b"
    Model = "qwen/qwen3.5-9b"
    LoadKey = "qwen/qwen3.5-9b"
  },
  [pscustomobject]@{
    Id = "glm_46v_flash"
    Model = "zai-org/glm-4.6v-flash"
    LoadKey = "zai-org/glm-4.6v-flash"
  },
  [pscustomobject]@{
    Id = "minicpm_v45"
    Model = "minicpm-v-4_5"
    LoadKey = "minicpm-v-4_5"
  }
)

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
New-Item -ItemType Directory -Force -Path $ScoredRoot | Out-Null

function Get-SelectedExperimentSpecs {
  $specs = @()
  foreach ($loss in $selectedLosses) {
    $teachersForLoss = $selectedTeachers
    if ($loss.Id -eq "D-mse") {
      $teachersForLoss = @(Get-BaseTeacher)
    }

    foreach ($teacher in $teachersForLoss) {
      $teacherSlug = Get-TeacherProbeSlug -TeacherProbeId $teacher.Id
      $studentBest = Join-Path (Join-Path $RunsRoot "student_${teacherSlug}_vitbase_$($loss.Suffix)") "best"
      $reportPath = Join-Path $ReportRoot "qwen_report_${teacherSlug}_vitbase_$($loss.Suffix).json"
      $judgedPath = Join-Path $ScoredRoot "qwen_report_${teacherSlug}_vitbase_$($loss.Suffix)_judged.json"
      $specs += [pscustomobject]@{
        Teacher = $teacher
        Loss = $loss
        TeacherSlug = $teacherSlug
        StudentBest = $studentBest
        ReportPath = $reportPath
        JudgedPath = $judgedPath
      }
    }
  }
  return $specs
}

$experimentSpecs = @(Get-SelectedExperimentSpecs)

if (-not $SkipReportGeneration) {
  foreach ($spec in $experimentSpecs) {
    Assert-PathExists -Path $spec.StudentBest -Description "Checkpoint student $($spec.Teacher.Id) x $($spec.Loss.Id)"
    Invoke-ExpStep "Generate report $($spec.Teacher.Id) x $($spec.Loss.Id)" {
      Invoke-Uv run compvis-qwen-student-prompt-report `
        --qwen-model $QwenModel `
        --preprocessed-dataset-dir $PreprocessedDatasetDir `
        --student-checkpoint-dir $spec.StudentBest `
        --report-path $spec.ReportPath `
        --samples-per-class $SamplesPerClass `
        --split $Split `
        --max-new-tokens $MaxNewTokens `
        --torch-dtype $TorchDtype `
        --seed $Seed
    }

    Invoke-ExpStep "Add negative baselines $($spec.Teacher.Id) x $($spec.Loss.Id)" {
      Invoke-Uv run compvis-augment-qwen-report-baselines `
        --report-path $spec.ReportPath `
        --in-place `
        --baselines no_image mismatched
    }
  }
}

foreach ($spec in $experimentSpecs) {
  Assert-PathExists -Path $spec.ReportPath -Description "Report generazione $($spec.Teacher.Id) x $($spec.Loss.Id)"
  Copy-Item -LiteralPath $spec.ReportPath -Destination $spec.JudgedPath -Force
}

$forceArgs = @()
if (-not $NoForce) {
  $forceArgs = @("--force")
}

$limitArgs = @()
if ($JudgeLimit -gt 0) {
  $limitArgs = @("--limit", "$JudgeLimit")
}

foreach ($judge in $localJudges) {
  Invoke-ExpStep "Load LM Studio judge $($judge.Id)" {
    lms unload --all
    if ($LASTEXITCODE -ne 0) {
      throw "lms unload --all fallito con exit code $LASTEXITCODE"
    }
    lms server start --port $LmStudioPort --bind $LmStudioBind
    if ($LASTEXITCODE -ne 0) {
      throw "lms server start fallito con exit code $LASTEXITCODE"
    }
    lms load $judge.LoadKey `
      --gpu max `
      --context-length $LmStudioContextLength `
      --parallel $LmStudioParallel `
      --identifier $judge.Model `
      -y
    if ($LASTEXITCODE -ne 0) {
      throw "lms load $($judge.LoadKey) fallito con exit code $LASTEXITCODE"
    }
  }

  foreach ($spec in $experimentSpecs) {
    if (-not $SkipTextJudges) {
      Invoke-ExpStep "Text judge $($judge.Id) $($spec.Teacher.Id) x $($spec.Loss.Id)" {
        Invoke-Uv run compvis-llm-judge-qwen-report `
          --report-path $spec.JudgedPath `
          --in-place `
          --backend lmstudio `
          --judge-id "$($judge.Id)_text" `
          --lmstudio-model $judge.Model `
          --lmstudio-base-url $LmStudioBaseUrl `
          --lmstudio-auto-load `
          --lmstudio-gpu max `
          --lmstudio-context-length $LmStudioContextLength `
          --lmstudio-parallel $LmStudioParallel `
          --modes blind `
          --lmstudio-timeout-seconds $LmStudioTimeoutSeconds `
          @forceArgs `
          @limitArgs
      }
    }

    if (-not $SkipVisionJudges) {
      Invoke-ExpStep "Vision judge $($judge.Id) $($spec.Teacher.Id) x $($spec.Loss.Id)" {
        Invoke-Uv run compvis-vlm-judge-qwen-report `
          --report-path $spec.JudgedPath `
          --in-place `
          --backend lmstudio `
          --judge-id "$($judge.Id)_vision" `
          --lmstudio-model $judge.Model `
          --lmstudio-base-url $LmStudioBaseUrl `
          --lmstudio-auto-load `
          --lmstudio-gpu max `
          --lmstudio-context-length $LmStudioContextLength `
          --lmstudio-parallel $LmStudioParallel `
          --preprocessed-dataset-dir $PreprocessedDatasetDir `
          --lmstudio-timeout-seconds $LmStudioTimeoutSeconds `
          @forceArgs `
          @limitArgs
      }
    }
  }
}

if (-not $SkipAggregation) {
  foreach ($spec in $experimentSpecs) {
    Invoke-ExpStep "Aggregate text judges $($spec.Teacher.Id) x $($spec.Loss.Id)" {
      Invoke-Uv run compvis-aggregate-qwen-judges `
        --report-path $spec.JudgedPath `
        --in-place `
        --namespace llm_judges `
        --mode blind `
        --judges qwen35_9b_text glm_46v_flash_text minicpm_v45_text
    }

    Invoke-ExpStep "Aggregate vision judges $($spec.Teacher.Id) x $($spec.Loss.Id)" {
      Invoke-Uv run compvis-aggregate-qwen-judges `
        --report-path $spec.JudgedPath `
        --in-place `
        --namespace vlm_judges `
        --mode vision_blind `
        --judges qwen35_9b_vision glm_46v_flash_vision minicpm_v45_vision
    }
  }
}

Write-Host ""
Write-Host "Done. Reports: $ScoredRoot" -ForegroundColor Green
