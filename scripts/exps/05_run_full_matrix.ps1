param(
  [string]$QwenModel = "Qwen/Qwen2.5-VL-3B-Instruct",
  [string]$PreprocessedDatasetDir = "./runs/mini_imagenet_preprocessed_392",
  [string]$RunsRoot = "./runs/exps",
  [string]$ReportRoot = "./runs/exps/reports",
  [string]$ScoredRoot = "./runs/exps/scored_reports",
  [string[]]$TeacherProbes = @("T-mean-full", "T-attn-full", "T-cross-full"),
  [string[]]$Losses = @("D-mse", "D-expl-only", "D-mse-expl"),
  [string[]]$Baselines = @("no_image", "mismatched"),
  [string]$JudgeBackend = "hf",
  [string]$JudgeDevice = "auto",
  [string]$JudgeQuantization = "bnb4",
  [string[]]$JudgeIds = @("qwen3_8b"),
  [string[]]$JudgeModels = @("Qwen/Qwen3-8B"),
  [string[]]$JudgeBackends = @(),
  [int]$JudgeLimit = 0,
  [string[]]$JudgeModes = @("direct", "reverse", "blind"),
  [string]$WandbRunSuffix = "valtest-es",
  [switch]$SkipTeacherProbes,
  [switch]$SkipDistillation,
  [switch]$SkipReports,
  [switch]$SkipBaselines,
  [switch]$SkipScoring,
  [switch]$NoWandb
)

. "$PSScriptRoot/common.ps1"

if (-not $SkipTeacherProbes) {
  Invoke-ExpStep "Block A: teacher probes" {
    & "$PSScriptRoot/01_train_teacher_probes.ps1" `
      -QwenModel $QwenModel `
      -PreprocessedDatasetDir $PreprocessedDatasetDir `
      -Only $TeacherProbes `
      -NoWandb:$NoWandb
  }
}

if (-not $SkipDistillation) {
  Invoke-ExpStep "Block B: distillation matrix" {
    & "$PSScriptRoot/02_run_distillation_matrix.ps1" `
      -QwenModel $QwenModel `
      -PreprocessedDatasetDir $PreprocessedDatasetDir `
      -RunsRoot $RunsRoot `
      -TeacherProbes $TeacherProbes `
      -Losses $Losses `
      -WandbRunSuffix $WandbRunSuffix `
      -NoWandb:$NoWandb
  }
}

if (-not $SkipReports) {
  Invoke-ExpStep "Block C: generation reports" {
    & "$PSScriptRoot/03_generate_student_reports.ps1" `
      -QwenModel $QwenModel `
      -PreprocessedDatasetDir $PreprocessedDatasetDir `
      -RunsRoot $RunsRoot `
      -ReportRoot $ReportRoot `
      -TeacherProbes $TeacherProbes `
      -Losses $Losses `
      -Baselines $Baselines `
      -SkipBaselines:$SkipBaselines `
      -NoWandb:$NoWandb
  }
}

if (-not $SkipScoring) {
  Invoke-ExpStep "Block C: report scoring" {
    & "$PSScriptRoot/04_score_generation_reports.ps1" `
      -ReportRoot $ReportRoot `
      -ScoredRoot $ScoredRoot `
      -TeacherProbes $TeacherProbes `
      -Losses $Losses `
      -JudgeBackend $JudgeBackend `
      -JudgeDevice $JudgeDevice `
      -JudgeQuantization $JudgeQuantization `
      -JudgeIds $JudgeIds `
      -JudgeModels $JudgeModels `
      -JudgeBackends $JudgeBackends `
      -JudgeLimit $JudgeLimit `
      -JudgeModes $JudgeModes `
      -NoWandb:$NoWandb
  }
}
