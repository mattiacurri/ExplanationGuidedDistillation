param(
  [Parameter(Mandatory = $true)]
  [string]$TeacherProbe,
  [Parameter(Mandatory = $true)]
  [string]$Loss,
  [string]$QwenModel = "Qwen/Qwen2.5-VL-3B-Instruct",
  [string]$PreprocessedDatasetDir = "./runs/mini_imagenet_preprocessed_392",
  [string]$RunsRoot = "./runs/exps",
  [string]$ReportRoot = "./runs/exps/reports",
  [string]$ScoredRoot = "./runs/exps/scored_reports",
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
  [switch]$SkipTeacherProbe,
  [switch]$SkipDistillation,
  [switch]$SkipReport,
  [switch]$SkipBaselines,
  [switch]$SkipScoring,
  [switch]$NoWandb
)

$ErrorActionPreference = "Stop"
$expsRoot = Split-Path -Parent $PSScriptRoot

if (-not $SkipTeacherProbe -and $TeacherProbe -ne "T-base") {
  & "$expsRoot/01_train_teacher_probes.ps1" `
    -QwenModel $QwenModel `
    -PreprocessedDatasetDir $PreprocessedDatasetDir `
    -Only $TeacherProbe `
    -NoWandb:$NoWandb
}

if (-not $SkipDistillation) {
  & "$expsRoot/02_run_distillation_matrix.ps1" `
    -QwenModel $QwenModel `
    -PreprocessedDatasetDir $PreprocessedDatasetDir `
    -RunsRoot $RunsRoot `
    -TeacherProbes $TeacherProbe `
    -Losses $Loss `
    -WandbRunSuffix $WandbRunSuffix `
    -NoWandb:$NoWandb
}

if (-not $SkipReport) {
  & "$expsRoot/03_generate_student_reports.ps1" `
    -QwenModel $QwenModel `
    -PreprocessedDatasetDir $PreprocessedDatasetDir `
    -RunsRoot $RunsRoot `
    -ReportRoot $ReportRoot `
    -TeacherProbes $TeacherProbe `
    -Losses $Loss `
    -Baselines $Baselines `
    -SkipBaselines:$SkipBaselines `
    -NoWandb:$NoWandb
}

if (-not $SkipScoring) {
  & "$expsRoot/04_score_generation_reports.ps1" `
    -ReportRoot $ReportRoot `
    -ScoredRoot $ScoredRoot `
    -TeacherProbes $TeacherProbe `
    -Losses $Loss `
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
