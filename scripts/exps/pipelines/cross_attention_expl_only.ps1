param(
  [switch]$SkipTeacherProbe,
  [switch]$SkipDistillation,
  [switch]$SkipReport,
  [switch]$SkipScoring,
  [switch]$NoWandb
)

& "$PSScriptRoot/run_single_pipeline.ps1" `
  -TeacherProbe T-cross-full `
  -Loss D-expl-only `
  -SkipTeacherProbe:$SkipTeacherProbe `
  -SkipDistillation:$SkipDistillation `
  -SkipReport:$SkipReport `
  -SkipScoring:$SkipScoring `
  -NoWandb:$NoWandb
