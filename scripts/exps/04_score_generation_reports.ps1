param(
  [string]$ReportRoot = "./runs/exps/reports",
  [string]$ScoredRoot = "./runs/exps/scored_reports",
  [string]$BertScoreDevice = "cuda",
  [string]$JudgeBackend = "hf",
  [string]$JudgeDevice = "auto",
  [string]$JudgeQuantization = "bnb4",
  [string[]]$JudgeIds = @("qwen3_8b"),
  [string[]]$JudgeModels = @("Qwen/Qwen3-8B"),
  [string[]]$JudgeBackends = @(),
  [int]$JudgeLimit = 0,
  [string[]]$JudgeModes = @("direct", "reverse", "blind"),
  [string[]]$TeacherProbes = @("T-attn-full"),
  [string[]]$Losses = @("D-mse", "D-expl-only", "D-mse-expl"),
  [switch]$NoWandb,
  [string]$WandbProject = "compvis-generation-reports",
  [string]$WandbEntity = "",
  [string]$WandbMode = "",
  [switch]$SkipBertScore,
  [switch]$SkipJudge
)

. "$PSScriptRoot/common.ps1"

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

New-Item -ItemType Directory -Force -Path $ScoredRoot | Out-Null

foreach ($loss in $selectedLosses) {
  $teachersForLoss = $selectedTeachers
  if ($loss.Id -eq "D-mse") {
    $teachersForLoss = @(Get-BaseTeacher)
  }

  foreach ($teacher in $teachersForLoss) {
    $teacherSlug = Get-TeacherProbeSlug -TeacherProbeId $teacher.Id
    $reportPath = Join-Path $ReportRoot "qwen_report_${teacherSlug}_vitbase_$($loss.Suffix).json"
    Assert-PathExists -Path $reportPath -Description "Report generazione $($teacher.Id) x $($loss.Id)"

    $currentReport = $reportPath
    if (-not $SkipBertScore) {
      $bertReport = Join-Path $ScoredRoot "qwen_report_${teacherSlug}_vitbase_$($loss.Suffix)_bertscore.json"
      Invoke-ExpStep "BERTScore $($teacher.Id) x $($loss.Id)" {
        Invoke-Uv run compvis-bertscore-qwen-report `
          --report-path $currentReport `
          --output-report $bertReport `
          --device $BertScoreDevice
      }
      $currentReport = $bertReport

      if (-not $NoWandb) {
        $metadata = @{
          stage = "bertscore_report"
          teacher_probe_id = $teacher.Id
          teacher_probe_pooling = $teacher.Pooling
          teacher_probe_slug = $teacherSlug
          distillation_loss_id = $loss.Id
          distillation_loss_suffix = $loss.Suffix
          bertscore_device = $BertScoreDevice
          source_report_path = $reportPath
          local_path = $bertReport
        }
        $artifactName = "qwen-report-${teacherSlug}-vitbase-$($loss.Suffix)-bertscore"
        Publish-WandbArtifact `
          -Path $bertReport `
          -Project $WandbProject `
          -RunName "upload-$artifactName" `
          -ArtifactName $artifactName `
          -ArtifactType "scored-report" `
          -Metadata $metadata `
          -Entity $WandbEntity `
          -Mode $WandbMode
      }
    }

    if (-not $SkipJudge) {
      $judgeReport = Join-Path $ScoredRoot "qwen_report_${teacherSlug}_vitbase_$($loss.Suffix)_judged.json"
      if ($JudgeIds.Count -ne $JudgeModels.Count) {
        throw "JudgeIds e JudgeModels devono avere la stessa lunghezza."
      }
      if ($JudgeBackends.Count -eq 0) {
        $JudgeBackends = @($JudgeIds | ForEach-Object { $JudgeBackend })
      }
      if ($JudgeIds.Count -ne $JudgeBackends.Count) {
        throw "JudgeIds e JudgeBackends devono avere la stessa lunghezza."
      }
      for ($judgeIndex = 0; $judgeIndex -lt $JudgeIds.Count; $judgeIndex++) {
        $judgeId = $JudgeIds[$judgeIndex]
        $judgeModel = $JudgeModels[$judgeIndex]
        $judgeBackendCurrent = $JudgeBackends[$judgeIndex]
        $limitArgs = @()
        if ($JudgeLimit -gt 0) {
          $limitArgs = @("--limit", "$JudgeLimit")
        }
        $backendModelArgs = @()
        if ($judgeBackendCurrent -eq "hf") {
          $backendModelArgs = @(
            "--hf-model", $judgeModel,
            "--hf-device", $JudgeDevice,
            "--hf-quantization", $JudgeQuantization
          )
        } elseif ($judgeBackendCurrent -eq "gemini") {
          $backendModelArgs = @("--gemini-model", $judgeModel)
        } elseif ($judgeBackendCurrent -eq "openrouter") {
          $backendModelArgs = @("--openrouter-model", $judgeModel)
        } elseif ($judgeBackendCurrent -eq "opencode") {
          $backendModelArgs = @("--opencode-model", $judgeModel)
        } else {
          throw "Judge backend non supportato: $judgeBackendCurrent"
        }
        $modeArgs = @("--modes") + $JudgeModes
        $outputArgs = @("--output-report", $judgeReport)
        $inputReport = $currentReport
        if ($judgeIndex -gt 0) {
          $inputReport = $judgeReport
          $outputArgs = @("--in-place")
        }
        Invoke-ExpStep "LLM judge $judgeId $($teacher.Id) x $($loss.Id)" {
          Invoke-Uv run compvis-llm-judge-qwen-report `
            --report-path $inputReport `
            @outputArgs `
            --backend $judgeBackendCurrent `
            --judge-id $judgeId `
            --student-key student `
            --teacher-key teacher `
            @backendModelArgs `
            @modeArgs `
            @limitArgs
        }
      }

      if ($JudgeIds.Count -ge 2 -and ($JudgeModes -contains "blind")) {
        Invoke-ExpStep "Aggregate multi-judge blind $($teacher.Id) x $($loss.Id)" {
          Invoke-Uv run compvis-aggregate-qwen-judges `
            --report-path $judgeReport `
            --in-place `
            --judges $JudgeIds
        }
      }

      if (-not $NoWandb) {
        $metadata = @{
          stage = "llm_judge_report"
          teacher_probe_id = $teacher.Id
          teacher_probe_pooling = $teacher.Pooling
          teacher_probe_slug = $teacherSlug
          distillation_loss_id = $loss.Id
          distillation_loss_suffix = $loss.Suffix
          judge_backend = $JudgeBackend
          judge_backends = $JudgeBackends
          judge_device = $JudgeDevice
          judge_quantization = $JudgeQuantization
          judge_ids = $JudgeIds
          judge_models = $JudgeModels
          judge_limit = $JudgeLimit
          judge_modes = $JudgeModes
          student_key = "student"
          teacher_key = "teacher"
          source_report_path = $currentReport
          local_path = $judgeReport
        }
        $artifactName = "qwen-report-${teacherSlug}-vitbase-$($loss.Suffix)-judged"
        Publish-WandbArtifact `
          -Path $judgeReport `
          -Project $WandbProject `
          -RunName "upload-$artifactName" `
          -ArtifactName $artifactName `
          -ArtifactType "scored-report" `
          -Metadata $metadata `
          -Entity $WandbEntity `
          -Mode $WandbMode
      }
    }
  }
}
