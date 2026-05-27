$ErrorActionPreference = "Stop"

function Invoke-ExpStep {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [scriptblock]$Body
  )

  Write-Host ""
  Write-Host "==> $Name" -ForegroundColor Cyan
  & $Body
}

function Invoke-Uv {
  param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
  )

  uv @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "uv fallito con exit code ${LASTEXITCODE}: uv $($Arguments -join ' ')"
  }
}

function Assert-PathExists {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$Description
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    throw "$Description non trovato: $Path"
  }
}

function Get-TeacherProbeMatrix {
  return @(
    [pscustomobject]@{
      Id = "T-mean-full"
      Pooling = "mean"
      OutputDir = "./runs/vlm_qwen25vl3b_probe_mean_full"
    },
    [pscustomobject]@{
      Id = "T-attn-full"
      Pooling = "attention"
      OutputDir = "./runs/vlm_qwen25vl3b_probe_attention_full"
    },
    [pscustomobject]@{
      Id = "T-cross-full"
      Pooling = "cross_attention"
      OutputDir = "./runs/vlm_qwen25vl3b_probe_cross_attention_full"
    }
  )
}

function Get-DistillationLossMatrix {
  return @(
    [pscustomobject]@{
      Id = "D-mse"
      Suffix = "mse"
      LambdaGlobal = "1.0"
      LambdaExpl = "0.0"
      LambdaTokenMse = "0.0"
    },
    [pscustomobject]@{
      Id = "D-expl-only"
      Suffix = "expl_only"
      LambdaGlobal = "0.0"
      LambdaExpl = "1.0"
      LambdaTokenMse = "0.0"
    },
    [pscustomobject]@{
      Id = "D-mse-expl"
      Suffix = "mse_expl"
      LambdaGlobal = "1.0"
      LambdaExpl = "1.0"
      LambdaTokenMse = "0.0"
    }
  )
}

function Get-BaseTeacher {
  return [pscustomobject]@{
    Id = "T-base"
    Pooling = "none"
    OutputDir = ""
    IsBase = $true
  }
}

function Get-TeacherProbeSlug {
  param(
    [Parameter(Mandatory = $true)]
    [string]$TeacherProbeId
  )

  return $TeacherProbeId.ToLower().Replace("t-", "").Replace("-full", "").Replace("-", "_")
}

function ConvertTo-ArtifactJson {
  param(
    [Parameter(Mandatory = $true)]
    [hashtable]$Metadata
  )

  return ($Metadata | ConvertTo-Json -Compress -Depth 8)
}

function Publish-WandbArtifact {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$Project,
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [Parameter(Mandatory = $true)]
    [string]$ArtifactName,
    [Parameter(Mandatory = $true)]
    [string]$ArtifactType,
    [Parameter(Mandatory = $true)]
    [hashtable]$Metadata,
    [string]$Entity = "",
    [string]$Mode = ""
  )

  $metadataFile = New-TemporaryFile
  $metadataJson = ConvertTo-ArtifactJson -Metadata $Metadata
  $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
  [System.IO.File]::WriteAllText($metadataFile, $metadataJson, $utf8NoBom)
  $args = @(
    "run", "compvis-upload-wandb-artifact",
    "--path", $Path,
    "--project", $Project,
    "--run-name", $RunName,
    "--artifact-name", $ArtifactName,
    "--artifact-type", $ArtifactType,
    "--metadata-file", $metadataFile
  )
  try {
    if ($Entity) {
      $args += @("--entity", $Entity)
    }
    if ($Mode) {
      $args += @("--mode", $Mode)
    }
    Invoke-Uv @args
  } finally {
    Remove-Item -LiteralPath $metadataFile -Force -ErrorAction SilentlyContinue
  }
}
