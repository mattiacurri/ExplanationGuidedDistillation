# Single-Pipeline Experiment Scripts

Each file runs one complete pipeline:

1. teacher probe training when the selected loss needs Grad-CAM,
2. one distillation loss variant,
3. Qwen generation report,
4. BERTScore + LLM judge scoring.

The experiment scripts use 20 distillation epochs by default.
LLM judge scoring runs `direct`, `reverse`, and `blind` modes by default.

For a run checklist and naming cheat sheet, see
[`../experiment_checklist.md`](../experiment_checklist.md).

W&B is enabled by default for training runs and JSON report artifacts. Report
artifacts include metadata for the teacher probe, distillation loss, local path,
student checkpoint, dataset split and seed. Pass `-NoWandb` to disable both
training logging and JSON artifact uploads.

Use `-SkipTeacherProbe`, `-SkipDistillation`, `-SkipReport`, or `-SkipScoring`
when an earlier stage already exists.

## Files

| Script | Teacher probe | Loss |
| --- | --- | --- |
| `baseline_mse.ps1` | `T-base` | `D-mse` |
| `mean_expl_only.ps1` | `T-mean-full` | `D-expl-only` |
| `mean_mse_expl.ps1` | `T-mean-full` | `D-mse-expl` |
| `attention_expl_only.ps1` | `T-attn-full` | `D-expl-only` |
| `attention_mse_expl.ps1` | `T-attn-full` | `D-mse-expl` |
| `cross_attention_expl_only.ps1` | `T-cross-full` | `D-expl-only` |
| `cross_attention_mse_expl.ps1` | `T-cross-full` | `D-mse-expl` |

Example:

```powershell
.\scripts\exps\pipelines\attention_expl_only.ps1
```

Reuse an already trained teacher probe:

```powershell
.\scripts\exps\pipelines\attention_expl_only.ps1 -SkipTeacherProbe
```

Run only scoring for one pipeline, including the new judge modes:

```powershell
.\scripts\exps\pipelines\cross_attention_expl_only.ps1 `
  -SkipTeacherProbe `
  -SkipDistillation `
  -SkipReport
```
