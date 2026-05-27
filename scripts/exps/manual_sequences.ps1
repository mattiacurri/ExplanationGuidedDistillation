# Manual experiment sequences for the current matrix.
#
# This file is intentionally not an orchestrator with logic. It is a set of
# explicit commands to run by hand, one block at a time.

# ---------------------------------------------------------------------------
# 1) Teacher probes: full dataset
# ---------------------------------------------------------------------------

.\scripts\exps\01_train_teacher_probes.ps1 -Only T-mean-full

.\scripts\exps\01_train_teacher_probes.ps1 -Only T-attn-full

.\scripts\exps\01_train_teacher_probes.ps1 -Only T-cross-full


# ---------------------------------------------------------------------------
# 2) Distillation baseline: base Qwen, no probe
# ---------------------------------------------------------------------------

.\scripts\exps\02_run_distillation_matrix.ps1 -TeacherProbes T-base -Losses D-mse


# ---------------------------------------------------------------------------
# 3) Distillation matrix: T-mean-full
# ---------------------------------------------------------------------------

.\scripts\exps\02_run_distillation_matrix.ps1 -TeacherProbes T-mean-full -Losses D-expl-only

.\scripts\exps\02_run_distillation_matrix.ps1 -TeacherProbes T-mean-full -Losses D-mse-expl


# ---------------------------------------------------------------------------
# 4) Distillation matrix: T-attn-full
# ---------------------------------------------------------------------------

.\scripts\exps\02_run_distillation_matrix.ps1 -TeacherProbes T-attn-full -Losses D-expl-only

.\scripts\exps\02_run_distillation_matrix.ps1 -TeacherProbes T-attn-full -Losses D-mse-expl


# ---------------------------------------------------------------------------
# 5) Distillation matrix: T-cross-full
# ---------------------------------------------------------------------------

.\scripts\exps\02_run_distillation_matrix.ps1 -TeacherProbes T-cross-full -Losses D-expl-only

.\scripts\exps\02_run_distillation_matrix.ps1 -TeacherProbes T-cross-full -Losses D-mse-expl


# ---------------------------------------------------------------------------
# 6) Generate language reports
# ---------------------------------------------------------------------------

.\scripts\exps\03_generate_student_reports.ps1 -TeacherProbes T-base -Losses D-mse

.\scripts\exps\03_generate_student_reports.ps1 -TeacherProbes T-mean-full -Losses D-expl-only

.\scripts\exps\03_generate_student_reports.ps1 -TeacherProbes T-mean-full -Losses D-mse-expl

.\scripts\exps\03_generate_student_reports.ps1 -TeacherProbes T-attn-full -Losses D-expl-only

.\scripts\exps\03_generate_student_reports.ps1 -TeacherProbes T-attn-full -Losses D-mse-expl

.\scripts\exps\03_generate_student_reports.ps1 -TeacherProbes T-cross-full -Losses D-expl-only

.\scripts\exps\03_generate_student_reports.ps1 -TeacherProbes T-cross-full -Losses D-mse-expl


# ---------------------------------------------------------------------------
# 7) Score language reports: BERTScore + LLM judge, no judge limit
# ---------------------------------------------------------------------------

.\scripts\exps\04_score_generation_reports.ps1 -TeacherProbes T-base -Losses D-mse

.\scripts\exps\04_score_generation_reports.ps1 -TeacherProbes T-mean-full -Losses D-expl-only

.\scripts\exps\04_score_generation_reports.ps1 -TeacherProbes T-mean-full -Losses D-mse-expl

.\scripts\exps\04_score_generation_reports.ps1 -TeacherProbes T-attn-full -Losses D-expl-only

.\scripts\exps\04_score_generation_reports.ps1 -TeacherProbes T-attn-full -Losses D-mse-expl

.\scripts\exps\04_score_generation_reports.ps1 -TeacherProbes T-cross-full -Losses D-expl-only

.\scripts\exps\04_score_generation_reports.ps1 -TeacherProbes T-cross-full -Losses D-mse-expl
