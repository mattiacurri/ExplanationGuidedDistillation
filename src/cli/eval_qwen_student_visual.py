"""Evaluate Qwen-VL linguistic output with distilled student visual embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.config import SMALL_VLM_MODEL_ID
from src.models.qwen_student_visual import (
    build_student_qwen_image_embeddings,
    install_qwen_visual_passthrough,
)
from src.metrics import compute_bertscore
from src.utils import get_device
from src.utils.torch_utils import cleanup_vram, parse_torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Confronta l'output linguistico Qwen teacher vs Qwen con visual "
            "embeddings dello student distillato"
        )
    )
    parser.add_argument("--image", type=str, required=True, help="Path immagine input.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image.",
        help="Prompt testuale per Qwen-VL.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=392,
        help=(
            "Resize quadrato applicato prima del processor Qwen "
            "(default: 392, 0 = immagine originale)."
        ),
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default=SMALL_VLM_MODEL_ID,
        help=f"Model ID Qwen-VL base (default: {SMALL_VLM_MODEL_ID})",
    )
    parser.add_argument(
        "--student-checkpoint-dir",
        type=str,
        default=str(Path("expl-distilled-student") / "best"),
        help="Checkpoint student distillato usato per produrre gli embedding visuali.",
    )
    parser.add_argument(
        "--teacher-vision-checkpoint",
        type=str,
        default=str(Path("runs") / "vlm_qwen25vl3b_finetuned" / "best"),
        help=(
            "Checkpoint VLMVisionClassifier da cui caricare il visual/merger "
            "finetuned. Usa 'none' per lasciare il visual base."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help=(
            "Calcola BERTScore. Senza --bertscore-reference confronta "
            "Student-direct contro Teacher."
        ),
    )
    parser.add_argument(
        "--bertscore-reference",
        type=str,
        default="",
        help="Reference testuale esplicita per BERTScore.",
    )
    parser.add_argument(
        "--bertscore-lang",
        type=str,
        default="en",
        help="Lingua usata da BERTScore quando non passi --bertscore-model-type.",
    )
    parser.add_argument(
        "--bertscore-model-type",
        type=str,
        default="",
        help="Encoder BERTScore esplicito, es. microsoft/deberta-xlarge-mnli.",
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=16,
        help="Batch size per BERTScore.",
    )
    parser.add_argument(
        "--bertscore-rescale-with-baseline",
        action="store_true",
        help="Applica la baseline rescaling di BERTScore.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--student-only", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser.parse_args()


def _load_qwen_model(
    model_id: str,
    *,
    torch_dtype: torch.dtype | None,
    trust_remote_code: bool,
):
    kwargs = {}
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        **kwargs,
    )


def _load_image(path: str, *, image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image_size > 0:
        resample = getattr(Image, "Resampling", Image).BICUBIC
        image = image.resize((image_size, image_size), resample=resample)
    return image


def _prepare_inputs(processor, image: Image.Image, prompt: str, device: torch.device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in inputs.items()}


@torch.no_grad()
def _generate_from_inputs(
    model,
    processor,
    inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> str:
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    input_ids = inputs["input_ids"]
    generated_trimmed = [
        output_ids[len(input_row) :]
        for input_row, output_ids in zip(input_ids, generated_ids, strict=True)
    ]
    return processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


@torch.no_grad()
def _generate_text(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    inputs = _prepare_inputs(processor, image, prompt, device)
    return _generate_from_inputs(
        model,
        processor,
        inputs,
        max_new_tokens=max_new_tokens,
    )


def _print_bertscore(
    label: str,
    *,
    candidate: str,
    reference: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    result = compute_bertscore(
        [candidate],
        [reference],
        lang=args.bertscore_lang,
        model_type=args.bertscore_model_type or None,
        batch_size=args.bertscore_batch_size,
        device=str(device),
        rescale_with_baseline=args.bertscore_rescale_with_baseline,
    )
    print(
        f"{label}: "
        f"P={result.precision_mean:.4f} "
        f"R={result.recall_mean:.4f} "
        f"F1={result.f1_mean:.4f}"
    )


def main() -> None:
    args = parse_args()
    trust_remote_code = not args.no_trust_remote_code
    torch_dtype = parse_torch_dtype(args.torch_dtype)
    device = get_device()
    wants_bertscore = args.bertscore or bool(args.bertscore_reference.strip())
    if wants_bertscore and args.student_only and not args.bertscore_reference.strip():
        raise SystemExit(
            "--bertscore con --student-only richiede --bertscore-reference, "
            "perche' non viene generato il teacher."
        )

    image = _load_image(args.image, image_size=args.image_size)
    processor = AutoProcessor.from_pretrained(
        args.qwen_model,
        trust_remote_code=trust_remote_code,
    )

    teacher_text = None
    if not args.student_only:
        print("Caricamento Qwen teacher...")
        teacher = _load_qwen_model(
            args.qwen_model,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        teacher.eval()
        teacher_text = _generate_text(
            teacher,
            processor,
            image,
            args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        print("\n[Teacher]")
        print(teacher_text)
        del teacher
        cleanup_vram()

    print("\nCaricamento Qwen student direct-embeddings...")
    student_model = _load_qwen_model(
        args.qwen_model,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)
    student_model.eval()
    student_inputs = _prepare_inputs(processor, image, args.prompt, device)
    image_embeddings, metadata = build_student_qwen_image_embeddings(
        student_model,
        student_checkpoint_dir=args.student_checkpoint_dir,
        image=image,
        image_grid_thw=student_inputs["image_grid_thw"],
    )
    student_inputs["pixel_values"] = image_embeddings.to(device=device)
    install_qwen_visual_passthrough(
        student_model,
        spatial_merge_size=metadata.spatial_merge_size,
    )
    print(
        "Student visual embeddings preparati: "
        f"{metadata.student_model_name}, image_size={metadata.student_image_size}, "
        f"patch={metadata.qwen_patch_size}, merge={metadata.spatial_merge_size}"
    )
    student_text = _generate_from_inputs(
        student_model,
        processor,
        student_inputs,
        max_new_tokens=args.max_new_tokens,
    )
    print("\n[Student-direct]")
    print(student_text)

    if wants_bertscore:
        print("\n[BERTScore]")
        reference = args.bertscore_reference.strip()
        if reference:
            if teacher_text is not None:
                _print_bertscore(
                    "Teacher vs reference",
                    candidate=teacher_text,
                    reference=reference,
                    args=args,
                    device=device,
                )
            _print_bertscore(
                "Student-direct vs reference",
                candidate=student_text,
                reference=reference,
                args=args,
                device=device,
            )
        else:
            assert teacher_text is not None
            _print_bertscore(
                "Student-direct vs teacher",
                candidate=student_text,
                reference=teacher_text,
                args=args,
                device=device,
            )


if __name__ == "__main__":
    main()
