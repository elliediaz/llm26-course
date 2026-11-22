"""개인 샌드박스의 LoRA 학습 실행기. 노트북의 lab.train() 이 별도 프로세스로 부르고, 진행 상황을 JSON 한 줄씩 받아 그림을 그림.

직접 돌릴 수도 있음:
  python src/lab_train.py --train my/train.jsonl --valid my/valid.jsonl --out my/adapters/v1 --base skt/A.X-4.0-Light --run v1
학습 데이터는 한 줄에 {"prompt": [대화], "completion": [{"role": "assistant", "content": 답}]} 형식(답 부분만 학습함).

메모(11/22 실측, A.X-4.0-Light · 학습 156건 · rank 16 · 에폭 3 · 배치 8)
  - RTX PRO 6000(96GB) 약 38초, 학습 메모리 16.2GB, Adapter 38.5MB
  - RTX PRO 4500(32GB) 약 224초. 노트북에 올려 둔 모델을 잠시 내렸다가 다시 올리는 시간이 들어간 값임
  - 4bit 본체(QLoRA) · 배치 4 는 32GB 에서 약 151초, 학습 메모리 8.3GB
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

import config
import model_utils
import utils

TARGETS = {"qv": ["q_proj", "v_proj"], "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
           "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def emit(**kw) -> None:
    print(json.dumps({"ev": kw.pop("ev"), **kw}, ensure_ascii=False), flush=True)


class Report(TrainerCallback):
    """학습 기록이 생길 때마다 한 줄씩 내보냄."""

    def __init__(self):
        self.history: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        vram = round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None
        if "loss" in logs:
            self.history.append({"step": state.global_step, "loss": round(float(logs["loss"]), 4)})
            emit(ev="log", step=state.global_step, total=state.max_steps, loss=round(float(logs["loss"]), 4), vram=vram)
        if "eval_loss" in logs:
            self.history.append({"step": state.global_step, "eval_loss": round(float(logs["eval_loss"]), 4)})
            emit(ev="eval", step=state.global_step, total=state.max_steps, eval_loss=round(float(logs["eval_loss"]), 4), vram=vram)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="v1")
    ap.add_argument("--base", default=config.MODEL_ID)
    ap.add_argument("--quant", default="bf16", choices=["bf16", "nf4"])
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--target", choices=list(TARGETS), default="qkvo")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args()

    utils.set_seed(a.seed)
    ds = load_dataset("json", data_files={"train": a.train, "valid": a.valid})
    keep = [c for c in ds["train"].column_names if c in ("prompt", "completion")]
    train = ds["train"].select_columns(keep)
    valid = ds["valid"].select_columns(keep)
    if a.limit:
        train = train.select(range(min(a.limit, len(train))))
    utils.vram_reset()
    model, tok = model_utils.load_llm(a.base, a.quant)
    model.train()
    if a.quant == "nf4":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05, target_modules=TARGETS[a.target], task_type="CAUSAL_LM")
    gpu = torch.cuda.is_available()
    steps = max(1, int(len(train) / a.batch * a.epochs))
    args = SFTConfig(
        output_dir=str(config.PATHS["runs"] / "lab_tmp" / a.run), max_length=a.max_length,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch, learning_rate=a.lr, num_train_epochs=a.epochs,
        lr_scheduler_type="cosine", warmup_steps=min(5, max(1, steps // 10)), logging_steps=max(1, steps // 40),
        eval_strategy="steps", eval_steps=max(2, steps // 8), save_strategy="no", bf16=gpu, gradient_checkpointing=gpu,
        report_to="none", seed=a.seed, disable_tqdm=True,
    )
    rep = Report()
    trainer = SFTTrainer(model=model, args=args, train_dataset=train, eval_dataset=valid, processing_class=tok,
                         peft_config=lora, callbacks=[rep])
    cnt = model_utils.count_params(trainer.model)
    t0 = time.perf_counter()
    trainer.train()
    minutes = round((time.perf_counter() - t0) / 60, 2)
    out = Path(a.out)
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    size_mb = round(sum(p.stat().st_size for p in out.glob("adapter_model.*")) / 2**20, 1)
    losses = [h["loss"] for h in rep.history if "loss" in h]
    evals = [h["eval_loss"] for h in rep.history if "eval_loss" in h]
    info = {"run": a.run, "base_model": a.base, "quant": a.quant, "r": a.r, "alpha": a.alpha, "lr": a.lr, "epochs": a.epochs,
            "batch": a.batch, "target": a.target, "max_length": a.max_length, "seed": a.seed, "n_train": len(train),
            "minutes": minutes, "trainable_params": cnt["trainable"], "trainable_ratio": cnt["ratio"], "adapter_mb": size_mb,
            "final_loss": losses[-1] if losses else None, "eval_loss": evals[-1] if evals else None,
            "vram_peak_gb": utils.vram_peak_gb(), "finished_at": utils.now()}
    utils.write_json(out / "train_config.json", info)
    utils.write_json(out / "log.json", rep.history)
    emit(ev="done", **{k: info[k] for k in ("minutes", "adapter_mb", "final_loss", "eval_loss", "vram_peak_gb")})


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:                                  # 노트북 화면에 원인 한 줄이 보이게 함
        print(f"{type(ex).__name__}: {ex}", flush=True)
        sys.exit(1)
