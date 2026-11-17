"""7회차 실습 1 ③④⑤: LoRA/QLoRA 학습 → Adapter 저장.

  python src/build_train_data.py --augment 300                 # 먼저 학습 데이터 준비
  python src/train_lora.py --team team3 --run r16               # 기본: config.QUANT로 본체 로딩(nf4이면 QLoRA)
  python src/train_lora.py --team team3 --run r8 --r 8          # 도전과제 ①: rank 바꾸기
  python src/train_lora.py --team team3 --run n100 --limit 100  # 데이터 양 바꾸기
  python src/train_lora.py --team team3 --run lr1e3 --lr 1e-3     # 학습률 바꾸기
  python src/train_lora.py --team team3 --run qv --target qv       # LoRA를 붙일 층 바꾸기(qv · qkvo · all)
  python src/train_lora.py --team team3 --run noise30 --data train_noise30.jsonl   # build_train_data.py가 만든 다른 학습 파일
  python src/train_lora.py --team team3 --run r16_s1 --seed 1  # 같은 설정, 다른 시드(흔들림 폭)
  python src/train_lora.py --team team3 --run q4 --quant nf4      # 본체 4비트(QLoRA): 메모리와 점수 비교

학습은 휴식 시간과 겹치게 돌린다. config.LORA["max_minutes"]를 넘기면 그때까지 학습한 Adapter를 저장하고 끝낸다.
학습이 끝나지 않으면: 미리 학습해 둔 Adapter(adapters/_instructor/<도메인>/)로 재평가를 이어 간다.

결과: adapters/<팀>/<run>/ (adapter_model.safetensors, train_config.json), results/train_log_<팀>_<run>.json
다음: python src/plot_loss.py --log results/train_log_<팀>_<run>.json
      python src/benchmark.py --mode lora --adapter adapters/<팀>/<run>
"""
from __future__ import annotations

import argparse
import time

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


class TimeLimit(TrainerCallback):
    def __init__(self, minutes: float):
        self.limit, self.t0, self.stopped = minutes * 60, None, False

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        if time.perf_counter() - self.t0 > self.limit:
            control.should_training_stop = True
            self.stopped = True
        return control


def main() -> None:
    L = config.LORA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True)
    ap.add_argument("--run", default="r16")
    ap.add_argument("--r", type=int, default=L["r"])
    ap.add_argument("--alpha", type=int, default=None, help="기본: r의 2배")
    ap.add_argument("--epochs", type=float, default=L["epochs"])
    ap.add_argument("--lr", type=float, default=L["learning_rate"])
    ap.add_argument("--batch", type=int, default=L["batch_size"])
    ap.add_argument("--max-length", type=int, default=L["max_length"])
    ap.add_argument("--max-minutes", type=float, default=L["max_minutes"])
    ap.add_argument("--limit", type=int, help="학습 데이터 앞에서 N건만")
    ap.add_argument("--quant", default=config.QUANT, choices=["bf16", "nf4"])
    ap.add_argument("--target", choices=["qv", "qkvo", "all"], default="qkvo",
                    help="LoRA를 붙일 층: qv(q·v) · qkvo(어텐션 넷, 기본) · all(어텐션 + MLP)")
    ap.add_argument("--data", default="train.jsonl", help="data/<도메인>/ 안의 학습 파일 이름")
    ap.add_argument("--seed", type=int, default=config.SEED, help="같은 설정을 시드만 바꿔 다시 학습하면 흔들림 폭이 보임")
    a = ap.parse_args()
    utils.set_seed(a.seed)
    out_dir = config.PATHS["adapters"] / a.team / a.run
    d = config.data_dir()
    ds = load_dataset("json", data_files={"train": str(d / a.data), "valid": str(d / "valid.jsonl")})
    train = ds["train"].select(range(min(a.limit, len(ds["train"])))) if a.limit else ds["train"]
    print(f"[학습] 도메인 {config.DOMAIN} · {a.data} {len(train)}건 · 검증 {len(ds['valid'])}건 · {a.quant} · r={a.r} · lr={a.lr} · 층 {a.target}")

    utils.vram_reset()
    model, tok = model_utils.load_llm(quant=a.quant)
    model.train()
    if a.quant == "nf4":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=a.r, lora_alpha=a.alpha or 2 * a.r, lora_dropout=L["dropout"],
                      target_modules=TARGETS[a.target], task_type="CAUSAL_LM")
    gpu = torch.cuda.is_available()
    args = SFTConfig(
        output_dir=str(config.PATHS["runs"] / "lora_tmp" / a.team / a.run),
        max_length=a.max_length, per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=L["grad_accum"], learning_rate=a.lr, num_train_epochs=a.epochs,
        lr_scheduler_type="cosine", warmup_steps=5, logging_steps=5, eval_strategy="steps", eval_steps=20,
        save_strategy="no", bf16=gpu, gradient_checkpointing=gpu, report_to="none", seed=a.seed,
    )
    timer = TimeLimit(a.max_minutes)
    trainer = SFTTrainer(model=model, args=args, train_dataset=train, eval_dataset=ds["valid"],
                         processing_class=tok, peft_config=lora, callbacks=[timer])
    trainable = model_utils.count_params(trainer.model)
    print(f"  학습 파라미터 {trainable['trainable']:,} / 전체 {trainable['total']:,} ({trainable['ratio']:.3%})")
    with utils.Timer() as t:
        trainer.train()
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    size_mb = sum(p.stat().st_size for p in out_dir.glob("adapter_model.*")) / 1024**2
    info = {"team": a.team, "run": a.run, "domain": config.DOMAIN, "base_model": config.MODEL_ID, "quant": a.quant,
            "r": a.r, "alpha": a.alpha or 2 * a.r, "epochs": a.epochs, "lr": a.lr, "batch": a.batch, "target": a.target, "data": a.data, "seed": a.seed,
            "max_length": a.max_length, "n_train": len(train), "minutes": round(t.sec / 60, 2),
            "stopped_by_time_limit": timer.stopped, "trainable_params": trainable["trainable"],
            "adapter_mb": round(size_mb, 1), "vram_peak_gb": utils.vram_peak_gb(), "finished_at": utils.now()}
    utils.write_json(out_dir / "train_config.json", info)
    utils.write_json(utils.result_path(f"train_log_{a.team}_{a.run}.json"),
                     {"info": info, "log_history": trainer.state.log_history})
    print(f"\n[완료] {t.sec / 60:.1f}분{' (시간 제한으로 중단)' if timer.stopped else ''} · Adapter {size_mb:.1f}MB → {out_dir}")
    print(f"다음: python src/plot_loss.py --log results/train_log_{a.team}_{a.run}.json")


if __name__ == "__main__":
    main()
