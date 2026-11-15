"""1·2회차 노트북용: 'LLM으로 오기까지' 각 단계의 문제와 해결을 짧게 확인하는 함수 모음.

노트북 셀을 짧게 유지하려고 계산 부분을 여기에 모았다. 각 함수는 GPU에서 수 초~1분 안에 끝나도록 작게 잡았다.

1회차  bigram_lm·sentence_prob  n-gram 희소성(본 적 없는 조합 = 확률 0)
       bow                      단어 주머니 벡터의 어순 소실
       static_token_ids         단어 하나에 벡터 하나(정적 임베딩의 한계)
       keyword_vs_meaning       키워드 겹침(TF-IDF) vs 문장 의미(임베딩) 유사도
2회차  rnn_vs_attention_timing  RNN 순차 처리 vs Self-Attention 병렬 처리, 길이 제곱 메모리
       oov_rate·MiniBPE         어휘집 밖 단어(OOV)와 서브워드 토큰화
       fill_mask·next_token     사전학습 목표의 분기: 빈칸 맞히기(MLM) vs 다음 토큰 예측
       icl_accuracy             문맥 내 학습: 가중치 변경 없이 예시만으로 과업 수행
       chat_raw_vs_template     정렬: Base 모델과 Instruct 모델의 응답 차이
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict

import config
import utils


# ================================================================ 1회차 ① n-gram 언어모델
def words(s: str) -> list[str]:
    return re.findall(r"[가-힣A-Za-z0-9]+", s)


def domain_sentences(domain: str | None = None) -> list[str]:
    d = config.data_dir(domain)
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted((d / "corpus").glob("*.md")))
    text += "\n" + "\n".join(f["question"] + " " + f["answer"] for f in utils.read_json(d / "faq.json"))
    return [s for s in re.split(r"[.\n?!]", text) if len(words(s)) >= 3]


def bigram_lm(sents: list[str]):
    uni, bi = Counter(), defaultdict(Counter)
    for s in sents:
        w = ["<s>"] + words(s) + ["</s>"]
        uni.update(w)
        for a, b in zip(w, w[1:]):
            bi[a][b] += 1
    return uni, bi


def sentence_prob(s: str, uni, bi, smoothing: float = 0.0) -> dict:
    """P(문장) = Π P(w_t | w_{t-1}). smoothing=1이면 add-one(라플라스) 보정."""
    w = ["<s>"] + words(s) + ["</s>"]
    V = len(uni)
    logp, zero = 0.0, []
    for a, b in zip(w, w[1:]):
        p = (bi[a][b] + smoothing) / (uni[a] + smoothing * V) if (uni[a] + smoothing * V) else 0.0
        if p == 0:
            zero.append(f"{a}→{b}")
            logp = float("-inf")
        elif logp != float("-inf"):
            logp += math.log(p)
    return {"문장": s, "확률 0 원인": ", ".join(zero) or "-",
            "log P": round(logp, 2) if logp != float("-inf") else "-inf"}


# ================================================================ 1회차 ② 단어 주머니
def bow(s: str) -> dict:
    return dict(sorted(Counter(s.split()).items()))


# ================================================================ 1회차 ③ 키워드 vs 의미(문장 임베딩)
def static_token_ids(word: str, sentences: list[str], model_name: str | None = None) -> list[dict]:
    """정적 임베딩의 한계: 같은 글자열이면 문장이 달라도 같은 토큰 ID → 같은 입력 벡터(표에서 같은 행)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name or config.EMBED_MODEL)
    rows = []
    for s in sentences:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        pieces = [tok.decode([i]).strip() for i in ids]
        rows.append({"문장": s, "토큰": " | ".join(pieces), f"'{word}' 토큰 ID": ids[pieces.index(word)] if word in pieces else None})
    return rows


def keyword_vs_meaning(query: str, candidates: list[str], embedder=None) -> list[dict]:
    """같은 질의에 대해 글자 TF-IDF(키워드 겹침)와 문장 임베딩(의미)의 유사도를 나란히 본다."""
    from faq_search import TfidfIndex
    t = TfidfIndex(candidates)
    kw = (t.encode([query]) @ t.mat.T)[0]
    if embedder is None:
        import model_utils
        embedder = model_utils.load_embedder()
    e = embedder.encode([query] + candidates, normalize_embeddings=True)
    mn = e[1:] @ e[0]
    return [{"후보 문장": c, "키워드(TF-IDF)": round(float(a), 3), "의미(임베딩)": round(float(b), 3)}
            for c, a, b in zip(candidates, kw, mn)]


# ================================================================ 2회차 ① RNN vs Self-Attention
def rnn_vs_attention_timing(lengths=(128, 256, 512, 1024, 2048), d: int = 512, batch: int = 8):
    """같은 크기의 GRU 층과 Self-Attention 층에 길이만 바꿔 넣고 forward 시간·메모리를 잰다."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gru = nn.GRU(d, d, batch_first=True).to(dev)
    att = nn.MultiheadAttention(d, 8, batch_first=True).to(dev)
    rows = []
    for L in lengths:
        x = torch.randn(batch, L, d, device=dev)
        r = {"길이": L}
        for name, fn in (("GRU", lambda: gru(x)), ("Self-Attention", lambda: att(x, x, x, need_weights=False))):
            with torch.no_grad():
                fn()                                   # 준비 실행
                utils.vram_reset()
                with utils.Timer() as t:
                    for _ in range(3):
                        fn()
            r[f"{name} 시간(ms)"] = round(t.sec / 3 * 1000, 1)
            r[f"{name} 메모리(MB)"] = round((utils.vram_peak_gb() or 0) * 1024, 1) if dev == "cuda" else None
        r["어텐션 점수 행렬 크기(MB, fp32)"] = round(batch * 8 * L * L * 4 / 1024**2, 1)
        rows.append(r)
    return rows


# ================================================================ 2회차 ② OOV와 서브워드
def oov_rate(train_sents: list[str], test_sents: list[str]) -> dict:
    vocab = {w for s in train_sents for w in s.split()}
    toks = [w for s in test_sents for w in s.split()]
    oov = [w for w in toks if w not in vocab]
    return {"어휘 크기": len(vocab), "시험 어절": len(toks), "OOV": len(oov), "OOV 비율": round(len(oov) / len(toks), 3),
            "OOV 예": oov[:8]}


class MiniBPE:
    """가장 자주 붙어 나오는 글자 쌍을 반복해서 합치는 BPE(Sennrich et al. 2016)의 최소 구현."""

    def __init__(self, merges: int = 300):
        self.n = merges
        self.rules: list[tuple[str, str]] = []

    def fit(self, sents: list[str]):
        vocab = Counter(tuple(w) + ("▁",) for s in sents for w in s.split())
        for _ in range(self.n):
            pairs = Counter()
            for w, c in vocab.items():
                for a, b in zip(w, w[1:]):
                    pairs[(a, b)] += c
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            self.rules.append(best)
            new = {}
            for w, c in vocab.items():
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                        out.append(w[i] + w[i + 1])
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                new[tuple(out)] = c
            vocab = new
        return self

    def tokenize(self, s: str) -> list[str]:
        out = []
        for w in s.split():
            sym = list(w) + ["▁"]
            for a, b in self.rules:
                i, merged = 0, []
                while i < len(sym):
                    if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                        merged.append(a + b)
                        i += 2
                    else:
                        merged.append(sym[i])
                        i += 1
                sym = merged
            out += sym
        return out


# ================================================================ 2회차 ③ 사전학습 목표
def fill_mask(text: str, model_name: str | None = None, k: int = 5) -> list[tuple[str, float]]:
    """빈칸(<mask>) 양쪽 문맥을 모두 보고 맞힌다(BERT 계열 MLM)."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    model_name = model_name or config.MLM_MODEL_ID
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForMaskedLM.from_pretrained(model_name).eval()
    enc = tok(text.replace("<mask>", tok.mask_token), return_tensors="pt")
    pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
    with torch.no_grad():
        p = m(**enc).logits[0, pos].softmax(-1)
    top = p.topk(k)
    return [(tok.decode([i]).strip(), round(float(v), 3)) for v, i in zip(top.values, top.indices)]


def next_token(text: str, model, tok, k: int = 5) -> list[tuple[str, float]]:
    """왼쪽 문맥만 보고 다음 토큰을 예측한다(GPT 계열).
    바이트 단위 토크나이저는 한글 한 글자가 토큰 2개로 갈라지기도 해서, 글자가 완성될 때까지 1~2토큰 더 이어 붙여 보여 준다."""
    import torch
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        p = model(**enc).logits[0, -1].float().softmax(-1)
    top = p.topk(k)
    out = []
    for v, i in zip(top.values, top.indices):
        ids = enc["input_ids"][0].tolist() + [int(i)]
        piece = tok.decode([int(i)])
        extra = 0
        while "\ufffd" in piece and extra < 2:
            with torch.no_grad():
                nxt = model(input_ids=torch.tensor([ids], device=model.device)).logits[0, -1].argmax()
            ids.append(int(nxt))
            extra += 1
            piece = tok.decode(ids[len(enc["input_ids"][0]):])
        out.append((piece + ("(+조각)" if extra else ""), round(float(v), 3)))
    return out


# ================================================================ 2회차 ④ 문맥 내 학습
CODES = {"버그": "BUG", "밸런스": "BAL", "결제": "PAY", "계정": "ACC", "이벤트": "EVT", "제재": "BAN", "기타": "ETC"}


def icl_accuracy(model, tok, items: list[dict], shots: list[dict]) -> dict:
    """Base 모델에 '문의 → 분류 코드' 예시를 k개 붙였을 때의 형식 준수율·정확도. 가중치는 그대로다.
    코드(PAY, BUG …)는 예시를 보지 않으면 알 수 없으므로, 0개일 때와 비교하면 예시의 효과가 드러난다."""
    import torch
    head = "고객 문의를 분류 코드로 바꾼다.\n\n"
    demo = "".join(f"문의: {s['input']}\n코드: {CODES[s['gold']['category']]}\n\n" for s in shots)
    hit, fmt, outs = 0, 0, []
    for it in items:
        prompt = head + demo + f"문의: {it['input']}\n코드:"
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = text.split()[0] if text.split() else ""
        gold = CODES[it["gold"]["category"]]
        fmt += pred in CODES.values()
        hit += pred == gold
        outs.append({"문의": it["input"][:28], "정답": gold, "출력": pred, "판정": "O" if pred == gold else "X"})
    n = len(items)
    return {"shots": len(shots), "format_ok": round(fmt / n, 3), "accuracy": round(hit / n, 3), "rows": outs}


# ================================================================ 2회차 ⑤ 정렬: Base vs Instruct
def chat_raw_vs_template(model, tok, question: str, use_template: bool, max_new_tokens: int = 80) -> str:
    import torch
    if use_template:
        enc = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt", **config.CHAT_TEMPLATE_KWARGS)
    else:
        enc = tok(question, return_tensors="pt")
    enc = enc.to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    import model_utils
    return model_utils.strip_think(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))


def timer() -> float:
    return time.perf_counter()
