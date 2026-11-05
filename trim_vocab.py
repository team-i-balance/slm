"""어휘 축소: 25만 다국어 사전에서 한국어·영어·숫자·기호 토큰만 남겨 임베딩을 줄인다.

  python trim_vocab.py models/v2/final models/v2/trim [--ascii 50000]

mdeberta 의 파라미터 2.89억 중 1.92억이 단어 임베딩(250,105 × 768)이다. 남기는 것:
  - 한글 토큰 전부 (약 4천 개) + 특수·추가 토큰 전부
  - ASCII 토큰 중 빈도 점수(Unigram score) 상위 N개 + 학습·평가 데이터에 실제로 나온 토큰
  - 자주 쓰는 기호 (전각, 원화, 화살표, 원문자 등)
Unigram 은 없는 조각을 더 작은 조각으로 쪼개므로, 한 글자 토큰이 남아 있는 한 [UNK] 가 거의 늘지 않는다.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import torch

HERE = Path(__file__).parent
HANGUL = re.compile(r"^▁?[가-힣ㄱ-ㆎ]+$")
ASCII = re.compile(r"^▁?[\x21-\x7E]+$")
SYMBOL = re.compile(r"^▁?[· -⁯₩←-⇿①-⓿■-◿☀-⛿"
                    r"　-〿！-～ -¿×÷]+$")


def corpus_texts():
    out = []
    for p in list((HERE / "data").glob("*.jsonl")) + list((HERE / "data" / "eval_raw").glob("*.jsonl")) \
            + list((HERE / "data" / "train_raw").glob("*.samples.jsonl")):
        out += [json.loads(l)["text"] for l in p.read_text().splitlines() if l.strip()]
    for p in [HERE / "data" / "rule_bank.json"]:
        out += [r["label"] for r in json.loads(p.read_text())]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--ascii", type=int, default=50000, help="빈도 상위로 남길 ASCII 토큰 수")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)

    from gliner import GLiNER
    from tokenizers import Tokenizer
    tj = json.loads((src / "tokenizer.json").read_text())
    vocab = tj["model"]["vocab"]
    added = {t["id"]: t for t in tj["added_tokens"]}

    used = set()
    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    for enc in tok.encode_batch(corpus_texts()):
        used.update(enc.ids)

    asc = sorted((i for i, (p, _) in enumerate(vocab) if ASCII.match(p)), key=lambda i: -vocab[i][1])[:a.ascii]
    keep = set(asc) | used | {i for i in added if i < len(vocab)}
    keep |= {i for i, (p, _) in enumerate(vocab) if HANGUL.match(p) or SYMBOL.match(p)}
    keep = sorted(keep)
    extra = sorted(i for i in added if i >= len(vocab))   # GLiNER 가 덧붙인 [MASK], <<ENT>> 등
    old_ids = keep + extra
    new_id = {o: n for n, o in enumerate(old_ids)}

    tj["model"]["vocab"] = [vocab[i] for i in keep]
    tj["model"]["unk_id"] = new_id[tj["model"]["unk_id"]]
    for t in tj["added_tokens"]:
        t["id"] = new_id[t["id"]]
    for k in ("single", "pair"):
        pass  # TemplateProcessing 은 토큰 문자열로 참조, special_tokens 의 ids 만 갱신
    for name, st in (tj.get("post_processor") or {}).get("special_tokens", {}).items():
        st["ids"] = [new_id[i] for i in st["ids"]]

    m = GLiNER.from_pretrained(str(src))
    emb = m.model.token_rep_layer.bert_layer.model.embeddings.word_embeddings
    new = torch.nn.Embedding(len(old_ids), emb.weight.shape[1], padding_idx=emb.padding_idx)
    new.weight.data = emb.weight.data[torch.tensor(old_ids)].clone()
    m.model.token_rep_layer.bert_layer.model.embeddings.word_embeddings = new
    m.config.vocab_size = len(old_ids)
    m.config.class_token_index = new_id[m.config.class_token_index]
    enc_cfg = m.model.token_rep_layer.bert_layer.model.config
    enc_cfg.vocab_size = len(old_ids)
    if isinstance(getattr(m.config, "encoder_config", None), dict):
        m.config.encoder_config["vocab_size"] = len(old_ids)
    elif getattr(m.config, "encoder_config", None) is not None:
        m.config.encoder_config.vocab_size = len(old_ids)

    dst.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(dst))
    (dst / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    cfg["unk_id"] = new_id[cfg.get("unk_id", 3)]
    if "added_tokens_decoder" in cfg:
        cfg["added_tokens_decoder"] = {str(new_id[int(k)]): v for k, v in cfg["added_tokens_decoder"].items()}
    (dst / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    for f in ("train_meta.json",):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"어휘 {len(vocab) + len(extra)} → {len(old_ids)} (데이터에 나온 토큰 {len(used)}), 파라미터 {n_params / 1e6:.0f}M")

    # 확인: 같은 문장이 같은 토큰 문자열로 쪼개지는지
    t2 = Tokenizer.from_file(str(dst / "tokenizer.json"))
    texts = corpus_texts()
    diff = sum(tok.encode(t).tokens != t2.encode(t).tokens for t in texts)
    print(f"토큰화 달라진 문장 {diff}/{len(texts)}")


if __name__ == "__main__":
    main()
