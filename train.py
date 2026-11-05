"""GLiNER 미세조정: 규칙 이름(+설명)을 읽고 문장에서 그 규칙의 기밀 구간을 찾게 한다.

  python train.py --out models/v1 [--desc] [--steps 1200] [--max-width 24]

문장마다 넣는 규칙 목록 = 정답 규칙 + near(헷갈림) + 같은 업종 다른 규칙 + 다른 업종 무작위.
목록은 꺼낼 때마다 새로 뽑는다 (같은 문장도 매번 다른 규칙들과 섞여 나오게).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from splitter import EojeolSplitter

DATA = Path(__file__).parent / "data"


def rule_text(r, desc):
    return f"{r['label']}: {r['description']}" if desc else r["label"]


def to_words(splitter, text, spans):
    """문자 위치 구간 → (단어 목록, [[시작 단어, 끝 단어(포함), 규칙 ID]]).
    경계가 단어 중간에 걸리면 그 단어까지 넓힌다."""
    toks = list(splitter(text))
    out = []
    for s, e, rid in spans:
        ids = [i for i, (_, a, b) in enumerate(toks) if a < e and b > s]
        if ids:
            out.append([ids[0], ids[-1], rid])
    return [t for t, _, _ in toks], out


class RuleSpanDataset(Dataset):
    def __init__(self, rows, bank, desc, max_width, n_labels=(12, 30), fixed_seed=None):
        self.bank, self.desc, self.n_labels = bank, desc, n_labels
        self.by_ind = defaultdict(list)
        for rid, r in bank.items():
            self.by_ind[r["industry"]].append(rid)
        self.all_ids = list(bank)
        sp = EojeolSplitter()
        self.items, self.too_wide = [], 0
        for r in rows:
            words, ner = to_words(sp, r["text"], r["spans"])
            keep = [x for x in ner if x[1] - x[0] < max_width]
            self.too_wide += len(ner) - len(keep)
            self.items.append({"words": words, "ner": keep, "near": r["near"], "industry": r["industry"]})
        # dev 는 매번 같은 목록이어야 손실을 비교할 수 있다
        self.fixed = None
        if fixed_seed is not None:
            rng = random.Random(fixed_seed)
            self.fixed = [self.sample(i, rng) for i in range(len(self.items))]

    def __len__(self):
        return len(self.items)

    def sample(self, i, rng):
        it = self.items[i]
        ids = list(dict.fromkeys([x[2] for x in it["ner"]] + it["near"]))
        n = rng.randint(*self.n_labels)
        same = [x for x in self.by_ind[it["industry"]] if x not in ids]
        rng.shuffle(same)
        ids += same[: max(0, n // 2 - len(ids))]
        while len(ids) < n:
            x = rng.choice(self.all_ids)
            if x not in ids:
                ids.append(x)
        rng.shuffle(ids)
        name = {x: rule_text(self.bank[x], self.desc) for x in ids}
        # 업종이 달라도 이름이 같은 규칙이 있다 → 같은 문자열은 한 번만
        return {"tokenized_text": it["words"],
                "ner": [[s, e, name[rid]] for s, e, rid in it["ner"]],
                "ner_labels": list(dict.fromkeys(name[x] for x in ids))}

    def __getitem__(self, i):
        return self.fixed[i] if self.fixed else self.sample(i, random)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="urchade/gliner_multi-v2.1")
    ap.add_argument("--out", default="models/v1")
    ap.add_argument("--desc", action="store_true", help="규칙 문자열에 설명까지 넣기")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-width", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=200)
    a = ap.parse_args()

    from gliner import GLiNER
    bank = {r["id"]: r for r in json.loads((DATA / "rule_bank.json").read_text())}
    load = lambda n: [json.loads(l) for l in (DATA / f"{n}.jsonl").read_text().splitlines()]
    train = RuleSpanDataset(load("train"), bank, a.desc, a.max_width)
    dev = RuleSpanDataset(load("dev"), bank, a.desc, a.max_width, fixed_seed=0)
    print(f"train {len(train)} / dev {len(dev)} / 폭 초과로 뺀 구간 {train.too_wide + dev.too_wide}")

    model = GLiNER.from_pretrained(a.base, max_width=a.max_width)
    model.data_processor.words_splitter = EojeolSplitter()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    args = model.create_training_args(
        output_dir=a.out, learning_rate=1e-5, others_lr=5e-5, weight_decay=0.01,
        focal_loss_alpha=0.75, focal_loss_gamma=2, max_steps=a.steps, warmup_ratio=0.1,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        save_steps=a.eval_every, eval_steps=a.eval_every, eval_strategy="steps",
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=20, dataloader_num_workers=0, use_cpu=device == "cpu")
    model.train_model(train_dataset=train, eval_dataset=dev, training_args=args)
    model.save_pretrained(Path(a.out) / "final")
    (Path(a.out) / "final" / "train_meta.json").write_text(json.dumps(
        {"base": a.base, "desc": a.desc, "steps": a.steps, "max_width": a.max_width, "splitter": "eojeol"},
        ensure_ascii=False))
    print("저장:", Path(a.out) / "final")


if __name__ == "__main__":
    main()
