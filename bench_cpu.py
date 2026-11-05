"""CPU 2스레드 대리 측정 (AWS 4 vCPU·물리 2코어 서버를 맥북에서 흉내).

  python bench_cpu.py models/v2/trim models/v3/trim [models/v3/trim_onnx:model_quantized.onnx ...]

같은 문장 20개(dev.jsonl 앞쪽)를 모아랩스 정책(규칙 22 + 고정 라벨 3)으로 detect 한다.
폴더:파일 형식이면 ONNX 로 불러온다 (onnxruntime 도 2스레드). 결과: p50·평균 지연, 프로세스 최대 메모리.
모델마다 새 프로세스에서 재야 메모리가 섞이지 않는다 → 인자 하나씩 자식 프로세스로 돈다.
"""
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def run_one(spec):
    import torch
    torch.set_num_threads(2)
    from detector import SlmDetector, load_policy
    from splitter import EojeolSplitter

    path, _, onnx = spec.partition(":")
    if onnx:
        import onnxruntime as ort
        from gliner import GLiNER
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = 2, 1
        det = SlmDetector.__new__(SlmDetector)
        det.m = GLiNER.from_pretrained(path, local_files_only=True, load_onnx_model=True, onnx_model_file=onnx,
                                       session_options=so)
        det.m.data_processor.words_splitter = EojeolSplitter()
        det.splitter, det.desc = EojeolSplitter(), False
        from detector import DEFAULT_THRESHOLDS
        det.th = dict(DEFAULT_THRESHOLDS)
        det.fixed = json.loads((HERE / "data" / "fixed_labels.json").read_text())["labels"]
        det.set_policy(load_policy(HERE / "data" / "rules.json"))
    else:
        det = SlmDetector(path, policy=HERE / "data" / "rules.json", device="cpu")
    texts = [json.loads(l)["text"] for l in (HERE / "data" / "dev.jsonl").read_text().splitlines()[:20]]
    det.detect(texts[0])
    det.detect(texts[1])
    ts = []
    for t in texts:
        s = time.perf_counter()
        det.detect(t)
        ts.append((time.perf_counter() - s) * 1000)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # macOS 는 바이트
    size = sum(f.stat().st_size for f in Path(path).iterdir() if not onnx or f.name == onnx or f.suffix != ".onnx") / 2**20
    print(json.dumps({"model": spec, "p50_ms": round(statistics.median(ts), 1), "mean_ms": round(statistics.mean(ts), 1),
                      "max_rss_mb": round(rss), "size_mb": round(size)}, ensure_ascii=False))


def main():
    if sys.argv[1] == "--one":
        run_one(sys.argv[2])
        return
    for spec in sys.argv[1:]:
        out = subprocess.run([sys.executable, __file__, "--one", spec], capture_output=True, text=True, cwd=HERE)
        print(out.stdout.strip().splitlines()[-1] if out.returncode == 0 else out.stderr[-2000:])


if __name__ == "__main__":
    main()
