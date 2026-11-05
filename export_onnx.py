"""PyTorch 모델 → ONNX fp32 + INT8.

  python export_onnx.py models/v3/trim models/v3/trim_onnx

GLiNER 의 export_to_onnx(quantize=True) 는 가중치를 QUInt8 로 양자화하는데, 이 모델에서는 점수가 무너진다
(fp32 대비 구간 일치 11%, 평가 재현율 0%). 가중치를 QInt8(부호 있음)·채널별로 양자화하면 fp32 와 거의 같다.
불러올 때: GLiNER.from_pretrained(dir, load_onnx_model=True, onnx_model_file="model_quantized.onnx") 후
words_splitter 를 EojeolSplitter 로 바꿔야 한다 (detector.py 처럼).
"""
import argparse
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    from gliner import GLiNER

    src, dst = Path(a.src), Path(a.dst)
    m = GLiNER.from_pretrained(str(src), local_files_only=True, map_location="cpu")
    m.export_to_onnx(dst, quantize=False)
    quantize_dynamic(str(dst / "model.onnx"), str(dst / "model_quantized.onnx"),
                     weight_type=QuantType.QInt8, per_channel=True)
    if (src / "train_meta.json").exists():
        shutil.copy(src / "train_meta.json", dst / "train_meta.json")
    for f in ("model.onnx", "model_quantized.onnx"):
        print(f, (dst / f).stat().st_size // 2**20, "MB")


if __name__ == "__main__":
    main()
