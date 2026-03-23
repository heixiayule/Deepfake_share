import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import tensorflow as tf
import shutil
import tf2onnx
import onnx
from model.lcnn import build_lcnn

EXPORT_DIR = "../onnx"
WIN_DIR = "/mnt/d/Documents/BaiduSyncdisk/WORKS/Deepfake2026/Deepfake/artifacts/onnx"

BATCH = 1
INPUT_SHAPE = (128, 128, 1)   # 如果你们确定了其他尺寸，可以在这里改

model = build_lcnn(list(INPUT_SHAPE), n_label=2)
spec = (tf.TensorSpec((BATCH, *INPUT_SHAPE), tf.float32, name="input"),)
_ = model(tf.random.normal([BATCH, *INPUT_SHAPE]), training=False)

out_path = os.path.join(EXPORT_DIR, "lcnn_test.onnx")
win_path = os.path.join(WIN_DIR, "lcnn_test.onnx")
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=17,
    output_path=out_path,
)
onnx.checker.check_model(onnx_model)
shutil.copy2(out_path, win_path)
print(f"Exported to: {out_path} and {win_path}")
