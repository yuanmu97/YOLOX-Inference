# YOLOX-Inference

A simple script of using YOLOX model just for inference.

I kept only necessary code to do inference in `yolox.py`.

To load a pretrained model and do inference:

```python
from yolox import create_yolox_model, load_image
import torch

conf_threshold = 0.25
nms_threshold = 0.45

model = create_yolox_model("yolox-s", "yolox_s.pth", conf_threshold, nms_threshold)
print("yolox model loaded.")

inp, ratio = load_image("xzl.jpg")

with torch.no_grad():
    outputs = model(inp).cpu()
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    bboxes = outputs[:, 0:4]
    bboxes /= ratio
    cls_idxes = outputs[:, 6]
    scores = outputs[:, 4] * outputs[:, 5]
    for bbox, cls_idx, score in zip(bboxes, cls_idxes, scores):
        print(f"class={cls_idx}, conf={score}, bbox={bbox}")
```

Terminla Outputs

```bash
yolox model loaded.
class=0.0, conf=0.4121667146682739, bbox=tensor([1175.3997,  325.5804, 1246.6086,  473.2525])
class=33.0, conf=0.37325724959373474, bbox=tensor([466.2440, 819.2187, 508.7644, 896.0631])
class=6.0, conf=0.2587979733943939, bbox=tensor([   8.5561,    5.7042, 1976.1226, 1082.2935])
```
