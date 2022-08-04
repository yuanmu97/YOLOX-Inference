import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
