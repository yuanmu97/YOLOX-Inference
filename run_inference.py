import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from yolox import create_yolox_model
import numpy as np
import torch

conf_threshold = 0.25
nms_threshold = 0.45

model = create_yolox_model("yolox-s", "yolox_s.pth", conf_threshold, nms_threshold)
print("yolox model loaded.")

inp = np.load("input.npy")
inp = torch.from_numpy(inp).cuda()

with torch.no_grad():
    outputs = model(inp)
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    print(outputs.cpu().numpy())