# YOLOX-Inference

A simple script of using YOLOX model just for inference.

I kept only necessary code to do inference in `yolox.py`.

To load a pretrained model and do inference:

```python
from yolox import create_yolox_model
import numpy as np
import torch

conf_threshold = 0.25
nms_threshold = 0.45

model = create_yolox_model("yolox-s", "yolox_s.pth", conf_threshold, nms_threshold)

inp = np.load("input.npy")
inp = torch.from_numpy(inp).cuda()

with torch.no_grad():
    outputs = model(inp)
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    print(outputs.cpu().numpy())
```

The `input.npy` is a numpy array after applying original preprocessing on a sample image.