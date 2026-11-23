# Fire and smoke model

`fire_smoke.pt` is the YOLOv8n fire and smoke model from
https://github.com/luminous0219/fire-and-smoke-detection-yolov8 (commit c30923f, `weights/best.pt`),
trained on the Roboflow "fire-and-smoke" dataset. Licensed under AGPL-3.0, see `LICENSE` here.
It was saved again with only the network and its class names (the original needed the `dill` library to load).
The program checks its SHA-256 before loading it (see `app/hazard.py`).
