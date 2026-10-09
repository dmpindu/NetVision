# Model weights

Weights are not stored in git. NetVision looks for these files:

| File | What it detects | If missing |
|---|---|---|
| `player_detector.pt` | Players (and Hoop, Ref, Ball if trained on the tutorial dataset) | Falls back to stock `yolo11n.pt` (`person`) |
| `ball_detector_model.pt` | The basketball | Uses the player model (`sports ball` on stock weights) |
| `court_keypoint_detector.pt` | 18 court keypoints (YOLO pose) | No tactical view, speed or distance unless you pass `--court-calibration` |
| `rim_detector.pt` | Rim / hoop | Uses the player model's `Hoop` class if it has one, otherwise shot results are `unknown` |

Two ways to get them:

1. Train them on Google Colab with `training/NetVision_Training_Colab.ipynb` (or `python training/train.py ...`).
2. Download the pretrained weights linked in the tutorial's repository
   (github.com/abdullahtarek/basketball_analysis, "Download the Pretrained Weights")
   and save them here with the same file names.
