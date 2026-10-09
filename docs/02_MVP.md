# NetVision: Minimum Viable Product (MVP)

**Project:** NetVision
**Author:** Doutche Mpindu
**Document version:** 1.0
**Related:** `01_Project_Scope.md`, `03_Test_Plan_and_Results.md`

---

## 1. MVP Goal

The MVP is the smallest version of NetVision that proves the full idea works from start to finish: a user runs one command on a basketball clip and gets back an annotated video, a tactical top-down view, and a stats file with possession, passes, interceptions, player movement, and shot metrics.

The MVP has to run on a normal laptop CPU with no GPU and no custom trained weights, because that is the environment it was built and tested in. Every piece that needs a fine-tuned model has a working fallback, and the training scripts are included so better models can be dropped in later without code changes.

## 2. Target User and Core Problem

**User:** a basketball analyst, coach, or fan with a short broadcast clip.

**Problem:** broadcast video has no data attached to it. Answering questions like "which team controlled the ball", "how many passes led to that shot", or "how far did this player run" means watching and counting by hand.

**MVP promise:** point NetVision at the clip and those answers come back automatically, both as a video you can watch and as data you can analyze.

## 3. Feature Prioritization (MoSCoW)

### Must have

| # | Feature | Notes |
|---|---|---|
| M1 | Read a video and write an annotated video | Streaming, two passes |
| M2 | Player detection and tracking with stable IDs | YOLO11 + ByteTrack |
| M3 | Ball detection, jump rejection, interpolation | Tutorial logic: 25 px per frame of gap, pandas interpolation |
| M4 | Team assignment | CLIP zero-shot when available, otherwise jersey color clustering, majority vote per track |
| M5 | Ball possession per frame | 50 px distance threshold, 80 percent containment, 11 consecutive frames |
| M6 | Team ball control percentage | Running percentage overlay |
| M7 | Pass and interception detection | Same team change is a pass, other team change is an interception |
| M8 | JSON stats and CSV tracking export | Team, player, event, and shot data |
| M9 | CLI with config file | `python -m netvision input.mp4` |
| M10 | Automated tests and CI | pytest + GitHub Actions |

### Should have

| # | Feature | Notes |
|---|---|---|
| S1 | Court keypoint detection and validation | YOLO pose model when weights exist |
| S2 | Manual court calibration file | Works today without a trained keypoint model |
| S3 | Tactical top-down mini map | Drawn programmatically, no image asset needed |
| S4 | Speed (km/h) and distance (m) per player | Needs S1 or S2 |
| S5 | Shot attempt detection | Ball flight after release that rises above the shooter |
| S6 | Shot arc, release angle, contest distance | Arc and angle from a parabola fit, contest distance from the court map |
| S7 | Detection cache (stubs) | Fast reruns while tuning |
| S8 | Training scripts for player, ball, rim, keypoints | Colab ready |

### Could have

| # | Feature | Notes |
|---|---|---|
| C1 | Make or miss classification | Only when a rim detector model is supplied |
| C2 | Rebound events | A possession change right after a shot is labeled a rebound, not a pass or interception |
| C3 | Docker image | Easy setup on any machine |

### Won't have (this release)

| # | Feature | Why deferred |
|---|---|---|
| W1 | DeepSORT re-identification | ByteTrack is good enough for short clips. Phase 2 |
| W2 | Pose estimation on the shooter | Needs more compute and labeled validation. Phase 2 |
| W3 | Jersey number OCR and player names | Separate research problem |
| W4 | Web dashboard | Phase 3 |
| W5 | Live stream processing | Out of scope |

## 4. User Stories and Acceptance Criteria

**US-1 Run the analysis.**
As an analyst, I want to run one command on a clip so that I get all outputs without writing code.
Acceptance: `python -m netvision clip.mp4 --output-dir out/` exits with code 0 and creates `out/clip_annotated.mp4`, `out/clip_stats.json`, `out/clip_tracking.csv`, and `out/clip_events.csv`.

**US-2 See who is who.**
As a viewer, I want each player marked with a colored ellipse and an ID that stays the same across frames so that I can follow individual players.
Acceptance: players have ellipses colored by team, IDs come from the tracker, and on the synthetic test video every player keeps one ID for the whole clip.

**US-3 Know who has the ball.**
As an analyst, I want the current ball holder highlighted and each team's possession percentage shown so that I can see which team controls the game.
Acceptance: possession percentages add up to 100 percent over frames where someone has the ball, and on the synthetic video the per frame holder matches ground truth after the confirmation delay.

**US-4 Count passes and steals.**
As a coach, I want passes and interceptions counted per team so that I can judge ball movement and turnovers.
Acceptance: on the synthetic video the pass and interception counts match ground truth exactly, and each event lists the frame, the passer, the receiver, and the teams.

**US-5 Measure movement.**
As a trainer, I want distance covered and current speed for each player in real units so that I can judge workload.
Acceptance: with a court mapping available, distances are reported in meters and speeds in km/h, and on the synthetic video total distance is within 5 percent of ground truth.

**US-6 Break down shots.**
As a shooting coach, I want each shot's arc, release angle, and how close the nearest defender was so that I can evaluate shot quality.
Acceptance: on the synthetic video the shot is detected at the right release frame (within 3 frames), the release angle is within 5 degrees of ground truth, and contest distance is within 0.5 m.

**US-7 Work with no GPU and no custom models.**
As a developer, I want the tool to run on stock weights so that I can try it before training anything.
Acceptance: with only `yolo11n.pt` available the pipeline runs on a real NBA clip without errors and skips the features that need missing models with a clear log message.

**US-8 Reuse the data.**
As a data analyst, I want structured output so that I can load it into pandas or a dashboard.
Acceptance: the JSON validates against the documented schema, and the CSV has one row per tracked player per frame with frame, track ID, team, box, court position, and speed.

## 5. MVP Architecture Summary

```
Pass 1  frames --> PlayerTracker, BallTracker, RimDetector?, KeypointDetector?, TeamAssigner
Analyze ball cleaning --> possession --> events --> court mapping --> speed/distance --> shots --> stats
Pass 2  frames --> drawers --> annotated video ; exporters --> JSON / CSV
```

Every model is wrapped in a small class with a `detect_frame` or `predict` method. The pipeline accepts these objects through its constructor, so the test suite swaps them for fakes that return known ground truth. This is what makes the analysis logic testable without a GPU or any downloads.

## 6. Key Algorithms in the MVP

**Ball cleaning.** Walk through frames and keep the last trusted ball position. A new detection is rejected when it is further than `25 px x frames since last good detection`. Gaps are then filled with linear interpolation, with back fill for the first frames.

**Possession.** For each frame, find players whose box contains at least 80 percent of the ball box. If none, measure the distance from the ball center to eight key points on each player box (corners, edge midpoints, and the nearest edge points at the ball's height or width). The closest player within 50 px is the candidate. A candidate only becomes the holder after 11 consecutive frames.

**Passes and interceptions.** Walk the possession timeline and remember the last holder. When a new holder appears, the change is a pass if both players are on the same team and an interception if not. If a shot happened between the two holders, the change is recorded as a rebound instead.

**Court mapping.** The 18 court keypoints are defined in meters. Detected keypoints are validated by comparing their pixel distance ratios to the real court ratios (80 percent tolerance). With at least four valid points, OpenCV computes a homography that maps each player's foot point to court meters. The last good homography is reused for short gaps.

**Speed and distance.** Court positions are smoothed with a 9 frame moving average to remove box jitter, then sampled every 10 frames. Distance is the sum of steps in meters, ignoring impossible jumps (faster than 40 km/h, usually an ID switch). Speed is distance over elapsed time, converted to km/h.

**Shots.** A flight segment starts when the holder loses the ball. It is a shot when the ball rises above the shooter's head by at least 30 percent of the shooter's box height, comes back down after its peak, and stays visible for at least 8 frames. A parabola is fitted to the ball's image path, and the analyzer walks back along the arc to the true launch frame, because the ball is still within reach of the shooter for a few frames after it leaves the hand. Release angle comes from the launch velocity of the fit. Apex height is converted to meters using the shooter's box height as a scale (default player height 2.0 m). Contest distance is the court distance from the shooter to the closest opponent at the release frame.

## 7. Outputs

| File | Contents |
|---|---|
| `<name>_annotated.mp4` | Video with all overlays and the tactical mini map |
| `<name>_stats.json` | Metadata, team stats, player stats, event list, shot list |
| `<name>_tracking.csv` | One row per player per frame |
| `<name>_events.csv` | One row per pass, interception, rebound, or shot |

## 8. Definition of Done

1. All Must have and Should have items are implemented.
2. The test suite passes locally and in GitHub Actions.
3. Line coverage of the `netvision` package is at least 85 percent, excluding thin model wrappers that need downloads.
4. The synthetic end to end test matches ground truth for possession, passes, interceptions, and shots.
5. A smoke test on at least one real NBA clip with stock YOLO11 weights finishes without errors.
6. README explains install, usage, training, and testing.
7. Test results are written up in `03_Test_Plan_and_Results.md`.

## 9. Known MVP Limitations

1. Stock COCO weights detect referees and people off the court, which lowers team and possession accuracy until a fine-tuned player model is used.
2. Without a court keypoint model or calibration file there is no tactical view and no metric speed, distance, or contest distance. Shot arc and release angle still work.
3. Make or miss needs a rim detector. It is reported as `unknown` otherwise.
4. Shot arc and release angle are measured in 2D and depend on camera angle.
5. Long occlusions can cause ID switches with ByteTrack.
