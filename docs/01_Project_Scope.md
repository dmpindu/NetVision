# NetVision: Project Scope

**Project:** NetVision, an NBA computer vision shot tracker and game analytics pipeline
**Author:** Doutche Mpindu
**Document version:** 1.0
**Status:** Approved for MVP build

---

## 1. Purpose

NetVision turns basketball broadcast footage into structured tracking data and game statistics. A user gives the program a video clip. NetVision finds the players, the ball, and the court in every frame, follows each player over time, works out which team each player is on, and then measures what happened: who had the ball, when it was passed or stolen, how far and how fast each player moved, and what each shot looked like (arc, release angle, and how closely it was defended).

The output is an annotated video, a top-down tactical map, and machine-readable files (JSON and CSV) that can feed dashboards or further analysis.

This document defines what the project will and will not do, the requirements it has to meet, and the plan for building it. The companion document `02_MVP.md` narrows this scope down to the first shippable version.

## 2. Background

Professional teams pay for optical tracking systems that use many fixed cameras in the arena. Those systems are expensive and the data is not public. Broadcast video, on the other hand, is everywhere. The question this project explores is how much useful tracking data can be recovered from a single broadcast camera using open source models.

The design is based on a public YOLO and OpenCV basketball analysis tutorial (the transcript supplied with this project). The tutorial covers player and ball detection, tracking, jersey-based team assignment, ball possession, pass and interception detection, court keypoints, a tactical view, and speed and distance. NetVision follows that architecture and extends it with shot analytics, structured data export, a command line interface, configuration, automated tests, and continuous integration.

## 3. Objectives

1. Detect players and the basketball in every frame of a broadcast clip using YOLO11 models.
2. Track players across frames so each player keeps one ID, and clean up noisy ball detections.
3. Assign every player to one of two teams based on jersey appearance.
4. Determine ball possession frame by frame and compute team possession percentage.
5. Detect passes and interceptions from changes in possession.
6. Detect court keypoints and use a homography to map the broadcast view onto a top-down court in real-world meters.
7. Measure distance covered and speed for each player in meters and km/h.
8. Detect shot attempts and measure shot arc, release angle, and contest distance.
9. Export everything as an annotated video plus JSON and CSV files.
10. Ship with automated tests, documentation, and a reproducible setup.

## 4. Stakeholders and Users

| Stakeholder | Interest |
|---|---|
| Project owner (Doutche Mpindu) | Portfolio project that shows end-to-end ML engineering and software engineering practice |
| Recruiters and interviewers | Clear README, readable code, evidence of testing and design decisions |
| Analysts and fans (end users) | Run the tool on a clip and get stats and a visual breakdown |
| Future contributors | Modular code that is easy to extend (new detectors, new metrics) |

## 5. In Scope

### 5.1 Functional scope

| ID | Capability | Description |
|---|---|---|
| F1 | Video I/O | Read common video formats frame by frame and write an annotated output video |
| F2 | Player detection and tracking | YOLO11 detection plus ByteTrack multi-object tracking with stable IDs |
| F3 | Ball detection | Pick the single most confident ball per frame |
| F4 | Ball track cleaning | Reject impossible jumps and interpolate short gaps |
| F5 | Team assignment | Zero-shot CLIP jersey classifier (as in the tutorial) with a no-download color clustering fallback, and majority voting per track |
| F6 | Ball possession | Distance and containment based possession with a minimum consecutive frame rule |
| F7 | Team ball control | Running possession percentage per team |
| F8 | Pass and interception detection | Possession changes classified as pass (same team) or interception (other team) |
| F9 | Court keypoint detection | YOLO pose model for 18 court landmarks, or a manual calibration file for fixed cameras |
| F10 | Keypoint validation | Drop keypoints whose distance ratios disagree with real court geometry |
| F11 | Tactical view | Homography from image to court coordinates and a top-down mini map |
| F12 | Speed and distance | Per player distance in meters and speed in km/h |
| F13 | Shot detection | Detect shot attempts from ball flight after a release |
| F14 | Shot metrics | Arc apex height, release angle, flight time, contest distance to the nearest defender, optional make or miss when a rim model is supplied |
| F15 | Statistics export | JSON summary (team and player stats, events, shots) and per frame CSV tracking data |
| F16 | Caching | Save and reuse detection results ("stubs") to skip slow inference on reruns |
| F17 | CLI and configuration | One command to run the pipeline, settings in a YAML file |
| F18 | Training tooling | Scripts to fine-tune player, ball, rim, and court keypoint models on a GPU (Colab) |

### 5.2 Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| N1 | Portability | Runs on Windows, macOS, and Linux with Python 3.10 or newer |
| N2 | Hardware | Works on CPU. Uses a GPU automatically when one is available |
| N3 | Memory | Does not hold the full video in memory. Frames are streamed twice (analyze, then draw) |
| N4 | Robustness | Missing models (court, rim, CLIP) degrade gracefully. The pipeline still produces the outputs it can |
| N5 | Testability | Core logic is pure Python and covered by unit tests. Model calls are behind small interfaces that tests can replace with fakes |
| N6 | Maintainability | One module per concern, type hints, docstrings, PEP 8 style |
| N7 | Reproducibility | Pinned minimum dependency versions, deterministic synthetic test video |
| N8 | Quality gate | Automated test suite runs in GitHub Actions on every push |

## 6. Out of Scope

These items are intentionally excluded from this project or deferred to later phases.

1. Real-time or live stream processing. NetVision works on recorded clips.
2. Multi-camera fusion and 3D ball reconstruction.
3. Automatic camera cut detection in full broadcasts. Input clips are expected to be a single continuous camera shot.
4. Player identification by name or jersey number recognition (OCR).
5. A web front end or hosted service (possible Phase 3).
6. Referee rule enforcement such as traveling or double dribble (needs pose, Phase 2).
7. Shipping fine-tuned model weights inside the repository. Weights are trained separately and placed in `models/`.

## 7. Assumptions and Constraints

1. The development environment for the MVP has no GPU, so fine-tuning happens on Google Colab using the provided training scripts.
2. Without fine-tuned weights the system uses stock YOLO11 COCO weights (`person` and `sports ball`). Stock weights also detect referees and fans, so accuracy is lower until fine-tuned models are added.
3. Clips show a standard broadcast angle with most of the half court or full court visible.
4. The court is assumed to be FIBA size (28 m by 15 m). NBA courts (28.65 m by 15.24 m) are close enough for MVP purposes and can be set in config.
5. Metric measurements (meters, km/h, contest distance) require a working court mapping, either from the keypoint model or a manual calibration file.
6. Shot arc and release angle are measured in the image plane and scaled using the shooter's bounding box height. They are estimates, not ground truth.

## 8. System Architecture

The pipeline runs in two passes over the video.

**Pass 1 (analyze):** stream frames, run detectors, update the tracker, collect jersey crops and court keypoints.
**Analysis stage:** clean the ball track, assign teams, compute possession, events, court positions, speed, distance, and shots. This stage is pure Python and does not touch frames.
**Pass 2 (render):** stream frames again, draw every overlay, and write the output video. Then write JSON and CSV.

```
input.mp4
   |
   v
[Pass 1] PlayerTracker (YOLO + ByteTrack) --> player tracks
         BallTracker   (YOLO)             --> raw ball boxes
         RimDetector   (optional YOLO)    --> rim boxes
         CourtKeypointDetector (YOLO pose or manual file) --> keypoints
         TeamAssigner  (CLIP or color)    --> per track team votes
   |
   v
[Analysis] ball cleaning + interpolation
           possession (BallAcquisitionDetector)
           events (PassInterceptionDetector)
           TacticalViewConverter (keypoint validation + homography)
           SpeedDistanceCalculator
           ShotAnalyzer (arc, release angle, contest distance)
           StatsAggregator
   |
   v
[Pass 2] Drawers --> annotated video
         Exporters --> stats.json, tracking.csv, events.csv
```

### Package layout

| Package | Responsibility |
|---|---|
| `netvision.utils` | Video streaming, bounding box geometry, stub cache |
| `netvision.trackers` | Player tracker, ball tracker, rim detector |
| `netvision.team_assigner` | Jersey classifiers and per track voting |
| `netvision.possession` | Ball acquisition (possession) detector |
| `netvision.events` | Pass and interception detection |
| `netvision.court` | Court geometry, keypoint detector, keypoint validation, homography, tactical view |
| `netvision.metrics` | Speed and distance, shot analysis, statistics aggregation |
| `netvision.drawers` | All video overlays |
| `netvision.pipeline` | Orchestrates the passes |
| `netvision.cli` | Command line entry point |

## 9. Technology Stack

| Area | Choice | Reason |
|---|---|---|
| Language | Python 3.10+ | Ecosystem for CV and ML |
| Detection | Ultralytics YOLO11 | Strong accuracy and speed, simple training API |
| Tracking | ByteTrack via Supervision | Fast, robust, no re-ID model needed for MVP |
| Team classification | Hugging Face `patrickjohncyh/fashion-clip` zero-shot, scikit-learn KMeans fallback | Matches the tutorial, works without labeled data |
| Geometry | OpenCV `findHomography`, NumPy | Standard perspective transform |
| Data | pandas | Interpolation and tabular export |
| Testing | pytest, pytest-cov | Unit, integration, and end-to-end tests |
| CI | GitHub Actions | Runs tests on every push |
| Training | Ultralytics + Roboflow datasets on Colab | GPU access |

## 10. Development Process

The project follows a lightweight iterative SDLC.

| Phase | Activities | Output |
|---|---|---|
| 1. Requirements | Review the tutorial transcript and resume goals, define scope | This document |
| 2. MVP definition | Prioritize features with MoSCoW, write user stories and acceptance criteria | `02_MVP.md` |
| 3. Design | Module boundaries, data model, two pass pipeline | Section 8 above, code skeleton |
| 4. Implementation | Build modules bottom up: utils, trackers, team assignment, possession, events, court, metrics, drawers, pipeline, CLI | `netvision/` package |
| 5. Testing | Unit tests per module, integration tests on a synthetic video with known ground truth, smoke test on real NBA clips | `tests/`, `03_Test_Plan_and_Results.md` |
| 6. Release | README, CI workflow, tagged commit on GitHub | GitHub repository |
| 7. Iteration | Fine-tune models on Colab, then Phase 2 features | Roadmap in Section 12 |

## 11. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Stock YOLO weights detect fans and referees as players | Wrong teams and possession | High | Fine-tune player model, filter by court mask when a homography exists, confidence and size filters |
| The ball is small and blurred | Missing ball frames | High | Jump rejection plus interpolation, fine-tuned ball model with motion blur augmentation |
| ID switches after occlusion | Inflated pass counts | Medium | Minimum possession frames, ByteTrack buffer, DeepSORT re-ID in Phase 2 |
| Court keypoints not visible or wrong | Bad metric numbers | Medium | Geometric validation, minimum 4 points, carry the last good homography forward for short gaps |
| No GPU for training | Cannot fine-tune in the dev environment | Certain | Colab training scripts, stock weights for development |
| Model downloads blocked (Hugging Face) | CLIP unavailable | Medium | Color clustering fallback selected automatically |
| 2D occlusion makes the ball look held when it is in the air | False possession | Medium | Containment threshold plus the 11 frame rule from the tutorial |

## 12. Roadmap

**Phase 1 (MVP):** everything in Section 5 except the Phase 2 items listed below.

**Phase 2:**
1. DeepSORT tracker with appearance re-identification as a selectable tracker.
2. Pose estimation (YOLO11-pose) on the shooter for a skeleton based release point, elbow angle, and jump height.
3. Fine-tuned rim detector shipped through the training script, enabling reliable make or miss.
4. Shot chart output on the tactical court.
5. Possession sequences (start, end, outcome) as a first class export.

**Phase 3:**
1. Streamlit or React dashboard for uploaded clips.
2. Batch processing of many clips and season level aggregation.

## 13. Success Criteria

1. One command processes a clip end to end and produces the video, JSON, and CSV outputs.
2. The automated test suite passes with at least 85 percent line coverage of the core logic.
3. On the synthetic ground truth video, possession, pass, interception, and shot counts match ground truth exactly, and distance error is under 5 percent.
4. On a real NBA clip with stock weights, the pipeline runs without errors and produces plausible output.
5. Documentation lets a new developer install, test, and run the project in under 15 minutes.

## 14. Deliverables

1. Source code in the `netvision/` package with a CLI.
2. Training scripts and a Colab notebook in `training/`.
3. Test suite in `tests/` and a synthetic video generator in `scripts/`.
4. Documents: Project Scope (this file), MVP, Test Plan and Results.
5. README with setup and usage, GitHub Actions workflow, MIT license.
