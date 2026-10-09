# NetVision: Test Plan and Results

**Project:** NetVision
**Author:** Doutche Mpindu
**Document version:** 1.0
**Related:** `01_Project_Scope.md`, `02_MVP.md`

---

## 1. Purpose

This document explains how NetVision was tested, lists every test case, and records the results. It also lists the defects found while testing, how each was fixed, and the limitations that are still open.

## 2. Test Strategy

Basketball video has no built-in answer key. Nobody labels "this was a pass at frame 72" in a broadcast clip, so the usual approach of comparing output to labeled data is not available for most of the features. NetVision handles this in four layers.

| Level | What it covers | How |
|---|---|---|
| Unit tests | Each module on its own: geometry, ball cleaning, tracking, team colors, possession, events, homography, speed, shots, stats, drawers, config, video I/O | Small hand-built inputs where the right answer can be worked out by hand |
| End to end tests (synthetic) | The whole pipeline from video file to JSON, CSV and annotated video | A generated 11 second clip where every pass, shot, position and distance is known exactly. Only the YOLO models are replaced by fakes that replay the ground truth with noise. Tracking, team assignment from real pixels, and everything after it is the real code |
| Real model smoke tests | The YOLO wrappers and the full pipeline on real NBA footage | Opt-in tests that run when weights and a clip are available, plus manual runs on the three tutorial clips |
| Visual inspection | Overlays, tactical map, readability | Frames pulled from the annotated videos and reviewed by eye |

### 2.1 The synthetic clip

`netvision/testing/synthetic.py` renders a full court from a broadcast style angle using a known homography, six players (three white jerseys, three dark blue), a referee in gray, and an orange ball. The script is:

| Frames | What happens | Expected event |
|---|---|---|
| 0 to 59 | Player A (white) dribbles | |
| 60 to 71 | A passes to B | Pass, team 1 |
| 72 to 129 | B holds | |
| 130 to 141 | B passes toward C, D (blue) cuts in | Interception, team 2 |
| 142 to 189 | D holds | |
| 190 to 201 | D passes to E | Pass, team 2 |
| 202 to 239 | E holds, then shoots | |
| 240 to 275 | Ball in a parabolic arc (50 degree launch) | Shot, team 2 |
| 276 to 285 | Ball drops to A | Defensive rebound, team 1 |
| 286 to 329 | A holds | |

To make the test realistic, the fake detector adds 1 px of random noise to every box, drops the ball in six frames, and adds a false ball detection in the crowd at frame 120. The fake keypoint model adds 0.5 px noise, moves one keypoint to a wrong spot in frame 5, and returns nothing for frames 200 to 209.

### 2.2 Test environment

| Item | Value |
|---|---|
| OS | Linux (cloud container), 2 CPU cores, 7 GB RAM, no GPU |
| Python | 3.13.16 (full suite), 3.10.20 (full suite again, without PyTorch installed) |
| Key packages | ultralytics 8.4.174, torch 2.14.1 (CPU), opencv 5.0, supervision 0.30.9, trackers 2.6.1, numpy 2.5.3 |
| Models | Stock `yolo11n.pt` and `yolo11n-pose.pt` (COCO). Fine-tuned basketball weights and the Hugging Face CLIP model could not be downloaded in this environment (see Section 7) |
| Real clips | The three tutorial clips, 1280x720 at 30 fps: video_1 (117 frames), video_2 (174), video_3 (243) |

### 2.3 Entry and exit criteria

Testing starts when every module in the MVP has code and imports cleanly. The MVP passes testing when all of the following hold: every test passes, line coverage is at least 85 percent, the synthetic run matches ground truth within the tolerances in `02_MVP.md`, and the pipeline completes on a real clip with stock weights.

## 3. Results Summary

| Measure | Result |
|---|---|
| Test functions | 91 |
| Test cases run (including parameterized cases) | 102 |
| Passed | 102 |
| Failed | 0 |
| Line coverage of the `netvision` package | 98 percent (2007 statements, 45 missed) |
| Default run, as in CI (real model tests skipped) | 99 passed, 3 skipped, about 37 seconds |
| Python 3.10 run without PyTorch | 99 passed, 3 skipped |
| Exit criteria met | Yes |

### 3.1 Accuracy against ground truth (synthetic clip)

| Metric | Ground truth | NetVision | Error | Target | Met |
|---|---|---|---|---|---|
| Events found (type, players, team) | pass, interception, pass, defensive rebound | identical sequence | none | exact | Yes |
| Shot release frame | 240 | 240 | 0 frames | within 3 | Yes |
| Release angle | 50.0 deg | 50.07 deg | 0.07 deg | within 5 | Yes |
| Arc apex above release | 2.073 m | 2.117 m | 2.1 percent | within 10 percent | Yes |
| Apex frame | 257 | 257 | 0 | | Yes |
| Contest distance | 1.972 m | 1.901 m | 0.07 m | within 0.5 m | Yes |
| Closest defender | player C | player C | | exact | Yes |
| Shooter court position | (21.00, 10.30) m | (21.00, 10.36) m | 0.06 m | within 0.3 m | Yes |
| Team assignment (7 people) | 3 white, 3 blue, 1 referee | all correct, referee marked team 0 | none | exact | Yes |
| Track IDs | 7 people | 7 IDs, no switches | none | no switches | Yes |
| Per frame ball holder (away from hand-offs) | | 228 of 229 frames | 99.6 percent agree | 95 percent | Yes |
| Total distance (all players) | 32.93 m | 33.26 m | 1.0 percent | within 5 percent | Yes |

Per player distance:

| Player | Team | Ground truth (m) | NetVision (m) | Error (m) |
|---|---|---|---|---|
| A | 1 | 9.75 | 9.84 | +0.08 |
| B | 1 | 2.94 | 2.92 | -0.02 |
| C | 1 | 2.00 | 2.29 | +0.29 |
| D | 2 | 9.99 | 9.78 | -0.21 |
| E | 2 | 2.12 | 2.34 | +0.22 |
| F | 2 | 1.12 | 1.24 | +0.12 |
| Referee | 0 | 5.00 | 4.85 | -0.15 |

Slow moving players show a small upward bias (about 0.1 to 0.3 m over the clip) because box jitter that survives smoothing still adds a little distance. Fast movers are slightly under because sampling every 10 frames cuts corners. Both stay inside the 0.5 m tolerance.

Possession percentage came out 60.9 percent for team 1 against 65.3 percent in the ground truth. The gap comes from frames right after a release, when the ball is still within 50 px of the passer and is counted as possession. Events and shots are not affected because the shot analyzer walks the release back to the true frame.

### 3.2 Real NBA clips with stock weights

These runs used only the stock COCO `yolo11n.pt` model (no basketball fine-tuning), the color team classifier, and no court model.

| Clip | Frames | Ball visible | Tracks created | Tracks covering half the clip | Processing speed (CPU) | Outcome |
|---|---|---|---|---|---|---|
| video_1 | 117 | 59.0 percent | 43 | 5 | 2.2 fps | Completed. Many "ball" hits were shoes, so possession is wrong |
| video_2 | 174 | 2.3 percent | 44 | 6 | 2.1 fps | Completed. Ball almost never found, so no possession or events |
| video_3 | 243 | 3.7 percent | 43 | 3 | 2.1 fps | Completed. Same as video_2 |

The pipeline itself worked on real footage: it ran end to end, wrote every output, and the team colors split the two teams correctly for most players on review of frames. The analytics were not meaningful, and that was expected. The stock COCO ball class barely detects a basketball in broadcast footage, and the person class also picks up fans and referees. This confirms the plan in the scope document: real footage needs the fine-tuned player, ball, and court models (Section 7).

## 4. Test Case Catalog

Every test function has an ID in its docstring, so a failing test in CI can be traced back to this table. "Cases" is more than 1 when a test is parameterized.

### Utilities and configuration

| ID | What is checked | Test function | Cases | Result |
|---|---|---|---|---|
| TC-UTL-01 | center, width, height and foot point of a box. | `test_bbox_center_width_height_foot` | 1 | Pass |
| TC-UTL-02 | Euclidean distance (3-4-5 triangle) and signed xy distance. | `test_measure_distance` | 1 | Pass |
| TC-UTL-03 | containment ratio for full, partial, no overlap and zero-area boxes. | `test_containment_ratio` | 4 | Pass |
| TC-UTL-04 | boxes are clipped to the image and converted to ints. | `test_clip_box_to_image` | 1 | Pass |
| TC-CFG-01 | default thresholds match the tutorial (25 px, 50 px, 11 frames, 80 percent). | `test_config_defaults_match_tutorial_thresholds` | 1 | Pass |
| TC-CFG-02 | overrides apply, lists become tuples, unknown keys are rejected. | `test_config_from_dict_overrides_and_validates` | 1 | Pass |
| TC-CFG-03 | the shipped configs/default.yaml loads and equals the defaults. | `test_config_yaml_round_trip` | 1 | Pass |
| TC-CFG-04 | pixel thresholds scale with video height (720p = 1.0). | `test_pixel_scale` | 1 | Pass |
| TC-VID-01 | saved video can be read back with the same size and frame count. | `test_video_write_read_round_trip` | 1 | Pass |
| TC-VID-02 | frames are streamed in batches with a smaller last batch. | `test_iter_batches` | 1 | Pass |
| TC-VID-03 | missing file, bad extension and empty frame list raise clear errors. | `test_video_errors` | 1 | Pass |
| TC-STB-01 | stubs round trip, and missing, wrong length or corrupt stubs return None. | `test_stub_round_trip_and_validation` | 1 | Pass |
| TC-STB-02 | the cache key changes when model settings change. | `test_stub_key_depends_on_settings` | 1 | Pass |

### Detection wrappers and tracking

| ID | What is checked | Test function | Cases | Result |
|---|---|---|---|---|
| TC-DET-01 | class names match regardless of case, underscores or dashes. | `test_class_name_matching_is_case_and_separator_insensitive` | 1 | Pass |
| TC-DET-02 | an existing weights file is used, otherwise the fallback. | `test_resolve_weights` | 1 | Pass |
| TC-DET-03 | a calibration JSON gives the same keypoints every frame; bad files are rejected. | `test_static_court_calibration` | 1 | Pass |
| TC-TRK-01 | two moving players keep the same track IDs over 30 frames. | `test_player_tracker_keeps_ids_for_moving_players` | 1 | Pass |
| TC-TRK-02 | only player classes above the confidence threshold are tracked. | `test_player_tracker_filters_classes_and_confidence` | 1 | Pass |
| TC-TRK-03 | a player missed for a few frames comes back with the same ID. | `test_player_tracker_survives_short_dropout` | 1 | Pass |
| TC-TRK-04 | asking for a tracker that is not implemented raises an error. | `test_unsupported_tracker_rejected` | 1 | Pass |
| TC-BAL-01 | the most confident ball is kept; other classes and weak detections are ignored. | `test_ball_selector_picks_most_confident` | 1 | Pass |
| TC-BAL-02 | a detection that jumps further than 25 px per frame is removed. | `test_remove_wrong_detections_drops_jumps` | 1 | Pass |
| TC-BAL-03 | after missing frames the allowed travel grows (25 px times the gap). | `test_remove_wrong_detections_scales_with_gap` | 1 | Pass |
| TC-BAL-04 | short gaps are filled linearly; long gaps and the edges stay empty. | `test_interpolate_positions` | 1 | Pass |

### Team assignment

| ID | What is checked | Test function | Cases | Result |
|---|---|---|---|---|
| TC-TEAM-01 | the torso color is found even with floor pixels in the crop. | `test_dominant_jersey_color_ignores_background` | 1 | Pass |
| TC-TEAM-02 | white and dark blue tracks split into team 1 (lighter) and team 2. | `test_color_classifier_splits_teams_lighter_is_team1` | 1 | Pass |
| TC-TEAM-03 | a track whose jersey matches neither team (referee) gets team 0. | `test_color_classifier_flags_referee_as_unknown` | 1 | Pass |
| TC-TEAM-04 | no tracks gives {}, one track gives team 1, empty crop lists are skipped. | `test_color_classifier_edge_cases` | 1 | Pass |
| TC-TEAM-05 | crops are sampled every N frames up to a maximum per track. | `test_team_assigner_sampling_limits` | 1 | Pass |
| TC-TEAM-06 | tiny boxes are skipped and large crops are resized to 128 px high. | `test_crop_player_handles_tiny_and_large_boxes` | 1 | Pass |
| TC-TEAM-07 | the CLIP classifier labels each crop and takes a majority vote per track. | `test_clip_classifier_majority_vote` | 1 | Pass |
| TC-TEAM-08 | 'auto' uses jersey colors when CLIP cannot be loaded; bad names raise. | `test_build_classifier_auto_falls_back_to_color` | 1 | Pass |

### Possession and events

| ID | What is checked | Test function | Cases | Result |
|---|---|---|---|---|
| TC-POS-01 | key points include the box edges at the ball's height and width. | `test_key_points_include_nearest_edge_points` | 1 | Pass |
| TC-POS-02 | a ball inside a player's box (>80 percent) goes to that player. | `test_contained_ball_wins_over_distance` | 1 | Pass |
| TC-POS-03 | the closest player within 50 px gets the ball, nobody beyond that. | `test_distance_threshold` | 1 | Pass |
| TC-POS-04 | possession needs 11 straight frames; with backfill the lead-up frames are filled in. | `test_minimum_consecutive_frames_and_backfill` | 1 | Pass |
| TC-POS-05 | a frame without a ball resets the consecutive frame counter. | `test_missing_ball_breaks_the_streak` | 1 | Pass |
| TC-POS-06 | holder IDs map to teams and percentages ignore frames with no holder. | `test_team_possession_and_percentages` | 1 | Pass |
| TC-EVT-01 | same-team change is a pass, other-team change is an interception. | `test_pass_and_interception` | 1 | Pass |
| TC-EVT-02 | a change after the previous holder's shot is a rebound (defensive or offensive). | `test_rebounds_after_shot` | 1 | Pass |
| TC-EVT-03 | changes involving team 0 and regains by the same player without a shot create no event. | `test_no_event_for_unknown_team_or_same_player` | 1 | Pass |
| TC-EVT-04 | running pass and interception counts per team for the overlay. | `test_cumulative_event_counts` | 1 | Pass |

### Court mapping

| ID | What is checked | Test function | Cases | Result |
|---|---|---|---|---|
| TC-CRT-01 | 18 keypoints, mirrored left/right, inside the court. | `test_geometry_keypoints` | 1 | Pass |
| TC-CRT-02 | rim positions and distance to the nearest rim. | `test_geometry_rims_and_distance` | 1 | Pass |
| TC-CRT-03 | with exact keypoints the homography maps pixels back to meters within 1 cm. | `test_homography_recovers_court_positions` | 1 | Pass |
| TC-CRT-04 | a keypoint far from where the court says it should be is dropped. | `test_validation_removes_a_corrupted_keypoint` | 1 | Pass |
| TC-CRT-05 | fewer than 4 keypoints, or none at all, gives no homography. | `test_not_enough_keypoints` | 1 | Pass |
| TC-CRT-06 | the last good homography covers up to N missing frames, then stops. | `test_homography_carry_forward` | 1 | Pass |
| TC-CRT-07 | foot points become court meters; people outside the court (fans) are dropped. | `test_players_to_court_drops_off_court_people` | 1 | Pass |
| TC-CRT-08 | the top-down court image has the expected size and white lines. | `test_render_court` | 1 | Pass |

### Speed, distance and shots

| ID | What is checked | Test function | Cases | Result |
|---|---|---|---|---|
| TC-SPD-01 | a player moving 5 m/s for 2 s covers 10 m at 18 km/h. | `test_constant_speed` | 1 | Pass |
| TC-SPD-02 | a teleport faster than 40 km/h (ID switch) is not counted as distance. | `test_impossible_jump_is_ignored` | 1 | Pass |
| TC-SPD-03 | a standing player with 3 cm jitter shows far less distance with smoothing. | `test_smoothing_reduces_jitter_distance` | 1 | Pass |
| TC-SPD-04 | invalid fps raises; players appear and disappear without errors. | `test_speed_distance_edge_cases` | 1 | Pass |
| TC-SHT-01 | a parabolic flight is a shot with the right release frame, angle and apex height. | `test_shot_detected_with_correct_metrics` | 1 | Pass |
| TC-SHT-02 | if possession lasts a few frames into the flight, the release is walked back to the true frame. | `test_shot_release_found_when_possession_ends_late` | 1 | Pass |
| TC-SHT-03 | a flight that never rises above the shooter's head is not a shot. | `test_flat_pass_is_not_a_shot` | 1 | Pass |
| TC-SHT-04 | a straight pass toward a player higher on screen (no fall after the peak) is not a shot. | `test_pass_upward_on_screen_is_not_a_shot` | 1 | Pass |
| TC-SHT-05 | flights shorter than 8 visible frames, or with no shooter box, are ignored. | `test_short_or_unseen_flight_is_not_a_shot` | 1 | Pass |
| TC-SHT-06 | contest distance is the court distance to the closest opponent at release. | `test_contest_distance_and_shot_location` | 1 | Pass |
| TC-SHT-07 | the ball dropping through the rim is a make; seen rim without that is a miss. | `test_make_and_miss_with_rim` | 1 | Pass |

### Statistics and drawing

| ID | What is checked | Test function | Cases | Result |
|---|---|---|---|---|
| TC-STA-01 | team and player stats add up (possession percent, passes, interceptions, shots, distance). | `test_build_stats_values` | 1 | Pass |
| TC-STA-02 | without a court mapping, distance and speed are null instead of failing. | `test_build_stats_without_court_mapping` | 1 | Pass |
| TC-STA-03 | JSON, tracking CSV (one row per player per frame) and events CSV are written correctly. | `test_export_files` | 1 | Pass |
| TC-DRW-01 | every drawer draws something on a frame without errors. | `test_each_drawer_changes_the_frame` | 9 | Pass |
| TC-DRW-02 | drawers that need court data or keypoints leave the frame alone when it is missing. | `test_drawers_skip_missing_data` | 1 | Pass |
| TC-DRW-03 | the ball control overlay shows the running share up to the current frame. | `test_team_ball_control_running_percentages` | 1 | Pass |
| TC-DRW-04 | keypoint and tactical view drawers can be switched off. | `test_default_drawers_options` | 1 | Pass |

### End to end on the synthetic clip

| ID | What is checked | Test function | Cases | Result |
|---|---|---|---|---|
| TC-E2E-01 | the generated clip has the scripted events, all keypoints in frame and a readable video. | `test_synthetic_ground_truth_is_consistent` | 1 | Pass |
| TC-E2E-02 | annotated video (same length and size), JSON, tracking CSV and events CSV exist. | `test_outputs_are_written` | 1 | Pass |
| TC-E2E-03 | each of the 7 people keeps a single track ID for the whole clip (no ID switches). | `test_every_player_keeps_one_id` | 1 | Pass |
| TC-E2E-04 | teams come out right from the rendered jersey colors, and the referee is team 0. | `test_team_assignment_matches_jerseys` | 1 | Pass |
| TC-E2E-05 | pass, interception, pass and defensive rebound are found with the right players and teams. | `test_events_match_ground_truth` | 1 | Pass |
| TC-E2E-06 | the per-frame ball holder matches ground truth away from the moments the ball changes hands. | `test_possession_matches_ground_truth` | 1 | Pass |
| TC-E2E-07 | one shot with release frame +/-3, angle +/-5 deg, arc +/-10 percent, contest +/-0.5 m. | `test_shot_metrics_match_ground_truth` | 1 | Pass |
| TC-E2E-08 | total distance within 5 percent, and each player within 0.5 m or 5 percent. | `test_distance_matches_ground_truth` | 1 | Pass |
| TC-E2E-09 | the false far-away ball is removed and dropped ball frames are filled in. | `test_ball_cleaning_in_pipeline` | 1 | Pass |
| TC-E2E-10 | a corrupted keypoint is rejected and 10 frames without keypoints reuse the last homography. | `test_court_mapping_survives_bad_and_missing_keypoints` | 1 | Pass |
| TC-E2E-11 | annotated frames differ from the input and the tactical map area is drawn. | `test_annotated_video_has_overlays` | 1 | Pass |
| TC-E2E-12 | with stubs on, the second run loads cached detections and never calls the detector. | `test_stub_cache_skips_detection` | 1 | Pass |
| TC-E2E-13 | with no keypoints the pipeline still finds events and shots and explains what is off. | `test_runs_without_court_model` | 1 | Pass |
| TC-E2E-14 | a 6-point manual calibration gives the same court positions as the keypoint model. | `test_manual_calibration_matches_keypoint_model` | 1 | Pass |
| TC-E2E-15 | with a rim placed on the ball's way down, the shot is reported as made. | `test_rim_detector_makes_the_shot` | 1 | Pass |
| TC-E2E-16 | pixel thresholds scale with resolution (same events on a 1080p-scaled detection set). | `test_scaled_video_uses_scaled_thresholds` | 1 | Pass |

### Command line

| ID | What is checked | Test function | Cases | Result |
|---|---|---|---|---|
| TC-CLI-01 | command line flags override the YAML config. | `test_cli_arguments_override_config` | 1 | Pass |
| TC-CLI-02 | a missing input video prints an error and exits with code 1. | `test_cli_missing_video_returns_error` | 1 | Pass |
| TC-CLI-03 | the CLI runs the whole pipeline, prints a summary and writes the files. | `test_cli_end_to_end` | 1 | Pass |
| TC-CLI-04 | the synthetic clip generator writes a video and a ground truth JSON. | `test_synthetic_generator_cli` | 1 | Pass |

### Real model smoke tests (opt-in)

| ID | What is checked | Test function | Cases | Result |
|---|---|---|---|---|
| TC-REAL-01 | the real YOLO wrapper returns detections in the expected format (needs weights). | `test_real_yolo_detector_on_frame` | 1 | Pass |
| TC-REAL-02 | the full pipeline with real YOLO weights runs on a real NBA clip (needs weights and video). | `test_real_video_smoke` | 1 | Pass |
| TC-REAL-03 | the YOLO pose wrapper returns one (18, 2) array per frame with NaN for missing points (needs pose weights). | `test_real_keypoint_detector_format` | 1 | Pass |


## 5. Defects Found and Fixed During Testing

| ID | Found by | Problem | Fix |
|---|---|---|---|
| D-01 | Synthetic E2E | A normal pass to a player farther from the camera was reported as a shot, because on screen the ball rose above the passer's head | A shot must now come back down after its peak by at least 25 percent of its rise. A pass toward a far player only goes up. Covered by TC-SHT-04 |
| D-02 | Synthetic E2E | Shot release frame was 4 to 6 frames late and the angle was 6 degrees low, because the ball stays within 50 px of the shooter for a few frames after leaving the hand | After fitting the arc, the analyzer walks back through earlier frames while they still sit on the arc to find the true launch point. Covered by TC-SHT-02 |
| D-03 | Synthetic E2E | The rebound after the shot was labeled an interception | The event detector now looks for a shot anywhere in the previous holder's possession run, not only after it ended. Covered by TC-EVT-02 |
| D-04 | Synthetic E2E | One blue player far from the camera was marked as belonging to neither team, because the upper half of the box held crowd and floor colors | Jersey color now comes from the torso region (rows 20 to 50 percent, middle half), using the largest color cluster. Covered by TC-TEAM-01 |
| D-05 | Synthetic E2E | The referee pulled the team color average toward gray and was assigned to a team | Team colors use the median of each group, and tracks far from both colors become team 0. Covered by TC-TEAM-03 |
| D-06 | Synthetic E2E | Distance was overstated by 20 to 180 percent for slow players because 1 px box jitter adds up | Court positions are smoothed with a 9 frame moving average and sampled every 10 frames. Covered by TC-SPD-03 and TC-E2E-08 |
| D-07 | Unit test setup | `supervision.ByteTrack` is deprecated and will be removed | Switched to the maintained `trackers` package, with supervision kept as a fallback |
| D-08 | Real clip review | Possession candidates included fans and people on the bench | When a court mapping exists, only players standing on the court can have the ball |

## 6. How to Run the Tests

```bash
pip install -r requirements-dev.txt
pytest                                  # full suite, about 40 seconds on CPU
pytest -m "not slow"                    # skip the end to end tests, under 10 seconds
pytest --cov=netvision --cov-report=term-missing
```

Optional real model tests:

```bash
export NETVISION_YOLO_WEIGHTS=yolo11n.pt
export NETVISION_POSE_WEIGHTS=yolo11n-pose.pt
export NETVISION_REAL_VIDEO=input_videos/video_2.mp4
pytest -m real_model
```

GitHub Actions runs the default suite on Python 3.10 and 3.12 for every push.

## 7. Open Issues and Limitations

1. **Fine-tuned weights not validated yet.** The Google Drive weights from the tutorial and the Hugging Face fashion-clip model were blocked by this environment's network rules. The code paths are covered by tests with fakes and by the stock model wrappers, but accuracy on real footage needs one run with the fine-tuned models (download them, or train on Colab with the included notebook).
2. **Track fragmentation on panning cameras.** Stock detections plus a moving broadcast camera produced about 43 track IDs for roughly 10 people per clip. Tuning ByteTrack thresholds did not help much. Phase 2 should try BoT-SORT with camera motion compensation (already in the `trackers` package) or DeepSORT with appearance re-identification.
3. **2D occlusion.** If the ball flies through the screen area of a far player for 11 or more frames, that player can be credited with possession. The 11 frame rule from the tutorial limits this but does not remove it. The synthetic script keeps players out of the shot path, so the end to end test does not cover this case.
4. **Shot geometry is 2D.** Release angle and arc are measured on screen and are most accurate when the shot crosses the screen. Shots toward or away from the camera will read flatter or steeper.
5. **First frame has no tracks.** ByteTrack confirms tracks on the second frame, so frame 0 has no players. This has no effect on stats.
6. **Make or miss** needs a rim detector. The fine-tuned player model from the tutorial dataset includes a Hoop class, which NetVision picks up automatically.
