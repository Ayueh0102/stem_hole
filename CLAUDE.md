# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Stamp perforation hole defect detection (郵票打洞瑕疵檢測). Single-file Python tool (`main.py`) using OpenCV to detect misaligned, weak, broken, or missing holes in stamp perforation grids. Sample images in `sample/` (`good.png`, `bad.png`).

## Commands

```bash
# Setup
uv venv && uv sync

# Run local (default) mode — outputs result_defect.jpg + result_defect_bw.jpg
uv run main.py --input_img sample/bad.png

# Run curve mode — outputs result + debug dir with overlays, metrics JSON, crops
uv run main.py --input_img sample/bad.png --mode curve --debug_dir debug_curve_bad --output result_curve_bad.jpg

# Curve mode with all Stage 2 features enabled
uv run main.py --input_img sample/bad.png --mode curve \
  --paper_mask on --scorer template --grid_prior on \
  --metrics_baseline _baselines/bad_metrics.json \
  --debug_dir debug_curve_bad --output result_curve_bad.jpg
```

No test suite, linter, or build step. Python >=3.10, dependencies: `numpy`, `opencv-python`.

## Architecture

Everything lives in `main.py` (~2400 lines). Two detection modes share preprocessing and contour-based hole extraction:

**Shared pipeline:**
- `preprocess_image` — contrast boost, grayscale, binary threshold (or adaptive) → hole mask
- `detect_hole_centers` — `findContours` filtered by diameter (6–16 px) and circularity (>0.45) → `HoleDetection` list
- `build_candidate_edges` — vectorized pairwise edges within `[15, dist_threshold]` px range
- `segment_paper_region` — convex hull of detected hole centres + dilation, used to clip predictions to the actual sheet

**`local` mode** (`detect_and_link_limited_holes`): Builds degree-capped adjacency graph, flags nodes with exactly 2 nearly-collinear neighbors whose perpendicular offset exceeds threshold.

**`curve` mode** (`analyze_curve_mode`): The main experimental mode. Pipeline:
1. Classify edges as row-like/col-like by angle; vote-based direction assignment per point
2. Cluster points by axis coordinate (`--line_cluster_gap`) into row/col chains
3. RANSAC-like robust polynomial fit (`robust_polyfit`) per chain → `CurveModel` (with cached `derivative_coeffs`)
4. Generate band masks from fitted curves (with endpoint tangent extrapolation)
5. **Six candidate detection strategies** run in parallel — see table below
6. Each candidate scored by selected scorer (area or template) → classified as WEAK/BROKEN/MISSING
7. Frame-line filtering (`is_frame_line_like_score`) suppresses false positives from stamp borders
8. Optional regression diff against a saved baseline

Key dataclasses: `HoleDetection` (detected hole), `CurveModel` (fitted curve with residuals + cached derivative).

### Candidate detection strategies

| Strategy | Function | What it does |
|----------|----------|--------------|
| Gap-based | `find_expected_hole_candidates` | Gaps between adjacent detected holes on same curve |
| Spacing lattice | `find_spacing_inferred_hole_candidates` | Fitted pitch/phase grid over observed slot range |
| Local gap | `find_local_gap_candidates` | Per-pair gap analysis with integer-slot validation, plus endpoint extension |
| Row/col consensus | `merge_row_col_consensus_candidates` | Cross-validates row and col local-gap candidates |
| Secondary origin | `find_secondary_origin_candidates` | Re-detects holes from alternative threshold masks |
| **Grid prior** (Phase 2D) | `find_grid_prior_candidates` | Spacing lattice extended to paper boundary, predicting positions beyond detected hull |

### Scorer backends

| Backend | Function | How it scores |
|---------|----------|---------------|
| `area` (default) | `score_expected_hole_roi` | Counts evidence pixels in ROI mask, classifies by area_ratio + template_overlap thresholds |
| `template` (Phase 2C) | `score_expected_hole_template` | NCC against a synthesized dark-disk template via `cv2.matchTemplate`. Median NCC on known-good holes ≈ 0.88 |

The two backends share an output schema so downstream filters (`is_frame_line_like_score`) and dict assemblers stay byte-compatible.

## Key parameters

### Existing (Stage 1 + earlier)

| Flag | Default | Effect |
|------|---------|--------|
| `--dist_threshold` | 32 | Max neighbor distance for edge building |
| `--line_cluster_gap` | 80 | Gap to split point clusters into separate lines |
| `--line_direction_vote_margin` | 1 | Margin for row/col vote assignment |
| `--curve_extend_pixels` | 80 | Tangent extrapolation beyond curve endpoints |
| `--curve_mask_expand_pixels` | 8 | Band mask thickness around curves |
| `--weak_gap_factor` | 1.55 | Minimum gap/pitch ratio to trigger candidate |
| `--local_gap_endpoint_steps` | 2 | Endpoint extension candidates per curve end |
| `--secondary_origin_thresholds` | 144 152 | Alternative binarization thresholds |

### Stage 2 flags (all opt-in, defaults preserve byte-equivalent JSON to Stage 1 baseline)

| Flag | Default | Effect |
|------|---------|--------|
| `--paper_mask {on,off}` | off | Convex-hull paper boundary; clips off-paper candidates and lattice extensions |
| `--threshold_mode {global,adaptive}` | global | Adaptive (Phase 2B) is **experimental** — over-detects on current samples; foundation only |
| `--adaptive_block_size` | 31 | Block size for adaptive threshold (forced odd) |
| `--adaptive_c` | 5 | C parameter subtracted from local mean |
| `--scorer {area,template}` | area | Template (Phase 2C) uses NCC against synthesized hole disk |
| `--grid_prior {on,off}` | off | Phase 2D v1: extends spacing lattice to paper boundary; predicts edge holes |
| `--metrics_baseline PATH` | None | Phase 2E: compares current `curve_metrics.json` to saved baseline, writes `metrics_baseline_diff.json` |

Baseline JSON files for regression detection live at `_baselines/{good,bad}_metrics.json`.

## Debug outputs (curve mode)

Written to `--debug_dir`. Key files:
- `curve_metrics.json` — full pipeline statistics and candidate counts. Stage 2 flags add their own keys only when enabled (preserves byte-equivalent default output).
- `metrics_baseline_diff.json` — written when `--metrics_baseline` is set; lists every numeric/class delta.
- `*_candidates.json` — candidate coordinates, classification, ROI scores per strategy
- `*_overlay.jpg` — visual overlays (curves, candidates, residuals)
- `*_crops.jpg` — zoomed candidate contact sheets
- `*_repaired_circles_mask.jpg` — mask with candidates filled in
- `paper_mask.jpg` / `paper_mask_overlay.jpg` — when `--paper_mask on`
- `adaptive_mask.jpg` — when `--threshold_mode adaptive`

## Recovery archive

`_recovery/` contains patches preserved from a prior NTFS corruption event:
- `stage1-and-phase2A.patch` — 7 commits covering Stage 1 refactor + Phase 2A
- `last8-commits.patch` — extends one commit further back
- `phase2b_wip.diff` — the in-progress Phase 2B adaptive-thresholding patch

`_baselines/` contains regression baseline metrics checked in for use with `--metrics_baseline`.

## Future work (not yet implemented)

- **Phase 2D v2** — true coupled 2D grid fit `position(i,j) = origin + i·R + j·C + Φ(i,j)` so evidence on row 1 informs predictions on row 0
- **Adaptive threshold tuning** — current `--threshold_mode adaptive` over-detects because the contrast-boosted grayscale has too much non-hole texture; alternatives include top-hat morphology or running adaptive only inside `paper_mask`
- **Cross-line consensus on grid_prior** — if a predicted position appears on both a row and col line lattice, score with higher confidence
