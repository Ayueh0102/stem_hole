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

# Curve mode with extended band masks
uv run main.py --input_img sample/bad.png --mode curve --curve_extend_pixels 300 --curve_mask_expand_pixels 8 --debug_dir debug_curve_bad --output result_curve_bad.jpg
```

No test suite, linter, or build step. Python >=3.10, dependencies: `numpy`, `opencv-python`.

## Architecture

Everything lives in `main.py` (~2090 lines). Two detection modes share preprocessing and contour-based hole extraction:

**Shared pipeline:**
- `preprocess_image` — contrast boost, grayscale, binary threshold → hole mask
- `detect_hole_centers` — `findContours` filtered by diameter (6–16 px) and circularity (>0.45) → `HoleDetection` dataclass list
- `build_candidate_edges` — pairwise edges within `[15, dist_threshold]` px range

**`local` mode** (`detect_and_link_limited_holes`): Builds degree-capped adjacency graph, flags nodes with exactly 2 nearly-collinear neighbors whose perpendicular offset exceeds threshold.

**`curve` mode** (`analyze_curve_mode`): The main experimental mode. Pipeline:
1. Classify edges as row-like/col-like by angle; vote-based direction assignment per point
2. Cluster points by axis coordinate (`--line_cluster_gap`) into row/col chains
3. RANSAC-like robust polynomial fit (`robust_polyfit`) per chain → `CurveModel` dataclass
4. Generate band masks from fitted curves (with endpoint tangent extrapolation)
5. Multiple candidate detection strategies run in parallel:
   - **Gap-based** (`find_expected_hole_candidates`) — gaps between adjacent detected holes on same curve
   - **Spacing lattice** (`find_spacing_inferred_hole_candidates`) — fitted pitch/phase grid
   - **Local gap** (`find_local_gap_candidates`) — per-pair gap analysis with integer-slot validation
   - **Row/col consensus** (`merge_row_col_consensus_candidates`) — cross-validates row and col candidates
   - **Secondary origin** (`find_secondary_origin_candidates`) — re-detects holes from alternative threshold masks
6. Each candidate scored by ROI mask evidence → classified as WEAK/BROKEN/MISSING
7. Frame-line filtering (`is_frame_line_like_score`) suppresses false positives from stamp borders

Key dataclasses: `HoleDetection` (detected hole), `CurveModel` (fitted curve with residuals and threshold).

## Key parameters

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

## Debug outputs (curve mode)

Written to `--debug_dir`. Key files:
- `curve_metrics.json` — full pipeline statistics and candidate counts
- `*_candidates.json` — candidate coordinates, classification, ROI scores
- `*_overlay.jpg` — visual overlays (curves, candidates, residuals)
- `*_crops.jpg` — zoomed candidate contact sheets
- `*_repaired_circles_mask.jpg` — mask with candidates filled in
