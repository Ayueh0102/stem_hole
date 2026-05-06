import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# MAD-to-sigma consistency factor for normally-distributed residuals.
MAD_TO_SIGMA = 1.4826


@dataclass
class HoleDetection:
    index: int
    center: tuple[int, int]
    radius: float
    diameter: float
    area: float
    circularity: float
    contour: np.ndarray


@dataclass
class CurveModel:
    line_id: int
    orientation: str
    ordered_indices: list[int]
    coeffs: np.ndarray
    parameter_min: float
    parameter_max: float
    residuals: np.ndarray
    inlier_mask: np.ndarray
    median_residual: float
    mad_residual: float
    offset_threshold: float
    derivative_coeffs: np.ndarray = field(init=False)

    def __post_init__(self):
        self.derivative_coeffs = np.polyder(self.coeffs)


def preprocess_image(
    image_path,
    threshold_mode="global",
    adaptive_block_size=31,
    adaptive_c=5,
    tophat_kernel=21,
    tophat_percentile=98.0,
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    contrast_img = np.clip(img * (80 / 127 + 1) - 80, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)

    if threshold_mode == "adaptive":
        # adaptiveThreshold's blockSize must be odd
        block = int(adaptive_block_size)
        if block % 2 == 0:
            block += 1
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            int(adaptive_c),
        )
        # Suppress single-pixel speckle from adaptive threshold without
        # eroding genuine 6-px holes.
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    elif threshold_mode == "tophat":
        # Black top-hat = morphological closing - input. Closing fills
        # dark holes with bright (because elliptical kernel > hole_dia
        # bridges across them); subtracting input leaves a positive
        # response only where holes used to be. Unlike adaptive
        # thresholding on the contrast-boosted gray, blackhat is
        # naturally selective: large bright/dark regions of the stamp
        # body cancel out, so only sub-kernel-sized dark structures
        # survive. Threshold by intensity percentile so the cut adapts
        # per-image instead of relying on a hard-coded value.
        raw_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel_size = max(5, int(tophat_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(raw_gray, cv2.MORPH_BLACKHAT, kernel)
        threshold_value = int(np.percentile(blackhat, float(tophat_percentile)))
        _, mask = cv2.threshold(blackhat, max(threshold_value, 1), 255, cv2.THRESH_BINARY)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    else:
        _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    return img, gray, mask


def segment_paper_region(image, points, padding_pixels=20):
    """Estimate the stamp-sheet area as the minimum-area rotated bounding
    rectangle of detected hole centres.

    Earlier versions used ``cv2.convexHull``, but a single outlier hole
    (e.g. a stray detection in printed marginalia) drags the hull into
    a "tent" shape that extends well beyond the actual sheet — the
    grid_prior strategy then over-extends past the paper. ``minAreaRect``
    is robust to a small number of outliers because it computes the
    smallest rotated rectangle enclosing all points: a single outlier
    can stretch one side but cannot create a corner peak. The padding
    keeps a small slack for boundary holes that genuinely lie just
    outside the rectangle of detected centres.

    Returns a uint8 mask (255 inside paper, 0 outside). When too few
    points are available, returns a fully-True mask so the caller can
    treat ``paper_mask`` as a no-op.
    """
    height, width = image.shape[:2]
    if len(points) < 3:
        return np.full((height, width), 255, dtype=np.uint8)

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box_int = np.round(box).astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, box_int, 255)

    if padding_pixels > 0:
        kernel_size = max(3, padding_pixels * 2 + 1)
        dilate_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, dilate_struct)

    return mask


def is_inside_paper(paper_mask, x, y):
    """Return True if (x, y) is inside the paper region (or paper_mask is None)."""
    if paper_mask is None:
        return True
    height, width = paper_mask.shape[:2]
    ix = int(round(float(x)))
    iy = int(round(float(y)))
    if ix < 0 or iy < 0 or ix >= width or iy >= height:
        return False
    return bool(paper_mask[iy, ix])


def detect_hole_centers(mask, min_diameter=6, max_diameter=16, min_circularity=0.45):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holes = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        diameter = radius * 2
        if not (min_diameter < diameter < max_diameter):
            continue

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity <= min_circularity:
            continue

        holes.append(
            HoleDetection(
                index=len(holes),
                center=(int(x), int(y)),
                radius=float(radius),
                diameter=float(diameter),
                area=float(area),
                circularity=float(circularity),
                contour=cnt,
            )
        )

    return holes


def hole_points(holes):
    return [hole.center for hole in holes]


def compute_hole_statistics(holes):
    if not holes:
        return {}

    areas = np.array([hole.area for hole in holes], dtype=float)
    radii = np.array([hole.radius for hole in holes], dtype=float)
    circularities = np.array([hole.circularity for hole in holes], dtype=float)

    def median_and_mad(values):
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median, mad

    area_median, area_mad = median_and_mad(areas)
    radius_median, radius_mad = median_and_mad(radii)
    circularity_median, circularity_mad = median_and_mad(circularities)

    return {
        "count": len(holes),
        "area_median": area_median,
        "area_mad": area_mad,
        "radius_median": radius_median,
        "radius_mad": radius_mad,
        "circularity_median": circularity_median,
        "circularity_mad": circularity_mad,
    }


def build_candidate_edges(points, min_dist=15, max_dist=32):
    num_points = len(points)
    if num_points < 2:
        return []

    pts = np.asarray(points, dtype=float)
    i_arr, j_arr = np.triu_indices(num_points, k=1)
    dx_arr = pts[j_arr, 0] - pts[i_arr, 0]
    dy_arr = pts[j_arr, 1] - pts[i_arr, 1]
    dist_arr = np.hypot(dx_arr, dy_arr)
    sel = (dist_arr >= min_dist) & (dist_arr <= max_dist)

    if not np.any(sel):
        return []

    i_sel = i_arr[sel]
    j_sel = j_arr[sel]
    dx_sel = dx_arr[sel]
    dy_sel = dy_arr[sel]
    dist_sel = dist_arr[sel]
    angle_sel = np.degrees(np.arctan2(dy_sel, dx_sel)) % 180

    edges = [
        {
            "points": (int(i_sel[k]), int(j_sel[k])),
            "distance": float(dist_sel[k]),
            "dx": int(dx_sel[k]),
            "dy": int(dy_sel[k]),
            "angle": float(angle_sel[k]),
        }
        for k in range(len(i_sel))
    ]
    edges.sort(key=lambda x: x["distance"])
    return edges


def build_limited_adjacency(points, edges, degree_cap=5):
    adjacency_list = {i: [] for i in range(len(points))}

    for edge in edges:
        idx1, idx2 = edge["points"]
        if len(adjacency_list[idx1]) < degree_cap and len(adjacency_list[idx2]) < degree_cap:
            adjacency_list[idx1].append(idx2)
            adjacency_list[idx2].append(idx1)

    return adjacency_list


def classify_edges_by_orientation(edges, angle_tolerance_degrees=35):
    row_edges = []
    col_edges = []
    tan_tol = float(np.tan(np.deg2rad(angle_tolerance_degrees)))

    for edge in edges:
        dx = abs(edge["dx"])
        dy = abs(edge["dy"])

        if (dx > 0 and dy <= dx * tan_tol) or dy == 0:
            row_edges.append(edge)

        if (dy > 0 and dx <= dy * tan_tol) or dx == 0:
            col_edges.append(edge)

    return row_edges, col_edges


def connected_components(num_nodes, edges):
    graph = [[] for _ in range(num_nodes)]
    for edge in edges:
        idx1, idx2 = edge["points"]
        graph[idx1].append(idx2)
        graph[idx2].append(idx1)

    visited = [False] * num_nodes
    components = []

    for start in range(num_nodes):
        if visited[start] or not graph[start]:
            continue

        stack = [start]
        visited[start] = True
        component = []

        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        components.append(component)

    return components


def extract_line_chains(points, edges, orientation, min_line_points=12, min_span=80):
    components = connected_components(len(points), edges)
    chains = []

    for component in components:
        if len(component) < min_line_points:
            continue

        if orientation == "row":
            ordered = sorted(component, key=lambda idx: (points[idx][0], points[idx][1]))
            span = points[ordered[-1]][0] - points[ordered[0]][0]
        else:
            ordered = sorted(component, key=lambda idx: (points[idx][1], points[idx][0]))
            span = points[ordered[-1]][1] - points[ordered[0]][1]

        if span < min_span:
            continue

        chains.append(ordered)

    return chains


def point_indices_from_edges(edges):
    indices = set()
    for edge in edges:
        idx1, idx2 = edge["points"]
        indices.add(idx1)
        indices.add(idx2)

    return sorted(indices)

def directional_point_indices_from_edges(row_edges, col_edges, vote_margin=1):
    row_votes = {}
    col_votes = {}

    for edge in row_edges:
        idx1, idx2 = edge["points"]
        row_votes[idx1] = row_votes.get(idx1, 0) + 1
        row_votes[idx2] = row_votes.get(idx2, 0) + 1

    for edge in col_edges:
        idx1, idx2 = edge["points"]
        col_votes[idx1] = col_votes.get(idx1, 0) + 1
        col_votes[idx2] = col_votes.get(idx2, 0) + 1

    all_indices = set(row_votes) | set(col_votes)
    row_indices = []
    col_indices = []
    for idx in all_indices:
        row_count = row_votes.get(idx, 0)
        col_count = col_votes.get(idx, 0)
        if row_count > 0 and row_count >= col_count + vote_margin:
            row_indices.append(idx)
        if col_count > 0 and col_count >= row_count + vote_margin:
            col_indices.append(idx)

    return sorted(row_indices), sorted(col_indices)


def cluster_line_chains_by_axis(
    points,
    orientation,
    min_line_points=12,
    min_span=80,
    cluster_gap=80,
    candidate_indices=None,
):
    indices = list(candidate_indices) if candidate_indices is not None else list(range(len(points)))
    if not indices:
        return []

    axis_idx = 1 if orientation == "row" else 0
    span_idx = 0 if orientation == "row" else 1
    sorted_indices = sorted(indices, key=lambda idx: (points[idx][axis_idx], points[idx][span_idx]))

    groups = []
    current_group = [sorted_indices[0]]
    previous_coordinate = points[sorted_indices[0]][axis_idx]

    for idx in sorted_indices[1:]:
        coordinate = points[idx][axis_idx]
        if coordinate - previous_coordinate > cluster_gap:
            groups.append(current_group)
            current_group = [idx]
        else:
            current_group.append(idx)
        previous_coordinate = coordinate

    groups.append(current_group)

    chains = []
    for group in groups:
        if len(group) < min_line_points:
            continue

        ordered = sorted(group, key=lambda idx: (points[idx][span_idx], points[idx][axis_idx]))
        span_values = [points[idx][span_idx] for idx in ordered]
        span = max(span_values) - min(span_values)
        if span < min_span:
            continue

        chains.append(ordered)

    return chains

def robust_polyfit(parameter_values, target_values, degree=2, residual_threshold=4.0, iterations=200, seed=0):
    parameter_values = np.asarray(parameter_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    count = len(parameter_values)

    if count < 2:
        return None

    degree = min(int(degree), count - 1)
    min_sample = degree + 1
    if count < min_sample:
        return None

    rng = np.random.default_rng(seed)
    best_coeffs = None
    best_inliers = None
    best_score = None

    sample_iterations = 1 if count == min_sample else iterations
    for _ in range(sample_iterations):
        if count == min_sample:
            sample_idx = np.arange(count)
        else:
            sample_idx = rng.choice(count, size=min_sample, replace=False)

        if np.ptp(parameter_values[sample_idx]) < 1.0:
            continue

        try:
            coeffs = np.polyfit(parameter_values[sample_idx], target_values[sample_idx], degree)
        except np.linalg.LinAlgError:
            continue

        residuals = np.abs(target_values - np.polyval(coeffs, parameter_values))
        inliers = residuals <= residual_threshold
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < min_sample:
            continue

        inlier_residuals = residuals[inliers]
        score = (inlier_count, -float(np.median(inlier_residuals)), -float(np.max(inlier_residuals)))
        if best_score is None or score > best_score:
            best_score = score
            best_coeffs = coeffs
            best_inliers = inliers

    if best_coeffs is None or best_inliers is None:
        try:
            best_coeffs = np.polyfit(parameter_values, target_values, degree)
        except np.linalg.LinAlgError:
            return None
        best_inliers = np.ones(count, dtype=bool)
    elif int(np.count_nonzero(best_inliers)) >= min_sample:
        try:
            best_coeffs = np.polyfit(parameter_values[best_inliers], target_values[best_inliers], degree)
        except np.linalg.LinAlgError:
            pass

    residuals = np.abs(target_values - np.polyval(best_coeffs, parameter_values))
    median_residual = float(np.median(residuals))
    mad_residual = float(np.median(np.abs(residuals - median_residual)))
    scaled_mad = MAD_TO_SIGMA * mad_residual
    offset_threshold = max(3.0, median_residual + 3.0 * max(scaled_mad, 0.5))

    return best_coeffs, best_inliers, residuals, median_residual, mad_residual, offset_threshold


def fit_curve_models(points, chains, orientation, degree=2, residual_threshold=4.0):
    models = []

    for line_id, chain in enumerate(chains):
        if orientation == "row":
            parameter_values = [points[idx][0] for idx in chain]
            target_values = [points[idx][1] for idx in chain]
        else:
            parameter_values = [points[idx][1] for idx in chain]
            target_values = [points[idx][0] for idx in chain]

        fit = robust_polyfit(
            parameter_values,
            target_values,
            degree=degree,
            residual_threshold=residual_threshold,
            seed=line_id + (0 if orientation == "row" else 10_000),
        )
        if fit is None:
            continue

        coeffs, inlier_mask, residuals, median_residual, mad_residual, offset_threshold = fit
        parameter_values = np.asarray(parameter_values, dtype=float)

        models.append(
            CurveModel(
                line_id=line_id,
                orientation=orientation,
                ordered_indices=chain,
                coeffs=coeffs,
                parameter_min=float(np.min(parameter_values)),
                parameter_max=float(np.max(parameter_values)),
                residuals=residuals,
                inlier_mask=inlier_mask,
                median_residual=median_residual,
                mad_residual=mad_residual,
                offset_threshold=offset_threshold,
            )
        )

    return models


def sample_curve_points(model, samples=300, image_shape=None, extend_pixels=0):
    core_start = model.parameter_min
    core_end = model.parameter_max
    if image_shape is not None:
        height, width = image_shape[:2]
        axis_limit = width - 1 if model.orientation == "row" else height - 1
        sample_start = -float(extend_pixels)
        sample_end = float(axis_limit + extend_pixels)
    else:
        sample_start = core_start - float(extend_pixels)
        sample_end = core_end + float(extend_pixels)

    derivative_coeffs = model.derivative_coeffs
    start_target = float(np.polyval(model.coeffs, core_start))
    end_target = float(np.polyval(model.coeffs, core_end))
    start_slope = float(np.polyval(derivative_coeffs, core_start))
    end_slope = float(np.polyval(derivative_coeffs, core_end))

    parameter_parts = []
    target_parts = []

    if sample_start < core_start:
        left_samples = max(8, int(abs(core_start - sample_start) / 2))
        left_parameters = np.linspace(sample_start, core_start, left_samples, endpoint=False)
        left_targets = start_target + start_slope * (left_parameters - core_start)
        parameter_parts.append(left_parameters)
        target_parts.append(left_targets)

    visible_core_start = max(sample_start, core_start)
    visible_core_end = min(sample_end, core_end)
    if visible_core_end >= visible_core_start:
        core_samples = max(20, int(samples))
        core_parameters = np.linspace(visible_core_start, visible_core_end, core_samples)
        core_targets = np.polyval(model.coeffs, core_parameters)
        parameter_parts.append(core_parameters)
        target_parts.append(core_targets)

    if sample_end > core_end:
        right_samples = max(8, int(abs(sample_end - core_end) / 2))
        right_parameters = np.linspace(core_end, sample_end, right_samples + 1, endpoint=True)[1:]
        right_targets = end_target + end_slope * (right_parameters - core_end)
        parameter_parts.append(right_parameters)
        target_parts.append(right_targets)

    if not parameter_parts:
        return np.empty((0, 2), dtype=np.int32)

    parameters = np.concatenate(parameter_parts)
    targets = np.concatenate(target_parts)

    if model.orientation == "row":
        pts = np.column_stack([parameters, targets])
    else:
        pts = np.column_stack([targets, parameters])

    pts = pts[np.isfinite(pts).all(axis=1)]
    return np.round(pts).astype(np.int32)

def draw_curve_models(image, models, color, thickness=2, extend_pixels=0):
    for model in models:
        pts = sample_curve_points(model, image_shape=image.shape, extend_pixels=extend_pixels)
        if len(pts) >= 2:
            cv2.polylines(image, [pts.reshape((-1, 1, 2))], False, color, thickness)


def build_curve_band_mask(image_shape, models, expand_pixels=8, extend_pixels=80):
    height, width = image_shape[:2]
    curve_mask = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, int(expand_pixels) * 2 + 1)

    for model in models:
        pts = sample_curve_points(
            model,
            image_shape=image_shape,
            extend_pixels=extend_pixels,
        )
        if len(pts) >= 2:
            cv2.polylines(curve_mask, [pts.reshape((-1, 1, 2))], False, 255, thickness)

    return curve_mask


def draw_curve_mask_overlay(mask, row_curve_mask, col_curve_mask):
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    row_area = row_curve_mask > 0
    col_area = col_curve_mask > 0
    both_area = row_area & col_area

    overlay[row_area] = (0, 160, 0)
    overlay[col_area] = (180, 0, 0)
    overlay[both_area] = (0, 180, 180)

    return cv2.addWeighted(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.45, overlay, 0.55, 0)


def write_debug_image(path, image, small_max_side=1400):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

    height, width = image.shape[:2]
    scale = small_max_side / max(height, width)
    if scale < 1:
        small = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path.with_name(f"{path.stem}_small{path.suffix}")), small)


def collect_curve_outliers(points, models):
    outliers = []
    for model in models:
        for position, point_idx in enumerate(model.ordered_indices):
            residual = float(model.residuals[position])
            if residual > model.offset_threshold:
                outliers.append(
                    {
                        "point_idx": point_idx,
                        "line_id": model.line_id,
                        "orientation": model.orientation,
                        "residual": residual,
                        "threshold": float(model.offset_threshold),
                        "center": points[point_idx],
                    }
                )

    return outliers

def curve_parameter_for_point(model, point):
    return float(point[0] if model.orientation == "row" else point[1])


def evaluate_curve_point(model, parameter):
    parameter = float(parameter)
    core_start = model.parameter_min
    core_end = model.parameter_max
    derivative_coeffs = model.derivative_coeffs

    if parameter < core_start:
        start_target = float(np.polyval(model.coeffs, core_start))
        start_slope = float(np.polyval(derivative_coeffs, core_start))
        target = start_target + start_slope * (parameter - core_start)
    elif parameter > core_end:
        end_target = float(np.polyval(model.coeffs, core_end))
        end_slope = float(np.polyval(derivative_coeffs, core_end))
        target = end_target + end_slope * (parameter - core_end)
    else:
        target = float(np.polyval(model.coeffs, parameter))

    if model.orientation == "row":
        return float(parameter), float(target)
    return float(target), float(parameter)


def estimate_curve_pitch(points, model, min_pitch=10.0, max_pitch=40.0):
    parameters = np.array(
        [curve_parameter_for_point(model, points[idx]) for idx in model.ordered_indices],
        dtype=float,
    )
    parameters.sort()
    diffs = np.diff(parameters)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None

    valid = diffs[(diffs >= min_pitch) & (diffs <= max_pitch)]
    if len(valid) == 0:
        valid = diffs

    return float(np.median(valid))


def estimate_local_pitch_statistics(points, model, min_pitch=10.0, max_pitch=40.0):
    parameters = np.array(
        [curve_parameter_for_point(model, points[idx]) for idx in model.ordered_indices],
        dtype=float,
    )
    parameters.sort()
    diffs = np.diff(parameters)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None

    valid = diffs[(diffs >= min_pitch) & (diffs <= max_pitch)]
    if len(valid) == 0:
        valid = diffs

    median_pitch = float(np.median(valid))
    near_pitch = valid[(valid >= median_pitch * 0.75) & (valid <= median_pitch * 1.25)]
    if len(near_pitch) == 0:
        near_pitch = valid

    return {
        "pitch": float(np.mean(near_pitch)),
        "pitch_median": median_pitch,
        "pitch_std": float(np.std(near_pitch)),
        "pitch_sample_count": int(len(near_pitch)),
    }

def estimate_spacing_lattice(points, model, min_pitch=10.0, max_pitch=40.0):
    parameters = np.array(
        [curve_parameter_for_point(model, points[idx]) for idx in model.ordered_indices],
        dtype=float,
    )
    parameters.sort()
    if len(parameters) < 3:
        return None

    pitch = estimate_curve_pitch(points, model, min_pitch=min_pitch, max_pitch=max_pitch)
    if pitch is None or pitch <= 0:
        return None

    slots = [0]
    for gap in np.diff(parameters):
        step = max(1, int(round(float(gap) / pitch)))
        slots.append(slots[-1] + step)

    slot_values = np.array(slots, dtype=float)
    if len(np.unique(slot_values)) < 2:
        return None

    fitted_pitch, phase = np.polyfit(slot_values, parameters, 1)
    if fitted_pitch <= 0 or fitted_pitch < min_pitch or fitted_pitch > max_pitch:
        fitted_pitch = pitch
        phase = float(np.median(parameters - slot_values * fitted_pitch))

    fitted_parameters = phase + slot_values * fitted_pitch
    residuals = np.abs(parameters - fitted_parameters)

    return {
        "pitch": float(fitted_pitch),
        "phase": float(phase),
        "slot_min": int(np.min(slot_values)),
        "slot_max": int(np.max(slot_values)),
        "observed_count": int(len(parameters)),
        "lattice_residual_median": float(np.median(residuals)),
        "lattice_residual_max": float(np.max(residuals)),
    }

def score_expected_hole_roi(
    mask,
    center,
    expected_area,
    expected_radius,
    roi_radius,
    weak_area_ratio=0.55,
    weak_template_overlap=0.35,
    broken_area_ratio=0.20,
    broken_template_overlap=0.16,
):
    height, width = mask.shape[:2]
    x, y = center
    x1 = max(0, int(round(x - roi_radius)))
    x2 = min(width, int(round(x + roi_radius + 1)))
    y1 = max(0, int(round(y - roi_radius)))
    y2 = min(height, int(round(y + roi_radius + 1)))

    if x1 >= x2 or y1 >= y2:
        return None

    roi = mask[y1:y2, x1:x2]
    yy, xx = np.ogrid[y1:y2, x1:x2]
    disk = ((xx - x) ** 2 + (yy - y) ** 2) <= expected_radius**2
    disk_pixels = int(np.count_nonzero(disk))
    if disk_pixels == 0:
        return None

    evidence = (roi > 0) & disk
    observed_area = int(np.count_nonzero(evidence))
    area_ratio = float(observed_area / max(expected_area, 1.0))
    template_overlap = float(observed_area / disk_pixels)

    if observed_area > 0:
        ys, xs = np.nonzero(evidence)
        centroid_x = float(np.mean(xs + x1))
        centroid_y = float(np.mean(ys + y1))
        centroid_shift = float(np.hypot(centroid_x - x, centroid_y - y))

        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), 8)
        evidence_labels = labels[evidence]
        evidence_labels = evidence_labels[evidence_labels > 0]
        if len(evidence_labels) > 0:
            component_labels, component_counts = np.unique(evidence_labels, return_counts=True)
            component_label = int(component_labels[int(np.argmax(component_counts))])
            component_x = int(stats[component_label, cv2.CC_STAT_LEFT])
            component_y = int(stats[component_label, cv2.CC_STAT_TOP])
            component_width = int(stats[component_label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
            component_area = int(stats[component_label, cv2.CC_STAT_AREA])
        else:
            component_width = None
            component_height = None
            component_area = None
            component_x = None
            component_y = None
    else:
        centroid_x = None
        centroid_y = None
        centroid_shift = None
        component_width = None
        component_height = None
        component_area = None
        component_x = None
        component_y = None

    if component_width and component_height:
        component_aspect_ratio = float(max(component_width, component_height) / max(min(component_width, component_height), 1))
        component_roi_span_x = float(component_width / max(x2 - x1, 1))
        component_roi_span_y = float(component_height / max(y2 - y1, 1))
        component_fill_ratio = float(component_area / max(component_width * component_height, 1))
    else:
        component_aspect_ratio = None
        component_roi_span_x = None
        component_roi_span_y = None
        component_fill_ratio = None

    centered = centroid_shift is not None and centroid_shift <= expected_radius * 0.9
    loosely_centered = centroid_shift is not None and centroid_shift <= expected_radius * 1.35

    if area_ratio >= weak_area_ratio and template_overlap >= weak_template_overlap and centered:
        candidate_class = "WEAK"
    elif (area_ratio >= broken_area_ratio or template_overlap >= broken_template_overlap) and loosely_centered:
        candidate_class = "BROKEN"
    else:
        candidate_class = "MISSING"

    return {
        "class": candidate_class,
        "observed_area": observed_area,
        "expected_area": float(expected_area),
        "area_ratio": area_ratio,
        "template_overlap": template_overlap,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "centroid_shift": centroid_shift,
        "component_width": component_width,
        "component_height": component_height,
        "component_area": component_area,
        "component_aspect_ratio": component_aspect_ratio,
        "component_roi_span_x": component_roi_span_x,
        "component_roi_span_y": component_roi_span_y,
        "component_fill_ratio": component_fill_ratio,
    }


def build_hole_template(radius, feather=3.0):
    """Synthesize a circular hole template for cv2.matchTemplate.

    In the contrast-boosted grayscale produced by ``preprocess_image``,
    holes appear as DARK disks against a BRIGHT stamp body (the contrast
    boost stretches mid-tones such that punched-paper show-through ends
    up below the threshold while ink stays above). The template mirrors
    that polarity: dark disk on bright background. The feather edge
    improves NCC stability against minor radius/blur variation.

    Returns a uint8 image of odd size whose centre coincides with the
    disk centre.
    """
    radius = max(2.0, float(radius))
    feather = max(1.0, float(feather))
    size = int(np.ceil(radius * 2 + feather * 2 + 2))
    if size % 2 == 0:
        size += 1
    cy = cx = size // 2
    yy, xx = np.ogrid[:size, :size]
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    inside = np.clip(1.0 - (distance - radius) / feather, 0.0, 1.0)
    # Dark disk (0) on bright surround (255).
    return ((1.0 - inside) * 255).astype(np.uint8)


def score_expected_hole_template(
    gray,
    center,
    template,
    expected_radius,
    roi_radius,
    weak_ncc=0.55,
    broken_ncc=0.30,
):
    """Score a candidate position via normalized cross-correlation
    against a synthesized hole template.

    Returns a dict with the same key set as :func:`score_expected_hole_roi`
    so downstream filters and dict assemblers stay byte-compatible. NCC
    fields populate ``area_ratio`` and ``template_overlap`` so the legacy
    threshold-based logic and frame-line filter degrade gracefully.
    """
    height, width = gray.shape[:2]
    th, tw = template.shape[:2]
    cx_int = int(round(float(center[0])))
    cy_int = int(round(float(center[1])))

    # Search window must fit at least one template-sized stamp; expand
    # search radius around the candidate by roi_radius pixels.
    search_half = max(int(roi_radius), th // 2 + 2)
    x1 = max(0, cx_int - search_half)
    y1 = max(0, cy_int - search_half)
    x2 = min(width, cx_int + search_half + 1)
    y2 = min(height, cy_int + search_half + 1)
    window = gray[y1:y2, x1:x2]
    if window.shape[0] < th or window.shape[1] < tw:
        return None

    result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    ncc = float(max_val)

    # Center of best match in image coordinates
    best_x = float(x1 + max_loc[0] + tw / 2.0)
    best_y = float(y1 + max_loc[1] + th / 2.0)
    centroid_shift = float(np.hypot(best_x - float(center[0]), best_y - float(center[1])))

    if ncc >= weak_ncc and centroid_shift <= expected_radius * 0.9:
        candidate_class = "WEAK"
    elif ncc >= broken_ncc and centroid_shift <= expected_radius * 1.35:
        candidate_class = "BROKEN"
    else:
        candidate_class = "MISSING"

    # Mirror the ``score_expected_hole_roi`` schema so consumers that
    # spread ``**score`` into candidate dicts emit a stable key set
    # regardless of which scorer ran. Component-shape fields are not
    # meaningful for NCC-based scoring; expose them as ``None`` so
    # ``is_frame_line_like_score`` returns False (its early-return
    # guard rejects ``None`` shape inputs).
    return {
        "class": candidate_class,
        "observed_area": 0,
        "expected_area": 0.0,
        "area_ratio": ncc,
        "template_overlap": ncc,
        "centroid_x": best_x,
        "centroid_y": best_y,
        "centroid_shift": centroid_shift,
        "component_width": None,
        "component_height": None,
        "component_area": None,
        "component_aspect_ratio": None,
        "component_roi_span_x": None,
        "component_roi_span_y": None,
        "component_fill_ratio": None,
    }


def evaluate_candidate_position(
    mask,
    x,
    y,
    accepted_points,
    match_radius,
    expected_area,
    expected_radius,
    roi_radius,
    scorer="area",
    gray=None,
    template=None,
):
    """Run the shared NN-rejection + ROI scoring for a candidate position.

    Returns ``(score | None, nearest_distance)``. ``score`` is ``None`` if the
    candidate falls within ``match_radius`` of any accepted point or if the
    selected scorer rejects the ROI. Callers stay responsible for bounds
    checks, secondary gates (e.g. curve-mask, frame-line filters), and
    final dict construction so JSON key order is preserved per site.

    ``scorer`` selects between the legacy mask-area scorer (``"area"``,
    default, byte-equivalent to prior behaviour) and the NCC template
    scorer (``"template"``, requires ``gray`` and ``template`` kwargs).
    """
    distances = np.hypot(accepted_points[:, 0] - x, accepted_points[:, 1] - y)
    nearest_distance = float(np.min(distances))
    if nearest_distance <= match_radius:
        return None, nearest_distance

    if scorer == "template":
        score = score_expected_hole_template(
            gray,
            (x, y),
            template,
            expected_radius=expected_radius,
            roi_radius=roi_radius,
        )
    else:
        score = score_expected_hole_roi(
            mask,
            (x, y),
            expected_area=expected_area,
            expected_radius=expected_radius,
            roi_radius=roi_radius,
        )
    return score, nearest_distance


def is_frame_line_like_score(
    score,
    min_aspect_ratio=2.0,
    min_roi_span=0.82,
    min_block_major_span=0.95,
    min_block_minor_span=0.62,
    min_block_fill_ratio=0.85,
    min_block_template_overlap=0.85,
    min_block_area_ratio=1.0,
):
    aspect_ratio = score.get("component_aspect_ratio")
    span_x = score.get("component_roi_span_x")
    span_y = score.get("component_roi_span_y")
    fill_ratio = score.get("component_fill_ratio")
    template_overlap = score.get("template_overlap")
    area_ratio = score.get("area_ratio")
    if aspect_ratio is None or span_x is None or span_y is None:
        return False

    line_like = aspect_ratio >= min_aspect_ratio and (span_x >= min_roi_span or span_y >= min_roi_span)
    major_span = max(span_x, span_y)
    minor_span = min(span_x, span_y)
    block_like = (
        fill_ratio is not None
        and template_overlap is not None
        and area_ratio is not None
        and major_span >= min_block_major_span
        and minor_span >= min_block_minor_span
        and fill_ratio >= min_block_fill_ratio
        and template_overlap >= min_block_template_overlap
        and area_ratio >= min_block_area_ratio
    )

    return line_like or block_like


def _default_merge_source(candidate):
    return {
        "orientation": candidate["orientation"],
        "line_id": candidate["line_id"],
        "parameter": candidate["parameter"],
    }


def deduplicate_candidates(
    candidates,
    merge_radius,
    *,
    sources_key="merged_sources",
    source_extractor=_default_merge_source,
    sort_extras=None,
):
    class_rank = {"WEAK": 3, "BROKEN": 2, "MISSING": 1}

    def sort_key(item):
        base = (
            class_rank.get(item["class"], 0),
            item["area_ratio"],
            item["template_overlap"],
        )
        if sort_extras is None:
            return base
        return base + tuple(sort_extras(item))

    sorted_candidates = sorted(candidates, key=sort_key, reverse=True)

    kept = []
    for candidate in sorted_candidates:
        center = np.array(candidate["center"], dtype=float)
        duplicate = None
        for kept_candidate in kept:
            kept_center = np.array(kept_candidate["center"], dtype=float)
            if np.linalg.norm(center - kept_center) <= merge_radius:
                duplicate = kept_candidate
                break

        source = source_extractor(candidate)
        if duplicate is None:
            new_kept = dict(candidate)
            new_kept[sources_key] = [source]
            kept.append(new_kept)
        else:
            duplicate[sources_key].append(source)

    return kept


def find_expected_hole_candidates(
    mask,
    points,
    models,
    hole_stats,
    extend_steps=0,
    match_radius=8.0,
    roi_radius=14,
    gap_factor=1.55,
    paper_mask=None,
    scorer="area",
    gray=None,
    template=None,
    accepted_points_for_nn=None,
):
    if not points or not models:
        return []

    nn_source = accepted_points_for_nn if accepted_points_for_nn is not None else points
    accepted_points = np.array(nn_source, dtype=float) if len(nn_source) else np.array(points, dtype=float)
    height, width = mask.shape[:2]
    expected_area = float(hole_stats.get("area_median", 80.0) or 80.0)
    expected_radius = float(hole_stats.get("radius_median", 6.0) or 6.0)
    roi_radius = int(max(roi_radius, expected_radius * 2.0 + 2))

    candidates = []
    for model in models:
        pitch = estimate_curve_pitch(points, model)
        if pitch is None or pitch <= 0:
            continue
        assignment_threshold = max(match_radius, expected_radius * 1.25, model.offset_threshold)
        observed_parameters = []
        parameter_margin = pitch * max(1, extend_steps)
        for point in points:
            parameter = curve_parameter_for_point(model, point)
            if parameter < model.parameter_min - parameter_margin:
                continue
            if parameter > model.parameter_max + parameter_margin:
                continue

            expected_x, expected_y = evaluate_curve_point(model, parameter)
            if model.orientation == "row":
                residual = abs(point[1] - expected_y)
            else:
                residual = abs(point[0] - expected_x)

            if residual <= assignment_threshold:
                observed_parameters.append(parameter)

        observed_parameters = np.array(observed_parameters, dtype=float)
        observed_parameters.sort()
        if len(observed_parameters) == 0:
            continue

        expected_parameters = []
        for k in range(1, extend_steps + 1):
            expected_parameters.append(observed_parameters[0] - pitch * k)
            expected_parameters.append(observed_parameters[-1] + pitch * k)

        for left_parameter, right_parameter in zip(observed_parameters[:-1], observed_parameters[1:]):
            gap = right_parameter - left_parameter
            if gap < pitch * gap_factor:
                continue

            missing_count = max(1, int(round(gap / pitch)) - 1)
            local_spacing = gap / (missing_count + 1)
            for missing_idx in range(1, missing_count + 1):
                expected_parameters.append(left_parameter + local_spacing * missing_idx)

        for parameter in expected_parameters:
            x, y = evaluate_curve_point(model, parameter)
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            if not is_inside_paper(paper_mask, x, y):
                continue

            score, nearest_distance = evaluate_candidate_position(
                mask, x, y, accepted_points, match_radius,
                expected_area, expected_radius, roi_radius,
                scorer=scorer, gray=gray, template=template,
            )
            if score is None:
                continue

            candidates.append(
                {
                    "center": [float(x), float(y)],
                    "parameter": float(parameter),
                    "pitch": float(pitch),
                    "orientation": model.orientation,
                    "line_id": model.line_id,
                    "nearest_detected_distance": nearest_distance,
                    **score,
                }
            )

    return deduplicate_candidates(candidates, merge_radius=match_radius)


def find_spacing_inferred_hole_candidates(
    mask,
    points,
    models,
    hole_stats,
    match_radius=8.0,
    roi_radius=14,
    paper_mask=None,
    scorer="area",
    gray=None,
    template=None,
    accepted_points_for_nn=None,
):
    if not points or not models:
        return [], []

    nn_source = accepted_points_for_nn if accepted_points_for_nn is not None else points
    accepted_points = np.array(nn_source, dtype=float) if len(nn_source) else np.array(points, dtype=float)
    height, width = mask.shape[:2]
    expected_area = float(hole_stats.get("area_median", 80.0) or 80.0)
    expected_radius = float(hole_stats.get("radius_median", 6.0) or 6.0)
    roi_radius = int(max(roi_radius, expected_radius * 2.0 + 2))

    candidates = []
    lattices = []
    for model in models:
        lattice = estimate_spacing_lattice(points, model)
        if lattice is None:
            continue

        lattice_record = {
            "orientation": model.orientation,
            "line_id": model.line_id,
            **lattice,
        }
        lattices.append(lattice_record)

        pitch = lattice["pitch"]
        phase = lattice["phase"]
        for slot in range(lattice["slot_min"], lattice["slot_max"] + 1):
            parameter = phase + pitch * slot
            x, y = evaluate_curve_point(model, parameter)
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            if not is_inside_paper(paper_mask, x, y):
                continue

            score, nearest_distance = evaluate_candidate_position(
                mask, x, y, accepted_points, match_radius,
                expected_area, expected_radius, roi_radius,
                scorer=scorer, gray=gray, template=template,
            )
            if score is None:
                continue

            candidates.append(
                {
                    "center": [float(x), float(y)],
                    "parameter": float(parameter),
                    "pitch": float(pitch),
                    "slot": int(slot),
                    "source": "spacing_lattice",
                    "orientation": model.orientation,
                    "line_id": model.line_id,
                    "nearest_detected_distance": nearest_distance,
                    **score,
                }
            )

    return deduplicate_candidates(candidates, merge_radius=match_radius), lattices


def extend_lattice_to_paper(
    model,
    lattice,
    paper_mask,
    image_shape,
    max_extension_steps=120,
):
    """Walk outward from a lattice's observed slot range until the
    predicted position leaves the paper_mask (or the image bounds, when
    paper_mask is None). Returns ``(slot_min_extended, slot_max_extended)``.

    Used by the grid-prior strategy to predict hole positions BEYOND the
    detected hull. The walk stops the moment a lattice position falls
    outside the paper, so corner clipping is automatic.
    """
    height, width = image_shape[:2]
    pitch = lattice["pitch"]
    phase = lattice["phase"]
    slot_min = lattice["slot_min"]
    slot_max = lattice["slot_max"]

    def _on_paper(slot):
        x, y = evaluate_curve_point(model, phase + pitch * slot)
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        return is_inside_paper(paper_mask, x, y)

    # extend left
    extended_min = slot_min
    for _ in range(max_extension_steps):
        if not _on_paper(extended_min - 1):
            break
        extended_min -= 1
    # extend right
    extended_max = slot_max
    for _ in range(max_extension_steps):
        if not _on_paper(extended_max + 1):
            break
        extended_max += 1

    return extended_min, extended_max


def find_grid_prior_candidates(
    mask,
    points,
    models,
    hole_stats,
    paper_mask,
    image_shape,
    match_radius=8.0,
    roi_radius=14,
    scorer="area",
    gray=None,
    template=None,
    accepted_points_for_nn=None,
):
    """Predict hole positions from per-line spacing lattices, extended
    to the paper-mask boundary, and score each predicted position that
    has no nearby detected hole.

    This is the Phase 2D "prior-driven" strategy: instead of inferring
    candidates from gaps between detected holes (which can only fill
    interior misses), the lattice is treated as ground truth and every
    expected hole position is scored. Off-paper predictions are clipped
    by paper_mask, so the lattice never extrapolates into the scanner
    background.
    """
    if not points or not models:
        return [], []

    nn_source = accepted_points_for_nn if accepted_points_for_nn is not None else points
    accepted_points = np.array(nn_source, dtype=float) if len(nn_source) else np.array(points, dtype=float)
    expected_area = float(hole_stats.get("area_median", 80.0) or 80.0)
    expected_radius = float(hole_stats.get("radius_median", 6.0) or 6.0)
    roi_radius = int(max(roi_radius, expected_radius * 2.0 + 2))
    height, width = image_shape[:2]

    candidates = []
    lattices = []
    for model in models:
        lattice = estimate_spacing_lattice(points, model)
        if lattice is None:
            continue

        slot_min_ext, slot_max_ext = extend_lattice_to_paper(
            model, lattice, paper_mask, image_shape,
        )
        lattice_record = {
            "orientation": model.orientation,
            "line_id": model.line_id,
            **lattice,
            "slot_min_extended": int(slot_min_ext),
            "slot_max_extended": int(slot_max_ext),
            "extension_count": int(
                (lattice["slot_min"] - slot_min_ext) + (slot_max_ext - lattice["slot_max"])
            ),
        }
        lattices.append(lattice_record)

        pitch = lattice["pitch"]
        phase = lattice["phase"]
        for slot in range(slot_min_ext, slot_max_ext + 1):
            parameter = phase + pitch * slot
            x, y = evaluate_curve_point(model, parameter)
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            if not is_inside_paper(paper_mask, x, y):
                continue

            score, nearest_distance = evaluate_candidate_position(
                mask, x, y, accepted_points, match_radius,
                expected_area, expected_radius, roi_radius,
                scorer=scorer, gray=gray, template=template,
            )
            if score is None:
                continue

            candidates.append(
                {
                    "center": [float(x), float(y)],
                    "parameter": float(parameter),
                    "pitch": float(pitch),
                    "slot": int(slot),
                    "source": "grid_prior",
                    "orientation": model.orientation,
                    "line_id": model.line_id,
                    "nearest_detected_distance": nearest_distance,
                    **score,
                }
            )

    return deduplicate_candidates(candidates, merge_radius=match_radius), lattices


def annotate_grid_prior_consensus(candidates):
    """Tag each grid_prior candidate with cross-line support.

    Phase 2D v1 ran ``find_grid_prior_candidates`` over both row and
    col models, then deduped close candidates. The dedup helper records
    the originating line in ``merged_sources``: if a predicted position
    appears on BOTH a row lattice AND a col lattice, the merged
    candidate carries entries with both orientations. This function
    surfaces that signal as a top-level ``cross_supported`` flag and
    promotes the candidate's class one level (MISSING -> BROKEN ->
    WEAK) to reflect the stronger prior. The unpromoted class is kept
    in ``original_class`` for traceability.

    Mutates and returns the same list (a new dict per candidate).
    """
    annotated = []
    promote = {"MISSING": "BROKEN", "BROKEN": "WEAK", "WEAK": "WEAK"}
    for candidate in candidates:
        sources = candidate.get("merged_sources", [])
        orientations = {s.get("orientation") for s in sources if isinstance(s, dict)}
        cross_supported = "row" in orientations and "col" in orientations
        new_candidate = dict(candidate)
        new_candidate["cross_supported"] = bool(cross_supported)
        new_candidate["support_orientations"] = sorted(orientations)
        if cross_supported:
            new_candidate["original_class"] = candidate.get("class")
            new_candidate["class"] = promote.get(candidate.get("class"), candidate.get("class"))
        annotated.append(new_candidate)
    return annotated


def filter_points_by_band_mask(points, holes, row_models, col_models,
                                  image_shape, band_expand_pixels=12,
                                  paper_mask=None):
    """Build a perforation-band mask from fitted row/col curves and keep
    only the points (and matching holes) that fall inside the band.

    The user's domain insight: every real hole sits ON one of the
    perforation lines (row or col), so anything outside a thin strip
    centred on those lines is by definition NOT a perforation.
    Building the band mask requires curves to be already fitted from
    a first detection pass, so this is meant for a second-pass clean
    up: detect → fit → mask interior → re-filter.

    ``band_expand_pixels`` is the half-width of the strip in pixels;
    set it just larger than the hole radius so even slightly off-line
    detections survive (camera distortion, sub-pixel jitter), while
    text strokes inside stamp interiors are excluded.

    Returns ``(filtered_points, filtered_holes, kept_indices, band_mask)``.
    The kept_indices map back into the original ``points`` list so
    callers can keep references stable.
    """
    height, width = image_shape[:2]
    # Build row + col band mask using the existing helper.
    row_band = build_curve_band_mask(
        image_shape, row_models,
        expand_pixels=band_expand_pixels, extend_pixels=0,
    )
    col_band = build_curve_band_mask(
        image_shape, col_models,
        expand_pixels=band_expand_pixels, extend_pixels=0,
    )
    band_mask = cv2.bitwise_or(row_band, col_band)
    if paper_mask is not None:
        band_mask = cv2.bitwise_and(band_mask, paper_mask)

    kept_indices = []
    for idx, p in enumerate(points):
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < width and 0 <= y < height and band_mask[y, x]:
            kept_indices.append(idx)

    filtered_points = [points[i] for i in kept_indices]
    filtered_holes = [holes[i] for i in kept_indices]
    return filtered_points, filtered_holes, kept_indices, band_mask


def find_lattice_verified_real_holes(
    image,
    points,
    row_models,
    col_models,
    hole_stats,
    paper_mask,
    image_shape,
    hough_param1=80,
    hough_param2=12,
    hough_radius_slack=3,
    hough_min_dist_factor=1.5,
    fallback_pitch_factor=0.5,
    cross_line_dedup_factor=0.4,
    off_grid_radius_factor=1.5,
):
    """Verify lattice-predicted hole positions with cv2.HoughCircles.

    The user's domain insight (paraphrased): given the row/col counts,
    fit curves through the perforation grid (handling lens distortion
    via per-line polyfit), regress the slot pitch + phase from the
    middle (densely-correct) detections, predict every slot position,
    then **verify each predicted position with HoughCircles** — only
    positions where a real circular feature exists count as a true
    hole; positions where no circle is found are genuine defects.

    Hough is significantly more selective than threshold-based detection:
    it requires the local image gradient to form a closed circular
    boundary, so printed-text strokes (which are not circular) cannot
    impersonate holes the way they fool the area scorer. Conversely,
    weak holes that retain a faint circular edge still register with
    Hough even when their interior brightness is borderline.

    Returns a dict with:
      hough_circles_total      : int, all circles HoughCircles found
      slot_positions_total     : int, lattice slots after cross-line dedup
      verified                 : list of slot dicts where Hough confirmed
      fallback_recovered       : list of slots where Hough missed but a
                                 detected hole sits within slot tolerance
      true_defects             : list of slots with neither Hough hit nor
                                 detection nearby — the genuine misses
      off_grid_anomalies       : Hough circles that don't lie on any slot
                                 (mid-stamp text, illustration features,
                                 or genuine off-grid perforations)
    """
    raw_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    expected_radius = float(hole_stats.get("radius_median", 6.0) or 6.0)
    height, width = image_shape[:2]

    blurred = cv2.medianBlur(raw_gray, 5)
    hough = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=int(max(2, expected_radius * hough_min_dist_factor)),
        param1=int(hough_param1),
        param2=int(hough_param2),
        minRadius=max(2, int(expected_radius - hough_radius_slack)),
        maxRadius=int(expected_radius + hough_radius_slack),
    )
    hough_centers = hough[0, :, :2] if hough is not None else np.empty((0, 2), dtype=float)

    slot_positions = []
    refined_lattices = []
    for model in list(row_models) + list(col_models):
        lattice = estimate_spacing_lattice(points, model)
        if lattice is None:
            continue
        # Snap-to-slot refinement: re-fit pitch+phase using only the
        # densely-correct middle points (residual <= pitch * 0.2).
        # This excludes "fly-off" defective holes from biasing the fit.
        refined = refine_lattice_with_inliers(points, model, lattice)
        pitch = float(refined.get("refined_pitch", lattice["pitch"]))
        phase = float(refined.get("refined_phase", lattice["phase"]))
        refined_lattices.append({
            "orientation": model.orientation,
            "line_id": model.line_id,
            "original_pitch": float(lattice["pitch"]),
            "refined_pitch": pitch,
            "original_phase": float(lattice["phase"]),
            "refined_phase": phase,
            "inlier_count": refined.get("inlier_count", 0),
            "outlier_count": refined.get("outlier_count", 0),
            "residual_median": refined.get("residual_median", 0.0),
        })
        # Re-package for extend_lattice_to_paper which expects pitch/phase keys
        refined_lattice = {**lattice, "pitch": pitch, "phase": phase}
        slot_min, slot_max = extend_lattice_to_paper(model, refined_lattice, paper_mask, image_shape)
        for slot in range(slot_min, slot_max + 1):
            x, y = evaluate_curve_point(model, phase + pitch * slot)
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            if not is_inside_paper(paper_mask, x, y):
                continue
            slot_positions.append({
                "center": [float(x), float(y)],
                "pitch": pitch,
                "slot": int(slot),
                "orientation": model.orientation,
                "line_id": model.line_id,
            })

    deduped_slots = []
    for slot in slot_positions:
        x, y = slot["center"]
        is_dup = False
        for kept in deduped_slots:
            kx, ky = kept["center"]
            tol = max(slot["pitch"], kept["pitch"]) * cross_line_dedup_factor
            if (kx - x) ** 2 + (ky - y) ** 2 < tol * tol:
                is_dup = True
                break
        if not is_dup:
            deduped_slots.append(slot)

    accepted_points = np.array(points, dtype=float) if points else np.empty((0, 2), dtype=float)

    verified = []
    fallback = []
    defects = []
    for slot in deduped_slots:
        sx, sy = slot["center"]
        slot_tol = slot["pitch"] * fallback_pitch_factor

        hough_dist = float("inf")
        if len(hough_centers) > 0:
            d = np.hypot(hough_centers[:, 0] - sx, hough_centers[:, 1] - sy)
            hough_dist = float(d.min())

        if hough_dist <= slot_tol:
            verified.append({**slot, "hough_distance": hough_dist, "source": "hough"})
            continue

        det_dist = float("inf")
        if len(accepted_points) > 0:
            d = np.hypot(accepted_points[:, 0] - sx, accepted_points[:, 1] - sy)
            det_dist = float(d.min())

        if det_dist <= slot_tol:
            fallback.append({
                **slot,
                "hough_distance": hough_dist,
                "detection_distance": det_dist,
                "source": "fallback",
            })
            continue

        defects.append({
            **slot,
            "hough_distance": hough_dist,
            "detection_distance": det_dist,
        })

    off_grid = []
    if len(deduped_slots) > 0 and len(hough_centers) > 0:
        slot_xy = np.array([s["center"] for s in deduped_slots], dtype=float)
        for hx, hy in hough_centers:
            d = np.hypot(slot_xy[:, 0] - hx, slot_xy[:, 1] - hy)
            if float(d.min()) > expected_radius * off_grid_radius_factor:
                off_grid.append({"center": [float(hx), float(hy)]})

    return {
        "hough_circles_total": int(len(hough_centers)),
        "slot_positions_total": len(deduped_slots),
        "verified": verified,
        "fallback_recovered": fallback,
        "true_defects": defects,
        "off_grid_anomalies": off_grid,
        "refined_lattices": refined_lattices,
    }


def enforce_chain_count(chains, target_count):
    """If the user specified an expected line count and the auto-detected
    chains exceed it, keep only the largest ``target_count`` chains. Smaller
    chains in the over-segmented case are usually noise clusters that
    happened to pass min_line_points.

    Returns the (possibly trimmed) chain list. If ``target_count`` is None
    or already <= len(chains), the input is returned unchanged.
    """
    if target_count is None or target_count <= 0:
        return chains
    if len(chains) <= target_count:
        return chains
    return sorted(chains, key=len, reverse=True)[:target_count]


def cluster_with_target_count(
    points,
    orientation,
    candidate_indices,
    target_count,
    min_line_points,
    cluster_gap,
    min_span=80,
):
    """Cluster line chains, retrying with progressively looser parameters
    when the user-specified ``target_count`` is not yet reached.

    The single-shot ``cluster_line_chains_by_axis`` can over-segment
    (too many small chains) or under-segment (too few because the
    densest cells fall below ``min_line_points``). enforce_chain_count
    handles the over-segment case; this wrapper handles the under-segment
    case by relaxing ``min_line_points`` and ``cluster_gap`` in steps,
    stopping as soon as the chain count meets the target. Returns
    ``(chains, attempts)`` where attempts logs each retry's parameters
    and resulting count for diagnostic emission to curve_metrics.json.
    """
    attempts = []

    def _try(mlp, cg):
        chains = cluster_line_chains_by_axis(
            points,
            orientation,
            min_line_points=mlp,
            cluster_gap=cg,
            min_span=min_span,
            candidate_indices=candidate_indices,
        )
        attempts.append({
            "min_line_points": int(mlp),
            "cluster_gap": float(cg),
            "chain_count": len(chains),
        })
        return chains

    chains = _try(min_line_points, cluster_gap)

    if target_count is None or target_count <= 0:
        return chains, attempts
    if len(chains) >= target_count:
        return enforce_chain_count(chains, target_count), attempts

    # Under-segmented: relax min_line_points first (cheaper), then cluster_gap.
    relaxed_min = [
        max(3, int(min_line_points * 0.75)),
        max(3, int(min_line_points * 0.5)),
        max(3, int(min_line_points * 0.3)),
    ]
    for mlp in relaxed_min:
        if mlp >= min_line_points:
            continue
        chains = _try(mlp, cluster_gap)
        if len(chains) >= target_count:
            return enforce_chain_count(chains, target_count), attempts

    # Still short — try widening cluster_gap with the most relaxed min_line_points
    relaxed_gap = [cluster_gap * 1.5, cluster_gap * 2.0]
    final_min = max(3, int(min_line_points * 0.3))
    for cg in relaxed_gap:
        chains = _try(final_min, cg)
        if len(chains) >= target_count:
            return enforce_chain_count(chains, target_count), attempts

    # Give up and return the best-effort result
    return chains, attempts


def refine_lattice_with_inliers(points, model, lattice, residual_factor=0.2):
    """Refine a spacing lattice by snap-to-slot then re-fit on inliers.

    The user's domain insight (paraphrased): middle points have fixed
    spacing, only a few "fly off". The original ``estimate_spacing_lattice``
    uses median + np.polyfit on every chain point, so any flying-off
    outlier still pulls the fitted pitch slightly. This refinement does:

      1. Snap each chain point to the nearest slot using the initial
         (pitch, phase) from lattice.
      2. Compute residual for each point.
      3. Mark inliers as those with |residual| <= pitch * residual_factor.
      4. Re-fit pitch and phase using only inliers (np.polyfit).
      5. Recompute residuals against refined fit.

    Returns the lattice dict augmented with ``refined_pitch``,
    ``refined_phase``, ``inlier_count``, ``outlier_count``,
    ``residual_median`` (refined). When refinement fails (too few
    inliers, degenerate fit) the returned dict falls back to the
    original lattice values.
    """
    parameters = np.array(
        [curve_parameter_for_point(model, points[idx]) for idx in model.ordered_indices],
        dtype=float,
    )
    parameters.sort()
    pitch = float(lattice.get("pitch", 0.0))
    phase = float(lattice.get("phase", 0.0))

    if pitch <= 0 or len(parameters) < 3:
        return {**lattice, "refined_pitch": pitch, "refined_phase": phase,
                "inlier_count": int(len(parameters)), "outlier_count": 0,
                "residual_median": 0.0}

    slots = np.round((parameters - phase) / pitch).astype(float)
    residuals = parameters - (phase + slots * pitch)
    tolerance = pitch * residual_factor
    inlier_mask = np.abs(residuals) <= tolerance

    if int(np.sum(inlier_mask)) < 3:
        return {**lattice, "refined_pitch": pitch, "refined_phase": phase,
                "inlier_count": int(len(parameters)), "outlier_count": 0,
                "residual_median": float(np.median(np.abs(residuals)))}

    inlier_slots = slots[inlier_mask]
    inlier_params = parameters[inlier_mask]
    try:
        refined_pitch, refined_phase = np.polyfit(inlier_slots, inlier_params, 1)
    except np.linalg.LinAlgError:
        return {**lattice, "refined_pitch": pitch, "refined_phase": phase,
                "inlier_count": int(np.sum(inlier_mask)),
                "outlier_count": int(np.sum(~inlier_mask)),
                "residual_median": float(np.median(np.abs(residuals)))}

    if refined_pitch <= 0:
        return {**lattice, "refined_pitch": pitch, "refined_phase": phase,
                "inlier_count": int(np.sum(inlier_mask)),
                "outlier_count": int(np.sum(~inlier_mask)),
                "residual_median": float(np.median(np.abs(residuals)))}

    final_residuals = parameters - (refined_phase + slots * refined_pitch)
    final_inlier_mask = np.abs(final_residuals) <= tolerance

    return {
        **lattice,
        "refined_pitch": float(refined_pitch),
        "refined_phase": float(refined_phase),
        "inlier_count": int(np.sum(final_inlier_mask)),
        "outlier_count": int(np.sum(~final_inlier_mask)),
        "residual_median": float(np.median(np.abs(final_residuals))),
    }


def find_local_gap_candidates(
    mask,
    points,
    models,
    hole_stats,
    match_radius=8.0,
    roi_radius=14,
    gap_factor=1.55,
    integer_tolerance=0.35,
    max_slot_span=4,
    endpoint_extend_steps=2,
    paper_mask=None,
    scorer="area",
    gray=None,
    template=None,
    accepted_points_for_nn=None,
):
    if not points or not models:
        return [], []

    nn_source = accepted_points_for_nn if accepted_points_for_nn is not None else points
    accepted_points = np.array(nn_source, dtype=float) if len(nn_source) else np.array(points, dtype=float)
    height, width = mask.shape[:2]
    expected_area = float(hole_stats.get("area_median", 80.0) or 80.0)
    expected_radius = float(hole_stats.get("radius_median", 6.0) or 6.0)
    roi_radius = int(max(roi_radius, expected_radius * 2.0 + 2))

    candidates = []
    line_summaries = []
    for model in models:
        pitch_stats = estimate_local_pitch_statistics(points, model)
        if pitch_stats is None or pitch_stats["pitch"] <= 0:
            continue

        pitch = pitch_stats["pitch"]
        ordered = sorted(
            (
                (curve_parameter_for_point(model, points[idx]), idx)
                for idx in model.ordered_indices
            ),
            key=lambda item: item[0],
        )
        candidates_before_model = len(candidates)
        gap_count = 0
        def append_candidate(parameter, source, extra_fields, require_evidence=False):
            x, y = evaluate_curve_point(model, parameter)
            if x < 0 or y < 0 or x >= width or y >= height:
                return False, True
            if not is_inside_paper(paper_mask, x, y):
                return False, True

            score, nearest_distance = evaluate_candidate_position(
                mask, x, y, accepted_points, match_radius,
                expected_area, expected_radius, roi_radius,
                scorer=scorer, gray=gray, template=template,
            )
            if score is None:
                return False, nearest_distance > match_radius

            if require_evidence and score["class"] == "MISSING":
                return False, True
            if require_evidence and is_frame_line_like_score(score):
                return False, True

            candidates.append(
                {
                    "center": [float(x), float(y)],
                    "parameter": float(parameter),
                    "pitch": float(pitch),
                    "pitch_median": float(pitch_stats["pitch_median"]),
                    "source": source,
                    "orientation": model.orientation,
                    "line_id": model.line_id,
                    "nearest_detected_distance": nearest_distance,
                    **extra_fields,
                    **score,
                }
            )
            return True, False
        for (left_parameter, left_idx), (right_parameter, right_idx) in zip(ordered[:-1], ordered[1:]):
            gap = float(right_parameter - left_parameter)
            if gap < pitch * gap_factor:
                continue

            slot_span = int(round(gap / pitch))
            if slot_span <= 1 or slot_span > max_slot_span:
                continue

            gap_error = abs(gap / pitch - slot_span)
            if gap_error > integer_tolerance:
                continue

            gap_count += 1
            local_spacing = gap / slot_span
            gap_confidence = max(0.0, 1.0 - gap_error / max(integer_tolerance, 1e-6))
            for missing_slot in range(1, slot_span):
                parameter = left_parameter + local_spacing * missing_slot
                append_candidate(
                    parameter,
                    "local_gap",
                    {
                        "local_spacing": float(local_spacing),
                        "gap": float(gap),
                        "gap_factor": float(gap / pitch),
                        "slot_span": int(slot_span),
                        "missing_slot": int(missing_slot),
                        "gap_error": float(gap_error),
                        "gap_confidence": float(gap_confidence),
                        "left_point_idx": int(left_idx),
                        "right_point_idx": int(right_idx),
                        "left_center": [int(points[left_idx][0]), int(points[left_idx][1])],
                        "right_center": [int(points[right_idx][0]), int(points[right_idx][1])],
                    },
                )
        endpoint_count_before = len(candidates) - candidates_before_model
        if endpoint_extend_steps > 0 and len(ordered) >= 2:
            endpoints = [
                ("start", ordered[0], -1),
                ("end", ordered[-1], 1),
            ]
            for endpoint_side, (anchor_parameter, anchor_idx), direction in endpoints:
                for step in range(1, endpoint_extend_steps + 1):
                    parameter = anchor_parameter + direction * pitch * step
                    added, should_stop = append_candidate(
                        parameter,
                        "local_gap_endpoint",
                        {
                            "local_spacing": float(pitch),
                            "gap": float(pitch * step),
                            "gap_factor": float(step),
                            "slot_span": int(step + 1),
                            "missing_slot": int(step),
                            "gap_error": 0.0,
                            "gap_confidence": 0.55,
                            "endpoint_side": endpoint_side,
                            "endpoint_step": int(step),
                            "anchor_point_idx": int(anchor_idx),
                            "anchor_center": [int(points[anchor_idx][0]), int(points[anchor_idx][1])],
                        },
                        require_evidence=True,
                    )
                    if should_stop or not added:
                        break
        candidate_count = len(candidates) - candidates_before_model
        endpoint_candidate_count = candidate_count - endpoint_count_before

        line_summaries.append(
            {
                "orientation": model.orientation,
                "line_id": model.line_id,
                **pitch_stats,
                "gap_count": int(gap_count),
                "candidate_count": int(candidate_count),
                "endpoint_candidate_count": int(endpoint_candidate_count),
            }
        )

    return candidates, line_summaries


def compact_candidate_source(candidate):
    return {
        "orientation": candidate["orientation"],
        "line_id": candidate["line_id"],
        "parameter": candidate["parameter"],
        "pitch": candidate["pitch"],
        "gap": candidate["gap"],
        "slot_span": candidate["slot_span"],
        "missing_slot": candidate["missing_slot"],
        "gap_error": candidate["gap_error"],
        "gap_confidence": candidate["gap_confidence"],
        "class": candidate["class"],
        "area_ratio": candidate["area_ratio"],
        "template_overlap": candidate["template_overlap"],
        "center": candidate["center"],
    }


def merge_row_col_consensus_candidates(
    mask,
    points,
    row_candidates,
    col_candidates,
    hole_stats,
    consensus_radius=8.0,
    match_radius=8.0,
    roi_radius=14,
    paper_mask=None,
    scorer="area",
    gray=None,
    template=None,
    accepted_points_for_nn=None,
):
    if not row_candidates and not col_candidates:
        return [], []

    nn_source = accepted_points_for_nn if accepted_points_for_nn is not None else points
    accepted_points = np.array(nn_source, dtype=float) if len(nn_source) else np.array(points, dtype=float)
    expected_area = float(hole_stats.get("area_median", 80.0) or 80.0)
    expected_radius = float(hole_stats.get("radius_median", 6.0) or 6.0)
    roi_radius = int(max(roi_radius, expected_radius * 2.0 + 2))

    possible_pairs = []
    for row_idx, row_candidate in enumerate(row_candidates):
        row_center = np.array(row_candidate["center"], dtype=float)
        for col_idx, col_candidate in enumerate(col_candidates):
            col_center = np.array(col_candidate["center"], dtype=float)
            distance = float(np.linalg.norm(row_center - col_center))
            if distance <= consensus_radius:
                possible_pairs.append((distance, row_idx, col_idx))

    possible_pairs.sort(key=lambda item: item[0])
    used_rows = set()
    used_cols = set()
    consensus_candidates = []
    for consensus_distance, row_idx, col_idx in possible_pairs:
        if row_idx in used_rows or col_idx in used_cols:
            continue

        row_candidate = row_candidates[row_idx]
        col_candidate = col_candidates[col_idx]
        row_center = np.array(row_candidate["center"], dtype=float)
        col_center = np.array(col_candidate["center"], dtype=float)
        center = (row_center + col_center) / 2.0
        if not is_inside_paper(paper_mask, center[0], center[1]):
            continue
        score, nearest_distance = evaluate_candidate_position(
            mask, center[0], center[1], accepted_points, match_radius,
            expected_area, expected_radius, roi_radius,
            scorer=scorer, gray=gray, template=template,
        )
        if score is None:
            continue

        used_rows.add(row_idx)
        used_cols.add(col_idx)
        consensus_candidates.append(
            {
                "center": [float(center[0]), float(center[1])],
                "source": "row_col_local_gap_consensus",
                "support": "row_col",
                "support_count": 2,
                "consensus_distance": float(consensus_distance),
                "gap_confidence": float(
                    (row_candidate["gap_confidence"] + col_candidate["gap_confidence"]) / 2.0
                ),
                "nearest_detected_distance": nearest_distance,
                "row_candidate": compact_candidate_source(row_candidate),
                "col_candidate": compact_candidate_source(col_candidate),
                **score,
            }
        )

    debug_only_candidates = []
    for row_idx, row_candidate in enumerate(row_candidates):
        if row_idx in used_rows:
            continue
        debug_candidate = dict(row_candidate)
        debug_candidate["support"] = "row_only"
        debug_candidate["support_count"] = 1
        debug_only_candidates.append(debug_candidate)

    for col_idx, col_candidate in enumerate(col_candidates):
        if col_idx in used_cols:
            continue
        debug_candidate = dict(col_candidate)
        debug_candidate["support"] = "col_only"
        debug_candidate["support_count"] = 1
        debug_only_candidates.append(debug_candidate)

    return consensus_candidates, debug_only_candidates


def count_candidates_by_class(candidates):
    return {
        "WEAK": sum(1 for candidate in candidates if candidate["class"] == "WEAK"),
        "BROKEN": sum(1 for candidate in candidates if candidate["class"] == "BROKEN"),
        "MISSING": sum(1 for candidate in candidates if candidate["class"] == "MISSING"),
    }


def count_candidates_by_support(candidates):
    counts = {}
    for candidate in candidates:
        support = candidate.get("support", candidate.get("orientation", "unknown"))
        counts[support] = counts.get(support, 0) + 1
    return counts


def draw_row_col_consensus_candidates(mask, points, consensus_candidates, debug_only_candidates):
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(overlay, point, 2, (100, 100, 100), -1)

    for candidate in debug_only_candidates:
        x, y = candidate["center"]
        center = (int(round(x)), int(round(y)))
        color = (255, 80, 80) if candidate.get("support") == "row_only" else (80, 80, 255)
        cv2.circle(overlay, center, 8, color, 1)

    class_colors = {
        "WEAK": (0, 255, 255),
        "BROKEN": (0, 165, 255),
        "MISSING": (0, 0, 255),
    }
    for candidate in consensus_candidates:
        x, y = candidate["center"]
        center = (int(round(x)), int(round(y)))
        color = class_colors.get(candidate["class"], (255, 255, 255))
        cv2.circle(overlay, center, 14, color, 2)
        cv2.circle(overlay, center, 3, color, -1)
        cv2.putText(
            overlay,
            f"C:{candidate['class'][0]} d{candidate['consensus_distance']:.1f}",
            (center[0] + 10, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    return overlay
def draw_expected_hole_candidates(mask, points, candidates):
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(overlay, point, 2, (120, 120, 120), -1)

    colors = {
        "WEAK": (0, 255, 255),
        "BROKEN": (0, 165, 255),
        "MISSING": (0, 0, 255),
    }
    for candidate in candidates:
        x, y = candidate["center"]
        center = (int(round(x)), int(round(y)))
        color = colors.get(candidate["class"], (255, 255, 255))
        radius = 12 if candidate["class"] != "MISSING" else 8
        cv2.circle(overlay, center, radius, color, 2)
        cv2.putText(
            overlay,
            f"{candidate['class'][0]}:{candidate['area_ratio']:.2f}",
            (center[0] + 10, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    return overlay

def draw_repaired_circles(mask, holes, candidates, expected_radius):
    repaired_mask = mask.copy()
    base_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    expected_radius = max(2, int(round(expected_radius)))

    for hole in holes:
        cv2.circle(base_overlay, hole.center, max(2, int(round(hole.radius))), (120, 190, 190), 1)

    colors = {
        "WEAK": (0, 255, 255),
        "BROKEN": (0, 165, 255),
        "MISSING": (0, 0, 255),
    }
    filled_overlay = base_overlay.copy()
    for candidate in candidates:
        x, y = candidate["center"]
        center = (int(round(x)), int(round(y)))
        color = colors.get(candidate["class"], (255, 255, 255))
        cv2.circle(repaired_mask, center, expected_radius, 255, -1)
        cv2.circle(filled_overlay, center, expected_radius + 2, color, -1)

    overlay = cv2.addWeighted(filled_overlay, 0.62, base_overlay, 0.38, 0)
    for candidate in candidates:
        x, y = candidate["center"]
        center = (int(round(x)), int(round(y)))
        color = colors.get(candidate["class"], (255, 255, 255))
        cv2.circle(overlay, center, expected_radius + 2, (0, 0, 0), 3)
        cv2.circle(overlay, center, expected_radius + 2, color, 2)
        cv2.circle(overlay, center, 2, color, -1)

    return overlay, repaired_mask

def write_candidate_crops(mask, candidates, output_path, crop_radius=28, max_crops=24):
    non_missing = [c for c in candidates if c["class"] != "MISSING"]
    selected = sorted(
        non_missing,
        key=lambda item: (item["area_ratio"], item["template_overlap"]),
        reverse=True,
    )[:max_crops]
    if not selected:
        return

    height, width = mask.shape[:2]
    crops = []
    for candidate in selected:
        x, y = candidate["center"]
        cx = int(round(x))
        cy = int(round(y))
        x1 = max(0, cx - crop_radius)
        x2 = min(width, cx + crop_radius + 1)
        y1 = max(0, cy - crop_radius)
        y2 = min(height, cy + crop_radius + 1)
        crop = cv2.cvtColor(mask[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
        cv2.circle(crop, (cx - x1, cy - y1), 10, (0, 255, 255), 1)
        cv2.putText(
            crop,
            f"{candidate['class']} {candidate['area_ratio']:.2f}",
            (3, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
        )
        crops.append(cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST))

    columns = 4
    rows = []
    for i in range(0, len(crops), columns):
        row_crops = crops[i : i + columns]
        max_h = max(c.shape[0] for c in row_crops)
        padded = []
        for crop in row_crops:
            if crop.shape[0] < max_h:
                pad = np.zeros((max_h - crop.shape[0], crop.shape[1], 3), dtype=np.uint8)
                crop = np.vstack([crop, pad])
            padded.append(crop)
        rows.append(np.hstack(padded))

    max_w = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        padded_rows.append(row)

    contact_sheet = np.vstack(padded_rows)
    write_debug_image(output_path, contact_sheet, small_max_side=1600)


STRATEGY_ARTIFACT_PATHS = {
    "expected_hole": {
        "overlay": "expected_hole_candidates_overlay.jpg",
        "crops": "expected_hole_candidate_crops.jpg",
        "repaired_overlay": "repaired_circles_overlay.jpg",
        "repaired_mask": "repaired_circles_mask.jpg",
        "json": "expected_hole_candidates.json",
    },
    "spacing_inferred": {
        "overlay": "spacing_inferred_circles_overlay.jpg",
        "crops": "spacing_inferred_circle_crops.jpg",
        "repaired_overlay": "spacing_repaired_circles_overlay.jpg",
        "repaired_mask": "spacing_repaired_circles_mask.jpg",
        "json": "spacing_inferred_circles.json",
    },
    "local_gap": {
        "overlay": "local_gap_candidates_overlay.jpg",
        "crops": "local_gap_candidate_crops.jpg",
        "repaired_overlay": "local_gap_repaired_circles_overlay.jpg",
        "repaired_mask": "local_gap_repaired_circles_mask.jpg",
        "json": "local_gap_candidates.json",
    },
    "consensus": {
        "overlay": "row_col_consensus_overlay.jpg",
        "crops": "row_col_consensus_candidate_crops.jpg",
        "repaired_overlay": "consensus_repaired_circles_overlay.jpg",
        "repaired_mask": "consensus_repaired_circles_mask.jpg",
        "json": "row_col_consensus_candidates.json",
    },
    "secondary_origin": {
        "overlay": "secondary_origin_candidates_overlay.jpg",
        "crops": "secondary_origin_candidate_crops.jpg",
        "repaired_overlay": "secondary_origin_repaired_circles_overlay.jpg",
        "repaired_mask": "secondary_origin_repaired_circles_mask.jpg",
        "json": "secondary_origin_candidates.json",
    },
    "grid_prior": {
        "overlay": "grid_prior_candidates_overlay.jpg",
        "crops": "grid_prior_candidate_crops.jpg",
        "repaired_overlay": "grid_prior_repaired_circles_overlay.jpg",
        "repaired_mask": "grid_prior_repaired_circles_mask.jpg",
        "json": "grid_prior_candidates.json",
    },
}


def emit_strategy_artifacts(
    name,
    mask,
    holes,
    points,
    candidates,
    expected_radius,
    debug_path,
    json_payload,
    overlay_image=None,
):
    cfg = STRATEGY_ARTIFACT_PATHS[name]
    if overlay_image is None:
        overlay_image = draw_expected_hole_candidates(mask, points, candidates)
    write_debug_image(debug_path / cfg["overlay"], overlay_image)
    write_candidate_crops(mask, candidates, debug_path / cfg["crops"])
    repaired_overlay, repaired_mask_img = draw_repaired_circles(
        mask, holes, candidates, expected_radius=expected_radius,
    )
    write_debug_image(debug_path / cfg["repaired_overlay"], repaired_overlay)
    write_debug_image(debug_path / cfg["repaired_mask"], repaired_mask_img)
    with open(debug_path / cfg["json"], "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)


METRICS_BASELINE_SKIP_KEYS = frozenset({
    "image_path",
    "weak_match_radius",
    "weak_roi_radius",
    "weak_gap_factor",
    "weak_extend_steps",
    "local_gap_endpoint_steps",
    "curve_extend_pixels",
    "curve_extend_mode",
    "curve_mask_expand_pixels",
    "line_grouping",
    "line_cluster_gap",
    "line_direction_vote_margin",
    "secondary_origin_thresholds",
    "secondary_origin_blackhat_percentile",
    "secondary_origin_curve_extend_pixels",
    "paper_mask_mode",
    "threshold_mode",
    "adaptive_block_size",
    "adaptive_c",
})


def _diff_metrics_recursive(prefix, current, baseline, deltas):
    if isinstance(current, dict) and isinstance(baseline, dict):
        for key in sorted(set(current.keys()) | set(baseline.keys())):
            if not prefix and key in METRICS_BASELINE_SKIP_KEYS:
                continue
            sub_prefix = f"{prefix}.{key}" if prefix else key
            _diff_metrics_recursive(sub_prefix, current.get(key), baseline.get(key), deltas)
        return

    if isinstance(current, (int, float)) and isinstance(baseline, (int, float)):
        if current != baseline:
            deltas.append({
                "key": prefix,
                "baseline": baseline,
                "current": current,
                "delta": current - baseline,
            })
        return

    if current != baseline:
        deltas.append({
            "key": prefix,
            "baseline": baseline,
            "current": current,
        })


def compare_metrics_to_baseline(metrics, baseline_path, debug_path):
    """Diff current metrics against a saved baseline JSON. Writes a
    metrics_baseline_diff.json next to other debug artifacts and prints
    a human-readable summary. Returns the number of deltas detected."""
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        print(f"⚠ baseline not found: {baseline_path}")
        return -1

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    deltas = []
    _diff_metrics_recursive("", metrics, baseline, deltas)

    output = {
        "baseline_path": str(baseline_path),
        "delta_count": len(deltas),
        "deltas": deltas,
    }
    with open(debug_path / "metrics_baseline_diff.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if not deltas:
        print(f"✓ metrics match baseline {baseline_path}")
        return 0

    print(f"⚠ {len(deltas)} metric(s) differ from baseline {baseline_path}:")
    for delta in deltas[:15]:
        base = delta["baseline"]
        cur = delta["current"]
        if isinstance(cur, (int, float)) and isinstance(base, (int, float)):
            sign = "+" if delta["delta"] > 0 else ""
            print(f"  {delta['key']}: {base} → {cur} ({sign}{delta['delta']:g})")
        else:
            print(f"  {delta['key']}: {base!r} → {cur!r}")
    if len(deltas) > 15:
        print(f"  ... and {len(deltas) - 15} more (see metrics_baseline_diff.json)")
    return len(deltas)


def build_secondary_origin_masks(image, contrast_gray, thresholds=None, blackhat_percentile=None):
    masks = {}
    thresholds = thresholds if thresholds is not None else [144, 152]
    for threshold in thresholds:
        _, loose_mask = cv2.threshold(contrast_gray, int(threshold), 255, cv2.THRESH_BINARY_INV)
        masks[f"global_t{int(threshold)}"] = loose_mask

    if blackhat_percentile is not None:
        raw_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        blackhat = cv2.morphologyEx(raw_gray, cv2.MORPH_BLACKHAT, kernel)
        blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        threshold = int(np.percentile(blackhat, float(blackhat_percentile)))
        _, masks[f"blackhat21_p{float(blackhat_percentile):g}"] = cv2.threshold(
            blackhat,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )

    return masks


def find_secondary_origin_candidates(
    mask_variants,
    points,
    curve_mask,
    hole_stats,
    match_radius=8.0,
    roi_radius=14,
    paper_mask=None,
    scorer="area",
    gray=None,
    template=None,
    accepted_points_for_nn=None,
):
    if not mask_variants or not points:
        return []

    nn_source = accepted_points_for_nn if accepted_points_for_nn is not None else points
    accepted_points = np.array(nn_source, dtype=float) if len(nn_source) else np.array(points, dtype=float)
    expected_area = float(hole_stats.get("area_median", 80.0) or 80.0)
    expected_radius = float(hole_stats.get("radius_median", 6.0) or 6.0)
    roi_radius = int(max(roi_radius, expected_radius * 2.0 + 2))
    height, width = curve_mask.shape[:2]

    candidates = []
    for mask_source, candidate_mask in mask_variants.items():
        detected_holes = detect_hole_centers(candidate_mask)
        for hole in detected_holes:
            x, y = hole.center
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            if curve_mask[y, x] == 0:
                continue
            if not is_inside_paper(paper_mask, x, y):
                continue

            score, nearest_distance = evaluate_candidate_position(
                candidate_mask, x, y, accepted_points, match_radius,
                expected_area, expected_radius, roi_radius,
                scorer=scorer, gray=gray, template=template,
            )
            if score is None or score["class"] == "MISSING":
                continue
            if is_frame_line_like_score(score):
                continue

            candidates.append(
                {
                    "center": [float(x), float(y)],
                    "source": "secondary_origin",
                    "mask_source": mask_source,
                    "nearest_detected_distance": nearest_distance,
                    "detected_radius": float(hole.radius),
                    "detected_area": float(hole.area),
                    "detected_circularity": float(hole.circularity),
                    **score,
                }
            )

    return deduplicate_candidates(
        candidates,
        merge_radius=match_radius,
        sources_key="mask_sources",
        source_extractor=lambda c: c["mask_source"],
        sort_extras=lambda c: (c.get("detected_area", 0.0),),
    )


def analyze_curve_mode(
    image_path,
    output_path="result_curve.jpg",
    debug_dir="debug_curve",
    dist_threshold=32,
    poly_degree=2,
    min_line_points=12,
    line_cluster_gap=80,
    line_direction_vote_margin=1,
    curve_extend_pixels=80,
    curve_mask_expand_pixels=8,
    weak_extend_steps=0,
    weak_match_radius=8.0,
    weak_roi_radius=14,
    weak_gap_factor=1.55,
    local_gap_endpoint_steps=2,
    secondary_origin_thresholds=None,
    secondary_origin_blackhat_percentile=None,
    secondary_origin_curve_extend_pixels=0,
    paper_mask_mode="off",
    threshold_mode="global",
    adaptive_block_size=31,
    adaptive_c=5,
    tophat_kernel=21,
    tophat_percentile=98.0,
    metrics_baseline=None,
    scorer="area",
    grid_prior_mode="off",
    filter_orphan_holes_mode="off",
    row_lines=None,
    col_lines=None,
    hough_verify_mode="off",
    hough_param1=80,
    hough_param2=12,
    mask_stamp_interior_mode="off",
    band_expand_pixels=12,
):
    img, gray, mask = preprocess_image(
        image_path,
        threshold_mode=threshold_mode,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        tophat_kernel=tophat_kernel,
        tophat_percentile=tophat_percentile,
    )
    holes = detect_hole_centers(mask)
    points = hole_points(holes)
    hole_stats = compute_hole_statistics(holes)
    if scorer == "template":
        template_radius = float(hole_stats.get("radius_median", 6.0) or 6.0)
        hole_template = build_hole_template(template_radius)
    else:
        hole_template = None

    edges = build_candidate_edges(points, min_dist=15, max_dist=dist_threshold)
    row_edges, col_edges = classify_edges_by_orientation(edges)
    row_point_indices, col_point_indices = directional_point_indices_from_edges(
        row_edges,
        col_edges,
        vote_margin=line_direction_vote_margin,
    )
    if row_lines is not None or col_lines is not None:
        row_chains, row_chain_attempts = cluster_with_target_count(
            points, "row", row_point_indices,
            target_count=row_lines,
            min_line_points=min_line_points,
            cluster_gap=line_cluster_gap,
        )
        col_chains, col_chain_attempts = cluster_with_target_count(
            points, "col", col_point_indices,
            target_count=col_lines,
            min_line_points=min_line_points,
            cluster_gap=line_cluster_gap,
        )
    else:
        row_chains = cluster_line_chains_by_axis(
            points, "row",
            min_line_points=min_line_points,
            cluster_gap=line_cluster_gap,
            candidate_indices=row_point_indices,
        )
        col_chains = cluster_line_chains_by_axis(
            points, "col",
            min_line_points=min_line_points,
            cluster_gap=line_cluster_gap,
            candidate_indices=col_point_indices,
        )
        row_chain_attempts = []
        col_chain_attempts = []
    row_models = fit_curve_models(points, row_chains, "row", degree=poly_degree)
    col_models = fit_curve_models(points, col_chains, "col", degree=poly_degree)

    # Optional second pass: build a thin perforation-band mask from the
    # first-pass curves and drop every detected point that falls outside
    # the band. Stamp-interior content (illustrations, captions) cannot
    # sit on a perforation strip by definition, so this filter cleans
    # the point set without re-running expensive detection.
    band_filter_stats = None
    if mask_stamp_interior_mode == "on" and row_models and col_models:
        before_count = len(points)
        filtered_points, filtered_holes, kept_indices, band_mask_img = filter_points_by_band_mask(
            points, holes, row_models, col_models, mask.shape,
            band_expand_pixels=band_expand_pixels,
        )
        if len(filtered_points) >= max(50, len(row_models) + len(col_models)):
            points = filtered_points
            holes = filtered_holes
            # Refit curves on the cleaner point set so chains/lattices
            # downstream aren't anchored to noise positions.
            edges = build_candidate_edges(points, min_dist=15, max_dist=dist_threshold)
            row_edges, col_edges = classify_edges_by_orientation(edges)
            row_point_indices, col_point_indices = directional_point_indices_from_edges(
                row_edges, col_edges, vote_margin=line_direction_vote_margin,
            )
            if row_lines is not None or col_lines is not None:
                row_chains, _ = cluster_with_target_count(
                    points, "row", row_point_indices,
                    target_count=row_lines,
                    min_line_points=min_line_points, cluster_gap=line_cluster_gap,
                )
                col_chains, _ = cluster_with_target_count(
                    points, "col", col_point_indices,
                    target_count=col_lines,
                    min_line_points=min_line_points, cluster_gap=line_cluster_gap,
                )
            else:
                row_chains = cluster_line_chains_by_axis(
                    points, "row", min_line_points=min_line_points,
                    cluster_gap=line_cluster_gap, candidate_indices=row_point_indices,
                )
                col_chains = cluster_line_chains_by_axis(
                    points, "col", min_line_points=min_line_points,
                    cluster_gap=line_cluster_gap, candidate_indices=col_point_indices,
                )
            row_models = fit_curve_models(points, row_chains, "row", degree=poly_degree)
            col_models = fit_curve_models(points, col_chains, "col", degree=poly_degree)
            hole_stats = compute_hole_statistics(holes)
            band_filter_stats = {
                "before_count": int(before_count),
                "after_count": int(len(points)),
                "dropped": int(before_count - len(points)),
                "band_expand_pixels": int(band_expand_pixels),
            }

    chain_member_indices = set()
    for model in row_models:
        chain_member_indices.update(model.ordered_indices)
    for model in col_models:
        chain_member_indices.update(model.ordered_indices)
    chain_points = [points[idx] for idx in sorted(chain_member_indices)]

    if paper_mask_mode == "on":
        # Use chain-validated points only so noise detections in printed
        # marginalia cannot drag the bounding rectangle outward.
        paper_mask = segment_paper_region(img, chain_points or points)
    else:
        paper_mask = None

    # Optionally filter the accepted-points set used for nearest-neighbour
    # rejection during candidate finding. This drops orphan detections
    # (holes not on any row/col chain — usually printed text inside
    # stamps that black-top-hat picks up as small dark structures), so
    # real candidate positions near those orphans are not falsely
    # rejected as duplicates.
    if filter_orphan_holes_mode == "on" and chain_points:
        accepted_points_for_nn = chain_points
    else:
        accepted_points_for_nn = points

    row_outliers = collect_curve_outliers(points, row_models)
    col_outliers = collect_curve_outliers(points, col_models)
    curve_outliers = row_outliers + col_outliers

    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)

    write_debug_image(debug_path / "mask_raw.jpg", mask)

    row_curve_mask = build_curve_band_mask(
        mask.shape,
        row_models,
        expand_pixels=curve_mask_expand_pixels,
        extend_pixels=curve_extend_pixels,
    )
    col_curve_mask = build_curve_band_mask(
        mask.shape,
        col_models,
        expand_pixels=curve_mask_expand_pixels,
        extend_pixels=curve_extend_pixels,
    )
    combined_curve_mask = cv2.bitwise_or(row_curve_mask, col_curve_mask)
    write_debug_image(debug_path / "rows_curve_mask.jpg", row_curve_mask)
    write_debug_image(debug_path / "cols_curve_mask.jpg", col_curve_mask)
    write_debug_image(debug_path / "combined_curve_mask.jpg", combined_curve_mask)
    curve_mask_overlay = draw_curve_mask_overlay(mask, row_curve_mask, col_curve_mask)
    write_debug_image(debug_path / "curve_mask_overlay.jpg", curve_mask_overlay)

    if paper_mask is not None:
        write_debug_image(debug_path / "paper_mask.jpg", paper_mask)
        paper_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        paper_overlay[paper_mask == 0] = (0, 0, 200)
        write_debug_image(debug_path / "paper_mask_overlay.jpg", paper_overlay)

    if band_filter_stats is not None:
        write_debug_image(debug_path / "perforation_band_mask.jpg", band_mask_img)

    if threshold_mode == "adaptive":
        write_debug_image(debug_path / "adaptive_mask.jpg", mask)
    elif threshold_mode == "tophat":
        write_debug_image(debug_path / "tophat_mask.jpg", mask)

    secondary_origin_row_curve_mask = build_curve_band_mask(
        mask.shape,
        row_models,
        expand_pixels=curve_mask_expand_pixels,
        extend_pixels=secondary_origin_curve_extend_pixels,
    )
    secondary_origin_col_curve_mask = build_curve_band_mask(
        mask.shape,
        col_models,
        expand_pixels=curve_mask_expand_pixels,
        extend_pixels=secondary_origin_curve_extend_pixels,
    )
    secondary_origin_curve_mask = cv2.bitwise_or(
        secondary_origin_row_curve_mask,
        secondary_origin_col_curve_mask,
    )
    secondary_origin_masks = build_secondary_origin_masks(
        img,
        gray,
        thresholds=secondary_origin_thresholds,
        blackhat_percentile=secondary_origin_blackhat_percentile,
    )
    combined_secondary_origin_mask = np.zeros_like(mask)
    for mask_name, secondary_mask in secondary_origin_masks.items():
        combined_secondary_origin_mask = cv2.bitwise_or(combined_secondary_origin_mask, secondary_mask)
        write_debug_image(debug_path / f"secondary_origin_mask_{mask_name}.jpg", secondary_mask)
    if secondary_origin_masks:
        write_debug_image(debug_path / "secondary_origin_combined_mask.jpg", combined_secondary_origin_mask)
    secondary_origin_candidates = find_secondary_origin_candidates(
        secondary_origin_masks,
        points,
        secondary_origin_curve_mask,
        hole_stats,
        match_radius=weak_match_radius,
        roi_radius=weak_roi_radius,
        paper_mask=paper_mask,
        scorer=scorer, gray=gray, template=hole_template,
        accepted_points_for_nn=accepted_points_for_nn,
    )

    expected_hole_candidates = find_expected_hole_candidates(
        mask,
        points,
        row_models + col_models,
        hole_stats,
        extend_steps=weak_extend_steps,
        match_radius=weak_match_radius,
        roi_radius=weak_roi_radius,
        gap_factor=weak_gap_factor,
        paper_mask=paper_mask,
        scorer=scorer, gray=gray, template=hole_template,
        accepted_points_for_nn=accepted_points_for_nn,
    )
    spacing_inferred_candidates, spacing_lattices = find_spacing_inferred_hole_candidates(
        mask,
        points,
        row_models + col_models,
        hole_stats,
        match_radius=weak_match_radius,
        roi_radius=weak_roi_radius,
        paper_mask=paper_mask,
        scorer=scorer, gray=gray, template=hole_template,
        accepted_points_for_nn=accepted_points_for_nn,
    )
    local_gap_row_candidates, local_gap_row_lines = find_local_gap_candidates(
        mask,
        points,
        row_models,
        hole_stats,
        match_radius=weak_match_radius,
        roi_radius=weak_roi_radius,
        gap_factor=weak_gap_factor,
        endpoint_extend_steps=local_gap_endpoint_steps,
        paper_mask=paper_mask,
        scorer=scorer, gray=gray, template=hole_template,
        accepted_points_for_nn=accepted_points_for_nn,
    )
    local_gap_col_candidates, local_gap_col_lines = find_local_gap_candidates(
        mask,
        points,
        col_models,
        hole_stats,
        match_radius=weak_match_radius,
        roi_radius=weak_roi_radius,
        gap_factor=weak_gap_factor,
        endpoint_extend_steps=local_gap_endpoint_steps,
        paper_mask=paper_mask,
        scorer=scorer, gray=gray, template=hole_template,
        accepted_points_for_nn=accepted_points_for_nn,
    )
    local_gap_candidates = local_gap_row_candidates + local_gap_col_candidates
    consensus_candidates, consensus_debug_only_candidates = merge_row_col_consensus_candidates(
        mask,
        points,
        local_gap_row_candidates,
        local_gap_col_candidates,
        hole_stats,
        consensus_radius=weak_match_radius,
        match_radius=weak_match_radius,
        roi_radius=weak_roi_radius,
        paper_mask=paper_mask,
        scorer=scorer, gray=gray, template=hole_template,
        accepted_points_for_nn=accepted_points_for_nn,
    )
    if grid_prior_mode == "on":
        grid_prior_candidates, grid_prior_lattices = find_grid_prior_candidates(
            mask,
            points,
            row_models + col_models,
            hole_stats,
            paper_mask,
            mask.shape,
            match_radius=weak_match_radius,
            roi_radius=weak_roi_radius,
            scorer=scorer, gray=gray, template=hole_template,
        )
        grid_prior_candidates = annotate_grid_prior_consensus(grid_prior_candidates)
    else:
        grid_prior_candidates, grid_prior_lattices = [], []

    if hough_verify_mode == "on":
        hough_result = find_lattice_verified_real_holes(
            img,
            points,
            row_models,
            col_models,
            hole_stats,
            paper_mask,
            mask.shape,
            hough_param1=hough_param1,
            hough_param2=hough_param2,
        )
    else:
        hough_result = None
    weak_candidate_counts = {
        "WEAK": sum(1 for candidate in expected_hole_candidates if candidate["class"] == "WEAK"),
        "BROKEN": sum(1 for candidate in expected_hole_candidates if candidate["class"] == "BROKEN"),
        "MISSING": sum(1 for candidate in expected_hole_candidates if candidate["class"] == "MISSING"),
    }
    spacing_candidate_counts = {
        "WEAK": sum(1 for candidate in spacing_inferred_candidates if candidate["class"] == "WEAK"),
        "BROKEN": sum(1 for candidate in spacing_inferred_candidates if candidate["class"] == "BROKEN"),
        "MISSING": sum(1 for candidate in spacing_inferred_candidates if candidate["class"] == "MISSING"),
    }
    local_gap_candidate_counts = count_candidates_by_class(local_gap_candidates)
    local_gap_support_counts = count_candidates_by_support(local_gap_candidates)
    consensus_candidate_counts = count_candidates_by_class(consensus_candidates)
    consensus_debug_only_counts = count_candidates_by_support(consensus_debug_only_candidates)
    secondary_origin_candidate_counts = count_candidates_by_class(secondary_origin_candidates)
    secondary_origin_source_counts = {}
    for candidate in secondary_origin_candidates:
        for mask_source in candidate.get("mask_sources", [candidate.get("mask_source", "unknown")]):
            secondary_origin_source_counts[mask_source] = secondary_origin_source_counts.get(mask_source, 0) + 1
    expected_radius = hole_stats.get("radius_median", 6.0) or 6.0

    emit_strategy_artifacts(
        "expected_hole", mask, holes, points,
        expected_hole_candidates, expected_radius, debug_path,
        json_payload=expected_hole_candidates,
    )
    emit_strategy_artifacts(
        "spacing_inferred", mask, holes, points,
        spacing_inferred_candidates, expected_radius, debug_path,
        json_payload={
            "lattices": spacing_lattices,
            "candidates": spacing_inferred_candidates,
        },
    )
    emit_strategy_artifacts(
        "local_gap", mask, holes, points,
        local_gap_candidates, expected_radius, debug_path,
        json_payload={
            "row_lines": local_gap_row_lines,
            "col_lines": local_gap_col_lines,
            "row_candidates": local_gap_row_candidates,
            "col_candidates": local_gap_col_candidates,
        },
    )
    consensus_overlay = draw_row_col_consensus_candidates(
        mask,
        points,
        consensus_candidates,
        consensus_debug_only_candidates,
    )
    emit_strategy_artifacts(
        "consensus", mask, holes, points,
        consensus_candidates, expected_radius, debug_path,
        json_payload={
            "consensus_candidates": consensus_candidates,
            "debug_only_candidates": consensus_debug_only_candidates,
        },
        overlay_image=consensus_overlay,
    )
    emit_strategy_artifacts(
        "secondary_origin", mask, holes, points,
        secondary_origin_candidates, expected_radius, debug_path,
        json_payload={
            "masks": list(secondary_origin_masks.keys()),
            "source_counts": secondary_origin_source_counts,
            "candidates": secondary_origin_candidates,
        },
    )

    if grid_prior_mode == "on":
        emit_strategy_artifacts(
            "grid_prior", mask, holes, points,
            grid_prior_candidates, expected_radius, debug_path,
            json_payload={
                "lattices": grid_prior_lattices,
                "candidates": grid_prior_candidates,
            },
        )

    if hough_result is not None:
        hough_overlay = img.copy()
        radius_int = max(2, int(round(expected_radius)))
        for slot in hough_result["verified"]:
            cx, cy = int(round(slot["center"][0])), int(round(slot["center"][1]))
            cv2.circle(hough_overlay, (cx, cy), radius_int + 2, (0, 255, 0), 2)
        for slot in hough_result["fallback_recovered"]:
            cx, cy = int(round(slot["center"][0])), int(round(slot["center"][1]))
            cv2.circle(hough_overlay, (cx, cy), radius_int + 2, (0, 255, 255), 2)
        for slot in hough_result["true_defects"]:
            cx, cy = int(round(slot["center"][0])), int(round(slot["center"][1]))
            cv2.circle(hough_overlay, (cx, cy), radius_int + 3, (0, 0, 255), 3)
        for circle in hough_result["off_grid_anomalies"]:
            cx, cy = int(round(circle["center"][0])), int(round(circle["center"][1]))
            cv2.drawMarker(hough_overlay, (cx, cy), (255, 0, 255), cv2.MARKER_DIAMOND, 12, 1)
        write_debug_image(debug_path / "hough_verified_overlay.jpg", hough_overlay)

        height_, width_ = mask.shape[:2]
        hough_real_mask = np.zeros((height_, width_), dtype=np.uint8)
        for slot in hough_result["verified"]:
            cx, cy = int(round(slot["center"][0])), int(round(slot["center"][1]))
            cv2.circle(hough_real_mask, (cx, cy), radius_int, 255, -1)
        for slot in hough_result["fallback_recovered"]:
            cx, cy = int(round(slot["center"][0])), int(round(slot["center"][1]))
            cv2.circle(hough_real_mask, (cx, cy), radius_int, 255, -1)
        write_debug_image(debug_path / "hough_real_holes_mask.jpg", hough_real_mask)

        with open(debug_path / "hough_verified.json", "w", encoding="utf-8") as f:
            json.dump(hough_result, f, ensure_ascii=False, indent=2)

    centers_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Always render orphan detections (not on any chain) in red so the
    # source of any text-induced false positives is visible at a glance.
    orphan_indices = [i for i in range(len(points)) if i not in chain_member_indices]
    for idx in orphan_indices:
        cv2.circle(centers_overlay, points[idx], 3, (0, 0, 255), -1)
    for idx in sorted(chain_member_indices):
        cv2.circle(centers_overlay, points[idx], 3, (255, 255, 0), -1)
    write_debug_image(debug_path / "centers_overlay.jpg", centers_overlay)

    rows_overlay = centers_overlay.copy()
    draw_curve_models(rows_overlay, row_models, (0, 255, 0), thickness=2, extend_pixels=curve_extend_pixels)
    for outlier in row_outliers:
        cv2.circle(rows_overlay, outlier["center"], 13, (0, 165, 255), 2)
        cv2.putText(
            rows_overlay,
            f"{outlier['residual']:.1f}",
            (outlier["center"][0] + 12, outlier["center"][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
        )
    write_debug_image(debug_path / "rows_curve_overlay.jpg", rows_overlay)

    cols_overlay = centers_overlay.copy()
    draw_curve_models(cols_overlay, col_models, (255, 0, 0), thickness=2, extend_pixels=curve_extend_pixels)
    for outlier in col_outliers:
        cv2.circle(cols_overlay, outlier["center"], 13, (0, 165, 255), 2)
        cv2.putText(
            cols_overlay,
            f"{outlier['residual']:.1f}",
            (outlier["center"][0] + 12, outlier["center"][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
        )
    write_debug_image(debug_path / "cols_curve_overlay.jpg", cols_overlay)

    residual_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    draw_curve_models(residual_overlay, row_models, (0, 255, 0), thickness=2, extend_pixels=curve_extend_pixels)
    draw_curve_models(residual_overlay, col_models, (255, 0, 0), thickness=2, extend_pixels=curve_extend_pixels)
    for point in points:
        cv2.circle(residual_overlay, point, 2, (255, 255, 0), -1)
    for outlier in curve_outliers:
        color = (0, 165, 255) if outlier["orientation"] == "row" else (255, 0, 255)
        cv2.circle(residual_overlay, outlier["center"], 14, color, 2)
        cv2.putText(
            residual_overlay,
            f"{outlier['orientation'][0]}:{outlier['residual']:.1f}",
            (outlier["center"][0] + 12, outlier["center"][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    write_debug_image(debug_path / "curve_residual_overlay.jpg", residual_overlay)
    cv2.imwrite(output_path, residual_overlay)

    metrics = {
        "image_path": image_path,
        "hole_stats": hole_stats,
        "centers": len(points),
        "candidate_edges": len(edges),
        "row_edges": len(row_edges),
        "col_edges": len(col_edges),
        "row_grouping_points": len(row_point_indices),
        "col_grouping_points": len(col_point_indices),
        "row_chains": len(row_chains),
        "col_chains": len(col_chains),
        "line_grouping": "point_axis_cluster",
        "line_cluster_gap": line_cluster_gap,
        "line_direction_vote_margin": line_direction_vote_margin,
        "row_models": len(row_models),
        "col_models": len(col_models),
        "row_outliers": len(row_outliers),
        "col_outliers": len(col_outliers),
        "curve_outliers": len(curve_outliers),
        "curve_extend_pixels": curve_extend_pixels,
        "curve_extend_mode": "endpoint_tangent",
        "curve_mask_expand_pixels": curve_mask_expand_pixels,
        "row_curve_mask_pixels": int(np.count_nonzero(row_curve_mask)),
        "col_curve_mask_pixels": int(np.count_nonzero(col_curve_mask)),
        "combined_curve_mask_pixels": int(np.count_nonzero(combined_curve_mask)),
        "weak_extend_steps": weak_extend_steps,
        "weak_match_radius": weak_match_radius,
        "weak_roi_radius": weak_roi_radius,
        "weak_gap_factor": weak_gap_factor,
        "local_gap_endpoint_steps": local_gap_endpoint_steps,
        "secondary_origin_thresholds": secondary_origin_thresholds if secondary_origin_thresholds is not None else [144, 152],
        "secondary_origin_blackhat_percentile": secondary_origin_blackhat_percentile,
        "secondary_origin_curve_extend_pixels": secondary_origin_curve_extend_pixels,
        "expected_hole_candidates": len(expected_hole_candidates),
        "expected_hole_candidate_counts": weak_candidate_counts,
        "spacing_lattices": len(spacing_lattices),
        "spacing_inferred_candidates": len(spacing_inferred_candidates),
        "spacing_inferred_candidate_counts": spacing_candidate_counts,
        "local_gap_candidates": len(local_gap_candidates),
        "local_gap_candidate_counts": local_gap_candidate_counts,
        "local_gap_support_counts": local_gap_support_counts,
        "row_col_consensus_candidates": len(consensus_candidates),
        "row_col_consensus_candidate_counts": consensus_candidate_counts,
        "row_col_consensus_debug_only_candidates": len(consensus_debug_only_candidates),
        "row_col_consensus_debug_only_counts": consensus_debug_only_counts,
        "secondary_origin_candidates": len(secondary_origin_candidates),
        "secondary_origin_candidate_counts": secondary_origin_candidate_counts,
        "secondary_origin_source_counts": secondary_origin_source_counts,
        "repaired_circle_count": len(holes) + len(expected_hole_candidates),
        "spacing_repaired_circle_count": len(holes) + len(spacing_inferred_candidates),
        "local_gap_repaired_circle_count": len(holes) + len(local_gap_candidates),
        "consensus_repaired_circle_count": len(holes) + len(consensus_candidates),
        "secondary_origin_repaired_circle_count": len(holes) + len(secondary_origin_candidates),
        "row_model_residual_median": float(np.median([m.median_residual for m in row_models])) if row_models else None,
        "col_model_residual_median": float(np.median([m.median_residual for m in col_models])) if col_models else None,
    }

    if paper_mask is not None:
        metrics["paper_mask_mode"] = paper_mask_mode
        metrics["paper_mask_pixels"] = int(np.count_nonzero(paper_mask))

    if threshold_mode != "global":
        metrics["threshold_mode"] = threshold_mode
        if threshold_mode == "adaptive":
            metrics["adaptive_block_size"] = adaptive_block_size
            metrics["adaptive_c"] = adaptive_c
        elif threshold_mode == "tophat":
            metrics["tophat_kernel"] = tophat_kernel
            metrics["tophat_percentile"] = tophat_percentile

    if scorer != "area":
        metrics["scorer"] = scorer
        if hole_template is not None:
            metrics["template_size"] = int(hole_template.shape[0])

    if grid_prior_mode == "on":
        metrics["grid_prior_mode"] = grid_prior_mode
        metrics["grid_prior_candidates"] = len(grid_prior_candidates)
        metrics["grid_prior_candidate_counts"] = count_candidates_by_class(grid_prior_candidates)
        metrics["grid_prior_extension_total"] = int(sum(
            l.get("extension_count", 0) for l in grid_prior_lattices
        ))
        metrics["grid_prior_cross_supported"] = int(sum(
            1 for c in grid_prior_candidates if c.get("cross_supported")
        ))

    if hough_result is not None:
        metrics["hough_verify_mode"] = hough_verify_mode
        metrics["hough_param1"] = int(hough_param1)
        metrics["hough_param2"] = int(hough_param2)
        metrics["hough_circles_total"] = hough_result["hough_circles_total"]
        metrics["hough_slot_positions"] = hough_result["slot_positions_total"]
        metrics["hough_verified_real_holes"] = len(hough_result["verified"])
        metrics["hough_fallback_recovered"] = len(hough_result["fallback_recovered"])
        metrics["hough_true_defects"] = len(hough_result["true_defects"])
        metrics["hough_off_grid_anomalies"] = len(hough_result["off_grid_anomalies"])
        refined = hough_result.get("refined_lattices", [])
        if refined:
            metrics["lattice_refine_total_inliers"] = int(sum(l.get("inlier_count", 0) for l in refined))
            metrics["lattice_refine_total_outliers"] = int(sum(l.get("outlier_count", 0) for l in refined))
            pitch_shifts = [
                abs(l.get("refined_pitch", 0) - l.get("original_pitch", 0))
                for l in refined
            ]
            metrics["lattice_refine_pitch_shift_max"] = float(max(pitch_shifts)) if pitch_shifts else 0.0
            metrics["lattice_refine_pitch_shift_median"] = float(np.median(pitch_shifts)) if pitch_shifts else 0.0

    if row_lines is not None or col_lines is not None:
        metrics["row_lines_target"] = row_lines
        metrics["col_lines_target"] = col_lines
        metrics["row_chain_attempts"] = row_chain_attempts
        metrics["col_chain_attempts"] = col_chain_attempts

    if band_filter_stats is not None:
        metrics["mask_stamp_interior_mode"] = mask_stamp_interior_mode
        metrics["band_filter"] = band_filter_stats

    with open(debug_path / "curve_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if metrics_baseline is not None:
        compare_metrics_to_baseline(metrics, metrics_baseline, debug_path)

    print(
        "曲線模式完成："
        f"共找到 {len(points)} 個孔，"
        f"擬合 {len(row_models)} 條水平曲線、{len(col_models)} 條垂直曲線，"
        f"發現 {len(curve_outliers)} 個曲線殘差異常，"
        f"候選弱/破/缺孔 {weak_candidate_counts}，"
        f"spacing 推論候選 {spacing_candidate_counts}，"
        f"local gap 候選 {local_gap_candidate_counts}，"
        f"row/col 共識候選 {consensus_candidate_counts}，"
        f"secondary origin 候選 {secondary_origin_candidate_counts}。"
    )
    print(f"debug 輸出：{debug_path}")


def detect_and_link_limited_holes(
    image_path,
    output_path,
    dist_threshold=32,
    offset_threshold=3.0,
    threshold_mode="global",
    adaptive_block_size=31,
    adaptive_c=5,
    tophat_kernel=21,
    tophat_percentile=98.0,
):
    img, gray, mask = preprocess_image(
        image_path,
        threshold_mode=threshold_mode,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        tophat_kernel=tophat_kernel,
        tophat_percentile=tophat_percentile,
    )
    display_img = img.copy()
    bw_display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    holes = detect_hole_centers(mask)
    hole_centers = hole_points(holes)
    num_holes = len(hole_centers)

    possible_pairs = build_candidate_edges(hole_centers, min_dist=15, max_dist=dist_threshold)
    adjacency_list = build_limited_adjacency(hole_centers, possible_pairs, degree_cap=5)

    for pair in possible_pairs:
        idx1, idx2 = pair["points"]
        if idx2 in adjacency_list[idx1]:
            p1 = hole_centers[idx1]
            p2 = hole_centers[idx2]
            cv2.line(display_img, p1, p2, (0, 0, 255), 2)
            cv2.line(bw_display_img, p1, p2, (0, 0, 255), 2)

    defect_count = 0
    for i, neighbors in adjacency_list.items():
        if len(neighbors) == 2:
            p0 = hole_centers[i]
            p1 = hole_centers[neighbors[0]]
            p2 = hole_centers[neighbors[1]]

            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p0[0], p2[1] - p0[1])

            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = np.hypot(v1[0], v1[1])
            mag2 = np.hypot(v2[0], v2[1])

            if mag1 > 0 and mag2 > 0:
                cos_theta = dot_product / (mag1 * mag2)

                if cos_theta < -0.85:
                    num = abs(
                        (p0[0] - p1[0]) * (p2[1] - p1[1])
                        - (p0[1] - p1[1]) * (p2[0] - p1[0])
                    )
                    den = np.hypot(p2[0] - p1[0], p2[1] - p1[1])

                    if den > 0:
                        offset_dist = num / den
                        if offset_dist > offset_threshold:
                            defect_count += 1
                            cv2.circle(display_img, p0, 12, (0, 165, 255), 3)
                            cv2.circle(bw_display_img, p0, 12, (0, 165, 255), 3)
                            cv2.putText(
                                display_img,
                                f"NG:{offset_dist:.1f}",
                                (p0[0] + 15, p0[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 165, 255),
                                2,
                            )

    for center in hole_centers:
        cv2.circle(display_img, center, 4, (255, 255, 0), -1)
        cv2.circle(bw_display_img, center, 4, (255, 255, 0), -1)

    cv2.imwrite(output_path, display_img)
    base_name, ext = os.path.splitext(output_path)
    cv2.imwrite(f"{base_name}_bw{ext}", bw_display_img)

    print(f"檢測完成：共找到 {num_holes} 個孔，發現 {defect_count} 處異常偏移。")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", choices=["local", "curve"], default="local")
    parser.add_argument("--debug_dir", type=str, default="debug_curve")
    parser.add_argument("--dist_threshold", type=float, default=32)
    parser.add_argument("--offset_threshold", type=float, default=3.0)
    parser.add_argument("--poly_degree", type=int, default=2)
    parser.add_argument("--min_line_points", type=int, default=12)
    parser.add_argument("--line_cluster_gap", type=float, default=80)
    parser.add_argument("--line_direction_vote_margin", type=int, default=1)
    parser.add_argument("--curve_extend_pixels", type=int, default=80)
    parser.add_argument("--curve_mask_expand_pixels", type=int, default=8)
    parser.add_argument("--weak_extend_steps", type=int, default=0)
    parser.add_argument("--weak_match_radius", type=float, default=8.0)
    parser.add_argument("--weak_roi_radius", type=int, default=14)
    parser.add_argument("--weak_gap_factor", type=float, default=1.55)
    parser.add_argument("--local_gap_endpoint_steps", type=int, default=2)
    parser.add_argument("--secondary_origin_thresholds", type=int, nargs="*", default=[144, 152])
    parser.add_argument("--secondary_origin_blackhat_percentile", type=float, default=None)
    parser.add_argument("--secondary_origin_curve_extend_pixels", type=int, default=0)
    parser.add_argument("--paper_mask", choices=["on", "off"], default="off")
    parser.add_argument("--threshold_mode", choices=["global", "adaptive", "tophat"], default="global")
    parser.add_argument("--adaptive_block_size", type=int, default=31)
    parser.add_argument("--adaptive_c", type=int, default=5)
    parser.add_argument("--tophat_kernel", type=int, default=21,
                        help="Elliptical kernel size for black top-hat (forced odd)")
    parser.add_argument("--tophat_percentile", type=float, default=98.0,
                        help="Percentile of blackhat response used as threshold")
    parser.add_argument("--metrics_baseline", type=str, default=None,
                        help="Path to baseline curve_metrics.json for regression diff")
    parser.add_argument("--scorer", choices=["area", "template"], default="area",
                        help="Candidate scoring backend: area (legacy mask area) or template (NCC)")
    parser.add_argument("--grid_prior", choices=["on", "off"], default="off",
                        help="Predict hole positions from spacing lattice extended to paper boundary")
    parser.add_argument("--filter_orphan_holes", choices=["on", "off"], default="off",
                        help="Drop detections not on any row/col chain (e.g. printed-text noise) "
                             "from the nearest-neighbour rejection set")
    parser.add_argument("--row_lines", type=int, default=None,
                        help="Expected number of horizontal perforation lines; "
                             "if auto-detected chains exceed this, keep the largest N")
    parser.add_argument("--col_lines", type=int, default=None,
                        help="Expected number of vertical perforation lines")
    parser.add_argument("--hough_verify", choices=["on", "off"], default="off",
                        help="Verify lattice slot predictions with cv2.HoughCircles. "
                             "Outputs hough_verified_overlay.jpg + hough_real_holes_mask.jpg")
    parser.add_argument("--hough_param1", type=int, default=80,
                        help="HoughCircles param1 (Canny upper threshold)")
    parser.add_argument("--hough_param2", type=int, default=12,
                        help="HoughCircles param2 (accumulator threshold; lower = more circles)")
    parser.add_argument("--mask_stamp_interior", choices=["on", "off"], default="off",
                        help="Two-pass detection: build perforation-band mask from first-pass curves, "
                             "drop points outside the band (stamp interiors / illustrations)")
    parser.add_argument("--band_expand_pixels", type=int, default=12,
                        help="Half-width of the perforation strip in pixels")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "curve":
        output = args.output or "result_curve.jpg"
        analyze_curve_mode(
            args.input_img,
            output_path=output,
            debug_dir=args.debug_dir,
            dist_threshold=args.dist_threshold,
            poly_degree=args.poly_degree,
            min_line_points=args.min_line_points,
            line_cluster_gap=args.line_cluster_gap,
            line_direction_vote_margin=args.line_direction_vote_margin,
            curve_extend_pixels=args.curve_extend_pixels,
            curve_mask_expand_pixels=args.curve_mask_expand_pixels,
            weak_extend_steps=args.weak_extend_steps,
            weak_match_radius=args.weak_match_radius,
            weak_roi_radius=args.weak_roi_radius,
            weak_gap_factor=args.weak_gap_factor,
            local_gap_endpoint_steps=args.local_gap_endpoint_steps,
            secondary_origin_thresholds=args.secondary_origin_thresholds,
            secondary_origin_blackhat_percentile=args.secondary_origin_blackhat_percentile,
            secondary_origin_curve_extend_pixels=args.secondary_origin_curve_extend_pixels,
            paper_mask_mode=args.paper_mask,
            threshold_mode=args.threshold_mode,
            adaptive_block_size=args.adaptive_block_size,
            adaptive_c=args.adaptive_c,
            tophat_kernel=args.tophat_kernel,
            tophat_percentile=args.tophat_percentile,
            metrics_baseline=args.metrics_baseline,
            scorer=args.scorer,
            grid_prior_mode=args.grid_prior,
            filter_orphan_holes_mode=args.filter_orphan_holes,
            row_lines=args.row_lines,
            col_lines=args.col_lines,
            hough_verify_mode=args.hough_verify,
            hough_param1=args.hough_param1,
            hough_param2=args.hough_param2,
            mask_stamp_interior_mode=args.mask_stamp_interior,
            band_expand_pixels=args.band_expand_pixels,
        )
    else:
        output = args.output or "result_defect.jpg"
        detect_and_link_limited_holes(
            args.input_img,
            output,
            dist_threshold=args.dist_threshold,
            offset_threshold=args.offset_threshold,
            threshold_mode=args.threshold_mode,
            adaptive_block_size=args.adaptive_block_size,
            adaptive_c=args.adaptive_c,
            tophat_kernel=args.tophat_kernel,
            tophat_percentile=args.tophat_percentile,
        )