import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    contrast_img = np.clip(img * (80 / 127 + 1) - 80, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return img, gray, mask


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
    edges = []
    num_points = len(points)
    for i in range(num_points):
        p1 = points[i]
        for j in range(i + 1, num_points):
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = float(np.hypot(dx, dy))
            if min_dist <= dist <= max_dist:
                angle = float(np.degrees(np.arctan2(dy, dx)) % 180)
                edges.append(
                    {
                        "points": (i, j),
                        "distance": dist,
                        "dx": dx,
                        "dy": dy,
                        "angle": angle,
                    }
                )

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
    scaled_mad = 1.4826 * mad_residual
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

    derivative_coeffs = np.polyder(model.coeffs)
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
    derivative_coeffs = np.polyder(model.coeffs)

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

def score_expected_hole_roi(mask, center, expected_area, expected_radius, roi_radius):
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

    if area_ratio >= 0.55 and template_overlap >= 0.35 and centered:
        candidate_class = "WEAK"
    elif (area_ratio >= 0.20 or template_overlap >= 0.16) and loosely_centered:
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


def deduplicate_candidates(candidates, merge_radius):
    class_rank = {"WEAK": 3, "BROKEN": 2, "MISSING": 1}
    candidates = sorted(
        candidates,
        key=lambda item: (
            class_rank.get(item["class"], 0),
            item["area_ratio"],
            item["template_overlap"],
        ),
        reverse=True,
    )

    kept = []
    for candidate in candidates:
        center = np.array(candidate["center"], dtype=float)
        is_duplicate = False
        for kept_candidate in kept:
            kept_center = np.array(kept_candidate["center"], dtype=float)
            if np.linalg.norm(center - kept_center) <= merge_radius:
                kept_candidate.setdefault("merged_sources", []).append(
                    {
                        "orientation": candidate["orientation"],
                        "line_id": candidate["line_id"],
                        "parameter": candidate["parameter"],
                    }
                )
                is_duplicate = True
                break

        if not is_duplicate:
            candidate["merged_sources"] = [
                {
                    "orientation": candidate["orientation"],
                    "line_id": candidate["line_id"],
                    "parameter": candidate["parameter"],
                }
            ]
            kept.append(candidate)

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
):
    if not points or not models:
        return []

    accepted_points = np.array(points, dtype=float)
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

            distances = np.hypot(accepted_points[:, 0] - x, accepted_points[:, 1] - y)
            nearest_distance = float(np.min(distances))
            if nearest_distance <= match_radius:
                continue

            score = score_expected_hole_roi(
                mask,
                (x, y),
                expected_area=expected_area,
                expected_radius=expected_radius,
                roi_radius=roi_radius,
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
):
    if not points or not models:
        return [], []

    accepted_points = np.array(points, dtype=float)
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

            distances = np.hypot(accepted_points[:, 0] - x, accepted_points[:, 1] - y)
            nearest_distance = float(np.min(distances))
            if nearest_distance <= match_radius:
                continue

            score = score_expected_hole_roi(
                mask,
                (x, y),
                expected_area=expected_area,
                expected_radius=expected_radius,
                roi_radius=roi_radius,
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
):
    if not points or not models:
        return [], []

    accepted_points = np.array(points, dtype=float)
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
        gap_count = 0
        candidate_count = 0
        def append_candidate(parameter, source, extra_fields, require_evidence=False):
            nonlocal candidate_count
            x, y = evaluate_curve_point(model, parameter)
            if x < 0 or y < 0 or x >= width or y >= height:
                return False, True

            distances = np.hypot(accepted_points[:, 0] - x, accepted_points[:, 1] - y)
            nearest_distance = float(np.min(distances))
            if nearest_distance <= match_radius:
                return False, False

            score = score_expected_hole_roi(
                mask,
                (x, y),
                expected_area=expected_area,
                expected_radius=expected_radius,
                roi_radius=roi_radius,
            )
            if score is None:
                return False, True

            if require_evidence and score["class"] == "MISSING":
                return False, True
            if require_evidence and is_frame_line_like_score(score):
                return False, True

            candidate_count += 1
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
        endpoint_count_before = candidate_count
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
):
    if not row_candidates and not col_candidates:
        return [], []

    accepted_points = np.array(points, dtype=float)
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
        distances = np.hypot(accepted_points[:, 0] - center[0], accepted_points[:, 1] - center[1])
        nearest_distance = float(np.min(distances))
        if nearest_distance <= match_radius:
            continue

        score = score_expected_hole_roi(
            mask,
            (center[0], center[1]),
            expected_area=expected_area,
            expected_radius=expected_radius,
            roi_radius=roi_radius,
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


def deduplicate_secondary_origin_candidates(candidates, merge_radius):
    class_rank = {"WEAK": 3, "BROKEN": 2, "MISSING": 1}
    candidates = sorted(
        candidates,
        key=lambda item: (
            class_rank.get(item["class"], 0),
            item["area_ratio"],
            item["template_overlap"],
            item.get("detected_area", 0.0),
        ),
        reverse=True,
    )

    kept = []
    for candidate in candidates:
        center = np.array(candidate["center"], dtype=float)
        duplicate = None
        for kept_candidate in kept:
            kept_center = np.array(kept_candidate["center"], dtype=float)
            if np.linalg.norm(center - kept_center) <= merge_radius:
                duplicate = kept_candidate
                break

        if duplicate is None:
            candidate["mask_sources"] = [candidate["mask_source"]]
            kept.append(candidate)
        else:
            duplicate.setdefault("mask_sources", []).append(candidate["mask_source"])

    return kept


def find_secondary_origin_candidates(
    mask_variants,
    points,
    curve_mask,
    hole_stats,
    match_radius=8.0,
    roi_radius=14,
):
    if not mask_variants or not points:
        return []

    accepted_points = np.array(points, dtype=float)
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

            distances = np.hypot(accepted_points[:, 0] - x, accepted_points[:, 1] - y)
            nearest_distance = float(np.min(distances))
            if nearest_distance <= match_radius:
                continue

            score = score_expected_hole_roi(
                candidate_mask,
                hole.center,
                expected_area=expected_area,
                expected_radius=expected_radius,
                roi_radius=roi_radius,
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

    return deduplicate_secondary_origin_candidates(candidates, merge_radius=match_radius)


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
):
    img, gray, mask = preprocess_image(image_path)
    holes = detect_hole_centers(mask)
    points = hole_points(holes)
    hole_stats = compute_hole_statistics(holes)

    edges = build_candidate_edges(points, min_dist=15, max_dist=dist_threshold)
    row_edges, col_edges = classify_edges_by_orientation(edges)
    row_point_indices, col_point_indices = directional_point_indices_from_edges(
        row_edges,
        col_edges,
        vote_margin=line_direction_vote_margin,
    )
    row_chains = cluster_line_chains_by_axis(
        points,
        "row",
        min_line_points=min_line_points,
        cluster_gap=line_cluster_gap,
        candidate_indices=row_point_indices,
    )
    col_chains = cluster_line_chains_by_axis(
        points,
        "col",
        min_line_points=min_line_points,
        cluster_gap=line_cluster_gap,
        candidate_indices=col_point_indices,
    )
    row_models = fit_curve_models(points, row_chains, "row", degree=poly_degree)
    col_models = fit_curve_models(points, col_chains, "col", degree=poly_degree)

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
    )
    spacing_inferred_candidates, spacing_lattices = find_spacing_inferred_hole_candidates(
        mask,
        points,
        row_models + col_models,
        hole_stats,
        match_radius=weak_match_radius,
        roi_radius=weak_roi_radius,
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
    )
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
    weak_candidates_overlay = draw_expected_hole_candidates(mask, points, expected_hole_candidates)
    write_debug_image(debug_path / "expected_hole_candidates_overlay.jpg", weak_candidates_overlay)
    write_candidate_crops(mask, expected_hole_candidates, debug_path / "expected_hole_candidate_crops.jpg")
    repaired_circles_overlay, repaired_circles_mask = draw_repaired_circles(
        mask,
        holes,
        expected_hole_candidates,
        expected_radius=hole_stats.get("radius_median", 6.0) or 6.0,
    )
    write_debug_image(debug_path / "repaired_circles_overlay.jpg", repaired_circles_overlay)
    write_debug_image(debug_path / "repaired_circles_mask.jpg", repaired_circles_mask)
    with open(debug_path / "expected_hole_candidates.json", "w", encoding="utf-8") as f:
        json.dump(expected_hole_candidates, f, ensure_ascii=False, indent=2)

    spacing_candidates_overlay = draw_expected_hole_candidates(mask, points, spacing_inferred_candidates)
    write_debug_image(debug_path / "spacing_inferred_circles_overlay.jpg", spacing_candidates_overlay)
    write_candidate_crops(mask, spacing_inferred_candidates, debug_path / "spacing_inferred_circle_crops.jpg")
    spacing_repaired_overlay, spacing_repaired_mask = draw_repaired_circles(
        mask,
        holes,
        spacing_inferred_candidates,
        expected_radius=hole_stats.get("radius_median", 6.0) or 6.0,
    )
    write_debug_image(debug_path / "spacing_repaired_circles_overlay.jpg", spacing_repaired_overlay)
    write_debug_image(debug_path / "spacing_repaired_circles_mask.jpg", spacing_repaired_mask)
    with open(debug_path / "spacing_inferred_circles.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "lattices": spacing_lattices,
                "candidates": spacing_inferred_candidates,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    local_gap_overlay = draw_expected_hole_candidates(mask, points, local_gap_candidates)
    write_debug_image(debug_path / "local_gap_candidates_overlay.jpg", local_gap_overlay)
    write_candidate_crops(mask, local_gap_candidates, debug_path / "local_gap_candidate_crops.jpg")
    local_gap_repaired_overlay, local_gap_repaired_mask = draw_repaired_circles(
        mask,
        holes,
        local_gap_candidates,
        expected_radius=hole_stats.get("radius_median", 6.0) or 6.0,
    )
    write_debug_image(debug_path / "local_gap_repaired_circles_overlay.jpg", local_gap_repaired_overlay)
    write_debug_image(debug_path / "local_gap_repaired_circles_mask.jpg", local_gap_repaired_mask)
    with open(debug_path / "local_gap_candidates.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "row_lines": local_gap_row_lines,
                "col_lines": local_gap_col_lines,
                "row_candidates": local_gap_row_candidates,
                "col_candidates": local_gap_col_candidates,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    consensus_overlay = draw_row_col_consensus_candidates(
        mask,
        points,
        consensus_candidates,
        consensus_debug_only_candidates,
    )
    write_debug_image(debug_path / "row_col_consensus_overlay.jpg", consensus_overlay)
    write_candidate_crops(mask, consensus_candidates, debug_path / "row_col_consensus_candidate_crops.jpg")
    consensus_repaired_overlay, consensus_repaired_mask = draw_repaired_circles(
        mask,
        holes,
        consensus_candidates,
        expected_radius=hole_stats.get("radius_median", 6.0) or 6.0,
    )
    write_debug_image(debug_path / "consensus_repaired_circles_overlay.jpg", consensus_repaired_overlay)
    write_debug_image(debug_path / "consensus_repaired_circles_mask.jpg", consensus_repaired_mask)
    with open(debug_path / "row_col_consensus_candidates.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "consensus_candidates": consensus_candidates,
                "debug_only_candidates": consensus_debug_only_candidates,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    secondary_origin_overlay = draw_expected_hole_candidates(mask, points, secondary_origin_candidates)
    write_debug_image(debug_path / "secondary_origin_candidates_overlay.jpg", secondary_origin_overlay)
    write_candidate_crops(mask, secondary_origin_candidates, debug_path / "secondary_origin_candidate_crops.jpg")
    secondary_origin_repaired_overlay, secondary_origin_repaired_mask = draw_repaired_circles(
        mask,
        holes,
        secondary_origin_candidates,
        expected_radius=hole_stats.get("radius_median", 6.0) or 6.0,
    )
    write_debug_image(debug_path / "secondary_origin_repaired_circles_overlay.jpg", secondary_origin_repaired_overlay)
    write_debug_image(debug_path / "secondary_origin_repaired_circles_mask.jpg", secondary_origin_repaired_mask)
    with open(debug_path / "secondary_origin_candidates.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "masks": list(secondary_origin_masks.keys()),
                "source_counts": secondary_origin_source_counts,
                "candidates": secondary_origin_candidates,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    centers_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(centers_overlay, point, 3, (255, 255, 0), -1)
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

    with open(debug_path / "curve_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

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


def detect_and_link_limited_holes(image_path, output_path, dist_threshold=32, offset_threshold=3.0):
    img, gray, mask = preprocess_image(image_path)
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
        )
    else:
        output = args.output or "result_defect.jpg"
        detect_and_link_limited_holes(
            args.input_img,
            output,
            dist_threshold=args.dist_threshold,
            offset_threshold=args.offset_threshold,
        )