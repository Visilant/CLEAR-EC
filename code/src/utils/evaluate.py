import numpy as np
import pandas as pd
from skimage.draw import polygon
from skimage.measure import regionprops


def calculate_metrics_from_masks(masks, ID=""):
    areas = calculate_area(masks)

    cell_density = calculate_cell_density(areas)
    cv = calculate_cv(areas)
    hexagonality = calculate_hexagonality(masks)
    std_area = np.std(areas) if len(areas) > 0 else 0
    total_area = np.sum(areas)
    count = len(areas)

    predictions = {
        "ID": ID,
        "CD": cell_density,
        "CV": cv,
        "HEX": hexagonality,
        "SD": std_area,
        "Total Area (um^2)": total_area,
        "Number of Cells": count,
    }
    return predictions


# Functions to compute Metrics =========================


def calculate_area(masks):
    """Calculate the area of objects in a mask.

    Input: mask (numpy.ndarray): The mask image where objects are segmented
    Output: areas_um2 (list): List of areas of objects in um^2."""
    pix_to_um = 1000 / 1296  # conversion ratio between real scan and image.
    area_obj = regionprops(masks)
    areas_pixels = [region.area for region in area_obj]
    areas_um2 = [area * (pix_to_um**2) for area in areas_pixels]
    return areas_um2


def calculate_cell_density(areas) -> float:
    """Cell density = # of cells / total area

    Input: areas (list): List of areas of cells in um^2.
    Output: cell_density (float): Cell density in cells per mm^2.
    """
    if len(areas) == 0:
        return 0
    total_area = np.sum(areas)
    if total_area == 0:
        return 0
    cell_density = (len(areas) * 1e6) / total_area
    return cell_density


def calculate_cv(areas):
    """Coefficient of Variation = standard deviation / mean

    Input: areas (list): List of areas of cells in um^2.
    Output: cv (float): Coefficient of Variation.
    """
    if len(areas) == 0:
        return 0
    mean_area = np.mean(areas)
    if mean_area == 0:
        return 0
    std_area = np.std(areas)
    cv = std_area / mean_area
    return cv


def calculate_hexagonality(masks):
    """Calculate the hexagonality of cell shapes.

    Input: masks (numpy.ndarray): The mask image where objects are segmented.
    Output: hexagonality (float): Hexagonality of cell shapes.
    """
    hexagonalities = []
    stats = regionprops(masks)
    for cell in stats:

        min_row, min_col, max_row, max_col = cell.bbox
        mask_single_cell = masks[min_row:max_row, min_col:max_col].copy()
        mask_single_cell = mask_single_cell.astype(np.float32)
        mask_single_cell[mask_single_cell != cell.label] = 0.2
        mask_single_cell[mask_single_cell == cell.label] = 1.0

        major_axis = cell.axis_major_length * 0.8
        minor_axis = cell.axis_minor_length * 0.8
        angle = cell.orientation

        hex_points = np.array(
            [
                [major_axis / 2, 0],  # right
                [major_axis / 4, -minor_axis / 2],  # bottom right
                [-major_axis / 4, -minor_axis / 2],  # bottom left
                [-major_axis / 2, 0],  # left
                [-major_axis / 4, minor_axis / 2],  # top left
                [major_axis / 4, minor_axis / 2],  # top right
            ]
        )

        rotation_matrix = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        rotated_hex = hex_points @ rotation_matrix

        # Create and fill hexagon mask
        height, width = mask_single_cell.shape
        hex_mask = np.zeros((height, width))

        # Center the hexagon
        centroid_y, centroid_x = cell.centroid
        center_y, center_x = np.ceil(centroid_y - min_row), np.ceil(
            centroid_x - min_col
        )

        translated_hex = rotated_hex + np.array([center_y, center_x])

        # Draw filled polygon
        rr, cc = polygon(translated_hex[:, 0], translated_hex[:, 1], hex_mask.shape)
        hex_mask[rr, cc] = 1

        # Calculate IoU (Intersection over Union)
        intersection = np.logical_and(mask_single_cell == 1.0, hex_mask).sum()
        union = np.logical_or(mask_single_cell == 1.0, hex_mask).sum()
        iou = intersection / union if union > 0 else 0

        hexagonalities.append(iou)

    hexagonality = np.mean(hexagonalities) if len(hexagonalities) > 0 else 0

    return hexagonality


# =========================================================


def evaluate_results(pred_df: pd.DataFrame, gt_df: pd.DataFrame, round_to: int | None = 1):
    """
    Merge predictions and ground-truth by ID, compute per-file percent errors
    for a set of numeric labels, and return both the per-file error DataFrame and
    a one-row DataFrame containing the average error across files for each label.

    Returns:
    - error_df: DataFrame with one row per ID and percent error columns
    - avg_error_df: single-row DataFrame with the average percent error for each label
    """

    def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
        if "ID" not in df.columns:
            return df.copy()
        out = df.copy()
        out["ID"] = (
            out["ID"]
            .astype(str)
            .apply(lambda x: x if x.endswith(".bmp") else f"{x}.bmp")
        )
        return out

    pred_df_norm = _normalize_ids(pred_df)
    gt_df_norm = _normalize_ids(gt_df)

    comparison_df = pd.merge(
        pred_df_norm, gt_df_norm, on="ID", suffixes=("_pred", "_gt")
    )

    if comparison_df.empty:
        print(
            "No overlapping IDs between predictions and ground truth after normalization."
        )
        return pd.DataFrame(), pd.DataFrame()

    labels = ["CD", "CV", "HEX"]

    # Build per-file error rows
    error_rows = []
    for _, row in comparison_df.iterrows():
        row_err = {"ID": row["ID"]}
        for label in labels:
            pred_col = label + "_pred"
            gt_col = label + "_gt"

            # If columns are missing, mark as NaN
            if (
                pred_col not in comparison_df.columns
                or gt_col not in comparison_df.columns
            ):
                row_err[label + " Error (%)"] = np.nan
                continue

            gt_val = row[gt_col]
            pred_val = row[pred_col]

            # Avoid division by zero and treat missing/zero ground-truth as NaN
            if pd.isna(gt_val) or gt_val == 0:
                row_err[label + " Error (%)"] = np.nan
            else:
                row_err[label + " Error (%)"] = abs(pred_val - gt_val) / gt_val * 100

        error_rows.append(row_err)
    error_df = pd.DataFrame(error_rows)
    if round_to is not None:
        error_df = error_df.round(round_to)
    avg_series = error_df.drop(columns=["ID"], errors="ignore").mean(skipna=True)
    avg_percent_error_df = pd.DataFrame([avg_series])

    return error_df, avg_percent_error_df
