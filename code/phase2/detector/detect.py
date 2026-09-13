"""Detection wrappers for cpsam (Cellpose-SAM) and sam_vit_b (Segment Anything)."""
import os
import time
import numpy as np
from skimage.measure import regionprops

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
SAM_VIT_B_CKPT = os.path.join(WEIGHTS_DIR, "sam_vit_b_01ec64.pth")

_MODELS = {}
LAST_TIME_S = {}


def _get_cpsam():
    if "cpsam" not in _MODELS:
        from cellpose.models import CellposeModel
        _MODELS["cpsam"] = CellposeModel(gpu=True)
    return _MODELS["cpsam"]


def _get_sam_vit_b():
    if "sam_vit_b" not in _MODELS:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import segment_anything.automatic_mask_generator as amg_mod
        import torchvision.ops.boxes as tv_boxes
        # Upstream bug: amg_mod.batched_nms is called with a CPU `idxs` tensor
        # while `boxes`/`scores` are on CUDA. Wrap to move idxs to boxes.device.
        _orig_nms = tv_boxes.batched_nms

        def _nms_same_device(boxes, scores, idxs, iou_threshold):
            return _orig_nms(boxes, scores, idxs.to(boxes.device), iou_threshold)

        amg_mod.batched_nms = _nms_same_device
        sam = sam_model_registry["vit_b"](checkpoint=SAM_VIT_B_CKPT)
        sam = sam.to("cuda")
        _MODELS["sam_vit_b"] = SamAutomaticMaskGenerator(
            sam,
            points_per_side=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.8,
            min_mask_region_area=100,
        )
    return _MODELS["sam_vit_b"]


def _detect_cpsam(image_u8, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, clahe=False):
    model = _get_cpsam()
    if clahe:
        import cv2
        image_u8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image_u8)
    t0 = time.time()
    masks, flows, styles = model.eval(
        image_u8,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    dt = time.time() - t0
    props = regionprops(masks)
    centroids = np.array([[p.centroid[1], p.centroid[0]] for p in props], dtype=float)
    return centroids, dt


def _detect_sam_vit_b(image_u8):
    gen = _get_sam_vit_b()
    rgb = np.repeat(image_u8[:, :, None], 3, axis=2)
    t0 = time.time()
    masks = gen.generate(rgb)
    dt = time.time() - t0
    centroids = []
    for m in masks:
        area = m["area"]
        if area < 150 or area > 2500:
            continue
        ys, xs = np.nonzero(m["segmentation"])
        centroids.append([xs.mean(), ys.mean()])
    return np.array(centroids, dtype=float), dt


def detect(model_name, image_u8, **kw):
    if model_name == "cpsam":
        centroids, dt = _detect_cpsam(image_u8, **kw)
    elif model_name == "sam_vit_b":
        centroids, dt = _detect_sam_vit_b(image_u8)
    else:
        raise ValueError(f"unknown model_name {model_name}")
    LAST_TIME_S[model_name] = dt
    return centroids
