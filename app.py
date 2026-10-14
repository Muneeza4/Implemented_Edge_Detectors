from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import streamlit as st


st.set_page_config(page_title="Edge Detector Studio", layout="wide")


@dataclass
class EvaluationResult:
    f1: float
    accuracy: float


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.abs(arr)
    max_val = arr.max()
    if max_val == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.uint8((arr / max_val) * 255)


def sobel_edges(gray: np.ndarray, threshold: int) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = normalize_to_uint8(mag)
    _, edges = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
    return edges


def prewitt_edges(gray: np.ndarray, threshold: int) -> np.ndarray:
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    gx = cv2.filter2D(gray.astype(np.float32), -1, kernel_x)
    gy = cv2.filter2D(gray.astype(np.float32), -1, kernel_y)
    mag = cv2.magnitude(gx, gy)
    mag = normalize_to_uint8(mag)
    _, edges = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
    return edges


def laplacian_edges(gray: np.ndarray, threshold: int) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = normalize_to_uint8(lap)
    _, edges = cv2.threshold(lap, threshold, 255, cv2.THRESH_BINARY)
    return edges


def canny_edges(gray: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    return cv2.Canny(gray, low_threshold, high_threshold)


def evaluate(pred_edges: np.ndarray, gt_edges: np.ndarray) -> EvaluationResult:
    pred = (pred_edges > 0).astype(np.uint8)
    gt = (gt_edges > 0).astype(np.uint8)

    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return EvaluationResult(f1=f1, accuracy=accuracy)


def read_uploaded_image(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    bytes_data = uploaded_file.read()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def read_uploaded_gray(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    bytes_data = uploaded_file.read()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return image


def get_detectors(gray: np.ndarray, gradient_threshold: int, lap_threshold: int, canny_low: int, canny_high: int) -> Dict[str, np.ndarray]:
    return {
        "Sobel": sobel_edges(gray, gradient_threshold),
        "Prewitt": prewitt_edges(gray, gradient_threshold),
        "Laplacian": laplacian_edges(gray, lap_threshold),
        "Canny": canny_edges(gray, canny_low, canny_high),
    }


def to_png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image to PNG.")
    return bytes(buf)


def main() -> None:
    st.title("Implemented Edge Detectors — Streamlit App")
    st.write(
        "Upload an image to run Sobel, Prewitt, Laplacian, and Canny detectors. "
        "Optionally upload a ground-truth edge image to compute F1 and accuracy."
    )

    with st.sidebar:
        st.header("Parameters")
        gradient_threshold = st.slider("Sobel/Prewitt threshold", 0, 255, 100)
        lap_threshold = st.slider("Laplacian threshold", 0, 255, 60)
        canny_low = st.slider("Canny low threshold", 0, 255, 60)
        canny_high = st.slider("Canny high threshold", 0, 255, 150)

    if canny_low >= canny_high:
        st.sidebar.error("Canny low threshold must be less than high threshold.")

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        uploaded_image = st.file_uploader("Upload input image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    with col_u2:
        uploaded_gt = st.file_uploader("Upload ground-truth edge image (optional)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

    image_bgr = read_uploaded_image(uploaded_image)
    if image_bgr is None:
        st.info("Please upload an input image to start.")
        return

    gt_gray = read_uploaded_gray(uploaded_gt)
    gray = to_grayscale(image_bgr)

    detectors = get_detectors(gray, gradient_threshold, lap_threshold, canny_low, canny_high)

    st.subheader("Input")
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    st.subheader("Detected edges")
    cols = st.columns(2)
    metrics_rows = []

    for idx, (name, edges) in enumerate(detectors.items()):
        with cols[idx % 2]:
            st.markdown(f"**{name}**")
            st.image(edges, clamp=True, use_container_width=True)
            st.download_button(
                label=f"Download {name} PNG",
                data=to_png_bytes(edges),
                file_name=f"{name.lower()}_edges.png",
                mime="image/png",
            )

            if gt_gray is not None:
                if gt_gray.shape != edges.shape:
                    st.warning("Ground truth shape does not match input; metrics skipped.")
                else:
                    result = evaluate(edges, gt_gray)
                    metrics_rows.append(
                        {
                            "Detector": name,
                            "F1": round(result.f1, 4),
                            "Accuracy": round(result.accuracy * 100, 2),
                        }
                    )

    if gt_gray is not None and metrics_rows:
        st.subheader("Evaluation vs Ground Truth")
        st.dataframe(metrics_rows, use_container_width=True)


if __name__ == "__main__":
    main()
