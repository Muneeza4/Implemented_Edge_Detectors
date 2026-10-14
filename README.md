# Edge Detection and Evaluation

This project implements multiple classical **edge detection algorithms** on a set of images and evaluates their performance against ground truth edges using **F1 score and accuracy metrics**.

---

## **Implemented Edge Detectors**

- **Sobel Operator**: Gradient-based method detecting horizontal and vertical edges.  
- **Prewitt Operator**: Similar to Sobel, simple gradient-based edge detector.  
- **Laplacian Operator**: Detects edges using second-order derivatives.  
- **Canny Edge Detector**: Multi-stage detector with noise reduction, gradient calculation, non-maximum suppression, and hysteresis thresholding.  

---

## **Dataset & Input**

- Images are read from a folder (e.g., `images/`) or uploaded in the Streamlit app.  
- Corresponding ground truth edges can be provided to compute metrics.  
- Specific image IDs can be selected for processing in notebook workflows.  

---

## **Workflow**

1. Load images and ground truth edges.  
2. Convert images to grayscale.  
3. Apply Sobel, Prewitt, Laplacian, and Canny edge detectors.  
4. Evaluate each detector against ground truth using:
   - **F1 Score**  
   - **Accuracy (%)**
5. Visualize and save comparison results.  
6. Export quantitative results to CSV.  

---

## **Run Streamlit App**

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app lets you:
- Upload an input image.
- Tune detector thresholds.
- View outputs for all four detectors.
- Optionally upload ground truth edges and view F1/accuracy.
- Download edge outputs as PNG files.

---

## **Requirements**

```bash
pip install -r requirements.txt
```
