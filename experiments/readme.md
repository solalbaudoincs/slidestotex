# Comparison of Incremental Slide Detection Techniques for OCR Cost Reduction

## Introduction

This document outlines and compares different algorithms designed to identify subsequences of "incremental" slides within a presentation PDF. The goal is to reduce the cost associated with sending slides to advanced OCR APIs (like Vision Language Models) by only processing the *last* slide of each incremental sequence, assuming it contains the cumulative information from the preceding slides in that sequence.

An incremental sequence is defined as a series of consecutive slides where each slide is highly similar to the previous one but contains slightly *more* information (e.g., revealing bullet points one by one).

I evaluated three methods with varying levels of complexity and robustness.

(the method descriptions were generated by llms, based on the code I wrote)

---

## Method 1: MSE & Pixel Count Comparison

**Core Idea:** This method considers slides to be part of an incremental sequence if they are visually similar (low Mean Squared Error) *and* the newer slide has slightly more non-background content.

**How it Works:**
1. Load two adjacent slides as grayscale images.
2. Create a mask where either image has content (pixels below a brightness threshold, e.g., 240).
3. Calculate the **Mean Squared Error (MSE)** between the pixels of the two images, but only for pixels included in the mask.
4. Count "non-white" pixels (pixels below the threshold) in both images as a proxy for content quantity.
5. A slide pair is considered incremental if:
   * The MSE is below a defined `mse_threshold`.
   * The non-white pixel count in the newer slide is greater than or equal to the count in the previous slide multiplied by a small `pixel_increase_factor` (e.g., 1.0005).

**Key Parameters:**
* `mse_threshold`: Maximum allowed MSE. Lower values mean slides must be *more* similar (e.g., 100-200).
* `pixel_increase_factor`: Multiplier for pixel count comparison. Must be >= 1.0. Values closer to 1.0 allow for minimal content additions (e.g., 1.0005).
* `content_threshold`: Brightness threshold (0-255) for considering a pixel as "content" vs. "background".

**Pros:**
* Conceptually simple and computationally efficient.
* Focuses comparison only on content areas through masking.
* Handles empty or content-free areas intelligently.

**Cons:**
* Sensitive to minor variations in rendering or alignment.
* MSE is not a perceptually accurate measure of visual similarity.
* Simple pixel counting may not capture semantic content changes.

---

## Method 2: SSIM & Difference Image Analysis

**Core Idea:** This method uses the Structural Similarity Index (SSIM), which is better aligned with human perception of similarity, combined with an analysis of *where* the differences occur. It checks for high structural similarity and ensures the difference is small but non-zero.

**How it Works:**
1. Load two adjacent slides as grayscale NumPy arrays (uint8 format).
2. Calculate the **SSIM score** between the two images. SSIM values range from -1 to 1, with 1 indicating perfect structural similarity.
3. Calculate the absolute pixel-wise difference between the two images.
4. Count the number of pixels where the absolute difference exceeds a small `noise_threshold` (e.g., 10 intensity levels). Let this be `significant_diff_pixels`.
5. Calculate the ratio of `significant_diff_pixels` to the total number of pixels (`diff_ratio`).
6. A slide pair is considered incremental if:
   * The SSIM score is above a defined `ssim_threshold` (e.g., 0.97 or higher).
   * The `diff_ratio` is below a defined `diff_pixel_max_ratio` (e.g., 0.05, meaning less than 5% of pixels changed significantly).
   * The `significant_diff_pixels` count is above a defined `diff_change_threshold` (e.g., 50) to ensure the slides are not *identical*.

**Key Parameters:**
* `ssim_threshold`: Minimum SSIM score required. Closer to 1 means more structurally similar. (e.g., 0.97 - 0.99).
* `diff_pixel_max_ratio`: Maximum allowed ratio of significantly changed pixels. (e.g., 0.01 - 0.10).
* `diff_change_threshold`: Minimum number of significantly changed pixels required to avoid flagging identical slides. (e.g., 20 - 100).

**Pros:**
* More robust to minor illumination, contrast, or rendering variations compared to MSE due to SSIM.
* Considers image structure, making it less susceptible to random noise affecting similarity scores drastically.
* Difference analysis helps quantify the *amount* of change.

**Cons:**
* SSIM calculation is computationally more expensive than MSE.
* Still requires careful tuning of three thresholds.
* Doesn't explicitly verify *containment* (i.e., that slide N is fully part of slide N+1).

---

## Method 3: Perceptual Hashing & Template Matching

**Core Idea:** This approach combines a very fast perceptual hash for an initial similarity check with OpenCV's template matching to explicitly verify if the previous slide appears "inside" the current slide, unchanged, at the top-left corner. It also includes a difference check similar to Method 2.

**How it Works:**
1. Load two adjacent slides (requires both PIL Image for hashing and NumPy array for OpenCV).
2. Calculate the **perceptual hash (pHash)** for both images using `imagehash`.
3. Compute the **Hamming distance** between the two hashes. A small distance indicates perceptual similarity.
4. If the hash distance is below `hash_diff_threshold`:
   * Perform **template matching** using OpenCV (`cv2.matchTemplate` with `TM_CCOEFF_NORMED`), treating the previous slide (`img_prev_np`) as the template and the current slide (`img_curr_np`) as the image to search within.
   * Check if the maximum correlation score (`maxVal`) is above `template_match_threshold` *and* if the location of this best match (`maxLoc`) is at `(0, 0)`.
   * If both template matching conditions are met, perform a difference analysis similar to Method 2: calculate `significant_diff_pixels` and `diff_ratio`.
5. A slide pair is considered incremental if:
   * Hash distance <= `hash_diff_threshold`.
   * Template match score >= `template_match_threshold`.
   * Best match location == `(0, 0)`.
   * `diff_ratio` < `diff_pixel_max_ratio`.
   * `significant_diff_pixels` > `diff_change_threshold`.

**Key Parameters:**
* `hash_diff_threshold`: Maximum Hamming distance between pHashes. (e.g., 0-5).
* `template_match_threshold`: Minimum correlation score for template matching (close to 1.0). (e.g., 0.97 - 0.99).
* `diff_pixel_max_ratio`: Maximum allowed ratio of significantly changed pixels. (e.g., 0.01 - 0.10).
* `diff_change_threshold`: Minimum number of significantly changed pixels required. (e.g., 20 - 100).

**Pros:**
* Perceptual hashing provides a very fast initial check.
* Template matching explicitly verifies the "containment" aspect, making it robust if slide N content appears unmodified in slide N+1.
* Relatively robust to minor rendering artifacts if they don't significantly alter the hash or the core template match area.

**Cons:**
* Most complex implementation involving multiple libraries and concepts.
* Relies on several thresholds that need tuning.
* Template matching assumes no rotation, scaling, or significant translation (usually true for incremental slides but a limitation).
* Performance depends on image size (template matching step).

---

## Comparison Summary
<center>

| Feature             | Method 1 (MSE + Pixels) | Method 2 (SSIM + Diff) | Method 3 (pHash + Template) |
| :------------------ | :---------------------- | :--------------------- | :-------------------------- |
| **Primary Metric**  | MSE                     | SSIM                   | pHash, Template Matching    |
| **Speed**           | Potentially Fastest     | Moderate               | Fast (hash) + Moderate (TM) |
| **Robustness**      | Low                     | Moderate               | High (for containment)      |
| **Perceptual?**     | No                      | Yes (SSIM)             | Yes (pHash)                 |
| **Checks Containment?**| No                   | Indirectly (diff)      | Yes (Template Matching)     |
| **Complexity**      | Low                     | Moderate               | High                        |
| **Thresholds**      | 2                       | 3                      | 4                           |

</center>

---

## Experimental Results

I found SSIM to be the best technique, based on qualitative results from testing on multiple pdfs. In the following image V1 is MSE, V2 is SSIM, V3 is Perceptual Hashing & Template Matching

<p align="center">
<img src="images/comparison techniques.png" />
</p>