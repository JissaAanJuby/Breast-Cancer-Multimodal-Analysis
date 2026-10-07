# Interview Preparation — Q&A

### Image Enhancement
**Q: Why CLAHE instead of normal histogram equalization?**
Global histogram equalization redistributes intensities using the WHOLE image's
histogram, so in ultrasound (which has strong local contrast variation and
speckle noise) it tends to over-amplify noisy regions while doing little for
local lesion contrast. CLAHE operates on small tiles with a clipped histogram,
so it boosts local contrast (e.g. lesion vs. surrounding tissue) without
blowing out noise globally.

### Restoration
**Q: What is restoration (in image processing terms)?**
Restoration means reversing a KNOWN or estimated degradation model (e.g. a blur
kernel/PSF plus noise) to recover an estimate of the original, undegraded image
— as opposed to enhancement, which just makes an image look subjectively
better without reference to any specific degradation model.

**Q: Why is Gaussian blur not strictly restoration?**
Gaussian blur is itself a degradation/smoothing operation — applying it makes
an image blurrier, it doesn't reverse anything. Calling "denoise with a
Gaussian blur" restoration confuses smoothing with deconvolution. Real
restoration requires inverting a degradation model, which is what Wiener
filtering does.

**Q: How does Wiener filtering work?**
It inverts a known (or assumed) blur PSF in the frequency domain while damping
the noise amplification that a naive inverse filter would cause, using a
noise-to-signal ratio term (`balance`/`K`) to control that trade-off. Where the
signal-to-noise ratio is high, it behaves close to an inverse filter; where
noise dominates, it suppresses the correction to avoid amplifying noise.

### Segmentation
**Q: What is Otsu thresholding?**
An automatic global thresholding method that picks the threshold minimizing
intra-class intensity variance (equivalently maximizing inter-class variance)
between foreground and background, assuming the histogram is roughly bimodal.

**Q: Why use morphological operations after thresholding?**
Raw thresholded masks are noisy — small false-positive specks and holes inside
the true region. Morphological opening removes small specks; closing fills
small holes and smooths the boundary, giving a cleaner mask before contour
extraction.

### SIFT / RANSAC
**Q: What is SIFT?**
Scale-Invariant Feature Transform — detects keypoints at multiple scales and
computes a descriptor from local gradient orientation histograms around each
keypoint, giving a representation that survives changes in scale, rotation,
and moderate illumination change.

**Q: Why is SIFT scale/rotation invariant?**
Keypoints are detected across a scale-space pyramid (so the same physical
feature is found regardless of image scale), and each descriptor is computed
relative to the keypoint's own dominant orientation (so rotating the image
rotates the descriptor's reference frame along with it, keeping the descriptor
values themselves consistent).

**Q: Why BFMatcher?**
Brute-force matching compares every descriptor in image 1 against every
descriptor in image 2 by distance — simple and exact, which is fine at the
scale of a few hundred keypoints (as here); FLANN is preferred mainly at much
larger scale for speed.

**Q: Why Lowe's ratio test?**
For each descriptor, compare the distance to its best match against its
second-best match; if they're close, the match is ambiguous and likely wrong.
Keeping only matches where best << second-best filters out a large fraction of
incorrect matches before any geometric verification.

**Q: What is RANSAC?**
RANdom SAmple Consensus — repeatedly fits a model (here, a homography) from a
small random subset of matches, counts how many of the remaining matches agree
with that model within a threshold ("inliers"), and keeps the model with the
most inliers. It's a robust-fitting technique that tolerates a fraction of
badly wrong data.

**Q: Why is RANSAC required after feature matching?**
Even after Lowe's ratio test, some incorrect matches remain — especially in
ultrasound where repetitive speckle texture produces descriptor matches that
are locally distinctive but globally wrong. RANSAC fits a single consistent
geometric transform across all matches and rejects the matches that don't fit
it, rather than trusting each match's local descriptor score alone.

**Q: What is a homography?**
A homography is the 3×3 projective transform mapping points from one planar
view to a corresponding planar view (covering translation, rotation, scale,
and perspective) — the appropriate model for a rigid or near-rigid planar
scene captured from different viewpoints/transforms.

**Q: Why can SIFT/RANSAC fail on ultrasound images?**
Ultrasound speckle noise is repetitive and somewhat random by nature, so
descriptors extracted from speckle regions are not distinctive and often match
incorrectly; also, ultrasound images can be low-contrast and lack the strong,
stable corner-like structures SIFT relies on, leading to fewer reliable
keypoints than in natural images.

### CNN / Transfer Learning
**Q: Why pretrained ResNet18?**
A CNN needs thousands to millions of images to learn good low/mid-level
filters from scratch; a small dataset like BUSI (~780 images) can't support
that. A network pretrained on ImageNet already has useful general-purpose
visual filters (edges, textures, shapes) that transfer reasonably well.

**Q: Why freeze the backbone?**
With a small dataset and no GPU, fine-tuning millions of parameters is both
computationally expensive and prone to overfitting. Freezing the backbone
means gradients are only computed for the final layer — training is fast on
CPU and the risk of overfitting the (small) dataset is much lower.

**Q: Why not train from scratch?**
Training from scratch needs far more data and compute than is available here;
with ~780 images the model would badly overfit or simply fail to learn
generalizable features.

**Q: What is transfer learning?**
Reusing a model's representations learned on one (typically large, general)
task/dataset as the starting point for a different but related task, rather
than learning that representation from zero.

**Q: Why ImageNet weights specifically?**
ImageNet is a large, diverse natural-image dataset; convolutional filters
learned on it capture generally useful low/mid-level visual patterns (edges,
textures, blobs) that transfer reasonably well to other image domains,
including medical images, even though the domains differ.

**Q: How do you handle class imbalance?**
By weighting the loss function inversely proportional to class frequency
(`class_weights` in `cnn_model.py`), so the model isn't free to ignore the
minority class(es) to minimize loss.

### Evaluation
**Q: Why is accuracy insufficient?**
On an imbalanced dataset, a model can score high accuracy by always predicting
the majority class while completely failing the minority class(es) — accuracy
hides this. Precision/recall/F1 (per class) reveal it.

**Q: Precision vs recall?**
Precision = of everything predicted positive, how much was actually positive
(cost of false positives). Recall = of everything actually positive, how much
was correctly found (cost of false negatives). In cancer diagnosis, missing a
real malignant case (false negative, i.e. low recall) is typically far more
costly than a false alarm.

**Q: Why F1?**
F1 is the harmonic mean of precision and recall, giving one number that
penalizes a large imbalance between the two rather than letting one high value
mask a low other.

**Q: What is ROC-AUC?**
The area under the Receiver Operating Characteristic curve, which plots true
positive rate against false positive rate across all classification
thresholds; AUC summarizes how well the model ranks positives above negatives
regardless of any single chosen threshold.

**Q: What is PR-AUC?**
The area under the Precision-Recall curve — generally more informative than
ROC-AUC on strongly imbalanced datasets, since it focuses on performance
specifically on the positive (minority) class rather than being dominated by
the (large) number of true negatives.

### Data Leakage
**Q: What was the data leakage problem in the original code?**
The XGBoost model was trained on the full dataset `X, y`, and then "predicted"
on `X.iloc[:1]` — a row the model had already been trained on. Reporting that
probability as a meaningful prediction is invalid; the model has already
memorized that exact row.

**Q: How did you fix it?**
Split the data into train/test BEFORE fitting anything; fit the scaler on the
training set only and apply (not refit) it to the test set; evaluate the
trained model exclusively on the held-out test set it never saw during
training.

**Q: Why is predicting on training data invalid?**
A model can trivially memorize training examples (especially tree ensembles
with enough depth/estimators), so its performance on training data
overestimates — often dramatically — how it will perform on genuinely unseen
data. Only held-out evaluation reflects real generalization.

### Fusion
**Q: How does multimodal fusion work (in general)?**
Combine predictions or features from multiple modalities for the SAME
underlying case — either at the decision level (combine each modality's output
probability, e.g. a weighted average) or at the feature level (concatenate
learned representations and train a joint classifier).

**Q: Why can't Wisconsin and BUSI simply be treated as paired samples?**
They are two independent public datasets collected from different patient
populations with no shared patient identifiers — there is no ground truth
saying which BUSI image (if any) corresponds to which Wisconsin row. Pairing
them and reporting a "fused per-patient prediction" as a real result would be
fabricating a relationship the data doesn't support.

**Q: What are the limitations of the fusion approach here?**
The fusion mechanism (weighted late-fusion, stacked meta-classifier) is real
and correctly implemented, but it's demonstrated on an explicitly-labelled
illustrative random pairing of test-set predictions, not on genuinely paired
patient data — so its reported metrics show the mechanism works, not a
validated clinical fusion result.

### Pipeline Overview
**Q: How does the complete pipeline work, end to end?**
Two independent branches are built and evaluated: (1) Wisconsin tabular data →
clean → leakage-free train/test split → scale → XGBoost → held-out evaluation;
(2) BUSI images → classical image processing (enhancement/restoration/
segmentation/compression) and SIFT+RANSAC feature matching as a separate
analysis, plus a CNN branch (frozen pretrained ResNet18 + trainable head) with
its own train/val/test split and held-out evaluation. Both branches' real,
independently-measured metrics form the main results table; a clearly-labelled
illustrative demo then shows how decision-level fusion of the two would work
if patient-paired data were available.
