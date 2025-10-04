import numpy as np
import pandas as pd
from skimage import color, filters, measure

def _to_float01(img):
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr

def extract_image_features(img_rgb01):
    """
    img_rgb01: np.ndarray in [0,1], shape (H,W,3) or (H,W)
    Returns 9 semantic features:
      brightness, contrast, edge_density, entropy,
      sharpness(laplacian var), intensity_mean, intensity_std, intensity_skew, intensity_kurtosis
    """
    from scipy.stats import kurtosis, skew
    import cv2

    # ensure grayscale [0,1]
    if img_rgb01.ndim == 3 and img_rgb01.shape[-1] == 3:
        gray = color.rgb2gray(img_rgb01).astype(np.float32)
    elif img_rgb01.ndim == 2:
        gray = img_rgb01.astype(np.float32)
    else:
        raise ValueError("Unexpected image shape for feature extraction")

    brightness = float(np.mean(gray))
    contrast   = float(np.std(gray))
    edge_density = float(np.mean(filters.sobel(gray)))
    entropy = float(measure.shannon_entropy(gray))

    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    sharpness = float(lap.var())

    hist = gray.ravel()
    intensity_mean     = float(np.mean(hist))
    intensity_std      = float(np.std(hist))
    intensity_skew     = float(skew(hist))
    intensity_kurtosis = float(kurtosis(hist))

    feats = np.array([
        brightness, contrast, edge_density, entropy,
        sharpness, intensity_mean, intensity_std,
        intensity_skew, intensity_kurtosis
    ], dtype=np.float32)

    return feats

def _label_with_blackbox(multimodal_model, feat_vec_1d):
    """Expect multimodal_model.predict_proba on combined features -> probability of class 1."""
    proba = multimodal_model.predict_proba(feat_vec_1d.reshape(1, -1))
    # Binary model: take column 1; fallback to last column
    if proba.ndim == 2 and proba.shape[1] >= 2:
        p1 = float(proba[0, 1])
    else:
        p1 = float(proba.ravel()[-1])
    return 1 if p1 > 0.5 else 0, p1

def generate_surrogates(x_tab0, x_img0, multimodal_model, n=200, tab_noise=0.05, img_noise=0.01, seed=42,
                        max_tries=5, noise_growth=2.0):
    """
    Create local surrogate dataset around one (tabular, image) point.
    Auto-retries with stronger noise until both classes 0/1 are present.
    Returns: DataFrame with named columns + final 'class' (0/1)
    """
    rng = np.random.default_rng(seed)

    # Tabular base (1,row) → flat vector + names
    tab_base = x_tab0.values.flatten().astype(np.float32)
    tab_names = list(x_tab0.columns)

    # Prepare image base -> [0,1] RGB
    img_base = _to_float01(x_img0)
    if img_base.ndim == 2:
        img_base = np.stack([img_base, img_base, img_base], axis=-1)
    elif img_base.ndim == 3 and img_base.shape[-1] == 1:
        img_base = np.repeat(img_base, 3, axis=-1)
    elif img_base.ndim == 3 and img_base.shape[-1] == 3:
        pass
    else:
        raise ValueError("x_img0 must be (H,W) or (H,W,1) or (H,W,3)")

    img_feat_names = [
        "img_brightness", "img_contrast", "img_edge_density", "img_entropy",
        "img_sharpness", "img_intensity_mean", "img_intensity_std",
        "img_intensity_skew", "img_intensity_kurtosis"
    ]

    rows, labels = [], []

    t_noise = float(tab_noise)
    i_noise = float(img_noise)

    for attempt in range(1, max_tries + 1):
        rows.clear(); labels.clear()

        for _ in range(int(n)):
            # perturb tabular
            tab_surr = tab_base + rng.normal(0.0, t_noise, size=tab_base.shape).astype(np.float32)

            # perturb image (pixelwise noise)
            img_noise_mat = rng.normal(0.0, i_noise, size=img_base.shape).astype(np.float32)
            img_surr = np.clip(img_base + img_noise_mat, 0.0, 1.0)

            # image semantic features
            img_feats = extract_image_features(img_surr).astype(np.float32)

            # combine: [img_feats | tab_feats]
            feat_vec = np.concatenate([img_feats, tab_surr], axis=0).astype(np.float32)

            y, _ = _label_with_blackbox(multimodal_model, feat_vec)
            rows.append(feat_vec)
            labels.append(y)

        # check class balance
        uniq = set(labels)
        if len(uniq) >= 2:
            break  # good: both classes present
        # else, grow noise and try again
        t_noise *= noise_growth
        i_noise *= noise_growth

    if len(set(labels)) < 2:
        raise RuntimeError(
            "Surrogate generation produced a single class after retries. "
            f"Try increasing --n or base noise (tab_noise/img_noise)."
        )

    all_names = img_feat_names + tab_names
    df = pd.DataFrame(rows, columns=all_names)
    df["class"] = labels
    return df
