import numpy as np
import math
import scipy.ndimage

def _rng(seed=None):
    return np.random.default_rng(seed)

def _cubic_bezier(p0, p1, p2, p3, t):
    omt = 1.0 - t
    return (omt**3)*p0 + 3*omt*omt*t*p1 + 3*omt*t*t*p2 + (t**3)*p3

def _stamp_disc(arr, cy, cx, rad, value=1.0):
    h, w = arr.shape
    y0 = max(0, int(np.floor(cy - rad)))
    y1 = min(h, int(np.ceil (cy + rad + 1)))
    x0 = max(0, int(np.floor(cx - rad)))
    x1 = min(w, int(np.ceil (cx + rad + 1)))
    rr, cc = np.ogrid[y0:y1, x0:x1]
    mask = (rr - cy)**2 + (cc - cx)**2 <= rad*rad
    # Use maximum to avoid overwriting brighter spots with darker ones
    if y0 < y1 and x0 < x1:
        arr[y0:y1, x0:x1][mask] = np.maximum(arr[y0:y1, x0:x1][mask], value)

class CurveMakerDSA:
    def __init__(self, h=128, w=128, thickness=1.5, seed=None):
        self.h, self.w = h, w
        self.thickness = thickness
        self.rng = _rng(seed)

    def _random_point(self, margin=10):
        y = self.rng.integers(margin, self.h - margin)
        x = self.rng.integers(margin, self.w - margin)
        return np.array([y, x], dtype=np.float32)

    def _random_bezier(self, n_samples=400):
        p0 = self._random_point()
        p3 = self._random_point()
        center = (p0 + p3) / 2.0
        spread = np.array([self.h, self.w], dtype=np.float32) * 0.25
        p1 = center + self.rng.normal(0, 1, 2) * spread * 0.5
        p2 = center + self.rng.normal(0, 1, 2) * spread * 0.5
        ts = np.linspace(0, 1, n_samples, dtype=np.float32)
        pts = np.stack([_cubic_bezier(p0, p1, p2, p3, t) for t in ts], axis=0)
        pts[:, 0] = np.clip(pts[:, 0], 0, self.h - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, self.w - 1)
        return pts

    def _draw_path(self, pts, img, mask_layer):
        # Draw on a temp layer first
        temp_layer = np.zeros_like(img)
        for (y, x) in pts:
            _stamp_disc(temp_layer, y, x, self.thickness, 1.0)
            _stamp_disc(mask_layer, y, x, self.thickness, 1.0)
        
        # --- HANDLE VESSEL INTENSITY HERE ---
        # Randomly dim the vessel to simulate non-perfect contrast
        intensity = self.rng.uniform(0.4, 0.95)
        img[:] = np.maximum(img, temp_layer * intensity)

    def sample_curve(self, branches=False):
        img = np.zeros((self.h, self.w), dtype=np.float32)
        mask = np.zeros_like(img, dtype=np.float32)
        
        pts_main = self._random_bezier()
        self._draw_path(pts_main, img, mask)
        pts_all = [pts_main]

        # If you wanted global intensity scaling, you could do it here,
        # but _draw_path already handled the vessel brightness.
        # The line causing the error (img = img * intensity) is removed.

        # --- DSA AUGMENTATION ---
        
        # 1. Background Artifacts (Blobs)
        num_blobs = self.rng.integers(2, 6)
        for _ in range(num_blobs):
            by, bx = self._random_point(margin=0)
            b_rad = self.rng.uniform(10, 30)
            b_int = self.rng.uniform(0.1, 0.4)
            _stamp_disc(img, by, bx, b_rad, b_int)

        # 2. Gaussian Noise
        noise_level = self.rng.uniform(0.05, 0.15)
        noise = self.rng.normal(0, noise_level, img.shape)
        img = img + noise
        
        # 3. Gaussian Blur
        if self.rng.random() < 0.6:
            sigma = self.rng.uniform(0.5, 1.5)
            img = scipy.ndimage.gaussian_filter(img, sigma=sigma)

        # 4. Clamp and Normalize
        img = np.clip(img, 0.0, 1.0)

        return img.astype(np.float32), mask.astype(np.uint8), pts_all