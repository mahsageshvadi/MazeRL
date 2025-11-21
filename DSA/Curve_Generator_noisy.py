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
        temp_layer = np.zeros_like(img)
        for (y, x) in pts:
            _stamp_disc(temp_layer, y, x, self.thickness, 1.0)
            _stamp_disc(mask_layer, y, x, self.thickness, 1.0)
        
        intensity = self.rng.uniform(0.4, 0.95)
        img[:] = np.maximum(img, temp_layer * intensity)

    def sample_curve(self, branches=False):
        img = np.zeros((self.h, self.w), dtype=np.float32)
        mask = np.zeros_like(img, dtype=np.float32)
        
        pts_main = self._random_bezier()
        self._draw_path(pts_main, img, mask)
        pts_all = [pts_main]

        # --- DSA AUGMENTATION ---
        
        # 1. Background Artifacts (Soft Cloud-like Blobs)
        num_clouds = self.rng.integers(3, 8)
        for _ in range(num_clouds):
            # Get float coordinates and convert to INT immediately
            cy_float, cx_float = self._random_point()
            cy, cx = int(cy_float), int(cx_float)
            
            # Create a small patch of random noise
            size = int(self.rng.uniform(20, 50))
            noise_patch = self.rng.uniform(0, 1, (size, size))
            
            # Blur it to look like soft tissue
            noise_patch = scipy.ndimage.gaussian_filter(noise_patch, sigma=5.0)
            
            # Normalize intensity
            noise_patch = noise_patch / (np.max(noise_patch) + 1e-6) * self.rng.uniform(0.1, 0.4)
            
            # Calculate integer slice indices
            half_size = size // 2
            y0 = int(max(0, cy - half_size))
            y1 = int(min(self.h, cy + half_size))
            x0 = int(max(0, cx - half_size))
            x1 = int(min(self.w, cx + half_size))
            
            # Calculate patch slice indices
            ph = y1 - y0
            pw = x1 - x0
            
            # Only add if the patch is valid (positive width/height)
            if ph > 0 and pw > 0:
                # Add the patch (sliced to fit if near edge)
                img[y0:y1, x0:x1] += noise_patch[:ph, :pw]

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