import numpy as np
import scipy.ndimage

def _rng(seed=None):
    return np.random.default_rng(seed)

def _cubic_bezier(p0, p1, p2, p3, t):
    omt = 1.0 - t
    return (omt**3)*p0 + 3*omt*omt*t*p1 + 3*omt*t*t*p2 + (t**3)*p3

def _stamp_disc(arr, cy, cx, rad, value=1.0):
    h, w = arr.shape
    # Compute bounding box with integer casting
    y0 = int(max(0, cy - rad))
    y1 = int(min(h, cy + rad + 1))
    x0 = int(max(0, cx - rad))
    x1 = int(min(w, cx + rad + 1))
    
    if y0 >= y1 or x0 >= x1: return

    rr, cc = np.ogrid[y0:y1, x0:x1]
    mask = (rr - cy)**2 + (cc - cx)**2 <= rad*rad
    
    # Use maximum to combine strokes
    arr[y0:y1, x0:x1][mask] = np.maximum(arr[y0:y1, x0:x1][mask], value)

class CurveMakerFlexible:
    def __init__(self, h=128, w=128, seed=None):
        self.h, self.w = h, w
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
        
        # Clamp to image bounds
        pts[:, 0] = np.clip(pts[:, 0], 0, self.h - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, self.w - 1)
        return pts

    def _draw_path(self, pts, img, mask_layer, thickness):
        temp_layer = np.zeros_like(img)
        for (y, x) in pts:
            _stamp_disc(temp_layer, y, x, thickness, 1.0)
            # Mask is always binary, regardless of image intensity
            _stamp_disc(mask_layer, y, x, thickness, 1.0)
        
        # Random Vessel Intensity (Simulation of Contrast Dye)
        # 0.4 (faint) to 1.0 (bright)
        intensity = self.rng.uniform(0.4, 1.0)
        img[:] = np.maximum(img, temp_layer * intensity)

    def sample_curve(self, 
                     width_range=(1.5, 1.5),  # (Min, Max) Thickness
                     noise_prob=0.0,          # Probability of adding noise (0.0 to 1.0)
                     branches=False):
        
        img = np.zeros((self.h, self.w), dtype=np.float32)
        mask = np.zeros_like(img, dtype=np.float32)
        
        # 1. Determine Thickness for this specific image
        # We sample a single thickness for the whole curve
        thickness = self.rng.uniform(width_range[0], width_range[1])
        
        pts_main = self._random_bezier()
        self._draw_path(pts_main, img, mask, thickness)
        pts_all = [pts_main]

        # 2. Apply Noise (If enabled by probability)
        if self.rng.random() < noise_prob:
            self._apply_dsa_noise(img)

        # 3. Final Clamp
        img = np.clip(img, 0.0, 1.0)

        return img.astype(np.float32), mask.astype(np.uint8), pts_all

    def _apply_dsa_noise(self, img):
        """Applies random artifacts, gaussian noise, and blur."""
        
        # A. Background Artifacts (Clouds)
        num_clouds = self.rng.integers(0, 6)
        for _ in range(num_clouds):
            cy, cx = self._random_point()
            size = int(self.rng.uniform(20, 50))
            noise_patch = self.rng.uniform(0, 1, (size, size))
            noise_patch = scipy.ndimage.gaussian_filter(noise_patch, sigma=4.0)
            
            # Normalize and scale intensity
            noise_patch = noise_patch / (np.max(noise_patch) + 1e-6) * self.rng.uniform(0.1, 0.5)
            
            # Integer Slicing
            y0, y1 = int(max(0, cy - size//2)), int(min(self.h, cy + size//2))
            x0, x1 = int(max(0, cx - size//2)), int(min(self.w, cx + size//2))
            ph, pw = y1 - y0, x1 - x0
            
            if ph > 0 and pw > 0:
                img[y0:y1, x0:x1] += noise_patch[:ph, :pw]

        # B. Gaussian Noise
        noise_level = self.rng.uniform(0.02, 0.15)
        img += self.rng.normal(0, noise_level, img.shape)

        # C. Blur
        if self.rng.random() < 0.5:
            sigma = self.rng.uniform(0.5, 1.2)
            img[:] = scipy.ndimage.gaussian_filter(img, sigma=sigma)