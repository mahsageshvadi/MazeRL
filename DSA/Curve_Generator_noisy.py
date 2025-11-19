
from __future__ import annotations
import numpy as np
import math, random
from typing import Tuple, List
import scipy.ndimage

def _rng(seed=None):
    return np.random.default_rng(seed)

def _cubic_bezier(p0, p1, p2, p3, t):
    """Return point on cubic Bezier at parameter t (0..1)."""
    omt = 1.0 - t
    return (omt**3)*p0 + 3*omt*omt*t*p1 + 3*omt*t*t*p2 + (t**3)*p3

def _stamp_disc(arr, cy, cx, rad, value=1.0):
    """Stamp a filled disc with center (cy,cx) and radius 'rad' into arr (in-place)."""
    h, w = arr.shape
    y0 = max(0, int(np.floor(cy - rad)))
    y1 = min(h, int(np.ceil (cy + rad + 1)))
    x0 = max(0, int(np.floor(cx - rad)))
    x1 = min(w, int(np.ceil (cx + rad + 1)))
    rr, cc = np.ogrid[y0:y1, x0:x1]
    mask = (rr - cy)**2 + (cc - cx)**2 <= rad*rad
    arr[y0:y1, x0:x1][mask] = value

class CurveMaker:
    def __init__(self, h=128, w=128, thickness=1.5, seed: int|None=None):
        self.h, self.w = h, w
        self.thickness = thickness
        self.rng = _rng(seed)

    def _random_point(self, margin=10):
        y = self.rng.integers(margin, self.h - margin)
        x = self.rng.integers(margin, self.w - margin)
        return np.array([y, x], dtype=np.float32)

    def _random_bezier(self, n_samples=400):
        """Sample one cubic Bezier path; return list of (y,x) float coords."""
        p0 = self._random_point()
        p3 = self._random_point()
        # control points roughly between p0 and p3 with random offsets
        center = (p0 + p3) / 2.0
        spread = np.array([self.h, self.w], dtype=np.float32) * 0.25
        p1 = center + self.rng.normal(0, 1, 2) * spread * 0.5
        p2 = center + self.rng.normal(0, 1, 2) * spread * 0.5
        ts = np.linspace(0, 1, n_samples, dtype=np.float32)
        pts = np.stack([_cubic_bezier(p0, p1, p2, p3, t) for t in ts], axis=0)
        # clip to frame
        pts[:, 0] = np.clip(pts[:, 0], 0, self.h - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, self.w - 1)
        return pts

    def _draw_path(self, pts: np.ndarray, img: np.ndarray, mask: np.ndarray):
        for (y, x) in pts:
            _stamp_disc(mask, y, x, self.thickness, 1.0)
        img[:] = np.maximum(img, mask)  # here base image is just the centerline

    def sample_curve(self, branches: bool=False, max_extra=2):
        """Return (image, mask, points). If branches=True, add up to 'max_extra' branches
        that connect to a random point on the main curve.
        """
        img = np.zeros((self.h, self.w), dtype=np.float32)
        mask = np.zeros_like(img)
        pts_main = self._random_bezier()
        self._draw_path(pts_main, img, mask)

        pts_all = [pts_main]

        if branches:
            n_extra = int(self.rng.integers(1, max_extra + 1))
            for _ in range(n_extra):
                # pick an anchor on the main path
                idx = int(self.rng.integers(50, len(pts_main) - 50))
                anchor = pts_main[idx]

                # make a new bezier starting near anchor
                p0 = anchor + self.rng.normal(0, 1, 2) * 2.0
                p3 = self._random_point()
                center = (p0 + p3) / 2.0
                spread = np.array([self.h, self.w], dtype=np.float32) * 0.25
                p1 = center + self.rng.normal(0, 1, 2) * spread * 0.5
                p2 = center + self.rng.normal(0, 1, 2) * spread * 0.5

                ts = np.linspace(0, 1, 200, dtype=np.float32)
                pts_b = np.stack([_cubic_bezier(p0, p1, p2, p3, t) for t in ts], axis=0)
                pts_b[:, 0] = np.clip(pts_b[:, 0], 0, self.h - 1)
                pts_b[:, 1] = np.clip(pts_b[:, 1], 0, self.w - 1)
                self._draw_path(pts_b, img, mask)
                pts_all.append(pts_b)

                intensity = self.rng.uniform(0.4, 1.0)
        img = img * intensity
        
        # B. Add Background Noise (Gaussian)
        noise_level = self.rng.uniform(0.0, 0.15)
        noise = self.rng.normal(0, noise_level, img.shape)
        img = img + noise
        
        # C. Add "Blob" artifacts (Simulate bones/organs)
        # Create random blobs
        num_blobs = self.rng.integers(0, 5)
        for _ in range(num_blobs):
            by, bx = self._random_point()
            b_rad = self.rng.uniform(5, 20)
            # Draw a faint blob
            yy, xx = np.ogrid[:self.h, :self.w]
            dist = (yy - by)**2 + (xx - bx)**2
            blob_mask = dist < b_rad**2
            img[blob_mask] += self.rng.uniform(0.0, 0.3)

        if self.rng.random() < 0.5:
            sigma = self.rng.uniform(0.5, 1.5)
            img = scipy.ndimage.gaussian_filter(img, sigma=sigma)
            

        img = np.clip(img, 0.0, 1.0)

        return img.astype(np.float32), mask.astype(np.uint8), [pts_main]

    def dataset(self, n=1000, branches_ratio=0.3, seed=None):
        if seed is not None:
            self.rng = _rng(seed)
        images = np.zeros((n, self.h, self.w), dtype=np.float32)
        masks  = np.zeros_like(images, dtype=np.uint8)
        for i in range(n):
            use_br = (self.rng.random() < branches_ratio)
            img, m, _ = self.sample_curve(branches=use_br)
            images[i] = img
            masks[i]  = m
        return images, masks