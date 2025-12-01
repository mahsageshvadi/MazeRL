import numpy as np
import cv2

def _rng(seed=None):
    return np.random.default_rng(seed)

def _cubic_bezier(p0, p1, p2, p3, t):
    """Standard Cubic Bezier formula."""
    omt = 1.0 - t
    return (omt**3)*p0 + 3*omt*omt*t*p1 + 3*omt*t*t*p2 + (t**3)*p3

class CurveMakerFlexible:
    def __init__(self, h=128, w=128, seed=None):
        self.h = h
        self.w = w
        self.rng = _rng(seed)

    def _random_point(self, margin=10):
        y = self.rng.integers(margin, self.h - margin)
        x = self.rng.integers(margin, self.w - margin)
        return np.array([y, x], dtype=np.float32)

    def _generate_bezier_points(self, p0=None, n_samples=1000):
        if p0 is None:
            p0 = self._random_point(margin=10)
        
        # FIX: Ensure p3 is far enough from p0 (prevents blobs)
        for _ in range(20): 
            p3 = self._random_point(margin=10)
            dist = np.linalg.norm(p0 - p3)
            if dist > 40.0: 
                break
        else:
            p3 = np.array([self.h - p0[0], self.w - p0[1]], dtype=np.float32)

        center = (p0 + p3) / 2.0
        spread = np.array([self.h, self.w], dtype=np.float32) * 0.3
        p1 = center + self.rng.normal(0, 1, 2) * spread * 0.6
        p2 = center + self.rng.normal(0, 1, 2) * spread * 0.6
        
        ts = np.linspace(0, 1, n_samples, dtype=np.float32)
        pts = np.stack([_cubic_bezier(p0, p1, p2, p3, t) for t in ts], axis=0)
        return pts

    def _draw_aa_curve(self, img, pts, thickness, intensity):
        pts_xy = pts[:, ::-1] * 16 
        pts_int = pts_xy.astype(np.int32).reshape((-1, 1, 2))
        canvas = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.polylines(canvas, [pts_int], isClosed=False, color=255, 
                      thickness=int(thickness), lineType=cv2.LINE_AA, shift=4)
        canvas_float = canvas.astype(np.float32) / 255.0
        img[:] = np.maximum(img, canvas_float * intensity)

    def sample_with_distractors(self, width_range=(2, 8), noise_prob=1.0, invert_prob=0.5, min_intensity=0.6):
        img = np.zeros((self.h, self.w), dtype=np.float32)
        mask = np.zeros_like(img)
        thickness = self.rng.integers(width_range[0], width_range[1] + 1)
        thickness = max(1, int(thickness))
        intensity = self.rng.uniform(min_intensity, 1.0) 

        pts_target = self._generate_bezier_points()
        self._draw_aa_curve(img, pts_target, thickness, intensity)
        self._draw_aa_curve(mask, pts_target, thickness, 1.0)
        
        num_distractors = self.rng.integers(2, 5)
        for _ in range(num_distractors):
            d_thick = max(1, int(thickness * self.rng.uniform(0.5, 1.5)))
            d_int = self.rng.uniform(min_intensity, 1.0)
            idx = self.rng.integers(0, len(pts_target))
            seed_pt = pts_target[idx]
            jitter = self.rng.normal(0, 10, 2)
            p0_d = seed_pt + jitter
            if self.rng.random() < 0.5: p0_d = None
            pts_distractor = self._generate_bezier_points(p0=p0_d)
            self._draw_aa_curve(img, pts_distractor, d_thick, d_int)

        if self.rng.random() < noise_prob: self._apply_dsa_noise(img)
        if self.rng.random() < invert_prob: img = 1.0 - img
        img = np.clip(img, 0.0, 1.0)
        mask = (mask > 0.1).astype(np.uint8)
        return img, mask, [pts_target]

    def sample_curve(self, width_range=(2, 2), noise_prob=0.0, invert_prob=0.0, min_intensity=0.6, branches=False):       
        img = np.zeros((self.h, self.w), dtype=np.float32)
        mask = np.zeros_like(img) 
        thickness = self.rng.integers(width_range[0], width_range[1] + 1)
        thickness = max(1, int(thickness))
        intensity = self.rng.uniform(min_intensity, 1.0)

        pts_main = self._generate_bezier_points()
        self._draw_aa_curve(img, pts_main, thickness, intensity)
        self._draw_aa_curve(mask, pts_main, thickness, 1.0)
        pts_all = [pts_main]

        if branches:
            num_branches = self.rng.integers(1, 3)
            for _ in range(num_branches):
                idx = self.rng.integers(len(pts_main)*0.2, len(pts_main)*0.8)
                p0 = pts_main[idx]
                pts_branch = self._generate_bezier_points(p0=p0)
                b_thick = max(1, int(thickness * 0.7))
                self._draw_aa_curve(img, pts_branch, b_thick, intensity)
                self._draw_aa_curve(mask, pts_branch, b_thick, 1.0)
                pts_all.append(pts_branch)

        if self.rng.random() < noise_prob: self._apply_dsa_noise(img)
        if self.rng.random() < invert_prob: img = 1.0 - img
        img = np.clip(img, 0.0, 1.0)
        mask = (mask > 0.1).astype(np.uint8)
        return img, mask, pts_all

    def _apply_dsa_noise(self, img):
        num_blobs = self.rng.integers(1, 4)
        for _ in range(num_blobs):
            y, x = self._random_point(margin=0)
            sigma = self.rng.uniform(2, 8) 
            yy, xx = np.ogrid[:self.h, :self.w]
            dist_sq = (yy - y)**2 + (xx - x)**2
            blob = np.exp(-dist_sq / (2 * sigma**2))
            blob_int = self.rng.uniform(0.05, 0.2)
            img[:] = np.maximum(img, blob * blob_int)

        noise_level = self.rng.uniform(0.05, 0.15)
        noise = self.rng.normal(0, noise_level, img.shape)
        img[:] += noise

        # SPEED OPTIMIZATION: Use OpenCV instead of Scipy
        if self.rng.random() < 0.5:
            sigma = self.rng.uniform(0.5, 1.0)
            ksize = int(sigma * 3) | 1 # Ensure odd
            img[:] = cv2.GaussianBlur(img, (ksize, ksize), sigma)