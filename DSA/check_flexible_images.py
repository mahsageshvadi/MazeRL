import matplotlib.pyplot as plt
from Curve_Generator_Flexible_For_Ciruculum_learning import CurveMakerFlexible

cm = CurveMakerFlexible(h=128, w=128)

plt.figure(figsize=(15, 6))

# Sample 1: Thin, Clean, Dark BG
img1, _, _ = cm.sample_with_distractors(width_range=(2, 3), noise_prob=0.0, invert_prob=0.5)
plt.subplot(1, 3, 1); plt.imshow(img1, cmap='gray', vmin=0, vmax=1); plt.title("Phase 1: Thin/Clean")

# Sample 2: Thick, Noisy, Dark BG
img2, _, _ = cm.sample_with_distractors(width_range=(1, 20), noise_prob=0.0, invert_prob=0.5)
plt.subplot(1, 3, 2); plt.imshow(img2, cmap='gray', vmin=0, vmax=1); plt.title("Phase 2: Thick/Noisy")

# Sample 3: Medium, Noisy, Inverted (Light BG)
img3, _, _ = cm.sample_with_distractors(width_range=(1, 12), noise_prob=1.0, invert_prob=0.5)
plt.subplot(1, 3, 3); plt.imshow(img3, cmap='gray', vmin=0, vmax=1); plt.title("Phase 3: Inverted")

plt.show()

