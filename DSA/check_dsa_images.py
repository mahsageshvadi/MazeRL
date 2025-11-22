import matplotlib.pyplot as plt
import numpy as np
import random

# Import the DSA Generator
try:
    from Curve_Generator_noisy import CurveMakerDSA
except ImportError:
    print("ERROR: Could not find 'Curve_Generator_DSA.py'.")
    print("Make sure you saved the new generator code into a file named 'Curve_Generator_DSA.py'")
    exit()

def show_samples():
    # Initialize generator
    cm = CurveMakerDSA(h=128, w=128, thickness=1.5, seed=None)
    
    num_samples = 5
    
    plt.figure(figsize=(15, 6))
    
    for i in range(num_samples):
        # Generate
        img, mask, pts_all = cm.sample_curve(branches=False)
        gt_poly = pts_all[0]
        
        # Plot Image
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f"DSA Sample {i+1}")
        
        # Plot Mask/GT Overlay
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        
        # Plot the red centerline
        ys = [p[0] for p in gt_poly]
        xs = [p[1] for p in gt_poly]
        plt.plot(xs, ys, 'r--', linewidth=1, alpha=0.7)
        
        # Plot start point
        plt.plot(xs[0], ys[0], 'g.', markersize=5)
        
        plt.axis('off')
        plt.title(f"GT Overlay {i+1}")

    plt.tight_layout()
    plt.savefig("dsa_samples_preview.png")
    print("Saved 'dsa_samples_preview.png'. Open it to see the generated images.")
    plt.show()


from Synth_simple_v1_9_paper_version_gemini import CurveEnv
import matplotlib.pyplot as plt

env = CurveEnv(h=128, w=128)
count_inverted = 0

plt.figure(figsize=(12, 4))
for i in range(10):
    obs = env.reset()
    # Check the center pixel of the crop. 
    # If background is White (1.0), this should be high.
    # If background is Black (0.0), this should be low.
    img = env.ep.img
    is_inverted = img[0,0] > 0.5 # Check corner pixel
    if is_inverted: count_inverted += 1
    
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f"Inv: {is_inverted}")

plt.show()
print(f"Total Inverted in 10 resets: {count_inverted}")
