'''
Author: yufei Ji
Date: 2026-02-13 22:59:04
LastEditTime: 2026-02-13 23:00:47
Description: this script is used to 
FilePath: /VLMTraffic/scripts/view_image.py
'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Open the user's uploaded image
img_path = 'data/sft_dataset/JiNan/step_11/intersection_1_1_bev.jpg'
img = Image.open(img_path)

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)

# Remove axes for cleaner view
ax.axis('off')

# Image dimensions (estimating based on standard SUMO/simulation grids)
# Center is approx 512, 512. Lane width approx 35-40px.

# --- NORTH APPROACH (Top) ---
# Driving South (Down). Left turn lane is immediately to the LEFT of the center median line.
# Coordinates: x start approx 460, width 35. y start 50, height 350.
rect_n_left = patches.Rectangle((463, 50), 34, 380, linewidth=3, edgecolor='red', facecolor='none', label='Left Turn (Empty)')
# Straight lane is to the left of the left turn lane.
rect_n_straight = patches.Rectangle((425, 50), 34, 380, linewidth=3, edgecolor='#00FF00', facecolor='none', label='Straight (Occupied)')

# --- SOUTH APPROACH (Bottom) ---
# Driving North (Up). Left turn lane is immediately to the RIGHT of the center median line.
rect_s_left = patches.Rectangle((503, 600), 34, 380, linewidth=3, edgecolor='red', facecolor='none')
# Straight lane is to the right of the left turn lane.
rect_s_straight = patches.Rectangle((541, 600), 34, 380, linewidth=3, edgecolor='#00FF00', facecolor='none')

# Add patches to the plot
ax.add_patch(rect_n_left)
ax.add_patch(rect_n_straight)
ax.add_patch(rect_s_left)
ax.add_patch(rect_s_straight)

# Add Legend
# Create dummy patches for legend to avoid duplicate labels
red_patch = patches.Patch(color='red', label='NLSL: Left Turn Lane (Empty!)', fill=False, linewidth=3)
green_patch = patches.Patch(color='#00FF00', label='NTST: Straight Lane (Cars are here)', fill=False, linewidth=3)
plt.legend(handles=[red_patch, green_patch], loc='upper right', fontsize=12, framealpha=1)

# Save the annotated image
output_path = 'annotated_lanes.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

# Display the result
plt.show()