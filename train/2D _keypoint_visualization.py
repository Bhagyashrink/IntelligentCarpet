import numpy as np
import cv2
import matplotlib.pyplot as plt

print(keypoint_GT.shape)
print(keypoint_GT[0, :, :2])

#The visual representation of body keypoints during Push-up activity
keypoints = [
    [0.34316593, 0.70200557],
    [0.4135706,  0.6837526],
    [0.34515056, 0.6438854],
    [0.35617143, 0.58404464],
    [0.3208778,  0.5890489],
    [0.47814146, 0.74324375],
    [0.50907624, 0.7499999],
    [0.52592325, 0.7601185],
    [0.6086717,  0.50775605],
    [0.5874259,  0.44470233],
    [0.7218672,  0.32295802],
    [0.85328025, 0.17067498],
    [0.6456683,  0.49246114],
    [0.74151474, 0.3492055],
    [0.8865339,  0.191579],
    [1.,         0.14098561],
    [1.,         0.12801307],
    [0.9323948,  0.10413533],
    [0.795264,   0.1356977],
    [0.7175107,  0.16488203],
    [0.8973856,  0.12622681]
]

BODY_25_pairs = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9),
    (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (14, 15), (0, 16),
    (0, 18), (14, 19), (19, 20)
]

BODY_25_color = [
    [100,  0, 255],
    [  0,  0, 255],
    [  0, 100, 255],
    [  0, 200, 255],
    [  0, 255, 255],
    [  0, 255, 100],
    [  0, 255,   0],
    [100, 255,   0],
    [200, 255,   0],
    [255, 255,   0],
    [255, 200,   0],
    [255, 100,   0],
    [255,   0,   0],
    [255,   0, 100],
    [255,   0, 200],
    [200, 200, 200],
    [150, 150, 150],
    [100, 100, 100],
    [ 50,  50,  50]
]

fig, ax = plt.subplots(figsize=(10,10))

# Plot keypoints as red dots
ax.scatter([point[0] for point in keypoints], [point[1] for point in keypoints], c='red')

# Connect the keypoints based on the pairs with respective colors
for i, pair in enumerate(BODY_25_pairs):
    x_values = [keypoints[pair[0]][0], keypoints[pair[1]][0]]
    y_values = [keypoints[pair[0]][1], keypoints[pair[1]][1]]
    color_rgb = [value/255. for value in BODY_25_color[i]]  # normalize to [0, 1] range
    ax.plot(x_values, y_values, color=color_rgb)

ax.set_xlim(0, 1.1)  # some padding for better visualization
ax.set_ylim(1.1, 0)  # flip the y-axis since origin is top-left and some padding for better visualization
ax.set_aspect('equal')
plt.show()

#
#
#
#The 2D positions of keypoints across the 8 sets

def plot_single_frame(ax, keypoints):
    # Plot keypoints
    ax.scatter([point[0] for point in keypoints], [point[1] for point in keypoints], c='white', s=40)

    # Connect the keypoints based on the pairs with respective colors
    for i, pair in enumerate(BODY_25_pairs):
        x_values = [keypoints[pair[0]][0], keypoints[pair[1]][0]]
        y_values = [keypoints[pair[0]][1], keypoints[pair[1]][1]]
        color_rgb = [value/255. for value in BODY_25_color[i]]
        ax.plot(x_values, y_values, color=color_rgb, linewidth=2)

    ax.set_xlim(0, 1.1)
    ax.set_ylim(1.1, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('black')

fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows and 4 columns of plots
fig.patch.set_facecolor('black')

for i in range(8):
    row = i // 4
    col = i % 4
    plot_single_frame(axs[row, col], keypoint_GT[i, :, :2])

plt.tight_layout()
plt.show()
