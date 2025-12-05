import matplotlib.pyplot as plt
import numpy as np

image_loss = [2.7776, 2.7641, 2.7554, 2.7552, 2.7564, 2.7529, 2.7447, 2.7390, 2.7316, 2.7377]
seg_loss = [4.6531, 4.2883, 4.0520, 3.9864, 3.9783, 3.9658, 3.9249, 3.9385, 3.9296, 3.9409]
x_values = np.arange(1, 10+1)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x_values, image_loss, marker='o', color='blue', label='Image Loss')
# plt.plot(x_values, seg_loss, marker='s', color='red', label='Segmentation Loss')

# Labels and Title
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Comparison Over Epochs', fontsize=14)

# Grid and Legend
plt.grid(alpha=0.5)
plt.legend(fontsize=12)

# Show Plot
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x_values, seg_loss, marker='s', color='red', label='Segmentation Loss')

# Labels and Title
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Comparison Over Epochs', fontsize=14)

# Grid and Legend
plt.grid(alpha=0.5)
plt.legend(fontsize=12)

# Show Plot
plt.show()