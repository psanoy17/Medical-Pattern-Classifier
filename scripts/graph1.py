import matplotlib.pyplot as plt
import numpy as np

# Data from Full Factorial Sweep (Heart Dataset)
n_colonies = np.array([2, 3, 4, 5])
mean_acc = np.array([0.7770, 0.7841, 0.8025, 0.8049])
std_dev = np.array([0.0162, 0.0214, 0.0119, 0.0074])
max_acc = np.array([0.8080, 0.8333, 0.8351, 0.8225]) # Derived from "Range" and "Top Configs" logic

fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot Mean Accuracy (Blue)
color = 'tab:blue'
ax1.set_xlabel('Number of Colonies ($K$)', fontsize=12)
ax1.set_ylabel('Mean Accuracy (± Std Dev)', color=color, fontsize=12)
ax1.errorbar(n_colonies, mean_acc, yerr=std_dev, fmt='-o', color=color, capsize=5, label='Mean Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

# Plot Max Accuracy (Red)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Maximum Achieved Accuracy', color=color, fontsize=12)
ax2.plot(n_colonies, max_acc, '-s', color=color, label='Max Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

# Annotate the Winner
plt.annotate('Optimal Peak (0.8351)', xy=(4, 0.8351), xytext=(4, 0.8300),
             arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

plt.title('Impact of Colony Count (Heart Dataset - Full Factorial)', fontsize=14)
fig.tight_layout()
plt.savefig('colony_sensitivity_factorial.png', dpi=300)
plt.show()