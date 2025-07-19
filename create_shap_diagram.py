import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Create SHAP explanation visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('BioPath: SHAP-Based Cultural Bias Detection in Therapeutic Validation', 
             fontsize=16, fontweight='bold')

# Left panel: Traditional vs Cultural Knowledge Weighting
ax1.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')

# Sample SHAP values for demonstration
features = ['Clinical Trial Data', 'Molecular Structure', 'Traditional Knowledge', 
           'Cultural Context', 'Preparation Method', 'Historical Usage']
traditional_weights = [0.45, 0.35, 0.05, 0.03, 0.07, 0.05]
biopath_weights = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]

x_pos = np.arange(len(features))
width = 0.35

bars1 = ax1.barh(x_pos - width/2, traditional_weights, width, 
                label='Traditional AI System', color='#ff7f7f', alpha=0.8)
bars2 = ax1.barh(x_pos + width/2, biopath_weights, width,
                label='BioPath (Cultural Bias Corrected)', color='#7fbf7f', alpha=0.8)

ax1.set_xlabel('SHAP Feature Importance')
ax1.set_yticks(x_pos)
ax1.set_yticklabels(features)
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Add 30% minimum cultural knowledge line
ax1.axvline(x=0.30, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax1.text(0.31, len(features)-1, '30% Min Cultural\nKnowledge Threshold', 
         verticalalignment='top', color='red', fontweight='bold')

# Right panel: Validation Pipeline
ax2.set_title('Therapeutic Validation Pipeline', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Pipeline boxes
boxes = [
    {'xy': (1, 8), 'width': 2, 'height': 1, 'label': 'Input:\nTraditional Knowledge'},
    {'xy': (4, 8), 'width': 2, 'height': 1, 'label': 'Cultural Context\nEmbedding'},
    {'xy': (7, 8), 'width': 2, 'height': 1, 'label': 'SHAP Analysis'},
    {'xy': (1, 6), 'width': 2, 'height': 1, 'label': 'Clinical Data\nIntegration'},
    {'xy': (4, 6), 'width': 2, 'height': 1, 'label': 'Bias Detection\n& Correction'},
    {'xy': (7, 6), 'width': 2, 'height': 1, 'label': 'Validation\nOutput'},
    {'xy': (4, 4), 'width': 2, 'height': 1, 'label': 'EquiPath\nCompensation'},
]

for box in boxes:
    rect = FancyBboxPatch(box['xy'], box['width'], box['height'],
                         boxstyle="round,pad=0.1", facecolor='lightblue',
                         edgecolor='navy', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(box['xy'][0] + box['width']/2, box['xy'][1] + box['height']/2,
            box['label'], ha='center', va='center', fontweight='bold', fontsize=9)

# Arrows
arrows = [
    ((3, 8.5), (4, 8.5)),  # Traditional Knowledge → Cultural Context
    ((6, 8.5), (7, 8.5)),  # Cultural Context → SHAP
    ((3, 6.5), (4, 6.5)),  # Clinical Data → Bias Detection
    ((6, 6.5), (7, 6.5)),  # Bias Detection → Validation
    ((5, 6), (5, 5)),      # Bias Detection → Compensation
]

for start, end in arrows:
    ax2.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=2, color='navy'))

plt.tight_layout()
plt.savefig('assets/diagrams/biopath_shap_explanation.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ SHAP explanation diagram created: assets/diagrams/biopath_shap_explanation.png")
