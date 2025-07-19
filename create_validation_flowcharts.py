import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Arrow
import numpy as np

# Create comprehensive therapeutic validation flowchart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
fig.suptitle('BioPath: Complete Therapeutic Validation Pipeline', 
             fontsize=18, fontweight='bold')

# Left panel: Data Input and Processing Flow
ax1.set_title('Data Processing & Cultural Integration', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 16)
ax1.axis('off')

# Define colors
input_color = '#E3F2FD'      # Light blue
process_color = '#FFF3E0'    # Light orange  
cultural_color = '#E8F5E8'   # Light green
output_color = '#F3E5F5'     # Light purple
decision_color = '#FFEBEE'   # Light red

# Input Sources (Top)
inputs = [
    {'xy': (0.5, 14), 'width': 2, 'height': 1.2, 'label': 'Traditional\nKnowledge\nDatabase', 'color': input_color},
    {'xy': (3, 14), 'width': 2, 'height': 1.2, 'label': 'Clinical\nTrial\nData', 'color': input_color},
    {'xy': (5.5, 14), 'width': 2, 'height': 1.2, 'label': 'Molecular\nStructure\nData', 'color': input_color},
    {'xy': (8, 14), 'width': 1.5, 'height': 1.2, 'label': 'Cultural\nContext', 'color': input_color}
]

for inp in inputs:
    rect = FancyBboxPatch(inp['xy'], inp['width'], inp['height'],
                         boxstyle="round,pad=0.1", facecolor=inp['color'],
                         edgecolor='navy', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(inp['xy'][0] + inp['width']/2, inp['xy'][1] + inp['height']/2,
            inp['label'], ha='center', va='center', fontweight='bold', fontsize=9)

# Data Preprocessing Layer
preprocess = [
    {'xy': (1, 12), 'width': 2.5, 'height': 1, 'label': 'Traditional Knowledge\nEmbedding & Weighting', 'color': process_color},
    {'xy': (4, 12), 'width': 2.5, 'height': 1, 'label': 'Clinical Data\nNormalization', 'color': process_color},
    {'xy': (7, 12), 'width': 2.5, 'height': 1, 'label': 'Cultural Context\nAnnotation', 'color': process_color}
]

for proc in preprocess:
    rect = FancyBboxPatch(proc['xy'], proc['width'], proc['height'],
                         boxstyle="round,pad=0.1", facecolor=proc['color'],
                         edgecolor='#F57C00', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(proc['xy'][0] + proc['width']/2, proc['xy'][1] + proc['height']/2,
            proc['label'], ha='center', va='center', fontweight='bold', fontsize=8)

# Cultural Bias Detection Core
bias_detection = {'xy': (3, 9.5), 'width': 4, 'height': 1.5, 'label': 'SHAP-Based Cultural Bias Detection\n& Correction Engine', 'color': cultural_color}
rect = FancyBboxPatch(bias_detection['xy'], bias_detection['width'], bias_detection['height'],
                     boxstyle="round,pad=0.15", facecolor=bias_detection['color'],
                     edgecolor='#2E7D32', linewidth=3)
ax1.add_patch(rect)
ax1.text(bias_detection['xy'][0] + bias_detection['width']/2, bias_detection['xy'][1] + bias_detection['height']/2,
        bias_detection['label'], ha='center', va='center', fontweight='bold', fontsize=10)

# Decision Diamond - 30% Threshold Check
diamond_x, diamond_y = 5, 7.5
diamond_size = 0.8
diamond = mpatches.RegularPolygon((diamond_x, diamond_y), 4, radius=diamond_size,
                                 orientation=np.pi/4, facecolor=decision_color,
                                 edgecolor='red', linewidth=2)
ax1.add_patch(diamond)
ax1.text(diamond_x, diamond_y, '≥30%\nCultural\nWeight?', ha='center', va='center',
        fontweight='bold', fontsize=8)

# Correction Path (if needed)
correction = {'xy': (0.5, 6), 'width': 3, 'height': 1, 'label': 'Cultural Weight\nCorrection & Rebalancing', 'color': cultural_color}
rect = FancyBboxPatch(correction['xy'], correction['width'], correction['height'],
                     boxstyle="round,pad=0.1", facecolor=correction['color'],
                     edgecolor='#2E7D32', linewidth=2)
ax1.add_patch(rect)
ax1.text(correction['xy'][0] + correction['width']/2, correction['xy'][1] + correction['height']/2,
        correction['label'], ha='center', va='center', fontweight='bold', fontsize=9)

# Validation Output
validation = {'xy': (6.5, 6), 'width': 3, 'height': 1, 'label': 'Therapeutic Validation\nResult & Confidence', 'color': output_color}
rect = FancyBboxPatch(validation['xy'], validation['width'], validation['height'],
                     boxstyle="round,pad=0.1", facecolor=validation['color'],
                     edgecolor='#7B1FA2', linewidth=2)
ax1.add_patch(rect)
ax1.text(validation['xy'][0] + validation['width']/2, validation['xy'][1] + validation['height']/2,
        validation['label'], ha='center', va='center', fontweight='bold', fontsize=9)

# EquiPath Integration
equipath = {'xy': (3.5, 4), 'width': 3, 'height': 1, 'label': 'EquiPath Compensation\nDistribution', 'color': '#E1F5FE'}
rect = FancyBboxPatch(equipath['xy'], equipath['width'], equipath['height'],
                     boxstyle="round,pad=0.1", facecolor=equipath['color'],
                     edgecolor='#0277BD', linewidth=2)
ax1.add_patch(rect)
ax1.text(equipath['xy'][0] + equipath['width']/2, equipath['xy'][1] + equipath['height']/2,
        equipath['label'], ha='center', va='center', fontweight='bold', fontsize=9)

# Add arrows for flow
arrows = [
    # Input to preprocessing
    ((1.5, 14), (2.25, 13)),
    ((4, 14), (5.25, 13)),
    ((6.5, 14), (8.25, 13)),
    ((8.75, 14), (8.25, 13)),
    
    # Preprocessing to bias detection
    ((2.25, 12), (4, 10.75)),
    ((5.25, 12), (5, 10.75)),
    ((8.25, 12), (6, 10.75)),
    
    # Bias detection to decision
    ((5, 9.5), (5, 8.3)),
    
    # Decision paths
    ((4.2, 7.5), (2, 7)),  # To correction (No path)
    ((5.8, 7.5), (8, 7)),  # To validation (Yes path)
    
    # Correction back to bias detection
    ((2, 6), (3.5, 9)),
    
    # Validation to EquiPath
    ((8, 6), (5, 5))
]

for start, end in arrows:
    ax1.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=2, color='navy'))

# Add labels for decision paths
ax1.text(3, 7.8, 'NO', ha='center', va='center', fontweight='bold', color='red', fontsize=10)
ax1.text(7, 7.8, 'YES', ha='center', va='center', fontweight='bold', color='green', fontsize=10)

# Right panel: Deployment Modes and Scaling
ax2.set_title('Deployment Modes & Performance Scaling', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 16)
ax2.axis('off')

# Deployment mode boxes
deployments = [
    {'xy': (1, 13), 'width': 8, 'height': 2, 'label': 'STANDALONE RESEARCH MODE\nProcessing: 24,000 validations/hour | Accuracy: 87%\nIdeal for: Individual research projects, pilot studies', 'color': '#E8F5E8'},
    
    {'xy': (1, 10), 'width': 8, 'height': 2, 'label': 'CLINICAL BUNDLE INTEGRATION\nProcessing: 45,000 validations/hour | Accuracy: 90%\nIdeal for: Clinical trial support, pharmaceutical R&D', 'color': '#FFF3E0'},
    
    {'xy': (1, 7), 'width': 8, 'height': 2, 'label': 'FULL ECOSYSTEM COORDINATION\nProcessing: 78,000 validations/hour | Accuracy: 93%\nIdeal for: Enterprise deployment, multi-system integration', 'color': '#F3E5F5'}
]

for i, deploy in enumerate(deployments):
    rect = FancyBboxPatch(deploy['xy'], deploy['width'], deploy['height'],
                         boxstyle="round,pad=0.15", facecolor=deploy['color'],
                         edgecolor='#424242', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(deploy['xy'][0] + deploy['width']/2, deploy['xy'][1] + deploy['height']/2,
            deploy['label'], ha='center', va='center', fontweight='bold', fontsize=10)

# Integration connections
integration_box = {'xy': (2, 4), 'width': 6, 'height': 2, 'label': 'OmniPath Ecosystem Integration\n\n• EthnoPath: Cultural preservation protocols\n• ChemPath: Quantum-enhanced processing\n• EquiPath: Fair compensation distribution', 'color': '#E1F5FE'}

rect = FancyBboxPatch(integration_box['xy'], integration_box['width'], integration_box['height'],
                     boxstyle="round,pad=0.15", facecolor=integration_box['color'],
                     edgecolor='#0277BD', linewidth=2)
ax2.add_patch(rect)
ax2.text(integration_box['xy'][0] + integration_box['width']/2, integration_box['xy'][1] + integration_box['height']/2,
        integration_box['label'], ha='center', va='center', fontweight='bold', fontsize=9)

# Performance scaling arrows
scaling_arrows = [
    ((5, 13), (5, 12)),    # Standalone to Clinical
    ((5, 10), (5, 9)),     # Clinical to Ecosystem
    ((5, 7), (5, 6))       # Ecosystem to Integration
]

for start, end in scaling_arrows:
    ax2.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=3, color='#1976D2'))

plt.tight_layout()
plt.savefig('assets/diagrams/biopath_validation_flowcharts.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Therapeutic validation flowcharts created: assets/diagrams/biopath_validation_flowcharts.png")
