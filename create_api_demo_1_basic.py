import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Create focused demonstration for Basic Cultural Bias Detection
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1.2, 1], 
                      hspace=0.3, wspace=0.25)

fig.suptitle('BioPath API Demo 1: Basic Cultural Bias Detection', 
             fontsize=18, fontweight='bold', y=0.95)

# Code Example
ax_code = fig.add_subplot(gs[:, 0])
ax_code.set_title('Python Code Example', fontsize=14, fontweight='bold', pad=20)
ax_code.axis('off')

code = '''from biopath import CulturalBiasDetector, TherapeuticValidator

# Initialize BioPath with cultural bias correction
validator = TherapeuticValidator(
    cultural_bias_correction=True,
    min_traditional_knowledge_weight=0.30,
    deployment_mode="standalone"
)

# Load your research data
traditional_data = load_ethnobotanical_database()
clinical_data = load_clinical_trial_results()

# Run validation with SHAP explanation
result = validator.validate_therapeutic(
    traditional_knowledge=traditional_data,
    clinical_data=clinical_data,
    compound_id="therapeutic_x",
    explain=True
)

# Get bias detection results
bias_report = result.get_bias_analysis()
print(f"Traditional knowledge weight: {bias_report.traditional_weight:.1%}")
print(f"Cultural preservation score: {bias_report.cultural_score:.1%}")

# Check if bias threshold is met
if bias_report.traditional_weight >= 0.30:
    print("✅ Cultural bias check PASSED")
    print(f"Result: {result.validation_score:.1%} confidence")
else:
    print("❌ Cultural bias correction needed")
    corrected_result = validator.apply_bias_correction(result)
    print(f"Corrected result: {corrected_result.validation_score:.1%}")

# Visualize SHAP explanation
result.plot_shap_explanation(save_path="shap_analysis.png")

# Generate detailed report
report = result.generate_cultural_impact_report()
report.save("cultural_impact_analysis.pdf")'''

ax_code.text(0.05, 0.95, code, transform=ax_code.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=1.0", facecolor='#f8f8f8', alpha=0.9),
             linespacing=1.5)

# SHAP Output
ax_shap = fig.add_subplot(gs[0, 1])
ax_shap.set_title('SHAP Feature Analysis Output', fontsize=12, fontweight='bold', pad=15)

features = ['Clinical\nEfficacy', 'Molecular\nBinding', 'Traditional\nKnowledge', 'Cultural\nContext', 'Preparation\nMethod']
shap_values = [0.28, 0.24, 0.31, 0.12, 0.05]
colors = ['#4ecdc4' if val >= 0.30 and 'Traditional' in feat else '#ff6b6b' if 'Traditional' in feat else '#95a5a6' 
          for feat, val in zip(features, shap_values)]

bars = ax_shap.barh(features, shap_values, color=colors, alpha=0.8)
ax_shap.set_xlabel('SHAP Feature Importance', fontsize=11, fontweight='bold')
ax_shap.axvline(x=0.30, color='red', linestyle='--', linewidth=2)
ax_shap.text(0.32, 2, '30% Min\nThreshold\n✅ PASSED', fontsize=9, fontweight='bold', color='green')

for bar, val in zip(bars, shap_values):
    ax_shap.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.1%}', ha='left', va='center', fontweight='bold', fontsize=9)

ax_shap.grid(axis='x', alpha=0.3)

# Console Output
ax_console = fig.add_subplot(gs[1, 1])
ax_console.set_title('Console Output', fontsize=12, fontweight='bold', pad=15)
ax_console.axis('off')

console_output = '''Traditional knowledge weight: 31.0%
Cultural preservation score: 95.2%
✅ Cultural bias check PASSED
Result: 94.7% confidence

📄 Generated Files:
   • shap_analysis.png
   • cultural_impact_analysis.pdf

📊 Analysis Summary:
   • Bias detection: PASSED
   • Cultural integration: OPTIMAL
   • Validation confidence: HIGH
   • Ready for research use: ✅'''

ax_console.text(0.05, 0.95, console_output, transform=ax_console.transAxes, fontsize=11, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.8", facecolor='#e8f5e8', alpha=0.9))

plt.tight_layout()
plt.savefig('assets/images/biopath_api_demo_1_basic.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Demo 1 (Basic Cultural Bias Detection) created")
