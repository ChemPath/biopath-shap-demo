import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Create focused demonstration for Deployment Mode Optimization
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1.2, 1], 
                      hspace=0.3, wspace=0.25)

fig.suptitle('BioPath API Demo 2: Deployment Mode Optimization', 
             fontsize=18, fontweight='bold', y=0.95)

# Code Example
ax_code = fig.add_subplot(gs[:, 0])
ax_code.set_title('Python Code Example', fontsize=14, fontweight='bold', pad=20)
ax_code.axis('off')

code = '''from biopath import DeploymentOptimizer, TherapeuticValidator

# Initialize deployment optimizer
optimizer = DeploymentOptimizer()

# Test different deployment configurations
modes = ['standalone', 'clinical_bundle', 'ecosystem']
results = {}

for mode in modes:
    print(f"Testing {mode} deployment...")
    
    validator = TherapeuticValidator(deployment_mode=mode)
    
    # Run performance benchmark
    benchmark = validator.run_performance_test(
        sample_size=1000,
        complexity="high_cultural_content",
        duration_minutes=5
    )
    
    results[mode] = {
        'processing_speed': benchmark.validations_per_hour,
        'accuracy': benchmark.validation_accuracy,
        'cultural_preservation': benchmark.cultural_score,
        'cost_per_validation': benchmark.cost_estimate,
        'resource_usage': benchmark.resource_consumption
    }
    
    print(f"  Speed: {benchmark.validations_per_hour:,}/hour")
    print(f"  Accuracy: {benchmark.validation_accuracy:.1%}")
    print(f"  Cultural score: {benchmark.cultural_score:.1%}")

# Get recommendation based on your needs
recommendation = optimizer.recommend_deployment(
    research_scale="medium",           # small, medium, large
    cultural_content_level="high",     # low, medium, high
    budget_constraint=75000,           # annual budget
    accuracy_priority="high",          # low, medium, high
    speed_priority="medium"            # low, medium, high
)

print(f"\\n🎯 RECOMMENDED: {recommendation.mode}")
print(f"📊 Expected performance: {recommendation.expected_performance}")
print(f"💰 Estimated cost: ${recommendation.annual_cost:,}")

# Visualize deployment comparison
optimizer.plot_deployment_comparison(results, save_path="deployment_comparison.png")'''

ax_code.text(0.05, 0.95, code, transform=ax_code.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=1.0", facecolor='#f8f8f8', alpha=0.9),
             linespacing=1.5)

# Performance Comparison Chart
ax_perf = fig.add_subplot(gs[0, 1])
ax_perf.set_title('Performance Comparison Output', fontsize=12, fontweight='bold', pad=15)

modes = ['Standalone', 'Clinical\nBundle', 'Full\nEcosystem']
processing_speed = [24, 45, 78]  # thousands/hour
accuracy = [87, 90, 93]
colors = ['#3498db', '#f39c12', '#e74c3c']

ax_twin = ax_perf.twinx()

bars = ax_perf.bar([0, 1, 2], processing_speed, color=colors, alpha=0.7, width=0.6)
line = ax_twin.plot([0, 1, 2], accuracy, 'go-', linewidth=3, markersize=10)

ax_perf.set_ylabel('Processing Speed\n(K validations/hour)', fontsize=11, fontweight='bold')
ax_twin.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='green')
ax_perf.set_xticks([0, 1, 2])
ax_perf.set_xticklabels(modes, fontsize=10)

# Add value labels
for i, (bar, speed) in enumerate(zip(bars, processing_speed)):
    ax_perf.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{speed}K/hr', ha='center', va='bottom', fontweight='bold', fontsize=10)

for i, acc in enumerate(accuracy):
    ax_twin.text(i, acc + 1.5, f'{acc}%', ha='center', va='bottom', 
                fontweight='bold', color='green', fontsize=10)

ax_perf.grid(axis='y', alpha=0.3)

# Recommendation Output
ax_rec = fig.add_subplot(gs[1, 1])
ax_rec.set_title('Optimization Recommendation', fontsize=12, fontweight='bold', pad=15)
ax_rec.axis('off')

recommendation_output = '''Testing standalone deployment...
  Speed: 24,000/hour
  Accuracy: 87.0%
  Cultural score: 94.5%

Testing clinical_bundle deployment...
  Speed: 45,000/hour  
  Accuracy: 90.0%
  Cultural score: 95.8%

Testing ecosystem deployment...
  Speed: 78,000/hour
  Accuracy: 93.0%
  Cultural score: 97.2%

🎯 RECOMMENDED: clinical_bundle
📊 Expected performance: Optimal balance
💰 Estimated cost: $52,000/year

✅ Best fit for:
   • Medium-scale research projects
   • High cultural content requirements  
   • Balanced speed/accuracy needs
   • Budget-conscious deployment'''

ax_rec.text(0.05, 0.95, recommendation_output, transform=ax_rec.transAxes, fontsize=10, 
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.8", facecolor='#e3f2fd', alpha=0.9))

plt.tight_layout()
plt.savefig('assets/images/biopath_api_demo_2_deployment.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Demo 2 (Deployment Mode Optimization) created")
