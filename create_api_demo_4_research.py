import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Create focused demonstration for Complete Research Workflow
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1.2, 1], 
                      hspace=0.3, wspace=0.25)

fig.suptitle('BioPath API Demo 4: Complete Research Workflow', 
             fontsize=18, fontweight='bold', y=0.95)

# Code Example
ax_code = fig.add_subplot(gs[:, 0])
ax_code.set_title('Python Code Example', fontsize=14, fontweight='bold', pad=20)
ax_code.axis('off')

code = '''from biopath import ResearchWorkflow

# Initialize comprehensive research project
project = ResearchWorkflow(
    project_name="Traditional_Medicine_AI_Validation_Study",
    research_institution="University Research Lab",
    principal_investigator="Dr. Research Leader",
    ethics_approval="IRB-2025-047",
    funding_source="NSF Grant #2025-AI-TM-001"
)

# Configure validation pipeline
pipeline = project.create_validation_pipeline(
    traditional_sources=[
        "ethnobotanical_database",
        "community_knowledge_interviews", 
        "historical_usage_records"
    ],
    clinical_datasets=[
        "phase2_trials_database",
        "meta_analyses_collection",
        "pharmacological_studies"
    ],
    cultural_preservation_level="high",
    publication_ready=True,
    reproducibility_package=True
)

# Run comprehensive analysis
print("🔬 Running comprehensive therapeutic validation analysis...")

results = pipeline.run_comprehensive_analysis(
    compounds_of_interest=target_compounds,
    statistical_significance=0.05,
    cultural_bias_threshold=0.30,
    validation_iterations=1000,
    cross_validation_folds=5
)

print(f"📊 Analysis complete: {len(results.validated_compounds)} compounds validated")

# Generate publication materials
print("📄 Generating publication-ready materials...")

figures = results.generate_publication_figures(
    format="high_resolution",
    style="nature_medicine",
    include_supplementary=True
)

manuscript = results.generate_manuscript_draft(
    template="nature_medicine",
    include_methods=True,
    include_ethics_statement=True
)

# Create supplementary materials
supplementary = results.create_supplementary_data(
    include_raw_shap=True,
    include_cultural_metadata=True,
    anonymize_communities=True,
    include_code_repository=True
)

# Export complete research package
project.export_research_package(
    destination="publication_ready_package/",
    include_reproducibility_info=True,
    include_data_availability_statement=True,
    generate_zenodo_package=True
)

print("✅ Research package exported successfully!")
print("🎯 Ready for manuscript submission and peer review")'''

ax_code.text(0.05, 0.95, code, transform=ax_code.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=1.0", facecolor='#f8f8f8', alpha=0.9),
             linespacing=1.5)

# Research Results Summary
ax_results = fig.add_subplot(gs[0, 1])
ax_results.set_title('Research Analysis Results', fontsize=12, fontweight='bold', pad=15)

# Validation results chart
compounds = ['Compound A', 'Compound B', 'Compound C', 'Compound D', 'Compound E']
efficacy_scores = [0.94, 0.87, 0.92, 0.76, 0.89]
cultural_scores = [0.95, 0.91, 0.97, 0.88, 0.93]

x_pos = np.arange(len(compounds))
width = 0.35

bars1 = ax_results.bar(x_pos - width/2, efficacy_scores, width, 
                      label='Therapeutic Efficacy', color='#4CAF50', alpha=0.8)
bars2 = ax_results.bar(x_pos + width/2, cultural_scores, width,
                      label='Cultural Preservation', color='#2196F3', alpha=0.8)

ax_results.set_ylabel('Validation Score', fontsize=11, fontweight='bold')
ax_results.set_xticks(x_pos)
ax_results.set_xticklabels(compounds, rotation=45, fontsize=9)
ax_results.legend()
ax_results.grid(axis='y', alpha=0.3)

# Add score labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax_results.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# Publication Package Output
ax_package = fig.add_subplot(gs[1, 1])
ax_package.set_title('Publication-Ready Package', fontsize=12, fontweight='bold', pad=15)
ax_package.axis('off')

package_output = '''🔬 Running comprehensive therapeutic validation analysis...
📊 Analysis complete: 5 compounds validated

📄 Generating publication-ready materials...

📦 RESEARCH PACKAGE GENERATED

📄 Manuscript Materials:
   • Main manuscript draft (Nature Medicine format)
   • Figure 1: SHAP feature importance analysis
   • Figure 2: Cultural bias correction validation
   • Figure 3: Therapeutic efficacy comparison
   • Figure 4: Traditional knowledge attribution

📋 Supplementary Materials:
   • Dataset S1: Raw SHAP values (anonymized)
   • Dataset S2: Cultural metadata (IRB compliant)  
   • Dataset S3: Statistical analysis results
   • Code S1: Complete reproducibility package

📈 Key Statistical Results:
   • p < 0.001 for accuracy improvements
   • Effect size: Cohen's d = 1.24 (large effect)
   • 95% CI: [42.1%, 47.3%] accuracy improvement
   • Reproducibility score: 98.7%

✅ Ethics & Compliance:
   • IRB approval maintained throughout
   • Community consent protocols followed
   • Cultural sensitivity validation passed
   • Data availability statement included

🎯 Ready for manuscript submission and peer review
✅ Research package exported successfully!'''

ax_package.text(0.05, 0.95, package_output, transform=ax_package.transAxes, fontsize=9, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.8", facecolor='#fff3e0', alpha=0.9))

plt.tight_layout()
plt.savefig('assets/images/biopath_api_demo_4_research.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Demo 4 (Complete Research Workflow) created")
