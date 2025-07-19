import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Create focused demonstration for EquiPath Integration
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1.2, 1], 
                      hspace=0.3, wspace=0.25)

fig.suptitle('BioPath API Demo 3: EquiPath Compensation Integration', 
             fontsize=18, fontweight='bold', y=0.95)

# Code Example
ax_code = fig.add_subplot(gs[:, 0])
ax_code.set_title('Python Code Example', fontsize=14, fontweight='bold', pad=20)
ax_code.axis('off')

code = '''from biopath import TherapeuticValidator
from equipath import CompensationDistributor

# Initialize BioPath with EquiPath integration
validator = TherapeuticValidator(
    equipath_enabled=True,
    compensation_rate=0.15,        # 15% of validation value
    cultural_attribution=True,
    blockchain_tracking=True
)

# Load data with community attribution
traditional_data = load_ethnobotanical_database(
    include_community_metadata=True,
    require_consent_verification=True
)

clinical_data = load_clinical_trial_results()

# Run validation that tracks knowledge contributors
result = validator.validate_therapeutic(
    traditional_knowledge=traditional_data,
    clinical_data=clinical_data,
    compound_id="therapeutic_compound_x",
    track_contributors=True,
    generate_attribution_blockchain=True
)

# Process compensation if validation successful
if result.validation_successful:
    compensation = result.distribute_compensation()
    
    print(f"🎯 Validation Results:")
    print(f"   Therapeutic efficacy: {result.efficacy_score:.1%}")
    print(f"   Validation value: ${result.commercial_value:,.2f}")
    print(f"   Cultural preservation: {result.cultural_score:.1%}")
    
    print(f"\\n💰 Compensation Distribution:")
    print(f"   Total compensation: ${compensation.total_amount:,.2f}")
    print(f"   Communities benefited: {len(compensation.recipients)}")
    print(f"   Average per community: ${compensation.average_amount:,.2f}")
    
    # Generate detailed attribution report
    attribution = result.generate_attribution_report()
    attribution.save_pdf("cultural_attribution_report.pdf")
    
    # Export blockchain proof
    blockchain_proof = result.export_blockchain_proof()
    blockchain_proof.save("compensation_blockchain_proof.json")
    
    print(f"\\n📄 Generated Documentation:")
    print(f"   • Cultural attribution report (PDF)")
    print(f"   • Blockchain compensation proof (JSON)")
    print(f"   • Community consent verification (PDF)")'''

ax_code.text(0.05, 0.95, code, transform=ax_code.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=1.0", facecolor='#f8f8f8', alpha=0.9),
             linespacing=1.5)

# Compensation Distribution Chart
ax_comp = fig.add_subplot(gs[0, 1])
ax_comp.set_title('Compensation Distribution Output', fontsize=12, fontweight='bold', pad=15)

communities = ['Community A', 'Community B', 'Community C', 'Community D']
compensation_amounts = [2847, 1923, 3156, 2491]
knowledge_contributions = [25, 18, 30, 22]

x_pos = np.arange(len(communities))
width = 0.35

bars1 = ax_comp.bar(x_pos - width/2, compensation_amounts, width, 
                   label='Compensation ($)', color='#2ecc71', alpha=0.8)
ax_twin = ax_comp.twinx()
bars2 = ax_twin.bar(x_pos + width/2, knowledge_contributions, width,
                   label='Knowledge (%)', color='#3498db', alpha=0.8)

ax_comp.set_ylabel('Compensation\nAmount ($)', fontsize=11, fontweight='bold', color='#2ecc71')
ax_twin.set_ylabel('Knowledge\nContribution (%)', fontsize=11, fontweight='bold', color='#3498db')
ax_comp.set_xticks(x_pos)
ax_comp.set_xticklabels(communities, fontsize=10)

# Add value labels
for bar, amount in zip(bars1, compensation_amounts):
    ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'${amount:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

for bar, contrib in zip(bars2, knowledge_contributions):
    ax_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{contrib}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax_comp.grid(axis='y', alpha=0.3)
ax_comp.legend(loc='upper left')
ax_twin.legend(loc='upper right')

# Console Output
ax_console = fig.add_subplot(gs[1, 1])
ax_console.set_title('Validation & Compensation Results', fontsize=12, fontweight='bold', pad=15)
ax_console.axis('off')

console_output = '''🎯 Validation Results:
   Therapeutic efficacy: 94.7%
   Validation value: $18,947.00
   Cultural preservation: 96.2%

💰 Compensation Distribution:
   Total compensation: $2,842.05
   Communities benefited: 4
   Average per community: $710.51

📄 Generated Documentation:
   • Cultural attribution report (PDF)
   • Blockchain compensation proof (JSON)  
   • Community consent verification (PDF)

🔗 Blockchain Transaction:
   Hash: 0xa7b9c2d8e4f1...
   Block: 2,847,193
   Status: ✅ CONFIRMED

✅ Ethical Compliance:
   • Community consent verified
   • Fair compensation distributed
   • Cultural knowledge attributed
   • Blockchain proof generated'''

ax_console.text(0.05, 0.95, console_output, transform=ax_console.transAxes, fontsize=10, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.8", facecolor='#e8f5e8', alpha=0.9))

plt.tight_layout()
plt.savefig('assets/images/biopath_api_demo_3_equipath.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Demo 3 (EquiPath Compensation Integration) created")
