"""
Master script to regenerate all charts from benchmark data
"""
import subprocess
import sys

scripts = [
    "analysis/archetype_analysis.py",
    "analysis/wcag_analysis.py", 
    "analysis/csr_analysis.py",
    "analysis/coherence_analysis.py",
    "analysis/contribution_analysis.py",
    "analysis/compute_metrics.py",
]

print("Running all analysis scripts...")
for script in scripts:
    print(f"\n--- {script} ---")
    subprocess.run([sys.executable, script], check=False)
print("\nDone!")
