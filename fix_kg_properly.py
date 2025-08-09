#!/usr/bin/env python3
"""Properly fix the KG by updating the source code to handle all relation types"""

import os
import shutil

def fix_knowledge_graph_source():
    """Fix the knowledge_graph.py to include all constraints"""
    
    print("Fixing knowledge_graph.py to include all relation type constraints...")
    
    # Read the current file
    with open('src/knowledge_graph.py', 'r') as f:
        lines = f.readlines()
    
    # Find where setup_mathematical_constraints is defined
    for i, line in enumerate(lines):
        if 'def setup_mathematical_constraints(self):' in line:
            start_idx = i
            break
    else:
        print("Could not find setup_mathematical_constraints!")
        return False
    
    # Find the end of the function
    indent_level = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    for i in range(start_idx + 1, len(lines)):
        if lines[i].strip() and not lines[i].startswith(' ' * (indent_level + 4)):
            end_idx = i
            break
    else:
        end_idx = len(lines)
    
    print(f"Found setup_mathematical_constraints at lines {start_idx}-{end_idx}")
    
    # Create the new constraint setup
    new_constraints = '''    def setup_mathematical_constraints(self):
        """Setup mathematical constraints for each relation type"""
        self.mathematical_constraints = {
            RelationType.CATALYSIS: [
                MathematicalConstraint(
                    "michaelis_menten",
                    "(v_max * S) / (k_m + S)",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
                ),
                MathematicalConstraint(
                    "michaelis_menten_reversible",
                    "(v_max_f * S - v_max_r * P) / (k_m + S + P)",
                    {"v_max_f": (0.001, 1000.0), "v_max_r": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
                ),
                MathematicalConstraint(
                    "hill_equation",
                    "(v_max * (S ** n)) / ((k_m ** n) + (S ** n))",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "n": (1.0, 4.0)}
                ),
                MathematicalConstraint(
                    "allosteric_mm",
                    "(v_max * S) / (k_m * (1.0 + (A / k_a) ** n) + S)",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "k_a": (0.001, 100.0), "n": (1.0, 4.0)}
                )
            ],
            RelationType.SUBSTRATE_OF: [
                MathematicalConstraint(
                    "substrate_binding",
                    "(v_max * S) / (k_m + S)",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
                ),
                MathematicalConstraint(
                    "substrate_consumption",
                    "k_cat * E * S / (k_m + S)",
                    {"k_cat": (0.01, 1000.0), "E": (0.0001, 10.0), "k_m": (0.000001, 1000.0)}
                )
            ],
            RelationType.PRODUCT_OF: [
                MathematicalConstraint(
                    "product_formation",
                    "(v_max * S) / (k_m + S)",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
                ),
                MathematicalConstraint(
                    "product_inhibition",
                    "(v_max * S) / ((k_m + S) * (1.0 + P / k_p))",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "k_p": (0.001, 100.0)}
                )
            ],
            RelationType.COMPETITIVE_INHIBITION: [
                MathematicalConstraint(
                    "competitive_mm",
                    "(v_max * S) / (k_m * (1.0 + I / k_i) + S)",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "k_i": (0.000001, 1000.0)}
                ),
                MathematicalConstraint(
                    "competitive_hill",
                    "(v_max * (S ** n)) / ((k_m ** n) * (1.0 + I / k_i) + (S ** n))",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "k_i": (0.000001, 1000.0), "n": (1.0, 4.0)}
                )
            ],
            RelationType.NON_COMPETITIVE_INHIBITION: [
                MathematicalConstraint(
                    "non_competitive_mm",
                    "(v_max * S) / ((k_m + S) * (1.0 + I / k_i))",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "k_i": (0.000001, 1000.0)}
                )
            ],
            RelationType.ALLOSTERIC_REGULATION: [
                MathematicalConstraint(
                    "allosteric_hill",
                    "((v_max * (S ** n)) / ((k_m ** n) + (S ** n))) * ((1.0 + alpha * (A / k_a)) / (1.0 + A / k_a))",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "n": (1.0, 4.0), "alpha": (0.1, 10.0), "k_a": (0.001, 100.0)}
                )
            ],
            RelationType.BINDING: [
                MathematicalConstraint(
                    "simple_binding",
                    "B_max * S / (k_d + S)",
                    {"B_max": (0.001, 100.0), "k_d": (0.000001, 100.0)}
                ),
                MathematicalConstraint(
                    "cooperative_binding", 
                    "B_max * (S ** n) / ((k_d ** n) + (S ** n))",
                    {"B_max": (0.001, 100.0), "k_d": (0.000001, 100.0), "n": (1.0, 4.0)}
                ),
                MathematicalConstraint(
                    "kinetic_binding",
                    "(k_on * D * R) - (k_off * DR)",
                    {"k_on": (1000.0, 1000000000.0), "k_off": (0.001, 1000.0)}
                ),
                MathematicalConstraint(
                    "equilibrium_binding",
                    "R_total * S / (k_d + S)",
                    {"R_total": (0.001, 100.0), "k_d": (0.000001, 100.0)}
                )
            ],
            RelationType.BINDS_TO: [
                MathematicalConstraint(
                    "simple_binding",
                    "B_max * S / (k_d + S)",
                    {"B_max": (0.001, 100.0), "k_d": (0.000001, 100.0)}
                )
            ],
            RelationType.TRANSPORT: [
                MathematicalConstraint(
                    "facilitated_diffusion",
                    "(v_max * (S_out - S_in)) / (k_m + S_out + S_in)",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
                ),
                MathematicalConstraint(
                    "active_transport",
                    "(v_max * S * ATP) / ((k_m + S) * (k_atp + ATP))",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "k_atp": (0.001, 10.0)}
                ),
                MathematicalConstraint(
                    "symport",
                    "(v_max * S * Na) / ((k_m_s + S) * (k_m_na + Na))",
                    {"v_max": (0.001, 1000.0), "k_m_s": (0.000001, 1000.0), "k_m_na": (0.001, 100.0)}
                )
            ],
            RelationType.TRANSPORTS: [
                MathematicalConstraint(
                    "active_transport",
                    "(v_max * S) / (k_m + S)",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
                )
            ],
            RelationType.INDUCES: [
                MathematicalConstraint(
                    "gene_induction",
                    "(fold * (S ** n)) / ((EC50 ** n) + (S ** n))",
                    {"fold": (1.0, 100.0), "EC50": (0.001, 100.0), "n": (1.0, 4.0)}
                )
            ],
            RelationType.REPRESSES: [
                MathematicalConstraint(
                    "gene_repression",
                    "1.0 / (1.0 + ((S / IC50) ** n))",
                    {"IC50": (0.001, 100.0), "n": (1.0, 4.0)}
                )
            ],
            RelationType.INHIBITION: [
                MathematicalConstraint(
                    "inhibition",
                    "1.0 / (1.0 + ((S / IC50) ** n))",
                    {"IC50": (0.001, 100.0), "n": (1.0, 4.0)}
                )
            ],
            RelationType.INHIBITS: [
                MathematicalConstraint(
                    "inhibition",
                    "1.0 / (1.0 + ((S / IC50) ** n))",
                    {"IC50": (0.001, 100.0), "n": (1.0, 4.0)}
                )
            ],
            RelationType.PHOSPHORYLATES: [
                MathematicalConstraint(
                    "phosphorylation",
                    "k_cat * E * S / (k_m + S) * ATP / (k_atp + ATP)",
                    {"k_cat": (0.01, 1000.0), "E": (0.0001, 10.0), "k_m": (0.000001, 1000.0), "k_atp": (0.001, 10.0)}
                )
            ],
            RelationType.ENZYME_INDUCTION: [
                MathematicalConstraint(
                    "enzyme_induction",
                    "v_max_0 * (1.0 + (I_max * (S ** n)) / ((IC50 ** n) + (S ** n)))",
                    {"v_max_0": (0.001, 100.0), "I_max": (1.0, 10.0), "IC50": (0.001, 100.0), "n": (1.0, 4.0)}
                )
            ],
            RelationType.EXPRESSED_IN: [
                MathematicalConstraint(
                    "expression",
                    "basal + induced * S^n / (k_50^n + S^n)",
                    {"basal": (0.001, 10.0), "induced": (0.1, 100.0), "k_50": (0.001, 100.0), "n": (1.0, 4.0)}
                )
            ],
            RelationType.DRUG_DRUG_INTERACTION: [
                MathematicalConstraint(
                    "competitive_ddi",
                    "v_i = v_max * S_i / (k_m * (1 + sum(S_j/k_ij)) + S_i)",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
                ),
                MathematicalConstraint(
                    "non_competitive_ddi",
                    "v_i = v_max * S_i / ((k_m + S_i) * (1 + sum(S_j/k_ij)))",
                    {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
                )
            ],
            # Add default constraints for structural relationships
            RelationType.LOCATED_IN: [
                MathematicalConstraint(
                    "location_factor",
                    "c",
                    {"c": (0.1, 10.0)}
                )
            ],
            RelationType.PART_OF: [
                MathematicalConstraint(
                    "part_factor",
                    "c",
                    {"c": (0.1, 10.0)}
                )
            ],
            RelationType.BIOMARKER_FOR: [
                MathematicalConstraint(
                    "biomarker_response",
                    "baseline + delta * S / (EC50 + S)",
                    {"baseline": (0.0, 10.0), "delta": (0.1, 100.0), "EC50": (0.001, 100.0)}
                )
            ],
            RelationType.CAUSES_DISEASE: [
                MathematicalConstraint(
                    "disease_progression",
                    "severity * (1.0 - exp(-k * S))",
                    {"severity": (0.1, 10.0), "k": (0.001, 1.0)}
                )
            ],
            RelationType.TREATS: [
                MathematicalConstraint(
                    "treatment_effect",
                    "E_max * S / (EC50 + S)",
                    {"E_max": (0.1, 1.0), "EC50": (0.001, 100.0)}
                )
            ]
        }
'''
    
    # Replace the function
    lines[start_idx:end_idx] = [new_constraints + '\n']
    
    # Backup original
    shutil.copy('src/knowledge_graph.py', 'src/knowledge_graph.py.backup')
    
    # Write the fixed version
    with open('src/knowledge_graph.py', 'w') as f:
        f.writelines(lines)
    
    print("✓ Fixed src/knowledge_graph.py with all relation type constraints")
    
    # Delete the old cached KG so it rebuilds with new constraints
    cache_files = [
        './kg_cache/comprehensive_kg.json',
        './kg_cache/comprehensive_kg_fixed.json'
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"✓ Deleted old cache: {cache_file}")
    
    return True

if __name__ == "__main__":
    if fix_knowledge_graph_source():
        print("\n✅ SUCCESS! Knowledge graph source has been fixed.")
        print("Now run training with --use-kg-cache to build a new KG with all constraints.")
    else:
        print("\n❌ Failed to fix knowledge graph source.")