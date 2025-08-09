#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Builder Script
Builds a comprehensive biological knowledge graph with hundreds of relationships
Based on kg_builder.md specifications for 500 experimental systems
"""

import json
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import random

from src.knowledge_graph import KnowledgeGraph, BiologicalEntity, BiologicalRelationship, RelationType
from src.kg_loader_unified import KnowledgeGraphLoader, KnowledgeGraphBuilder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedKGBuilder:
    """Builds comprehensive knowledge graph with complete relationships"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.kg = KnowledgeGraph(config)
        
    def build_enzyme_kinetics_entities(self):
        """Add all enzyme kinetics entities (Section 4.1)"""
        logger.info("Building enzyme kinetics entities...")
        
        # Comprehensive enzyme list with their primary substrates
        enzyme_substrate_mapping = {
            # Glycolysis Enzymes
            "HK1": ("Hexokinase 1", "2.7.1.1", ["Glucose"], ["Glucose_6_phosphate"]),
            "HK2": ("Hexokinase 2", "2.7.1.1", ["Glucose"], ["Glucose_6_phosphate"]),
            "GPI": ("Glucose-6-phosphate isomerase", "5.3.1.9", ["Glucose_6_phosphate"], ["Fructose_6_phosphate"]),
            "PFK1": ("Phosphofructokinase 1", "2.7.1.11", ["Fructose_6_phosphate", "ATP"], ["Fructose_1_6_bisphosphate", "ADP"]),
            "ALDOA": ("Aldolase A", "4.1.2.13", ["Fructose_1_6_bisphosphate"], ["Glyceraldehyde_3_phosphate", "Dihydroxyacetone_phosphate"]),
            "TPI1": ("Triosephosphate isomerase", "5.3.1.1", ["Dihydroxyacetone_phosphate"], ["Glyceraldehyde_3_phosphate"]),
            "GAPDH": ("Glyceraldehyde-3-phosphate dehydrogenase", "1.2.1.12", ["Glyceraldehyde_3_phosphate", "NADplus", "Pi"], ["1_3_Bisphosphoglycerate", "NADH"]),
            "PGK1": ("Phosphoglycerate kinase 1", "2.7.2.3", ["1_3_Bisphosphoglycerate", "ADP"], ["3_Phosphoglycerate", "ATP"]),
            "ENO1": ("Enolase 1", "4.2.1.11", ["2_Phosphoglycerate"], ["Phosphoenolpyruvate", "H2O"]),
            "PKM": ("Pyruvate kinase M", "2.7.1.40", ["Phosphoenolpyruvate", "ADP"], ["Pyruvate", "ATP"]),
            
            # TCA Cycle Enzymes
            "CS": ("Citrate synthase", "2.3.3.1", ["Acetyl_CoA", "Oxaloacetate"], ["Citrate", "Coenzyme_A"]),
            "ACO2": ("Aconitase 2", "4.2.1.3", ["Citrate"], ["Isocitrate"]),
            "IDH1": ("Isocitrate dehydrogenase 1", "1.1.1.42", ["Isocitrate", "NADplus"], ["α_Ketoglutarate", "NADH", "CO2"]),
            "IDH2": ("Isocitrate dehydrogenase 2", "1.1.1.42", ["Isocitrate", "NADPplus"], ["α_Ketoglutarate", "NADPH", "CO2"]),
            "OGDH": ("α-Ketoglutarate dehydrogenase", "1.2.4.2", ["α_Ketoglutarate", "NADplus", "Coenzyme_A"], ["Succinyl_CoA", "NADH", "CO2"]),
            "SUCLA2": ("Succinate-CoA ligase", "6.2.1.5", ["Succinyl_CoA", "GDP", "Pi"], ["Succinate", "GTP", "Coenzyme_A"]),
            "SDH": ("Succinate dehydrogenase", "1.3.5.1", ["Succinate", "FAD"], ["Fumarate", "FADH2"]),
            "FH": ("Fumarase", "4.2.1.2", ["Fumarate", "H2O"], ["Malate"]),
            "MDH2": ("Malate dehydrogenase 2", "1.1.1.37", ["Malate", "NADplus"], ["Oxaloacetate", "NADH"]),
            
            # Amino Acid Metabolism Enzymes
            "GOT1": ("Glutamic-oxaloacetic transaminase 1", "2.6.1.1", ["Aspartate", "α_Ketoglutarate"], ["Oxaloacetate", "Glutamate"]),
            "GOT2": ("Glutamic-oxaloacetic transaminase 2", "2.6.1.1", ["Aspartate", "α_Ketoglutarate"], ["Oxaloacetate", "Glutamate"]),
            "GPT": ("Glutamic-pyruvic transaminase", "2.6.1.2", ["Alanine", "α_Ketoglutarate"], ["Pyruvate", "Glutamate"]),
            "GLS": ("Glutaminase", "3.5.1.2", ["Glutamine"], ["Glutamate", "Ammonia"]),
            "GLS2": ("Glutaminase 2", "3.5.1.2", ["Glutamine"], ["Glutamate", "Ammonia"]),
            "GLUD1": ("Glutamate dehydrogenase 1", "1.4.1.3", ["Glutamate", "NADplus"], ["α_Ketoglutarate", "NADH", "Ammonia"]),
            "ASS1": ("Argininosuccinate synthase 1", "6.3.4.5", ["Citrulline", "Aspartate", "ATP"], ["Argininosuccinate", "AMP", "PPi"]),
            "ASL": ("Argininosuccinate lyase", "4.3.2.1", ["Argininosuccinate"], ["Arginine", "Fumarate"]),
            
            # Lipid Metabolism Enzymes
            "FASN": ("Fatty acid synthase", "2.3.1.85", ["Acetyl_CoA", "Malonyl_CoA", "NADPH"], ["Palmitic_acid", "NADP+", "CO2"]),
            "ACC1": ("Acetyl-CoA carboxylase 1", "6.4.1.2", ["Acetyl_CoA", "ATP", "CO2"], ["Malonyl_CoA", "ADP", "Pi"]),
            "HMGCR": ("HMG-CoA reductase", "1.1.1.34", ["HMG_CoA", "NADPH"], ["Mevalonate", "NADP+"]),
            "CPT1A": ("Carnitine palmitoyltransferase 1A", "2.3.1.21", ["Palmitic_acid", "Carnitine"], ["Palmitoyl_carnitine", "Coenzyme_A"]),
            "HADHA": ("Hydroxyacyl-CoA dehydrogenase", "1.1.1.35", ["3_Hydroxyacyl_CoA", "NADplus"], ["3_Ketoacyl_CoA", "NADH"]),
            "ACACA": ("Acetyl-CoA carboxylase alpha", "6.4.1.2", ["Acetyl_CoA", "ATP", "CO2"], ["Malonyl_CoA", "ADP", "Pi"]),
            
            # Oxidoreductases
            "CAT": ("Catalase", "1.11.1.6", ["Hydrogen_peroxide"], ["H2O", "O2"]),
            "SOD1": ("Superoxide dismutase 1", "1.15.1.1", ["Superoxide"], ["Hydrogen_peroxide", "O2"]),
            "SOD2": ("Superoxide dismutase 2", "1.15.1.1", ["Superoxide"], ["Hydrogen_peroxide", "O2"]),
            "GPX1": ("Glutathione peroxidase 1", "1.11.1.9", ["Hydrogen_peroxide", "Glutathione"], ["H2O", "Oxidized_glutathione"]),
            "PRDX1": ("Peroxiredoxin 1", "1.11.1.15", ["Hydrogen_peroxide"], ["H2O"]),
            "NQO1": ("NAD(P)H dehydrogenase quinone 1", "1.6.5.2", ["NADPH", "Quinone"], ["NADPplus", "Hydroquinone"]),
            "G6PD": ("Glucose-6-phosphate dehydrogenase", "1.1.1.49", ["Glucose_6_phosphate", "NADPplus"], ["6_Phosphogluconate", "NADPH"]),
            "ALDH2": ("Aldehyde dehydrogenase 2", "1.2.1.3", ["Acetaldehyde", "NADplus"], ["Acetate", "NADH"]),
            
            # Add more transferases
            "GST": ("Glutathione S-transferase", "2.5.1.18", ["Glutathione"], ["Glutathione_conjugate"]),
            "UGT1A1": ("UDP glucuronosyltransferase 1A1", "2.4.1.17", ["UDP_glucuronate"], ["Glucuronide"]),
            "COMT": ("Catechol-O-methyltransferase", "2.1.1.6", ["S_Adenosylmethionine", "Catechol"], ["S_Adenosylhomocysteine", "O_Methylcatechol"]),
            "TPMT": ("Thiopurine S-methyltransferase", "2.1.1.67", ["S_Adenosylmethionine", "Thiopurine"], ["S_Adenosylhomocysteine", "Methylthiopurine"]),
            "NAT1": ("N-acetyltransferase 1", "2.3.1.5", ["Acetyl_CoA"], ["Acetylated_product", "Coenzyme_A"]),
            "NAT2": ("N-acetyltransferase 2", "2.3.1.5", ["Acetyl_CoA"], ["Acetylated_product", "Coenzyme_A"]),
            "SULT1A1": ("Sulfotransferase 1A1", "2.8.2.1", ["PAPS"], ["Sulfated_product", "PAP"]),
            "AANAT": ("Aralkylamine N-acetyltransferase", "2.3.1.87", ["Serotonin", "Acetyl_CoA"], ["N_Acetylserotonin", "Coenzyme_A"]),
        }
        
        # Create all enzymes with properties
        for enzyme_id, (name, ec_number, substrates, products) in enzyme_substrate_mapping.items():
            entity = BiologicalEntity(
                id=f"enzyme_{enzyme_id}",
                name=name,
                entity_type="enzyme",
                properties={
                    "ec_number": ec_number,
                    "substrates": substrates,
                    "products": products,
                    "source": "kg_builder"
                },
                confidence_score=0.95
            )
            self.kg.add_entity(entity)
        
        # Add CYP enzymes (important for drug metabolism)
        cyp_enzymes = [
            ("CYP3A4", "CYP3A4", "1.14.14.1"),
            ("CYP3A5", "CYP3A5", "1.14.14.1"),
            ("CYP2D6", "CYP2D6", "1.14.14.1"),
            ("CYP2C9", "CYP2C9", "1.14.14.1"),
            ("CYP2C19", "CYP2C19", "1.14.14.1"),
            ("CYP2C8", "CYP2C8", "1.14.14.1"),
            ("CYP1A2", "CYP1A2", "1.14.14.1"),
            ("CYP2B6", "CYP2B6", "1.14.14.1"),
            ("CYP2E1", "CYP2E1", "1.14.14.1"),
            ("CYP2A6", "CYP2A6", "1.14.14.1"),
        ]
        
        for cyp_id, name, ec_number in cyp_enzymes:
            entity = BiologicalEntity(
                id=f"cyp_{cyp_id}",
                name=name,
                entity_type="enzyme",
                properties={
                    "ec_number": ec_number,
                    "enzyme_class": "cytochrome_P450",
                    "drug_metabolism": True,
                    "source": "kg_builder"
                },
                confidence_score=0.93
            )
            self.kg.add_entity(entity)
        
        # Add comprehensive substrates (100 as specified)
        self._add_all_substrates()
        
        # Add comprehensive products (50 as specified)
        self._add_all_products()
        
        # Add inhibitors (30 as specified)
        self._add_all_inhibitors()
        
        # Add allosteric regulators (20 as specified)
        self._add_all_allosteric_regulators()
        
        logger.info(f"Added enzyme kinetics entities: {len([e for e in self.kg.entities.values() if e.entity_type in ['enzyme', 'substrate', 'product', 'inhibitor', 'allosteric_regulator']])} entities")
    
    def _add_all_substrates(self):
        """Add all 100 substrates as specified in kg_builder.md"""
        key_substrates = [
            # Glycolysis intermediates
            ("Glucose", 180.16, "C6H12O6"),
            ("Glucose_6_phosphate", 260.14, "C6H13O9P"),
            ("Fructose_6_phosphate", 260.14, "C6H13O9P"),
            ("Fructose_1_6_bisphosphate", 340.12, "C6H14O12P2"),
            ("Glyceraldehyde_3_phosphate", 170.06, "C3H7O6P"),
            ("Dihydroxyacetone_phosphate", 170.06, "C3H7O6P"),
            ("1_3_Bisphosphoglycerate", 266.04, "C3H8O10P2"),
            ("3_Phosphoglycerate", 186.06, "C3H7O7P"),
            ("2_Phosphoglycerate", 186.06, "C3H7O7P"),
            ("Phosphoenolpyruvate", 168.04, "C3H5O6P"),
            
            # TCA cycle intermediates
            ("Pyruvate", 88.06, "C3H4O3"),
            ("Acetyl_CoA", 809.57, "C23H38N7O17P3S"),
            ("Citrate", 192.12, "C6H8O7"),
            ("Isocitrate", 192.12, "C6H8O7"),
            ("α_Ketoglutarate", 146.11, "C5H6O5"),
            ("Succinyl_CoA", 867.61, "C25H40N7O19P3S"),
            ("Succinate", 118.09, "C4H6O4"),
            ("Fumarate", 116.07, "C4H4O4"),
            ("Malate", 134.09, "C4H6O5"),
            ("Oxaloacetate", 132.07, "C4H4O5"),
            
            # Amino acids
            ("Glutamate", 147.13, "C5H9NO4"),
            ("Glutamine", 146.14, "C5H10N2O3"),
            ("Aspartate", 133.10, "C4H7NO4"),
            ("Asparagine", 132.12, "C4H8N2O3"),
            ("Alanine", 89.09, "C3H7NO2"),
            ("Glycine", 75.07, "C2H5NO2"),
            ("Serine", 105.09, "C3H7NO3"),
            ("Threonine", 119.12, "C4H9NO3"),
            ("Cysteine", 121.16, "C3H7NO2S"),
            ("Methionine", 149.21, "C5H11NO2S"),
            ("Valine", 117.15, "C5H11NO2"),
            ("Leucine", 131.17, "C6H13NO2"),
            ("Isoleucine", 131.17, "C6H13NO2"),
            ("Lysine", 146.19, "C6H14N2O2"),
            ("Arginine", 174.20, "C6H14N4O2"),
            ("Histidine", 155.15, "C6H9N3O2"),
            ("Phenylalanine", 165.19, "C9H11NO2"),
            ("Tyrosine", 181.19, "C9H11NO3"),
            ("Tryptophan", 204.23, "C11H12N2O2"),
            ("Proline", 115.13, "C5H9NO2"),
            
            # Nucleotides and cofactors
            ("ATP", 507.18, "C10H16N5O13P3"),
            ("ADP", 427.20, "C10H15N5O10P2"),
            ("AMP", 347.22, "C10H14N5O7P"),
            ("GTP", 523.18, "C10H16N5O14P3"),
            ("GDP", 443.20, "C10H15N5O11P2"),
            ("NADplus", 663.43, "C21H27N7O14P2"),
            ("NADH", 665.44, "C21H29N7O14P2"),
            ("NADPplus", 743.41, "C21H28N7O17P3"),
            ("NADPH", 745.42, "C21H30N7O17P3"),
            ("FAD", 785.55, "C27H33N9O15P2"),
            ("FADH2", 787.54, "C27H35N9O15P2"),
            ("Coenzyme_A", 767.53, "C21H36N7O16P3S"),
            ("S_Adenosylmethionine", 398.44, "C15H22N6O5S"),
            ("Tetrahydrofolate", 445.43, "C19H23N7O6"),
            
            # Add more substrates to reach ~100
            ("Biotin", 244.31, "C10H16N2O3S"),
            ("Thiamine_pyrophosphate", 425.31, "C12H19N4O7P2S"),
            ("Pyridoxal_phosphate", 247.14, "C8H10NO6P"),
            ("Riboflavin", 376.36, "C17H20N4O6"),
            ("Pantothenic_acid", 219.23, "C9H17NO5"),
            ("Nicotinic_acid", 123.11, "C6H5NO2"),
            ("Folic_acid", 441.40, "C19H19N7O6"),
            ("Vitamin_B12", 1355.37, "C63H88CoN14O14P"),
            ("Ascorbic_acid", 176.12, "C6H8O6"),
            ("α_Tocopherol", 430.71, "C29H50O2"),
            
            # Lipids
            ("Palmitic_acid", 256.42, "C16H32O2"),
            ("Stearic_acid", 284.48, "C18H36O2"),
            ("Oleic_acid", 282.46, "C18H34O2"),
            ("Linoleic_acid", 280.45, "C18H32O2"),
            ("Arachidonic_acid", 304.47, "C20H32O2"),
            ("Cholesterol", 386.65, "C27H46O"),
            ("Phosphatidylcholine", 760.08, "C40H80NO8P"),
            ("Phosphatidylserine", 747.04, "C38H74NO10P"),
            ("Sphingomyelin", 703.03, "C39H79N2O6P"),
            ("Ceramide", 537.86, "C34H67NO3"),
            ("Diacylglycerol", 594.82, "C35H66O5"),
            
            # Signaling molecules
            ("Inositol_1_4_5_trisphosphate", 420.10, "C6H15O15P3"),
            ("cAMP", 329.21, "C10H12N5O6P"),
            ("cGMP", 345.21, "C10H12N5O7P"),
            
            # Ions
            ("Calcium_ions", 40.08, "Ca2+"),
            ("Magnesium_ions", 24.31, "Mg2+"),
            ("Iron_ions", 55.85, "Fe2+/Fe3+"),
            ("Zinc_ions", 65.38, "Zn2+"),
            ("Copper_ions", 63.55, "Cu2+"),
            ("Manganese_ions", 54.94, "Mn2+"),
            
            # Neurotransmitters
            ("Dopamine", 153.18, "C8H11NO2"),
            ("Serotonin", 176.21, "C10H12N2O"),
            ("GABA", 103.12, "C4H9NO2"),
            ("Acetylcholine", 146.21, "C7H16NO2"),
            ("Epinephrine", 183.20, "C9H13NO3"),
            ("Norepinephrine", 169.18, "C8H11NO3"),
            ("Histamine", 111.15, "C5H9N3"),
            ("Melatonin", 232.28, "C13H16N2O2"),
            
            # Additional metabolites
            ("Lactate", 90.08, "C3H6O3"),
            ("Uric_acid", 168.11, "C5H4N4O3"),
            ("Creatine", 131.13, "C4H9N3O2"),
            ("Creatinine", 113.12, "C4H7N3O"),
            ("Urea", 60.06, "CH4N2O"),
            ("Ammonia", 17.03, "NH3"),
            ("Nitric_oxide", 30.01, "NO"),
            ("Carbon_monoxide", 28.01, "CO"),
            ("Hydrogen_sulfide", 34.08, "H2S"),
            ("Hydrogen_peroxide", 34.01, "H2O2"),
            ("Superoxide", 32.00, "O2-"),
            ("Glutathione", 307.32, "C10H17N3O6S"),
            ("Oxidized_glutathione", 612.63, "C20H32N6O12S2"),
        ]
        
        for substrate_name, mw, formula in key_substrates:
            entity = BiologicalEntity(
                id=f"substrate_{substrate_name.replace(' ', '_').replace('-', '_').replace(',', '_').replace('+', 'plus')}",
                name=substrate_name.replace('_', ' '),
                entity_type="substrate",
                properties={
                    "molecular_weight": mw,
                    "formula": formula,
                    "source": "kg_builder"
                },
                confidence_score=0.95
            )
            self.kg.add_entity(entity)
    
    def _add_all_products(self):
        """Add all 50 products as specified"""
        key_products = [
            ("Glucose_1_phosphate", 260.14, "C6H13O9P"),
            ("6_Phosphogluconate", 276.14, "C6H13O10P"),
            ("Ribulose_5_phosphate", 230.11, "C5H11O8P"),
            ("Ribose_5_phosphate", 230.11, "C5H11O8P"),
            ("Xylulose_5_phosphate", 230.11, "C5H11O8P"),
            ("Sedoheptulose_7_phosphate", 290.16, "C7H15O10P"),
            ("Erythrose_4_phosphate", 200.08, "C4H9O7P"),
            ("CO2", 44.01, "CO2"),
            ("H2O", 18.02, "H2O"),
            ("Pi", 95.98, "PO4^3-"),
            ("PPi", 174.95, "P2O7^4-"),
            ("O2", 32.00, "O2"),
            ("Reduced_glutathione", 307.32, "C10H17N3O6S"),
            ("Citrulline", 175.19, "C6H13N3O3"),
            ("Ornithine", 132.16, "C5H12N2O2"),
            ("Homocysteine", 135.19, "C4H9NO2S"),
            ("Cystathionine", 222.26, "C7H14N2O4S"),
            ("α_Ketobutyrate", 102.09, "C4H6O3"),
            ("Succinyl_homoserine", 219.19, "C8H13NO6"),
            ("Glycerol", 92.09, "C3H8O3"),
            ("Acetoacetate", 102.09, "C4H6O3"),
            ("β_Hydroxybutyrate", 104.10, "C4H8O3"),
            ("Malonyl_CoA", 853.58, "C24H38N7O19P3S"),
            ("Acyl_CoA", 809.57, "Variable"),
            ("Phosphatidic_acid", 674.96, "C35H67O8P"),
            ("Inosine", 268.23, "C10H12N4O5"),
            ("Hypoxanthine", 136.11, "C5H4N4O"),
            ("Xanthine", 152.11, "C5H4N4O2"),
            ("dATP", 491.18, "C10H16N5O12P3"),
            ("dGTP", 507.18, "C10H16N5O13P3"),
            ("dCTP", 467.16, "C9H16N3O13P3"),
            ("dTTP", 482.17, "C10H17N2O14P3"),
            ("IP3", 420.10, "C6H15O15P3"),
            ("DAG", 594.82, "C35H66O5"),
            ("Arachidonic_acid", 304.47, "C20H32O2"),
            ("Prostaglandins", 352.47, "Variable"),
            ("Leukotrienes", 336.47, "Variable"),
            ("DOPAC", 168.15, "C8H8O4"),
            ("HVA", 182.17, "C9H10O4"),
            ("5_HIAA", 191.18, "C10H9NO3"),
            ("VMA", 198.17, "C9H12O5"),
            ("Bilirubin", 584.66, "C33H36N4O6"),
            ("Biliverdin", 582.65, "C33H34N4O6"),
            ("Heme", 616.49, "C34H32FeN4O4"),
            ("Methylglyoxal", 72.06, "C3H4O2"),
            ("D_Lactate", 90.08, "C3H6O3"),
            ("Acetate", 60.05, "C2H4O2"),
            ("HMG_CoA", 911.66, "C27H44O17P2S"),
            ("Mevalonate", 148.16, "C6H12O4"),
            ("Palmitoyl_carnitine", 399.61, "C23H45NO4"),
            ("3_Hydroxyacyl_CoA", 853.62, "Variable"),
        ]
        
        for product_name, mw, formula in key_products:
            entity = BiologicalEntity(
                id=f"product_{product_name.replace(' ', '_').replace('-', '_')}",
                name=product_name.replace('_', ' '),
                entity_type="product",
                properties={
                    "molecular_weight": mw,
                    "formula": formula,
                    "source": "kg_builder"
                },
                confidence_score=0.93
            )
            self.kg.add_entity(entity)
    
    def _add_all_inhibitors(self):
        """Add all 30 inhibitors"""
        inhibitors = [
            ("Metformin", 129.16, "complex_I_inhibitor", ["enzyme_COMPLEX_I"]),
            ("2_Deoxyglucose", 164.16, "hexokinase_inhibitor", ["enzyme_HK1", "enzyme_HK2"]),
            ("Dichloroacetate", 128.94, "PDK_inhibitor", ["enzyme_PDK"]),
            ("Oxamate", 89.05, "LDH_inhibitor", ["enzyme_LDH"]),
            ("CB_839", 571.5, "glutaminase_inhibitor", ["enzyme_GLS", "enzyme_GLS2"]),
            ("Etomoxir", 320.77, "CPT1_inhibitor", ["enzyme_CPT1A"]),
            ("Orlistat", 495.73, "FASN_inhibitor", ["enzyme_FASN"]),
            ("Allopurinol", 136.11, "xanthine_oxidase_inhibitor", ["enzyme_XOD"]),
            ("Disulfiram", 296.52, "ALDH_inhibitor", ["enzyme_ALDH2"]),
            ("Valproic_acid", 144.21, "HDAC_inhibitor", ["enzyme_HDAC"]),
            ("Vorinostat", 264.32, "HDAC_inhibitor", ["enzyme_HDAC"]),
            ("Azacitidine", 244.20, "DNMT_inhibitor", ["enzyme_DNMT"]),
            ("Tranylcypromine", 133.19, "MAO_inhibitor", ["enzyme_MAO"]),
            ("Selegiline", 187.28, "MAO_B_inhibitor", ["enzyme_MAO_B"]),
            ("Entacapone", 305.29, "COMT_inhibitor", ["enzyme_COMT"]),
            ("Tolcapone", 273.24, "COMT_inhibitor", ["enzyme_COMT"]),
            ("Mycophenolic_acid", 320.34, "IMPDH_inhibitor", ["enzyme_IMPDH"]),
            ("Ribavirin", 244.20, "IMPDH_inhibitor", ["enzyme_IMPDH"]),
            ("6_Mercaptopurine", 152.18, "purine_synthesis_inhibitor", ["enzyme_HGPRT"]),
            ("Rotenone", 394.42, "complex_I_inhibitor", ["enzyme_COMPLEX_I"]),
            ("Antimycin_A", 548.63, "complex_III_inhibitor", ["enzyme_COMPLEX_III"]),
            ("Oligomycin", 791.06, "ATP_synthase_inhibitor", ["enzyme_ATP_SYNTHASE"]),
            ("3_Bromopyruvate", 166.96, "GAPDH_inhibitor", ["enzyme_GAPDH"]),
            ("FX11", 295.66, "LDHA_inhibitor", ["enzyme_LDHA"]),
            ("BPTES", 533.06, "GLS1_inhibitor", ["enzyme_GLS"]),
            ("AOA", 133.10, "transaminase_inhibitor", ["enzyme_GOT1", "enzyme_GOT2", "enzyme_GPT"]),
            ("Phloretin", 274.27, "glucose_transporter_inhibitor", ["transporter_GLUT"]),
            ("BAY_876", 494.88, "GLUT1_inhibitor", ["transporter_GLUT1"]),
            ("UK5099", 407.40, "MPC_inhibitor", ["transporter_MPC"]),
            ("TOFA", 213.15, "ACC_inhibitor", ["enzyme_ACC1", "enzyme_ACACA"]),
        ]
        
        for inhibitor_name, mw, mechanism, targets in inhibitors:
            entity = BiologicalEntity(
                id=f"inhibitor_{inhibitor_name.replace(' ', '_').replace('-', '_')}",
                name=inhibitor_name.replace('_', ' '),
                entity_type="inhibitor",
                properties={
                    "molecular_weight": mw,
                    "mechanism": mechanism,
                    "targets": targets,
                    "source": "kg_builder"
                },
                confidence_score=0.91
            )
            self.kg.add_entity(entity)
    
    def _add_all_allosteric_regulators(self):
        """Add all 20 allosteric regulators"""
        regulators = [
            ("AMP", 347.22, "activator", ["enzyme_PFK1", "enzyme_AMPK"]),
            ("ADP", 427.20, "activator", ["enzyme_PFK1"]),
            ("ATP", 507.18, "inhibitor", ["enzyme_PFK1", "enzyme_CS"]),
            ("Citrate", 192.12, "mixed", ["enzyme_PFK1", "enzyme_ACC1"]),
            ("Fructose_2,6_bisphosphate", 340.12, "activator", ["enzyme_PFK1"]),
            ("Acetyl_CoA", 809.57, "activator", ["enzyme_PC"]),
            ("Malonyl_CoA", 853.58, "inhibitor", ["enzyme_CPT1A"]),
            ("Palmitate", 256.42, "inhibitor", ["enzyme_ACC1"]),
            ("cAMP", 329.21, "activator", ["enzyme_PKA"]),
            ("Ca2plus", 40.08, "activator", ["multiple_enzymes"]),
            ("Phosphoenolpyruvate", 168.04, "inhibitor", ["enzyme_PFK1"]),
            ("Glucose_6_phosphate", 260.14, "activator", ["enzyme_GS"]),
            ("NADH", 665.44, "inhibitor", ["multiple_dehydrogenases"]),
            ("Succinyl_CoA", 867.61, "inhibitor", ["enzyme_CS"]),
            ("GTP", 523.18, "activator", ["enzyme_PEPCK"]),
            ("IMP", 348.21, "feedback_inhibitor", ["enzyme_PRPP"]),
            ("CTP", 483.16, "inhibitor", ["enzyme_ATC"]),
            ("UTP", 484.14, "activator", ["enzyme_CPS2"]),
            ("S_Adenosylhomocysteine", 384.41, "inhibitor", ["methyltransferases"]),
            ("CoQ10", 863.34, "modulator", ["electron_transport"]),
        ]
        
        for regulator_name, mw, effect_type, targets in regulators:
            entity = BiologicalEntity(
                id=f"regulator_{regulator_name.replace(' ', '_').replace('-', '_').replace(',', '_').replace('+', 'plus')}",
                name=regulator_name.replace('_', ' '),
                entity_type="allosteric_regulator",
                properties={
                    "molecular_weight": mw,
                    "effect_type": effect_type,
                    "targets": targets,
                    "source": "kg_builder"
                },
                confidence_score=0.90
            )
            self.kg.add_entity(entity)
    
    def build_multiscale_entities(self):
        """Add multi-scale entities"""
        logger.info("Building multi-scale entities...")
        
        # Add 12 receptors with proper names
        receptors = [
            ("EGFR", "Epidermal growth factor receptor"),
            ("HER2", "Human epidermal growth factor receptor 2"),
            ("VEGFR1", "Vascular endothelial growth factor receptor 1"),
            ("VEGFR2", "Vascular endothelial growth factor receptor 2"),
            ("PDGFRA", "Platelet-derived growth factor receptor alpha"),
            ("PDGFRB", "Platelet-derived growth factor receptor beta"),
            ("FGFR1", "Fibroblast growth factor receptor 1"),
            ("IGF1R", "Insulin-like growth factor 1 receptor"),
            ("INSR", "Insulin receptor"),
            ("MET", "Hepatocyte growth factor receptor"),
            ("KIT", "Stem cell factor receptor"),
            ("ALK", "Anaplastic lymphoma kinase"),
        ]
        
        for receptor_id, name in receptors:
            entity = BiologicalEntity(
                id=f"receptor_{receptor_id}",
                name=name,
                entity_type="receptor",
                properties={
                    "receptor_type": "tyrosine_kinase",
                    "localization": "plasma_membrane",
                    "source": "kg_builder"
                },
                confidence_score=0.92
            )
            self.kg.add_entity(entity)
        
        # Add 12 ligands
        ligands = [
            ("EGF", "Epidermal growth factor"),
            ("TGFα", "Transforming growth factor alpha"),
            ("VEGF_A", "Vascular endothelial growth factor A"),
            ("VEGF_B", "Vascular endothelial growth factor B"),
            ("PDGF_AA", "Platelet-derived growth factor AA"),
            ("PDGF_BB", "Platelet-derived growth factor BB"),
            ("FGF1", "Fibroblast growth factor 1"),
            ("FGF2", "Fibroblast growth factor 2"),
            ("IGF1", "Insulin-like growth factor 1"),
            ("Insulin", "Insulin"),
            ("HGF", "Hepatocyte growth factor"),
            ("SCF", "Stem cell factor"),
        ]
        
        for ligand_id, name in ligands:
            entity = BiologicalEntity(
                id=f"ligand_{ligand_id}",
                name=name,
                entity_type="ligand",
                properties={
                    "ligand_type": "growth_factor",
                    "source": "kg_builder"
                },
                confidence_score=0.91
            )
            self.kg.add_entity(entity)
        
        # Add 11 signaling proteins
        signaling_proteins = [
            ("RAS", "RAS GTPase"),
            ("RAF1", "RAF proto-oncogene serine/threonine kinase"),
            ("BRAF", "B-Raf proto-oncogene"),
            ("MEK1", "Mitogen-activated protein kinase kinase 1"),
            ("MEK2", "Mitogen-activated protein kinase kinase 2"),
            ("ERK1", "Extracellular signal-regulated kinase 1"),
            ("ERK2", "Extracellular signal-regulated kinase 2"),
            ("PI3K", "Phosphoinositide 3-kinase"),
            ("AKT1", "AKT serine/threonine kinase 1"),
            ("mTOR", "Mechanistic target of rapamycin"),
            ("PTEN", "Phosphatase and tensin homolog"),
        ]
        
        for protein_id, name in signaling_proteins:
            entity = BiologicalEntity(
                id=f"signaling_{protein_id}",
                name=name,
                entity_type="signaling_protein",
                properties={
                    "pathway": "growth_signaling",
                    "source": "kg_builder"
                },
                confidence_score=0.90
            )
            self.kg.add_entity(entity)
        
        # Add compartments
        compartments = [
            ("plasma_membrane", "Plasma membrane"),
            ("cytoplasm", "Cytoplasm"),
            ("nucleus", "Nucleus"),
            ("mitochondria", "Mitochondria"),
            ("er", "Endoplasmic reticulum"),
            ("golgi", "Golgi apparatus"),
            ("lysosomes", "Lysosomes"),
            ("extracellular", "Extracellular space"),
        ]
        
        for comp_id, name in compartments:
            entity = BiologicalEntity(
                id=f"compartment_{comp_id}",
                name=name,
                entity_type="compartment",
                properties={
                    "cellular_location": True,
                    "source": "kg_builder"
                },
                confidence_score=0.95
            )
            self.kg.add_entity(entity)
        
        # Add cell types
        cell_types = [
            ("epithelial", "Epithelial cells"),
            ("endothelial", "Endothelial cells"),
            ("fibroblast", "Fibroblasts"),
            ("tcell", "T cells"),
            ("smooth_muscle", "Smooth muscle cells"),
        ]
        
        for cell_id, name in cell_types:
            entity = BiologicalEntity(
                id=f"celltype_{cell_id}",
                name=name,
                entity_type="cell_type",
                properties={
                    "tissue_type": "various",
                    "source": "kg_builder"
                },
                confidence_score=0.89
            )
            self.kg.add_entity(entity)
        
        logger.info(f"Added multi-scale entities: {len([e for e in self.kg.entities.values() if e.entity_type in ['receptor', 'ligand', 'signaling_protein', 'compartment', 'cell_type']])} entities")
    
    def build_disease_state_entities(self):
        """Add disease-state entities"""
        logger.info("Building disease-state entities...")
        
        # Add diseases
        diseases = [
            ("breast_cancer", "Breast cancer"),
            ("lung_cancer", "Lung cancer"),
            ("colorectal_cancer", "Colorectal cancer"),
            ("pancreatic_cancer", "Pancreatic cancer"),
            ("prostate_cancer", "Prostate cancer"),
            ("type2_diabetes", "Type 2 diabetes"),
            ("metabolic_syndrome", "Metabolic syndrome"),
            ("atherosclerosis", "Atherosclerosis"),
            ("heart_failure", "Heart failure"),
            ("alzheimers", "Alzheimer's disease"),
            ("parkinsons", "Parkinson's disease"),
            ("rheumatoid_arthritis", "Rheumatoid arthritis"),
        ]
        
        for disease_id, name in diseases:
            entity = BiologicalEntity(
                id=f"disease_{disease_id}",
                name=name,
                entity_type="disease",
                properties={
                    "disease_category": "chronic",
                    "source": "kg_builder"
                },
                confidence_score=0.88
            )
            self.kg.add_entity(entity)
        
        # Add biomarkers
        biomarkers = [
            ("CA15_3", "Cancer antigen 15-3"),
            ("CA125", "Cancer antigen 125"),
            ("CEA", "Carcinoembryonic antigen"),
            ("PSA", "Prostate-specific antigen"),
            ("HER2", "Human epidermal growth factor receptor 2"),
            ("PD_L1", "Programmed death-ligand 1"),
            ("BRCA1", "Breast cancer gene 1"),
            ("BRCA2", "Breast cancer gene 2"),
            ("CRP", "C-reactive protein"),
            ("IL_6", "Interleukin-6"),
            ("TNF_α", "Tumor necrosis factor alpha"),
            ("HbA1c", "Glycated hemoglobin"),
            ("Glucose", "Blood glucose"),
            ("LDL", "LDL cholesterol"),
            ("Troponin_I", "Troponin I"),
            ("BNP", "Brain natriuretic peptide"),
        ]
        
        for biomarker_id, name in biomarkers:
            entity = BiologicalEntity(
                id=f"biomarker_{biomarker_id}",
                name=name,
                entity_type="biomarker",
                properties={
                    "diagnostic_use": True,
                    "source": "kg_builder"
                },
                confidence_score=0.87
            )
            self.kg.add_entity(entity)
        
        # Add disease-associated proteins
        disease_proteins = [
            ("TP53", "Tumor protein p53"),
            ("MYC", "MYC proto-oncogene"),
            ("RAS", "RAS proteins"),
            ("PI3K", "Phosphoinositide 3-kinase"),
            ("PTEN", "Phosphatase and tensin homolog"),
            ("APC", "Adenomatous polyposis coli"),
            ("VHL", "Von Hippel-Lindau"),
            ("RB1", "Retinoblastoma protein"),
            ("CDKN2A", "Cyclin-dependent kinase inhibitor 2A"),
            ("BCL2", "B-cell lymphoma 2"),
            ("BAX", "BCL2 associated X"),
            ("NF_κB", "Nuclear factor kappa B"),
            ("HIF_1α", "Hypoxia-inducible factor 1-alpha"),
            ("AMPK", "AMP-activated protein kinase"),
        ]
        
        for protein_id, name in disease_proteins:
            entity = BiologicalEntity(
                id=f"disease_protein_{protein_id}",
                name=name,
                entity_type="disease_protein",
                properties={
                    "oncogene_or_suppressor": True,
                    "source": "kg_builder"
                },
                confidence_score=0.86
            )
            self.kg.add_entity(entity)
        
        # Add phenotypes
        phenotypes = [
            ("proliferation", "Cell proliferation"),
            ("apoptosis_resistance", "Apoptosis resistance"),
            ("angiogenesis", "Angiogenesis"),
            ("metastasis", "Metastasis"),
            ("drug_resistance", "Drug resistance"),
            ("metabolic_reprogramming", "Metabolic reprogramming"),
            ("immune_evasion", "Immune evasion"),
            ("senescence", "Senescence"),
            ("autophagy", "Autophagy"),
            ("EMT", "Epithelial-mesenchymal transition"),
            ("hypoxia_response", "Hypoxia response"),
            ("inflammation", "Inflammation"),
            ("fibrosis", "Fibrosis"),
        ]
        
        for phenotype_id, name in phenotypes:
            entity = BiologicalEntity(
                id=f"phenotype_{phenotype_id}",
                name=name,
                entity_type="phenotype",
                properties={
                    "cellular_process": True,
                    "source": "kg_builder"
                },
                confidence_score=0.85
            )
            self.kg.add_entity(entity)
        
        logger.info(f"Added disease-state entities: {len([e for e in self.kg.entities.values() if e.entity_type in ['disease', 'biomarker', 'disease_protein', 'phenotype']])} entities")
    
    def build_drug_interaction_entities(self):
        """Add drug interaction entities"""
        logger.info("Building drug interaction entities...")
        
        # Add 30 drugs with ChEMBL/PubChem-like IDs
        drugs = [
            # Tyrosine Kinase Inhibitors
            ("DB00619", "Imatinib"),
            ("DB00530", "Erlotinib"),
            ("DB00317", "Gefitinib"),
            ("DB01259", "Lapatinib"),
            ("DB01268", "Sunitinib"),
            ("DB00398", "Sorafenib"),
            ("DB06589", "Pazopanib"),
            ("DB08865", "Crizotinib"),
            ("DB09330", "Osimertinib"),
            ("DB08875", "Cabozantinib"),
            
            # Common CYP Substrates/Inhibitors
            ("DB00641", "Simvastatin"),
            ("DB01076", "Atorvastatin"),
            ("DB00682", "Warfarin"),
            ("DB00758", "Clopidogrel"),
            ("DB00338", "Omeprazole"),
            ("DB00264", "Metoprolol"),
            ("DB00472", "Fluoxetine"),
            ("DB01026", "Ketoconazole"),
            ("DB00537", "Clarithromycin"),
            ("DB01045", "Rifampin"),
            
            # Chemotherapy Agents
            ("DB01248", "Paclitaxel"),
            ("DB01248", "Docetaxel"),
            ("DB00531", "Cyclophosphamide"),
            ("DB00675", "Tamoxifen"),
            ("DB00997", "Doxorubicin"),
            ("DB00544", "5-Fluorouracil"),
            ("DB00563", "Methotrexate"),
            ("DB00541", "Vincristine"),
            ("DB00515", "Cisplatin"),
            ("DB00441", "Gemcitabine"),
        ]
        
        for drug_id, name in drugs:
            entity = BiologicalEntity(
                id=f"drug_{drug_id}",
                name=name,
                entity_type="drug",
                properties={
                    "drug_class": "various",
                    "source": "kg_builder"
                },
                confidence_score=0.91
            )
            self.kg.add_entity(entity)
        
        # CYP enzymes were already added in enzyme_kinetics section
        
        logger.info(f"Added drug interaction entities: {len([e for e in self.kg.entities.values() if e.entity_type == 'drug'])} drugs")
    
    def build_comprehensive_relationships(self):
        """Build comprehensive relationships between all entities"""
        logger.info("Building comprehensive relationships...")
        
        # Track relationship counts
        rel_counts = {
            "enzyme_substrate": 0,
            "enzyme_product": 0,
            "inhibition": 0,
            "allosteric": 0,
            "binding": 0,
            "phosphorylation": 0,
            "location": 0,
            "biomarker": 0,
            "disease_cause": 0,
            "drug_target": 0,
            "drug_metabolism": 0,
            "drug_drug": 0,
            "treats": 0
        }
        
        # 1. Comprehensive enzyme-substrate relationships (should be ~100+)
        enzyme_substrate_mappings = [
            # Glycolysis pathway - complete
            ("enzyme_HK1", ["substrate_Glucose", "substrate_ATP"], ["product_Glucose_6_phosphate", "substrate_ADP"]),
            ("enzyme_HK2", ["substrate_Glucose", "substrate_ATP"], ["product_Glucose_6_phosphate", "substrate_ADP"]),
            ("enzyme_GPI", ["substrate_Glucose_6_phosphate"], ["substrate_Fructose_6_phosphate"]),
            ("enzyme_PFK1", ["substrate_Fructose_6_phosphate", "substrate_ATP"], ["substrate_Fructose_1_6_bisphosphate", "substrate_ADP"]),
            ("enzyme_ALDOA", ["substrate_Fructose_1_6_bisphosphate"], ["substrate_Glyceraldehyde_3_phosphate", "substrate_Dihydroxyacetone_phosphate"]),
            ("enzyme_TPI1", ["substrate_Dihydroxyacetone_phosphate"], ["substrate_Glyceraldehyde_3_phosphate"]),
            ("enzyme_GAPDH", ["substrate_Glyceraldehyde_3_phosphate", "substrate_NADplus", "product_Pi"], ["substrate_1_3_Bisphosphoglycerate", "substrate_NADH"]),
            ("enzyme_PGK1", ["substrate_1_3_Bisphosphoglycerate", "substrate_ADP"], ["substrate_3_Phosphoglycerate", "substrate_ATP"]),
            ("enzyme_ENO1", ["substrate_2_Phosphoglycerate"], ["substrate_Phosphoenolpyruvate", "product_H2O"]),
            ("enzyme_PKM", ["substrate_Phosphoenolpyruvate", "substrate_ADP"], ["substrate_Pyruvate", "substrate_ATP"]),
            
            # TCA cycle - complete
            ("enzyme_CS", ["substrate_Acetyl_CoA", "substrate_Oxaloacetate"], ["substrate_Citrate", "substrate_Coenzyme_A"]),
            ("enzyme_ACO2", ["substrate_Citrate"], ["substrate_Isocitrate"]),
            ("enzyme_IDH1", ["substrate_Isocitrate", "substrate_NADplus"], ["substrate_α_Ketoglutarate", "substrate_NADH", "product_CO2"]),
            ("enzyme_IDH2", ["substrate_Isocitrate", "substrate_NADPplus"], ["substrate_α_Ketoglutarate", "substrate_NADPH", "product_CO2"]),
            ("enzyme_OGDH", ["substrate_α_Ketoglutarate", "substrate_NADplus", "substrate_Coenzyme_A"], ["substrate_Succinyl_CoA", "substrate_NADH", "product_CO2"]),
            ("enzyme_SUCLA2", ["substrate_Succinyl_CoA", "substrate_GDP", "product_Pi"], ["substrate_Succinate", "substrate_GTP", "substrate_Coenzyme_A"]),
            ("enzyme_SDH", ["substrate_Succinate", "substrate_FAD"], ["substrate_Fumarate", "substrate_FADH2"]),
            ("enzyme_FH", ["substrate_Fumarate", "product_H2O"], ["substrate_Malate"]),
            ("enzyme_MDH2", ["substrate_Malate", "substrate_NADplus"], ["substrate_Oxaloacetate", "substrate_NADH"]),
            
            # Amino acid metabolism
            ("enzyme_GOT1", ["substrate_Aspartate", "substrate_α_Ketoglutarate"], ["substrate_Oxaloacetate", "substrate_Glutamate"]),
            ("enzyme_GOT2", ["substrate_Aspartate", "substrate_α_Ketoglutarate"], ["substrate_Oxaloacetate", "substrate_Glutamate"]),
            ("enzyme_GPT", ["substrate_Alanine", "substrate_α_Ketoglutarate"], ["substrate_Pyruvate", "substrate_Glutamate"]),
            ("enzyme_GLS", ["substrate_Glutamine"], ["substrate_Glutamate", "product_Ammonia"]),
            ("enzyme_GLS2", ["substrate_Glutamine"], ["substrate_Glutamate", "product_Ammonia"]),
            ("enzyme_GLUD1", ["substrate_Glutamate", "substrate_NADplus"], ["substrate_α_Ketoglutarate", "substrate_NADH", "product_Ammonia"]),
            ("enzyme_ASS1", ["product_Citrulline", "substrate_Aspartate", "substrate_ATP"], ["product_Argininosuccinate", "substrate_AMP", "product_PPi"]),
            ("enzyme_ASL", ["product_Argininosuccinate"], ["substrate_Arginine", "substrate_Fumarate"]),
            
            # Lipid metabolism
            ("enzyme_FASN", ["substrate_Acetyl_CoA", "product_Malonyl_CoA", "substrate_NADPH"], ["substrate_Palmitic_acid", "substrate_NADPplus", "product_CO2"]),
            ("enzyme_ACC1", ["substrate_Acetyl_CoA", "substrate_ATP", "product_CO2"], ["product_Malonyl_CoA", "substrate_ADP", "product_Pi"]),
            ("enzyme_HMGCR", ["product_HMG_CoA", "substrate_NADPH"], ["product_Mevalonate", "substrate_NADPplus"]),
            ("enzyme_CPT1A", ["substrate_Palmitic_acid", "substrate_Carnitine"], ["product_Palmitoyl_carnitine", "substrate_Coenzyme_A"]),
            ("enzyme_HADHA", ["product_3_Hydroxyacyl_CoA", "substrate_NADplus"], ["product_3_Ketoacyl_CoA", "substrate_NADH"]),
            ("enzyme_ACACA", ["substrate_Acetyl_CoA", "substrate_ATP", "product_CO2"], ["product_Malonyl_CoA", "substrate_ADP", "product_Pi"]),
            
            # Oxidoreductases
            ("enzyme_CAT", ["substrate_Hydrogen_peroxide"], ["product_H2O", "product_O2"]),
            ("enzyme_SOD1", ["substrate_Superoxide"], ["substrate_Hydrogen_peroxide", "product_O2"]),
            ("enzyme_SOD2", ["substrate_Superoxide"], ["substrate_Hydrogen_peroxide", "product_O2"]),
            ("enzyme_GPX1", ["substrate_Hydrogen_peroxide", "substrate_Glutathione"], ["product_H2O", "substrate_Oxidized_glutathione"]),
            ("enzyme_PRDX1", ["substrate_Hydrogen_peroxide"], ["product_H2O"]),
            ("enzyme_NQO1", ["substrate_NADPH", "product_Quinone"], ["substrate_NADPplus", "product_Hydroquinone"]),
            ("enzyme_G6PD", ["substrate_Glucose_6_phosphate", "substrate_NADPplus"], ["product_6_Phosphogluconate", "substrate_NADPH"]),
            ("enzyme_ALDH2", ["product_Acetaldehyde", "substrate_NADplus"], ["product_Acetate", "substrate_NADH"]),
            
            # Transferases
            ("enzyme_GST", ["substrate_Glutathione"], ["product_Glutathione_conjugate"]),
            ("enzyme_UGT1A1", ["product_UDP_glucuronate"], ["product_Glucuronide"]),
            ("enzyme_COMT", ["substrate_S_Adenosylmethionine", "product_Catechol"], ["product_S_Adenosylhomocysteine", "product_O_Methylcatechol"]),
            ("enzyme_TPMT", ["substrate_S_Adenosylmethionine", "product_Thiopurine"], ["product_S_Adenosylhomocysteine", "product_Methylthiopurine"]),
            ("enzyme_NAT1", ["substrate_Acetyl_CoA"], ["product_Acetylated_product", "substrate_Coenzyme_A"]),
            ("enzyme_NAT2", ["substrate_Acetyl_CoA"], ["product_Acetylated_product", "substrate_Coenzyme_A"]),
            ("enzyme_SULT1A1", ["product_PAPS"], ["product_Sulfated_product", "product_PAP"]),
            ("enzyme_AANAT", ["substrate_Serotonin", "substrate_Acetyl_CoA"], ["product_N_Acetylserotonin", "substrate_Coenzyme_A"]),
        ]
        
        for enzyme, substrates, products in enzyme_substrate_mappings:
            if self.kg.get_entity(enzyme):
                # Add substrate relationships
                for substrate in substrates:
                    if self.kg.get_entity(substrate):
                        rel = BiologicalRelationship(
                            source=enzyme,
                            target=substrate,
                            relation_type=RelationType.SUBSTRATE_OF,
                            properties={"reaction_type": "enzymatic"},
                            mathematical_constraints=["michaelis_menten"],
                            confidence_score=0.9
                        )
                        self.kg.add_relationship(rel)
                        rel_counts["enzyme_substrate"] += 1
                
                # Add product relationships
                for product in products:
                    if self.kg.get_entity(product):
                        rel = BiologicalRelationship(
                            source=enzyme,
                            target=product,
                            relation_type=RelationType.PRODUCT_OF,
                            properties={"reaction_type": "enzymatic"},
                            mathematical_constraints=[],
                            confidence_score=0.88
                        )
                        self.kg.add_relationship(rel)
                        rel_counts["enzyme_product"] += 1
        
        # 2. Inhibitor relationships (30 inhibitors × ~2 targets each = ~60)
        inhibitor_mappings = [
            ("inhibitor_Metformin", ["enzyme_COMPLEX_I"], RelationType.NON_COMPETITIVE_INHIBITION),
            ("inhibitor_2_Deoxyglucose", ["enzyme_HK1", "enzyme_HK2"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Dichloroacetate", ["enzyme_PDK"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Oxamate", ["enzyme_LDH"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_CB_839", ["enzyme_GLS", "enzyme_GLS2"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Etomoxir", ["enzyme_CPT1A"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Orlistat", ["enzyme_FASN"], RelationType.NON_COMPETITIVE_INHIBITION),
            ("inhibitor_Allopurinol", ["enzyme_XOD"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Disulfiram", ["enzyme_ALDH2"], RelationType.NON_COMPETITIVE_INHIBITION),
            ("inhibitor_Valproic_acid", ["enzyme_HDAC"], RelationType.NON_COMPETITIVE_INHIBITION),
            ("inhibitor_Vorinostat", ["enzyme_HDAC"], RelationType.NON_COMPETITIVE_INHIBITION),
            ("inhibitor_Azacitidine", ["enzyme_DNMT"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Tranylcypromine", ["enzyme_MAO"], RelationType.NON_COMPETITIVE_INHIBITION),
            ("inhibitor_Selegiline", ["enzyme_MAO_B"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Entacapone", ["enzyme_COMT"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Tolcapone", ["enzyme_COMT"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Mycophenolic_acid", ["enzyme_IMPDH"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Ribavirin", ["enzyme_IMPDH"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_6_Mercaptopurine", ["enzyme_HGPRT"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Rotenone", ["enzyme_COMPLEX_I"], RelationType.NON_COMPETITIVE_INHIBITION),
            ("inhibitor_Antimycin_A", ["enzyme_COMPLEX_III"], RelationType.NON_COMPETITIVE_INHIBITION),
            ("inhibitor_Oligomycin", ["enzyme_ATP_SYNTHASE"], RelationType.NON_COMPETITIVE_INHIBITION),
            ("inhibitor_3_Bromopyruvate", ["enzyme_GAPDH"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_FX11", ["enzyme_LDHA"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_BPTES", ["enzyme_GLS"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_AOA", ["enzyme_GOT1", "enzyme_GOT2", "enzyme_GPT"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_Phloretin", ["transporter_GLUT"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_BAY_876", ["transporter_GLUT1"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_UK5099", ["transporter_MPC"], RelationType.COMPETITIVE_INHIBITION),
            ("inhibitor_TOFA", ["enzyme_ACC1", "enzyme_ACACA"], RelationType.COMPETITIVE_INHIBITION),
        ]
        
        for inhibitor, targets, inhibition_type in inhibitor_mappings:
            if self.kg.get_entity(inhibitor):
                for target in targets:
                    # Create target if it doesn't exist
                    if not self.kg.get_entity(target):
                        entity = BiologicalEntity(
                            id=target,
                            name=target.replace('_', ' '),
                            entity_type="enzyme" if "enzyme" in target else "transporter",
                            properties={"inhibition_target": True},
                            confidence_score=0.85
                        )
                        self.kg.add_entity(entity)
                    
                    rel = BiologicalRelationship(
                        source=inhibitor,
                        target=target,
                        relation_type=inhibition_type,
                        properties={"inhibition_mechanism": str(inhibition_type)},
                        mathematical_constraints=["competitive_mm"] if "COMPETITIVE" in str(inhibition_type) else ["non_competitive_mm"],
                        confidence_score=0.87
                    )
                    self.kg.add_relationship(rel)
                    rel_counts["inhibition"] += 1
        
        # 3. Allosteric regulation relationships (20 regulators × ~2 targets = ~40)
        allosteric_mappings = [
            ("regulator_AMP", ["enzyme_PFK1", "enzyme_AMPK"], RelationType.ALLOSTERIC_REGULATION, "activator"),
            ("regulator_ADP", ["enzyme_PFK1"], RelationType.ALLOSTERIC_REGULATION, "activator"),
            ("regulator_ATP", ["enzyme_PFK1", "enzyme_CS"], RelationType.ALLOSTERIC_REGULATION, "inhibitor"),
            ("regulator_Citrate", ["enzyme_PFK1", "enzyme_ACC1"], RelationType.ALLOSTERIC_REGULATION, "mixed"),
            ("regulator_Fructose_2_6_bisphosphate", ["enzyme_PFK1"], RelationType.ALLOSTERIC_REGULATION, "activator"),
            ("regulator_Acetyl_CoA", ["enzyme_PC"], RelationType.ALLOSTERIC_REGULATION, "activator"),
            ("regulator_Malonyl_CoA", ["enzyme_CPT1A"], RelationType.ALLOSTERIC_REGULATION, "inhibitor"),
            ("regulator_Palmitate", ["enzyme_ACC1"], RelationType.ALLOSTERIC_REGULATION, "inhibitor"),
            ("regulator_cAMP", ["enzyme_PKA"], RelationType.ALLOSTERIC_REGULATION, "activator"),
            ("regulator_Ca2plus", ["enzyme_PKC", "enzyme_CAM"], RelationType.ALLOSTERIC_REGULATION, "activator"),
            ("regulator_Phosphoenolpyruvate", ["enzyme_PFK1"], RelationType.ALLOSTERIC_REGULATION, "inhibitor"),
            ("regulator_Glucose_6_phosphate", ["enzyme_GS"], RelationType.ALLOSTERIC_REGULATION, "activator"),
            ("regulator_NADH", ["enzyme_CS", "enzyme_IDH1", "enzyme_OGDH"], RelationType.ALLOSTERIC_REGULATION, "inhibitor"),
            ("regulator_Succinyl_CoA", ["enzyme_CS", "enzyme_OGDH"], RelationType.ALLOSTERIC_REGULATION, "inhibitor"),
            ("regulator_GTP", ["enzyme_PEPCK"], RelationType.ALLOSTERIC_REGULATION, "activator"),
            ("regulator_IMP", ["enzyme_PRPP"], RelationType.ALLOSTERIC_REGULATION, "inhibitor"),
            ("regulator_CTP", ["enzyme_ATC"], RelationType.ALLOSTERIC_REGULATION, "inhibitor"),
            ("regulator_UTP", ["enzyme_CPS2"], RelationType.ALLOSTERIC_REGULATION, "activator"),
            ("regulator_S_Adenosylhomocysteine", ["enzyme_MT"], RelationType.ALLOSTERIC_REGULATION, "inhibitor"),
            ("regulator_CoQ10", ["enzyme_COMPLEX_I", "enzyme_COMPLEX_III"], RelationType.ALLOSTERIC_REGULATION, "modulator"),
        ]
        
        for regulator, targets, rel_type, effect in allosteric_mappings:
            if self.kg.get_entity(regulator):
                for target in targets:
                    # Create target if it doesn't exist
                    if not self.kg.get_entity(target):
                        entity = BiologicalEntity(
                            id=target,
                            name=target.replace('_', ' '),
                            entity_type="enzyme",
                            properties={"allosteric_target": True},
                            confidence_score=0.85
                        )
                        self.kg.add_entity(entity)
                    
                    rel = BiologicalRelationship(
                        source=regulator,
                        target=target,
                        relation_type=rel_type,
                        properties={"allosteric_effect": effect},
                        mathematical_constraints=["allosteric_hill"],
                        confidence_score=0.86
                    )
                    self.kg.add_relationship(rel)
                    rel_counts["allosteric"] += 1
        
        # 4. Receptor-ligand binding (complete pairing)
        binding_pairs = [
            ("ligand_EGF", "receptor_EGFR"),
            ("ligand_TGFα", "receptor_EGFR"),
            ("ligand_VEGF_A", "receptor_VEGFR1"),
            ("ligand_VEGF_A", "receptor_VEGFR2"),
            ("ligand_VEGF_B", "receptor_VEGFR1"),
            ("ligand_PDGF_AA", "receptor_PDGFRA"),
            ("ligand_PDGF_BB", "receptor_PDGFRB"),
            ("ligand_FGF1", "receptor_FGFR1"),
            ("ligand_FGF2", "receptor_FGFR1"),
            ("ligand_IGF1", "receptor_IGF1R"),
            ("ligand_Insulin", "receptor_INSR"),
            ("ligand_HGF", "receptor_MET"),
            ("ligand_SCF", "receptor_KIT"),
        ]
        
        for ligand, receptor in binding_pairs:
            if self.kg.get_entity(ligand) and self.kg.get_entity(receptor):
                rel = BiologicalRelationship(
                    source=ligand,
                    target=receptor,
                    relation_type=RelationType.BINDS_TO,
                    properties={"binding_type": "receptor_ligand"},
                    mathematical_constraints=["simple_binding", "receptor_occupancy"],
                    confidence_score=0.92
                )
                self.kg.add_relationship(rel)
                rel_counts["binding"] += 1
        
        # 5. Expanded signaling cascades
        signaling_cascades = [
            # EGFR pathway
            ("receptor_EGFR", "signaling_RAS", RelationType.PHOSPHORYLATES),
            ("signaling_RAS", "signaling_RAF1", RelationType.PHOSPHORYLATES),
            ("signaling_RAS", "signaling_BRAF", RelationType.PHOSPHORYLATES),
            ("signaling_RAF1", "signaling_MEK1", RelationType.PHOSPHORYLATES),
            ("signaling_RAF1", "signaling_MEK2", RelationType.PHOSPHORYLATES),
            ("signaling_BRAF", "signaling_MEK1", RelationType.PHOSPHORYLATES),
            ("signaling_BRAF", "signaling_MEK2", RelationType.PHOSPHORYLATES),
            ("signaling_MEK1", "signaling_ERK1", RelationType.PHOSPHORYLATES),
            ("signaling_MEK1", "signaling_ERK2", RelationType.PHOSPHORYLATES),
            ("signaling_MEK2", "signaling_ERK1", RelationType.PHOSPHORYLATES),
            ("signaling_MEK2", "signaling_ERK2", RelationType.PHOSPHORYLATES),
            
            # PI3K/AKT pathway
            ("receptor_EGFR", "signaling_PI3K", RelationType.PHOSPHORYLATES),
            ("receptor_IGF1R", "signaling_PI3K", RelationType.PHOSPHORYLATES),
            ("receptor_INSR", "signaling_PI3K", RelationType.PHOSPHORYLATES),
            ("signaling_PI3K", "signaling_AKT1", RelationType.PHOSPHORYLATES),
            ("signaling_AKT1", "signaling_mTOR", RelationType.PHOSPHORYLATES),
            ("signaling_PTEN", "signaling_PI3K", RelationType.INHIBITS),
            
            # VEGFR pathway
            ("receptor_VEGFR1", "signaling_PI3K", RelationType.PHOSPHORYLATES),
            ("receptor_VEGFR2", "signaling_PI3K", RelationType.PHOSPHORYLATES),
            ("receptor_VEGFR2", "signaling_RAS", RelationType.PHOSPHORYLATES),
        ]
        
        for source, target, rel_type in signaling_cascades:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"signaling_cascade": True},
                    mathematical_constraints=["phosphorylation_kinetics"] if rel_type == RelationType.PHOSPHORYLATES else [],
                    confidence_score=0.89
                )
                self.kg.add_relationship(rel)
                rel_counts["phosphorylation"] += 1
        
        # 6. Cellular localization
        localization_map = [
            ("receptor_EGFR", "compartment_plasma_membrane"),
            ("receptor_HER2", "compartment_plasma_membrane"),
            ("receptor_VEGFR1", "compartment_plasma_membrane"),
            ("receptor_VEGFR2", "compartment_plasma_membrane"),
            ("receptor_PDGFRA", "compartment_plasma_membrane"),
            ("receptor_PDGFRB", "compartment_plasma_membrane"),
            ("receptor_FGFR1", "compartment_plasma_membrane"),
            ("receptor_IGF1R", "compartment_plasma_membrane"),
            ("receptor_INSR", "compartment_plasma_membrane"),
            ("receptor_MET", "compartment_plasma_membrane"),
            ("receptor_KIT", "compartment_plasma_membrane"),
            ("receptor_ALK", "compartment_plasma_membrane"),
            ("signaling_RAS", "compartment_plasma_membrane"),
            ("signaling_RAF1", "compartment_cytoplasm"),
            ("signaling_BRAF", "compartment_cytoplasm"),
            ("signaling_MEK1", "compartment_cytoplasm"),
            ("signaling_MEK2", "compartment_cytoplasm"),
            ("signaling_ERK1", "compartment_cytoplasm"),
            ("signaling_ERK2", "compartment_nucleus"),
            ("signaling_PI3K", "compartment_plasma_membrane"),
            ("signaling_AKT1", "compartment_cytoplasm"),
            ("signaling_mTOR", "compartment_cytoplasm"),
            ("signaling_PTEN", "compartment_cytoplasm"),
            ("enzyme_HK1", "compartment_cytoplasm"),
            ("enzyme_HK2", "compartment_mitochondria"),
            ("enzyme_CS", "compartment_mitochondria"),
            ("enzyme_IDH1", "compartment_cytoplasm"),
            ("enzyme_IDH2", "compartment_mitochondria"),
            ("enzyme_FASN", "compartment_cytoplasm"),
            ("enzyme_CPT1A", "compartment_mitochondria"),
        ]
        
        for protein, compartment in localization_map:
            if self.kg.get_entity(protein) and self.kg.get_entity(compartment):
                rel = BiologicalRelationship(
                    source=protein,
                    target=compartment,
                    relation_type=RelationType.LOCATED_IN,
                    properties={"subcellular_localization": True},
                    mathematical_constraints=[],
                    confidence_score=0.88
                )
                self.kg.add_relationship(rel)
                rel_counts["location"] += 1
        
        # 7. Expanded disease associations
        biomarker_disease_map = [
            ("biomarker_CA15_3", "disease_breast_cancer"),
            ("biomarker_CA125", "disease_ovarian_cancer"),
            ("biomarker_CEA", "disease_colorectal_cancer"),
            ("biomarker_PSA", "disease_prostate_cancer"),
            ("biomarker_HER2", "disease_breast_cancer"),
            ("biomarker_PD_L1", "disease_lung_cancer"),
            ("biomarker_BRCA1", "disease_breast_cancer"),
            ("biomarker_BRCA2", "disease_breast_cancer"),
            ("biomarker_CRP", "disease_rheumatoid_arthritis"),
            ("biomarker_IL_6", "disease_rheumatoid_arthritis"),
            ("biomarker_TNF_α", "disease_rheumatoid_arthritis"),
            ("biomarker_HbA1c", "disease_type2_diabetes"),
            ("biomarker_Glucose", "disease_type2_diabetes"),
            ("biomarker_LDL", "disease_atherosclerosis"),
            ("biomarker_Troponin_I", "disease_heart_failure"),
            ("biomarker_BNP", "disease_heart_failure"),
        ]
        
        for biomarker, disease in biomarker_disease_map:
            if self.kg.get_entity(biomarker) and self.kg.get_entity(disease):
                rel = BiologicalRelationship(
                    source=biomarker,
                    target=disease,
                    relation_type=RelationType.BIOMARKER_FOR,
                    properties={"diagnostic_value": "high"},
                    mathematical_constraints=[],
                    confidence_score=0.87
                )
                self.kg.add_relationship(rel)
                rel_counts["biomarker"] += 1
        
        # Disease protein associations
        disease_protein_map = [
            ("disease_protein_TP53", ["disease_breast_cancer", "disease_lung_cancer", "disease_colorectal_cancer"]),
            ("disease_protein_MYC", ["disease_breast_cancer", "disease_lung_cancer"]),
            ("disease_protein_RAS", ["disease_colorectal_cancer", "disease_pancreatic_cancer", "disease_lung_cancer"]),
            ("disease_protein_PI3K", ["disease_breast_cancer", "disease_prostate_cancer"]),
            ("disease_protein_PTEN", ["disease_prostate_cancer", "disease_breast_cancer"]),
            ("disease_protein_APC", ["disease_colorectal_cancer"]),
            ("disease_protein_VHL", ["disease_renal_cancer"]),
            ("disease_protein_RB1", ["disease_retinoblastoma", "disease_lung_cancer"]),
            ("disease_protein_CDKN2A", ["disease_melanoma", "disease_pancreatic_cancer"]),
            ("disease_protein_BCL2", ["disease_lymphoma"]),
            ("disease_protein_NF_κB", ["disease_rheumatoid_arthritis"]),
            ("disease_protein_HIF_1α", ["disease_cancer_general"]),
            ("disease_protein_AMPK", ["disease_type2_diabetes", "disease_metabolic_syndrome"]),
        ]
        
        for protein, diseases in disease_protein_map:
            if self.kg.get_entity(protein):
                for disease in diseases:
                    # Create disease if doesn't exist
                    if not self.kg.get_entity(disease):
                        entity = BiologicalEntity(
                            id=disease,
                            name=disease.replace('_', ' '),
                            entity_type="disease",
                            properties={"disease_category": "chronic"},
                            confidence_score=0.85
                        )
                        self.kg.add_entity(entity)
                    
                    rel = BiologicalRelationship(
                        source=protein,
                        target=disease,
                        relation_type=RelationType.CAUSES_DISEASE,
                        properties={"association_strength": "strong"},
                        mathematical_constraints=[],
                        confidence_score=0.85
                    )
                    self.kg.add_relationship(rel)
                    rel_counts["disease_cause"] += 1
        
        # 8. Comprehensive drug-target relationships
        drug_target_map = [
            ("drug_DB00619", "protein_ABL1", RelationType.COMPETITIVE_INHIBITION),  # Imatinib
            ("drug_DB00619", "receptor_KIT", RelationType.COMPETITIVE_INHIBITION),
            ("drug_DB00619", "receptor_PDGFRA", RelationType.COMPETITIVE_INHIBITION),
            ("drug_DB00530", "receptor_EGFR", RelationType.COMPETITIVE_INHIBITION),  # Erlotinib
            ("drug_DB00317", "receptor_EGFR", RelationType.COMPETITIVE_INHIBITION),  # Gefitinib
            ("drug_DB01259", "receptor_EGFR", RelationType.COMPETITIVE_INHIBITION),  # Lapatinib
            ("drug_DB01259", "receptor_HER2", RelationType.COMPETITIVE_INHIBITION),
            ("drug_DB01268", "receptor_VEGFR1", RelationType.COMPETITIVE_INHIBITION),  # Sunitinib
            ("drug_DB01268", "receptor_VEGFR2", RelationType.COMPETITIVE_INHIBITION),
            ("drug_DB01268", "receptor_PDGFRA", RelationType.COMPETITIVE_INHIBITION),
            ("drug_DB01268", "receptor_PDGFRB", RelationType.COMPETITIVE_INHIBITION),
            ("drug_DB00398", "signaling_RAF1", RelationType.COMPETITIVE_INHIBITION),  # Sorafenib
            ("drug_DB00398", "signaling_BRAF", RelationType.COMPETITIVE_INHIBITION),
            ("drug_DB00398", "receptor_VEGFR2", RelationType.COMPETITIVE_INHIBITION),
            ("drug_DB06589", "receptor_VEGFR1", RelationType.COMPETITIVE_INHIBITION),  # Pazopanib
            ("drug_DB06589", "receptor_VEGFR2", RelationType.COMPETITIVE_INHIBITION),
            ("drug_DB08865", "receptor_ALK", RelationType.COMPETITIVE_INHIBITION),  # Crizotinib
            ("drug_DB08865", "receptor_MET", RelationType.COMPETITIVE_INHIBITION),
            ("drug_DB09330", "receptor_EGFR", RelationType.COMPETITIVE_INHIBITION),  # Osimertinib
            ("drug_DB08875", "receptor_MET", RelationType.COMPETITIVE_INHIBITION),  # Cabozantinib
            ("drug_DB08875", "receptor_VEGFR2", RelationType.COMPETITIVE_INHIBITION),
            ("drug_DB00641", "enzyme_HMGCR", RelationType.COMPETITIVE_INHIBITION),  # Simvastatin
            ("drug_DB01076", "enzyme_HMGCR", RelationType.COMPETITIVE_INHIBITION),  # Atorvastatin
            ("drug_DB00563", "enzyme_DHFR", RelationType.COMPETITIVE_INHIBITION),  # Methotrexate
            ("drug_DB00675", "receptor_ER", RelationType.COMPETITIVE_INHIBITION),  # Tamoxifen
        ]
        
        for drug, target, rel_type in drug_target_map:
            if self.kg.get_entity(drug):
                # Create target if doesn't exist
                if not self.kg.get_entity(target):
                    entity = BiologicalEntity(
                        id=target,
                        name=target.replace('_', ' '),
                        entity_type="protein" if "protein" in target else "receptor",
                        properties={"drug_target": True},
                        confidence_score=0.9
                    )
                    self.kg.add_entity(entity)
                
                rel = BiologicalRelationship(
                    source=drug,
                    target=target,
                    relation_type=rel_type,
                    properties={"interaction_type": "drug_target"},
                    mathematical_constraints=["competitive_mm"],
                    confidence_score=0.91
                )
                self.kg.add_relationship(rel)
                rel_counts["drug_target"] += 1
        
        # 9. Drug metabolism via CYP
        drug_cyp_map = [
            ("drug_DB00641", "cyp_CYP3A4", RelationType.SUBSTRATE_OF),  # Simvastatin
            ("drug_DB01076", "cyp_CYP3A4", RelationType.SUBSTRATE_OF),  # Atorvastatin
            ("drug_DB00682", "cyp_CYP2C9", RelationType.SUBSTRATE_OF),  # Warfarin
            ("drug_DB00758", "cyp_CYP2C19", RelationType.SUBSTRATE_OF),  # Clopidogrel
            ("drug_DB00338", "cyp_CYP2C19", RelationType.SUBSTRATE_OF),  # Omeprazole
            ("drug_DB00338", "cyp_CYP3A4", RelationType.SUBSTRATE_OF),
            ("drug_DB00264", "cyp_CYP2D6", RelationType.SUBSTRATE_OF),  # Metoprolol
            ("drug_DB00472", "cyp_CYP2D6", RelationType.COMPETITIVE_INHIBITION),  # Fluoxetine
            ("drug_DB01026", "cyp_CYP3A4", RelationType.COMPETITIVE_INHIBITION),  # Ketoconazole
            ("drug_DB00537", "cyp_CYP3A4", RelationType.COMPETITIVE_INHIBITION),  # Clarithromycin
            ("drug_DB01045", "cyp_CYP3A4", RelationType.INDUCES),  # Rifampin
            ("drug_DB01045", "cyp_CYP2C9", RelationType.INDUCES),
            ("drug_DB01045", "cyp_CYP2C19", RelationType.INDUCES),
            ("drug_DB01248", "cyp_CYP2C8", RelationType.SUBSTRATE_OF),  # Paclitaxel
            ("drug_DB01248", "cyp_CYP3A4", RelationType.SUBSTRATE_OF),
            ("drug_DB00531", "cyp_CYP2B6", RelationType.SUBSTRATE_OF),  # Cyclophosphamide
            ("drug_DB00531", "cyp_CYP3A4", RelationType.SUBSTRATE_OF),
            ("drug_DB00675", "cyp_CYP2D6", RelationType.SUBSTRATE_OF),  # Tamoxifen
            ("drug_DB00675", "cyp_CYP3A4", RelationType.SUBSTRATE_OF),
            ("drug_DB00541", "cyp_CYP3A4", RelationType.SUBSTRATE_OF),  # Vincristine
        ]
        
        for drug, cyp, rel_type in drug_cyp_map:
            if self.kg.get_entity(drug) and self.kg.get_entity(cyp):
                rel = BiologicalRelationship(
                    source=drug,
                    target=cyp,
                    relation_type=rel_type,
                    properties={"metabolism_pathway": True},
                    mathematical_constraints=["michaelis_menten"] if rel_type == RelationType.SUBSTRATE_OF else ["competitive_mm"],
                    confidence_score=0.88
                )
                self.kg.add_relationship(rel)
                rel_counts["drug_metabolism"] += 1
        
        # 10. Drug-drug interactions
        drug_interactions = [
            ("drug_DB00641", "drug_DB01026"),  # Simvastatin-Ketoconazole
            ("drug_DB00641", "drug_DB00537"),  # Simvastatin-Clarithromycin
            ("drug_DB00682", "drug_DB00472"),  # Warfarin-Fluoxetine
            ("drug_DB00682", "drug_DB01045"),  # Warfarin-Rifampin
            ("drug_DB00758", "drug_DB00338"),  # Clopidogrel-Omeprazole
            ("drug_DB00675", "drug_DB00472"),  # Tamoxifen-Fluoxetine
            ("drug_DB01248", "drug_DB01026"),  # Paclitaxel-Ketoconazole
            ("drug_DB00530", "drug_DB01045"),  # Erlotinib-Rifampin
        ]
        
        for drug1, drug2 in drug_interactions:
            if self.kg.get_entity(drug1) and self.kg.get_entity(drug2):
                rel = BiologicalRelationship(
                    source=drug1,
                    target=drug2,
                    relation_type=RelationType.DRUG_DRUG_INTERACTION,
                    properties={"severity": "major", "mechanism": "cyp_mediated"},
                    mathematical_constraints=["drug_interaction_competitive"],
                    confidence_score=0.86
                )
                self.kg.add_relationship(rel)
                rel_counts["drug_drug"] += 1
        
        # 11. Drug treats disease
        drug_disease_map = [
            ("drug_DB00619", "disease_leukemia"),  # Imatinib
            ("drug_DB00530", "disease_lung_cancer"),  # Erlotinib
            ("drug_DB00317", "disease_lung_cancer"),  # Gefitinib
            ("drug_DB01259", "disease_breast_cancer"),  # Lapatinib
            ("drug_DB01268", "disease_renal_cancer"),  # Sunitinib
            ("drug_DB00398", "disease_liver_cancer"),  # Sorafenib
            ("drug_DB00675", "disease_breast_cancer"),  # Tamoxifen
            ("drug_DB00563", "disease_rheumatoid_arthritis"),  # Methotrexate
            ("drug_DB00563", "disease_cancer_general"),  # Methotrexate
            ("drug_DB00997", "disease_breast_cancer"),  # Doxorubicin
            ("drug_DB00515", "disease_lung_cancer"),  # Cisplatin
            ("drug_DB00441", "disease_pancreatic_cancer"),  # Gemcitabine
        ]
        
        for drug, disease in drug_disease_map:
            if self.kg.get_entity(drug):
                # Create disease if doesn't exist
                if not self.kg.get_entity(disease):
                    entity = BiologicalEntity(
                        id=disease,
                        name=disease.replace('_', ' '),
                        entity_type="disease",
                        properties={"disease_category": "oncology"},
                        confidence_score=0.85
                    )
                    self.kg.add_entity(entity)
                
                rel = BiologicalRelationship(
                    source=drug,
                    target=disease,
                    relation_type=RelationType.TREATS,
                    properties={"indication": "approved"},
                    mathematical_constraints=[],
                    confidence_score=0.90
                )
                self.kg.add_relationship(rel)
                rel_counts["treats"] += 1
        
        # Log relationship summary
        logger.info("Comprehensive relationships added:")
        logger.info(f"  Enzyme-substrate: {rel_counts['enzyme_substrate']}")
        logger.info(f"  Enzyme-product: {rel_counts['enzyme_product']}")
        logger.info(f"  Inhibition: {rel_counts['inhibition']}")
        logger.info(f"  Allosteric: {rel_counts['allosteric']}")
        logger.info(f"  Binding: {rel_counts['binding']}")
        logger.info(f"  Phosphorylation: {rel_counts['phosphorylation']}")
        logger.info(f"  Localization: {rel_counts['location']}")
        logger.info(f"  Biomarker: {rel_counts['biomarker']}")
        logger.info(f"  Disease cause: {rel_counts['disease_cause']}")
        logger.info(f"  Drug target: {rel_counts['drug_target']}")
        logger.info(f"  Drug metabolism: {rel_counts['drug_metabolism']}")
        logger.info(f"  Drug-drug: {rel_counts['drug_drug']}")
        logger.info(f"  Treats: {rel_counts['treats']}")
        logger.info(f"  TOTAL RELATIONSHIPS: {sum(rel_counts.values())}")
    
    def build_complete_graph(self) -> KnowledgeGraph:
        """Build the complete knowledge graph with all entities and relationships"""
        logger.info("Building complete enhanced knowledge graph...")
        
        # Build all entity categories
        self.build_enzyme_kinetics_entities()
        self.build_multiscale_entities()
        self.build_disease_state_entities()
        self.build_drug_interaction_entities()
        
        # Add comprehensive relationships
        self.build_comprehensive_relationships()
        
        logger.info("Complete knowledge graph built:")
        logger.info(f"  Total entities: {len(self.kg.entities)}")
        logger.info(f"  Total relationships: {len(self.kg.relationships)}")
        
        # Log entity distribution
        entity_types = {}
        for entity in self.kg.entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
        
        logger.info("Entity distribution:")
        for entity_type, count in sorted(entity_types.items()):
            logger.info(f"  {entity_type}: {count}")
        
        return self.kg

def main():
    parser = argparse.ArgumentParser(description='Build enhanced biological knowledge graph')
    parser.add_argument('--cache-dir', type=str, default='data/kg_cache',
                        help='Directory to cache the knowledge graph')
    parser.add_argument('--output', type=str, default='data/enhanced_knowledge_graph.json',
                        help='Output file for the knowledge graph')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'cache_directory': args.cache_dir,
        'use_cache': True,
        'cache_ttl_hours': 24
    }
    
    # Build the enhanced knowledge graph
    builder = EnhancedKGBuilder(config)
    kg = builder.build_complete_graph()
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export graph data
    graph_data = {
        'entities': {
            entity_id: {
                'id': entity.id,
                'name': entity.name,
                'type': entity.entity_type,
                'properties': entity.properties,
                'confidence': entity.confidence_score
            }
            for entity_id, entity in kg.entities.items()
        },
        'relationships': [
            {
                'source': rel.source,
                'target': rel.target,
                'type': str(rel.relation_type),
                'properties': rel.properties,
                'constraints': rel.mathematical_constraints,
                'confidence': rel.confidence_score
            }
            for rel in kg.relationships
        ],
        'statistics': {
            'total_entities': len(kg.entities),
            'total_relationships': len(kg.relationships),
            'entity_types': len(set(e.entity_type for e in kg.entities.values())),
            'relationship_types': len(set(str(r.relation_type) for r in kg.relationships))
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    logger.info(f"Enhanced knowledge graph saved to {output_path}")
    logger.info(f"Total entities: {graph_data['statistics']['total_entities']}")
    logger.info(f"Total relationships: {graph_data['statistics']['total_relationships']}")

if __name__ == "__main__":
    main()