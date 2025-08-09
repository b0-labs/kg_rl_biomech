#!/usr/bin/env python3
"""
Knowledge Graph Builder Script
Builds and caches a comprehensive biological knowledge graph based on kg_builder.md specifications
"""

import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.knowledge_graph import KnowledgeGraph, BiologicalEntity, BiologicalRelationship, RelationType
from src.kg_loader_unified import KnowledgeGraphLoader, KnowledgeGraphBuilder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveKGBuilder:
    """Builds comprehensive knowledge graph with all entities from kg_builder.md"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.kg = KnowledgeGraph(config)
        
    def build_enzyme_kinetics_entities(self):
        """Add all enzyme kinetics entities (Section 4.1)"""
        logger.info("Building enzyme kinetics entities...")
        
        # Glycolysis Enzymes
        glycolysis_enzymes = [
            ("HK1", "Hexokinase 1", "2.7.1.1"),
            ("HK2", "Hexokinase 2", "2.7.1.1"),
            ("GPI", "Glucose-6-phosphate isomerase", "5.3.1.9"),
            ("PFK1", "Phosphofructokinase 1", "2.7.1.11"),
            ("ALDOA", "Aldolase A", "4.1.2.13"),
            ("TPI1", "Triosephosphate isomerase", "5.3.1.1"),
            ("GAPDH", "Glyceraldehyde-3-phosphate dehydrogenase", "1.2.1.12"),
            ("PGK1", "Phosphoglycerate kinase 1", "2.7.2.3"),
            ("ENO1", "Enolase 1", "4.2.1.11"),
            ("PKM", "Pyruvate kinase M", "2.7.1.40"),
        ]
        
        for enzyme_id, name, ec_number in glycolysis_enzymes:
            entity = BiologicalEntity(
                id=f"enzyme_{enzyme_id}",
                name=name,
                entity_type="enzyme",
                properties={
                    "ec_number": ec_number,
                    "pathway": "glycolysis",
                    "source": "kg_builder"
                },
                confidence_score=0.95
            )
            self.kg.add_entity(entity)
        
        # TCA Cycle Enzymes
        tca_enzymes = [
            ("CS", "Citrate synthase", "2.3.3.1"),
            ("ACO2", "Aconitase 2", "4.2.1.3"),
            ("IDH1", "Isocitrate dehydrogenase 1", "1.1.1.42"),
            ("IDH2", "Isocitrate dehydrogenase 2", "1.1.1.42"),
            ("OGDH", "α-Ketoglutarate dehydrogenase", "1.2.4.2"),
            ("SUCLA2", "Succinate-CoA ligase", "6.2.1.5"),
            ("SDH", "Succinate dehydrogenase", "1.3.5.1"),
            ("FH", "Fumarase", "4.2.1.2"),
            ("MDH2", "Malate dehydrogenase 2", "1.1.1.37"),
        ]
        
        for enzyme_id, name, ec_number in tca_enzymes:
            entity = BiologicalEntity(
                id=f"enzyme_{enzyme_id}",
                name=name,
                entity_type="enzyme",
                properties={
                    "ec_number": ec_number,
                    "pathway": "tca_cycle",
                    "source": "kg_builder"
                },
                confidence_score=0.95
            )
            self.kg.add_entity(entity)
        
        # Amino Acid Metabolism Enzymes
        amino_acid_enzymes = [
            ("GOT1", "Glutamic-oxaloacetic transaminase 1", "2.6.1.1"),
            ("GOT2", "Glutamic-oxaloacetic transaminase 2", "2.6.1.1"),
            ("GPT", "Glutamic-pyruvic transaminase", "2.6.1.2"),
            ("GLS", "Glutaminase", "3.5.1.2"),
            ("GLS2", "Glutaminase 2", "3.5.1.2"),
            ("GLUD1", "Glutamate dehydrogenase 1", "1.4.1.3"),
            ("ASS1", "Argininosuccinate synthase 1", "6.3.4.5"),
            ("ASL", "Argininosuccinate lyase", "4.3.2.1"),
        ]
        
        for enzyme_id, name, ec_number in amino_acid_enzymes:
            entity = BiologicalEntity(
                id=f"enzyme_{enzyme_id}",
                name=name,
                entity_type="enzyme",
                properties={
                    "ec_number": ec_number,
                    "pathway": "amino_acid_metabolism",
                    "source": "kg_builder"
                },
                confidence_score=0.93
            )
            self.kg.add_entity(entity)
        
        # Lipid Metabolism Enzymes
        lipid_enzymes = [
            ("FASN", "Fatty acid synthase", "2.3.1.85"),
            ("ACC1", "Acetyl-CoA carboxylase 1", "6.4.1.2"),
            ("HMGCR", "HMG-CoA reductase", "1.1.1.34"),
            ("CPT1A", "Carnitine palmitoyltransferase 1A", "2.3.1.21"),
            ("HADHA", "Hydroxyacyl-CoA dehydrogenase", "1.1.1.35"),
            ("ACACA", "Acetyl-CoA carboxylase alpha", "6.4.1.2"),
        ]
        
        for enzyme_id, name, ec_number in lipid_enzymes:
            entity = BiologicalEntity(
                id=f"enzyme_{enzyme_id}",
                name=name,
                entity_type="enzyme",
                properties={
                    "ec_number": ec_number,
                    "pathway": "lipid_metabolism",
                    "source": "kg_builder"
                },
                confidence_score=0.92
            )
            self.kg.add_entity(entity)
        
        # Oxidoreductases
        oxidoreductases = [
            ("CAT", "Catalase", "1.11.1.6"),
            ("SOD1", "Superoxide dismutase 1", "1.15.1.1"),
            ("SOD2", "Superoxide dismutase 2", "1.15.1.1"),
            ("GPX1", "Glutathione peroxidase 1", "1.11.1.9"),
            ("PRDX1", "Peroxiredoxin 1", "1.11.1.15"),
            ("NQO1", "NAD(P)H dehydrogenase quinone 1", "1.6.5.2"),
            ("G6PD", "Glucose-6-phosphate dehydrogenase", "1.1.1.49"),
            ("ALDH2", "Aldehyde dehydrogenase 2", "1.2.1.3"),
        ]
        
        for enzyme_id, name, ec_number in oxidoreductases:
            entity = BiologicalEntity(
                id=f"enzyme_{enzyme_id}",
                name=name,
                entity_type="enzyme",
                properties={
                    "ec_number": ec_number,
                    "enzyme_class": "oxidoreductase",
                    "source": "kg_builder"
                },
                confidence_score=0.94
            )
            self.kg.add_entity(entity)
        
        # Add comprehensive substrates (100 as specified in kg_builder.md)
        key_substrates = [
            # Glycolysis intermediates
            ("Glucose", 180.16, "C6H12O6"),
            ("Glucose-6-phosphate", 260.14, "C6H13O9P"),
            ("Fructose-6-phosphate", 260.14, "C6H13O9P"),
            ("Fructose-1,6-bisphosphate", 340.12, "C6H14O12P2"),
            ("Glyceraldehyde-3-phosphate", 170.06, "C3H7O6P"),
            ("Dihydroxyacetone phosphate", 170.06, "C3H7O6P"),
            ("1,3-Bisphosphoglycerate", 266.04, "C3H8O10P2"),
            ("3-Phosphoglycerate", 186.06, "C3H7O7P"),
            ("2-Phosphoglycerate", 186.06, "C3H7O7P"),
            ("Phosphoenolpyruvate", 168.04, "C3H5O6P"),
            
            # TCA cycle intermediates
            ("Pyruvate", 88.06, "C3H4O3"),
            ("Acetyl-CoA", 809.57, "C23H38N7O17P3S"),
            ("Citrate", 192.12, "C6H8O7"),
            ("Isocitrate", 192.12, "C6H8O7"),
            ("α-Ketoglutarate", 146.11, "C5H6O5"),
            ("Succinyl-CoA", 867.61, "C25H40N7O19P3S"),
            ("Succinate", 118.09, "C4H6O4"),
            ("Fumarate", 116.07, "C4H4O4"),
            ("Malate", 134.09, "C4H6O5"),
            ("Oxaloacetate", 132.07, "C4H4O5"),
            
            # Amino acids (20 standard)
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
            ("NAD+", 663.43, "C21H27N7O14P2"),
            ("NADH", 665.44, "C21H29N7O14P2"),
            ("NADP+", 743.41, "C21H28N7O17P3"),
            ("NADPH", 745.42, "C21H30N7O17P3"),
            ("FAD", 785.55, "C27H33N9O15P2"),
            ("FADH2", 787.54, "C27H35N9O15P2"),
            ("Coenzyme A", 767.53, "C21H36N7O16P3S"),
            ("S-Adenosylmethionine", 398.44, "C15H22N6O5S"),
            ("Tetrahydrofolate", 445.43, "C19H23N7O6"),
            
            # Vitamins and cofactors
            ("Biotin", 244.31, "C10H16N2O3S"),
            ("Thiamine pyrophosphate", 425.31, "C12H19N4O7P2S"),
            ("Pyridoxal phosphate", 247.14, "C8H10NO6P"),
            ("Riboflavin", 376.36, "C17H20N4O6"),
            ("Pantothenic acid", 219.23, "C9H17NO5"),
            ("Nicotinic acid", 123.11, "C6H5NO2"),
            ("Folic acid", 441.40, "C19H19N7O6"),
            ("Vitamin B12", 1355.37, "C63H88CoN14O14P"),
            ("Ascorbic acid", 176.12, "C6H8O6"),
            ("α-Tocopherol", 430.71, "C29H50O2"),
            
            # Lipids and fatty acids
            ("Palmitic acid", 256.42, "C16H32O2"),
            ("Stearic acid", 284.48, "C18H36O2"),
            ("Oleic acid", 282.46, "C18H34O2"),
            ("Linoleic acid", 280.45, "C18H32O2"),
            ("Arachidonic acid", 304.47, "C20H32O2"),
            ("Cholesterol", 386.65, "C27H46O"),
            ("Phosphatidylcholine", 760.08, "C40H80NO8P"),
            ("Phosphatidylserine", 747.04, "C38H74NO10P"),
            ("Sphingomyelin", 703.03, "C39H79N2O6P"),
            ("Ceramide", 537.86, "C34H67NO3"),
            ("Diacylglycerol", 594.82, "C35H66O5"),
            
            # Signaling molecules
            ("Inositol-1,4,5-trisphosphate", 420.10, "C6H15O15P3"),
            ("cAMP", 329.21, "C10H12N5O6P"),
            ("cGMP", 345.21, "C10H12N5O7P"),
            
            # Ions
            ("Calcium ions", 40.08, "Ca2+"),
            ("Magnesium ions", 24.31, "Mg2+"),
            ("Iron ions", 55.85, "Fe2+/Fe3+"),
            ("Zinc ions", 65.38, "Zn2+"),
            ("Copper ions", 63.55, "Cu2+"),
            ("Manganese ions", 54.94, "Mn2+"),
            
            # Neurotransmitters
            ("Dopamine", 153.18, "C8H11NO2"),
            ("Serotonin", 176.21, "C10H12N2O"),
            ("GABA", 103.12, "C4H9NO2"),
            ("Acetylcholine", 146.21, "C7H16NO2"),
            ("Epinephrine", 183.20, "C9H13NO3"),
            ("Norepinephrine", 169.18, "C8H11NO3"),
            ("Histamine", 111.15, "C5H9N3"),
            ("Melatonin", 232.28, "C13H16N2O2"),
            
            # Eicosanoids
            ("Prostaglandin E2", 352.47, "C20H32O5"),
            ("Leukotriene B4", 336.47, "C20H32O4"),
            ("Thromboxane A2", 352.47, "C20H32O5"),
            
            # Gaseous signaling
            ("Nitric oxide", 30.01, "NO"),
            ("Carbon monoxide", 28.01, "CO"),
            ("Hydrogen sulfide", 34.08, "H2S"),
            
            # Other metabolites
            ("Lactate", 90.08, "C3H6O3"),
            ("Uric acid", 168.11, "C5H4N4O3"),
        ]
        
        for substrate_name, mw, formula in key_substrates:
            entity = BiologicalEntity(
                id=f"substrate_{substrate_name.replace(' ', '_').replace('-', '_').replace(',', '_').replace('+', 'plus')}",
                name=substrate_name,
                entity_type="substrate",
                properties={
                    "molecular_weight": mw,
                    "formula": formula,
                    "source": "kg_builder"
                },
                confidence_score=0.95
            )
            self.kg.add_entity(entity)
        
        # Add expanded products (50 as specified in kg_builder.md)
        key_products = [
            # Primary metabolic products
            ("Glucose-1-phosphate", 260.14, "C6H13O9P"),
            ("6-Phosphogluconate", 276.14, "C6H13O10P"),
            ("Ribulose-5-phosphate", 230.11, "C5H11O8P"),
            ("Ribose-5-phosphate", 230.11, "C5H11O8P"),
            ("Xylulose-5-phosphate", 230.11, "C5H11O8P"),
            ("Sedoheptulose-7-phosphate", 290.16, "C7H15O10P"),
            ("Erythrose-4-phosphate", 200.08, "C4H9O7P"),
            
            # Energy products
            ("CO2", 44.01, "CO2"),
            ("H2O", 18.02, "H2O"),
            ("Pi", 95.98, "PO4^3-"),
            ("PPi", 174.95, "P2O7^4-"),
            
            # Reduced cofactors
            ("NADH", 665.44, "C21H29N7O14P2"),
            ("NADPH", 745.42, "C21H30N7O17P3"),
            ("FADH2", 787.54, "C27H35N9O15P2"),
            ("Reduced glutathione", 307.32, "C10H17N3O6S"),
            
            # Amino acid products
            ("Urea", 60.06, "CH4N2O"),
            ("Ammonia", 17.03, "NH3"),
            ("Citrulline", 175.19, "C6H13N3O3"),
            ("Ornithine", 132.16, "C5H12N2O2"),
            ("Homocysteine", 135.19, "C4H9NO2S"),
            ("Cystathionine", 222.26, "C7H14N2O4S"),
            ("α-Ketobutyrate", 102.09, "C4H6O3"),
            ("Succinyl-homoserine", 219.19, "C8H13NO6"),
            
            # Lipid products
            ("Glycerol", 92.09, "C3H8O3"),
            ("Acetoacetate", 102.09, "C4H6O3"),
            ("β-Hydroxybutyrate", 104.10, "C4H8O3"),
            ("Malonyl-CoA", 853.58, "C24H38N7O19P3S"),
            ("Acyl-CoA", 809.57, "Variable"),
            ("Phosphatidic acid", 674.96, "C35H67O8P"),
            
            # Nucleotide products
            ("Inosine", 268.23, "C10H12N4O5"),
            ("Hypoxanthine", 136.11, "C5H4N4O"),
            ("Xanthine", 152.11, "C5H4N4O2"),
            ("Uric acid", 168.11, "C5H4N4O3"),
            ("dATP", 491.18, "C10H16N5O12P3"),
            ("dGTP", 507.18, "C10H16N5O13P3"),
            ("dCTP", 467.16, "C9H16N3O13P3"),
            ("dTTP", 482.17, "C10H17N2O14P3"),
            
            # Signaling products
            ("IP3", 420.10, "C6H15O15P3"),
            ("DAG", 594.82, "C35H66O5"),
            ("Arachidonic acid", 304.47, "C20H32O2"),
            ("Prostaglandins", 352.47, "Variable"),
            ("Leukotrienes", 336.47, "Variable"),
            
            # Neurotransmitter metabolites
            ("DOPAC", 168.15, "C8H8O4"),
            ("HVA", 182.17, "C9H10O4"),
            ("5-HIAA", 191.18, "C10H9NO3"),
            ("VMA", 198.17, "C9H12O5"),
            
            # Other products
            ("Bilirubin", 584.66, "C33H36N4O6"),
            ("Biliverdin", 582.65, "C33H34N4O6"),
            ("Heme", 616.49, "C34H32FeN4O4"),
            ("Methylglyoxal", 72.06, "C3H4O2"),
            ("D-Lactate", 90.08, "C3H6O3"),
        ]
        
        for product_name, mw, formula in key_products:
            entity = BiologicalEntity(
                id=f"product_{product_name.replace(' ', '_').replace('-', '_')}",
                name=product_name,
                entity_type="product",
                properties={
                    "molecular_weight": mw,
                    "formula": formula,
                    "source": "kg_builder"
                },
                confidence_score=0.93
            )
            self.kg.add_entity(entity)
        
        # Add comprehensive inhibitors (30 as specified)
        inhibitors = [
            # Metabolic inhibitors
            ("Metformin", 129.16, "complex_I_inhibitor", 10000),
            ("2-Deoxyglucose", 164.16, "hexokinase_inhibitor", 5000),
            ("Dichloroacetate", 128.94, "PDK_inhibitor", 100),
            ("Oxamate", 89.05, "LDH_inhibitor", 20000),
            ("CB-839", 571.5, "glutaminase_inhibitor", 30),
            ("Etomoxir", 320.77, "CPT1_inhibitor", 50),
            ("Orlistat", 495.73, "FASN_inhibitor", 100),
            ("Allopurinol", 136.11, "xanthine_oxidase_inhibitor", 10),
            
            # Additional inhibitors
            ("Rotenone", 394.42, "complex_I_inhibitor", 5),
            ("Antimycin A", 548.63, "complex_III_inhibitor", 10),
            ("Oligomycin", 791.06, "ATP_synthase_inhibitor", 20),
            ("FCCP", 254.17, "uncoupler", 100),
            ("Iodoacetate", 185.95, "GAPDH_inhibitor", 1000),
            ("Fluorocitrate", 209.09, "aconitase_inhibitor", 50),
            ("Malonate", 104.06, "SDH_inhibitor", 5000),
            ("3-Bromopyruvate", 166.96, "hexokinase_inhibitor", 100),
            ("Lonidamine", 321.16, "hexokinase_inhibitor", 150),
            ("FX11", 408.88, "LDH_inhibitor", 20),
            ("UK5099", 407.25, "MPC_inhibitor", 50),
            ("BAY-876", 494.88, "GLUT1_inhibitor", 2),
            ("Fasentin", 388.46, "GLUT_inhibitor", 100),
            ("Phloretin", 274.27, "GLUT_inhibitor", 200),
            ("BPTES", 533.06, "glutaminase_inhibitor", 10),
            ("DON", 171.15, "glutamine_antagonist", 50),
            ("AOA", 167.16, "transaminase_inhibitor", 100),
            ("TOFA", 313.67, "ACC_inhibitor", 200),
            ("C75", 254.41, "FAS_inhibitor", 50),
            ("GW9662", 276.74, "PPARγ_antagonist", 100),
            ("MK-886", 525.88, "FLAP_inhibitor", 30),
            ("Indomethacin", 357.79, "COX_inhibitor", 100),
        ]
        
        for inhibitor_name, mw, mechanism, ic50_nm in inhibitors:
            entity = BiologicalEntity(
                id=f"inhibitor_{inhibitor_name.replace('-', '_').replace(' ', '_')}",
                name=inhibitor_name,
                entity_type="inhibitor",
                properties={
                    "molecular_weight": mw,
                    "mechanism": mechanism,
                    "ic50_nm": ic50_nm,
                    "source": "kg_builder"
                },
                confidence_score=0.91
            )
            self.kg.add_entity(entity)
        
        # Add allosteric regulators (20 as specified)
        allosteric_regulators = [
            # Positive allosteric modulators
            ("AMP", 347.22, "positive", "AMPK_PFK", 50),
            ("ADP", 427.20, "positive", "PFK", 100),
            ("Fructose-2,6-bisphosphate", 340.12, "positive", "PFK", 10),
            ("Acetyl-CoA", 809.57, "positive", "PC", 20),
            ("NAD+", 663.43, "positive", "SIRT", 50),
            ("Ca2+", 40.08, "positive", "multiple", 1),
            ("cAMP", 329.21, "positive", "PKA", 10),
            ("Insulin", 5808, "positive", "metabolism", 0.1),
            ("Glucagon", 3485, "positive", "glycogenolysis", 0.01),
            ("T3", 650.98, "positive", "metabolism", 0.001),
            
            # Negative allosteric modulators
            ("ATP", 507.18, "negative", "PFK_PK", 1000),
            ("Citrate", 192.12, "negative", "PFK", 500),
            ("Glucose-6-phosphate", 260.14, "negative", "HK", 100),
            ("NADH", 665.44, "negative", "PDH", 50),
            ("Palmitoyl-CoA", 1005.94, "negative", "ACC", 10),
            ("Malonyl-CoA", 853.58, "negative", "CPT1", 5),
            ("GTP", 523.18, "negative", "PEP_CK", 100),
            ("Cortisol", 362.46, "negative", "metabolism", 1),
            ("Leptin", 16024, "negative", "appetite", 0.01),
            ("Adiponectin", 26000, "negative", "gluconeogenesis", 0.1),
        ]
        
        for reg_name, mw, reg_type, target, ka_um in allosteric_regulators:
            entity = BiologicalEntity(
                id=f"regulator_{reg_name.replace(' ', '_').replace('-', '_').replace('+', 'plus')}",
                name=reg_name,
                entity_type="allosteric_regulator",
                properties={
                    "molecular_weight": mw,
                    "regulation_type": reg_type,
                    "target": target,
                    "ka_um": ka_um,
                    "source": "kg_builder"
                },
                confidence_score=0.88
            )
            self.kg.add_entity(entity)
        
        
        # Add relationships between enzymes and substrates
        self._add_enzyme_substrate_relationships()
        
        logger.info(f"Added enzyme kinetics entities: {len([e for e in self.kg.entities.values() if 'enzyme' in e.entity_type or 'substrate' in e.entity_type])} entities")
    
    def build_multiscale_entities(self):
        """Add all multi-scale system entities (Section 4.2)"""
        logger.info("Building multi-scale entities...")
        
        # Receptors
        receptors = [
            ("EGFR", "Epidermal growth factor receptor", "ErbB1"),
            ("HER2", "Human epidermal growth factor receptor 2", "ErbB2"),
            ("VEGFR1", "Vascular endothelial growth factor receptor 1", "FLT1"),
            ("VEGFR2", "Vascular endothelial growth factor receptor 2", "KDR"),
            ("PDGFRA", "Platelet-derived growth factor receptor alpha", None),
            ("PDGFRB", "Platelet-derived growth factor receptor beta", None),
            ("FGFR1", "Fibroblast growth factor receptor 1", None),
            ("IGF1R", "Insulin-like growth factor 1 receptor", None),
            ("INSR", "Insulin receptor", None),
            ("MET", "Hepatocyte growth factor receptor", "c-Met"),
            ("KIT", "Stem cell factor receptor", "c-Kit"),
            ("ALK", "Anaplastic lymphoma kinase", None),
        ]
        
        for receptor_id, name, alt_name in receptors:
            entity = BiologicalEntity(
                id=f"receptor_{receptor_id}",
                name=name,
                entity_type="receptor",
                properties={
                    "alt_name": alt_name,
                    "receptor_type": "tyrosine_kinase",
                    "source": "kg_builder"
                },
                confidence_score=0.95
            )
            self.kg.add_entity(entity)
        
        # Ligands
        ligands = [
            ("EGF", "Epidermal growth factor", 6.05),
            ("TGFα", "Transforming growth factor alpha", 5.5),
            ("VEGF-A", "Vascular endothelial growth factor A", 45.0),
            ("VEGF-B", "Vascular endothelial growth factor B", 21.0),
            ("PDGF-AA", "Platelet-derived growth factor AA", 28.0),
            ("PDGF-BB", "Platelet-derived growth factor BB", 24.0),
            ("FGF1", "Fibroblast growth factor 1", 15.5),
            ("FGF2", "Fibroblast growth factor 2", 17.2),
            ("IGF1", "Insulin-like growth factor 1", 7.6),
            ("Insulin", "Insulin", 5.8),
            ("HGF", "Hepatocyte growth factor", 84.0),
            ("SCF", "Stem cell factor", 18.5),
        ]
        
        for ligand_id, name, mw_kda in ligands:
            entity = BiologicalEntity(
                id=f"ligand_{ligand_id.replace('-', '_')}",
                name=name,
                entity_type="ligand",
                properties={
                    "molecular_weight_kda": mw_kda,
                    "ligand_type": "growth_factor",
                    "source": "kg_builder"
                },
                confidence_score=0.94
            )
            self.kg.add_entity(entity)
        
        # Signaling proteins
        signaling_proteins = [
            ("RAS", "RAS GTPase", "HRAS/KRAS/NRAS"),
            ("RAF1", "RAF proto-oncogene serine/threonine kinase", "c-Raf"),
            ("BRAF", "B-Raf proto-oncogene", None),
            ("MEK1", "Mitogen-activated protein kinase kinase 1", "MAP2K1"),
            ("MEK2", "Mitogen-activated protein kinase kinase 2", "MAP2K2"),
            ("ERK1", "Extracellular signal-regulated kinase 1", "MAPK3"),
            ("ERK2", "Extracellular signal-regulated kinase 2", "MAPK1"),
            ("PI3K", "Phosphoinositide 3-kinase", "PIK3CA"),
            ("AKT1", "AKT serine/threonine kinase 1", "PKB"),
            ("mTOR", "Mechanistic target of rapamycin", None),
            ("PTEN", "Phosphatase and tensin homolog", None),
        ]
        
        for protein_id, name, alt_name in signaling_proteins:
            entity = BiologicalEntity(
                id=f"signaling_{protein_id}",
                name=name,
                entity_type="signaling_protein",
                properties={
                    "alt_name": alt_name,
                    "pathway": "growth_signaling",
                    "source": "kg_builder"
                },
                confidence_score=0.93
            )
            self.kg.add_entity(entity)
        
        # Compartments
        compartments = [
            ("extracellular", "Extracellular space", 1000.0, 7.4),
            ("plasma_membrane", "Plasma membrane", 0.001, 7.0),
            ("cytoplasm", "Cytoplasm", 2.0, 7.2),
            ("nucleus", "Nucleus", 0.5, 7.8),
            ("mitochondria", "Mitochondria", 0.2, 7.8),
            ("er", "Endoplasmic reticulum", 0.3, 7.2),
            ("golgi", "Golgi apparatus", 0.1, 6.5),
            ("lysosomes", "Lysosomes", 0.05, 4.5),
        ]
        
        for comp_id, name, volume_ml, ph in compartments:
            entity = BiologicalEntity(
                id=f"compartment_{comp_id}",
                name=name,
                entity_type="compartment",
                properties={
                    "volume_ml": volume_ml,
                    "pH": ph,
                    "source": "kg_builder"
                },
                confidence_score=0.96
            )
            self.kg.add_entity(entity)
        
        # Cell types
        cell_types = [
            ("epithelial", "Epithelial cells", ["EGFR", "HER2"]),
            ("endothelial", "Endothelial cells", ["VEGFR1", "VEGFR2"]),
            ("fibroblast", "Fibroblasts", ["PDGFRA", "FGFR1"]),
            ("tcell", "T cells", ["CD3", "CD4", "CD8"]),
            ("smooth_muscle", "Smooth muscle cells", ["PDGFRB"]),
        ]
        
        for cell_id, name, markers in cell_types:
            entity = BiologicalEntity(
                id=f"celltype_{cell_id}",
                name=name,
                entity_type="cell_type",
                properties={
                    "markers": markers,
                    "source": "kg_builder"
                },
                confidence_score=0.92
            )
            self.kg.add_entity(entity)
        
        # Add multi-scale relationships
        self._add_multiscale_relationships()
        
        logger.info(f"Added multi-scale entities: {len([e for e in self.kg.entities.values() if e.entity_type in ['receptor', 'ligand', 'signaling_protein', 'compartment', 'cell_type']])} entities")
    
    def build_disease_state_entities(self):
        """Add all disease-state entities (Section 4.3)"""
        logger.info("Building disease-state entities...")
        
        # Diseases
        diseases = [
            ("breast_cancer", "Breast cancer", "cancer", 125.0),
            ("lung_cancer", "Lung cancer", "cancer", 228.0),
            ("colorectal_cancer", "Colorectal cancer", "cancer", 38.0),
            ("pancreatic_cancer", "Pancreatic cancer", "cancer", 13.0),
            ("prostate_cancer", "Prostate cancer", "cancer", 110.0),
            ("type2_diabetes", "Type 2 diabetes", "metabolic", 425.0),
            ("metabolic_syndrome", "Metabolic syndrome", "metabolic", 340.0),
            ("atherosclerosis", "Atherosclerosis", "cardiovascular", 200.0),
            ("heart_failure", "Heart failure", "cardiovascular", 64.0),
            ("alzheimers", "Alzheimer's disease", "neurodegenerative", 47.0),
            ("parkinsons", "Parkinson's disease", "neurodegenerative", 13.0),
            ("rheumatoid_arthritis", "Rheumatoid arthritis", "autoimmune", 40.0),
        ]
        
        for disease_id, name, category, prevalence in diseases:
            entity = BiologicalEntity(
                id=f"disease_{disease_id}",
                name=name,
                entity_type="disease",
                properties={
                    "category": category,
                    "prevalence_per_100k": prevalence,
                    "source": "kg_builder"
                },
                confidence_score=0.94
            )
            self.kg.add_entity(entity)
        
        # Biomarkers
        biomarkers = [
            ("CA15-3", "Cancer antigen 15-3", "protein", 30.0, "breast_cancer"),
            ("CA125", "Cancer antigen 125", "protein", 35.0, "ovarian_cancer"),
            ("CEA", "Carcinoembryonic antigen", "protein", 5.0, "colorectal_cancer"),
            ("PSA", "Prostate-specific antigen", "protein", 4.0, "prostate_cancer"),
            ("HER2", "Human epidermal growth factor receptor 2", "protein", None, "breast_cancer"),
            ("PD-L1", "Programmed death-ligand 1", "protein", None, "various_cancers"),
            ("BRCA1", "Breast cancer gene 1", "genetic", None, "breast_cancer"),
            ("BRCA2", "Breast cancer gene 2", "genetic", None, "breast_cancer"),
            ("KRAS", "KRAS mutation", "genetic", None, "various_cancers"),
            ("CRP", "C-reactive protein", "protein", 3.0, "inflammation"),
            ("IL-6", "Interleukin-6", "protein", 7.0, "inflammation"),
            ("TNF-α", "Tumor necrosis factor alpha", "protein", 8.5, "inflammation"),
            ("HbA1c", "Glycated hemoglobin", "metabolite", 5.7, "diabetes"),
            ("Glucose", "Blood glucose", "metabolite", 100.0, "diabetes"),
            ("LDL", "LDL cholesterol", "metabolite", 100.0, "cardiovascular"),
            ("Troponin-I", "Troponin I", "protein", 0.04, "cardiac"),
        ]
        
        for marker_id, name, marker_type, normal_range, indication in biomarkers:
            entity = BiologicalEntity(
                id=f"biomarker_{marker_id.replace('-', '_').replace(' ', '_')}",
                name=name,
                entity_type="biomarker",
                properties={
                    "marker_type": marker_type,
                    "normal_range": normal_range,
                    "indication": indication,
                    "source": "kg_builder"
                },
                confidence_score=0.93
            )
            self.kg.add_entity(entity)
        
        # Disease-associated proteins
        disease_proteins = [
            ("TP53", "Tumor protein p53", "tumor_suppressor"),
            ("MYC", "MYC proto-oncogene", "oncogene"),
            ("RAS", "RAS proteins", "oncogene"),
            ("PI3K", "Phosphoinositide 3-kinase", "oncogene"),
            ("PTEN", "Phosphatase and tensin homolog", "tumor_suppressor"),
            ("APC", "Adenomatous polyposis coli", "tumor_suppressor"),
            ("VHL", "Von Hippel-Lindau", "tumor_suppressor"),
            ("RB1", "Retinoblastoma protein", "tumor_suppressor"),
            ("CDKN2A", "Cyclin-dependent kinase inhibitor 2A", "tumor_suppressor"),
            ("BCL2", "B-cell lymphoma 2", "anti_apoptotic"),
            ("BAX", "BCL2 associated X", "pro_apoptotic"),
            ("NF-κB", "Nuclear factor kappa B", "transcription_factor"),
            ("HIF-1α", "Hypoxia-inducible factor 1-alpha", "transcription_factor"),
            ("AMPK", "AMP-activated protein kinase", "metabolic_regulator"),
        ]
        
        for protein_id, name, protein_type in disease_proteins:
            entity = BiologicalEntity(
                id=f"disease_protein_{protein_id.replace('-', '_')}",
                name=name,
                entity_type="disease_protein",
                properties={
                    "protein_type": protein_type,
                    "source": "kg_builder"
                },
                confidence_score=0.92
            )
            self.kg.add_entity(entity)
        
        # Phenotypes
        phenotypes = [
            ("proliferation", "Cell proliferation", 8),
            ("apoptosis_resistance", "Apoptosis resistance", 7),
            ("angiogenesis", "Angiogenesis", 6),
            ("metastasis", "Metastasis", 9),
            ("drug_resistance", "Drug resistance", 8),
            ("metabolic_reprogramming", "Metabolic reprogramming", 7),
            ("immune_evasion", "Immune evasion", 8),
            ("senescence", "Senescence", 5),
            ("autophagy", "Autophagy", 6),
            ("EMT", "Epithelial-mesenchymal transition", 7),
            ("hypoxia_response", "Hypoxia response", 6),
            ("inflammation", "Inflammation", 7),
            ("fibrosis", "Fibrosis", 6),
        ]
        
        for phenotype_id, name, severity in phenotypes:
            entity = BiologicalEntity(
                id=f"phenotype_{phenotype_id}",
                name=name,
                entity_type="phenotype",
                properties={
                    "severity_score": severity,
                    "source": "kg_builder"
                },
                confidence_score=0.91
            )
            self.kg.add_entity(entity)
        
        # Add disease relationships
        self._add_disease_relationships()
        
        logger.info(f"Added disease-state entities: {len([e for e in self.kg.entities.values() if e.entity_type in ['disease', 'biomarker', 'disease_protein', 'phenotype']])} entities")
    
    def build_drug_interaction_entities(self):
        """Add all drug interaction entities (Section 4.4)"""
        logger.info("Building drug interaction entities...")
        
        # Tyrosine Kinase Inhibitors
        tkis = [
            ("Imatinib", "DB00619", ["BCR-ABL", "KIT", "PDGFR"], 0.98),
            ("Erlotinib", "DB00530", ["EGFR"], 0.55),
            ("Gefitinib", "DB00317", ["EGFR"], 0.30),
            ("Lapatinib", "DB01259", ["EGFR", "HER2"], 0.11),
            ("Sunitinib", "DB01268", ["VEGFR", "PDGFR", "KIT"], 0.41),
            ("Sorafenib", "DB00398", ["VEGFR", "PDGFR", "RAF"], 0.23),
            ("Pazopanib", "DB06589", ["VEGFR", "PDGFR", "KIT"], 0.14),
            ("Crizotinib", "DB08865", ["ALK", "ROS1", "MET"], 0.43),
            ("Osimertinib", "DB09330", ["EGFR-T790M"], 0.70),
            ("Cabozantinib", "DB08875", ["MET", "VEGFR2"], 0.25),
        ]
        
        for drug_name, drugbank_id, targets, bioavailability in tkis:
            entity = BiologicalEntity(
                id=f"drug_{drugbank_id}",
                name=drug_name,
                entity_type="drug",
                properties={
                    "drugbank_id": drugbank_id,
                    "drug_class": "tyrosine_kinase_inhibitor",
                    "targets": targets,
                    "bioavailability": bioavailability,
                    "source": "kg_builder"
                },
                confidence_score=0.95
            )
            self.kg.add_entity(entity)
        
        # CYP Substrates/Inhibitors
        cyp_drugs = [
            ("Simvastatin", "DB00641", "CYP3A4", "substrate", 0.05),
            ("Atorvastatin", "DB01076", "CYP3A4", "substrate", 0.12),
            ("Warfarin", "DB00682", "CYP2C9", "substrate", 1.0),
            ("Clopidogrel", "DB00758", "CYP2C19", "substrate", 0.50),
            ("Omeprazole", "DB00338", "CYP2C19", "substrate_inhibitor", 0.35),
            ("Metoprolol", "DB00264", "CYP2D6", "substrate", 0.50),
            ("Fluoxetine", "DB00472", "CYP2D6", "inhibitor", 0.80),
            ("Ketoconazole", "DB01026", "CYP3A4", "strong_inhibitor", 0.01),
            ("Clarithromycin", "DB01211", "CYP3A4", "inhibitor", 0.55),
            ("Rifampin", "DB01045", "CYP3A4", "inducer", 0.95),
        ]
        
        for drug_name, drugbank_id, cyp, interaction_type, bioavailability in cyp_drugs:
            entity = BiologicalEntity(
                id=f"drug_{drugbank_id}",
                name=drug_name,
                entity_type="drug",
                properties={
                    "drugbank_id": drugbank_id,
                    "cyp_enzyme": cyp,
                    "cyp_interaction": interaction_type,
                    "bioavailability": bioavailability,
                    "source": "kg_builder"
                },
                confidence_score=0.94
            )
            self.kg.add_entity(entity)
        
        # Chemotherapy agents
        chemo_drugs = [
            ("Paclitaxel", "DB01229", ["CYP2C8", "CYP3A4"], 0.06),
            ("Docetaxel", "DB01248", ["CYP3A4"], 0.10),
            ("Cyclophosphamide", "DB00531", ["CYP2B6", "CYP3A4"], 0.75),
            ("Tamoxifen", "DB00675", ["CYP2D6"], 1.0),
            ("Doxorubicin", "DB00997", ["multiple"], 0.05),
            ("5-Fluorouracil", "DB00544", ["DPD"], 0.28),
            ("Methotrexate", "DB00563", ["folate"], 0.70),
            ("Vincristine", "DB00541", ["CYP3A4", "CYP3A5"], 0.01),
            ("Cisplatin", "DB00515", ["DNA"], 1.0),
            ("Gemcitabine", "DB00441", ["nucleoside"], 0.50),
        ]
        
        for drug_name, drugbank_id, metabolism, bioavailability in chemo_drugs:
            entity = BiologicalEntity(
                id=f"drug_{drugbank_id}",
                name=drug_name,
                entity_type="drug",
                properties={
                    "drugbank_id": drugbank_id,
                    "drug_class": "chemotherapy",
                    "metabolism": metabolism,
                    "bioavailability": bioavailability,
                    "source": "kg_builder"
                },
                confidence_score=0.93
            )
            self.kg.add_entity(entity)
        
        # CYP Enzymes
        cyp_enzymes = [
            ("CYP3A4", "Most abundant CYP, metabolizes ~50% of drugs", 50),
            ("CYP3A5", "Polymorphic expression", 15),
            ("CYP2D6", "Highly polymorphic", 25),
            ("CYP2C9", "Warfarin metabolism", 20),
            ("CYP2C19", "Clopidogrel activation", 15),
            ("CYP2C8", "Paclitaxel metabolism", 10),
            ("CYP1A2", "Caffeine, theophylline", 15),
            ("CYP2B6", "Efavirenz, cyclophosphamide", 10),
            ("CYP2E1", "Acetaminophen, ethanol", 7),
            ("CYP2A6", "Nicotine metabolism", 4),
        ]
        
        for cyp_id, description, percent_drugs in cyp_enzymes:
            entity = BiologicalEntity(
                id=f"cyp_{cyp_id}",
                name=cyp_id,
                entity_type="enzyme",
                properties={
                    "enzyme_family": "cytochrome_P450",
                    "description": description,
                    "percent_drugs_metabolized": percent_drugs,
                    "source": "kg_builder"
                },
                confidence_score=0.96
            )
            self.kg.add_entity(entity)
        
        # Add drug interaction relationships
        self._add_drug_interaction_relationships()
        
        logger.info(f"Added drug interaction entities: {len([e for e in self.kg.entities.values() if e.entity_type == 'drug' or 'cyp' in e.id])} entities")
    
    def build_complete_kg(self) -> KnowledgeGraph:
        """Backward-compatible alias for build_complete_graph."""
        return self.build_complete_graph()

    def _add_enzyme_substrate_relationships(self):
        """Add relationships between enzymes and their substrates"""
        # Glycolysis pathway relationships
        glycolysis_relations = [
            ("enzyme_HK1", "substrate_Glucose", RelationType.SUBSTRATE_OF),
            ("enzyme_HK1", "product_Glucose_6_phosphate", RelationType.PRODUCT_OF),
            ("enzyme_GPI", "substrate_Glucose_6_phosphate", RelationType.SUBSTRATE_OF),
            ("enzyme_GPI", "substrate_Fructose_6_phosphate", RelationType.PRODUCT_OF),
            ("enzyme_PFK1", "substrate_Fructose_6_phosphate", RelationType.SUBSTRATE_OF),
            ("enzyme_PFK1", "substrate_Fructose_1_6_bisphosphate", RelationType.PRODUCT_OF),
            ("enzyme_PKM", "substrate_Pyruvate", RelationType.PRODUCT_OF),
        ]
        
        for source, target, rel_type in glycolysis_relations:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"pathway": "glycolysis"},
                    mathematical_constraints=["michaelis_menten"] if rel_type == RelationType.SUBSTRATE_OF else [],
                    confidence_score=0.9
                )
                self.kg.add_relationship(rel)
        
        # TCA cycle relationships
        tca_relations = [
            ("enzyme_CS", "substrate_Acetyl_CoA", RelationType.SUBSTRATE_OF),
            ("enzyme_CS", "substrate_Oxaloacetate", RelationType.SUBSTRATE_OF),
            ("enzyme_CS", "substrate_Citrate", RelationType.PRODUCT_OF),
            ("enzyme_IDH1", "substrate_Citrate", RelationType.SUBSTRATE_OF),
            ("enzyme_OGDH", "substrate_α_Ketoglutarate", RelationType.SUBSTRATE_OF),
            ("enzyme_SDH", "substrate_Succinate", RelationType.SUBSTRATE_OF),
            ("enzyme_MDH2", "substrate_Malate", RelationType.SUBSTRATE_OF),
        ]
        
        for source, target, rel_type in tca_relations:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"pathway": "tca_cycle"},
                    mathematical_constraints=["michaelis_menten"] if rel_type == RelationType.SUBSTRATE_OF else [],
                    confidence_score=0.9
                )
                self.kg.add_relationship(rel)
    
    def _add_multiscale_relationships(self):
        """Add multi-scale relationships"""
        # Ligand-receptor binding
        binding_relations = [
            ("ligand_EGF", "receptor_EGFR", RelationType.BINDS_TO),
            ("ligand_VEGF_A", "receptor_VEGFR1", RelationType.BINDS_TO),
            ("ligand_VEGF_A", "receptor_VEGFR2", RelationType.BINDS_TO),
            ("ligand_PDGF_AA", "receptor_PDGFRA", RelationType.BINDS_TO),
            ("ligand_PDGF_BB", "receptor_PDGFRB", RelationType.BINDS_TO),
            ("ligand_FGF1", "receptor_FGFR1", RelationType.BINDS_TO),
            ("ligand_IGF1", "receptor_IGF1R", RelationType.BINDS_TO),
            ("ligand_Insulin", "receptor_INSR", RelationType.BINDS_TO),
            ("ligand_HGF", "receptor_MET", RelationType.BINDS_TO),
            ("ligand_SCF", "receptor_KIT", RelationType.BINDS_TO),
        ]
        
        for source, target, rel_type in binding_relations:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"binding_type": "receptor_ligand"},
                    mathematical_constraints=["simple_binding", "receptor_occupancy"],
                    confidence_score=0.92
                )
                self.kg.add_relationship(rel)
        
        # Receptor to signaling cascade
        signaling_relations = [
            ("receptor_EGFR", "signaling_RAS", RelationType.PHOSPHORYLATES),
            ("signaling_RAS", "signaling_RAF1", RelationType.PHOSPHORYLATES),
            ("signaling_RAF1", "signaling_MEK1", RelationType.PHOSPHORYLATES),
            ("signaling_MEK1", "signaling_ERK1", RelationType.PHOSPHORYLATES),
            ("receptor_EGFR", "signaling_PI3K", RelationType.PHOSPHORYLATES),
            ("signaling_PI3K", "signaling_AKT1", RelationType.PHOSPHORYLATES),
            ("signaling_AKT1", "signaling_mTOR", RelationType.PHOSPHORYLATES),
        ]
        
        for source, target, rel_type in signaling_relations:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"cascade": "growth_signaling"},
                    mathematical_constraints=["phosphorylation_kinetics"],
                    confidence_score=0.89
                )
                self.kg.add_relationship(rel)
        
        # Protein location in compartments
        location_relations = [
            ("receptor_EGFR", "compartment_plasma_membrane", RelationType.LOCATED_IN),
            ("signaling_RAS", "compartment_plasma_membrane", RelationType.LOCATED_IN),
            ("signaling_ERK1", "compartment_cytoplasm", RelationType.LOCATED_IN),
            ("signaling_ERK2", "compartment_nucleus", RelationType.LOCATED_IN),
        ]
        
        for source, target, rel_type in location_relations:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"subcellular_localization": True},
                    mathematical_constraints=[],
                    confidence_score=0.88
                )
                self.kg.add_relationship(rel)
    
    def _add_disease_relationships(self):
        """Add disease-related relationships"""
        # Biomarker for disease
        biomarker_relations = [
            ("biomarker_CA15_3", "disease_breast_cancer", RelationType.BIOMARKER_FOR),
            ("biomarker_PSA", "disease_prostate_cancer", RelationType.BIOMARKER_FOR),
            ("biomarker_HbA1c", "disease_type2_diabetes", RelationType.BIOMARKER_FOR),
            ("biomarker_CRP", "disease_rheumatoid_arthritis", RelationType.BIOMARKER_FOR),
            ("biomarker_Troponin_I", "disease_heart_failure", RelationType.BIOMARKER_FOR),
        ]
        
        for source, target, rel_type in biomarker_relations:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"diagnostic_value": "high"},
                    mathematical_constraints=[],
                    confidence_score=0.87
                )
                self.kg.add_relationship(rel)
        
        # Disease protein associations
        disease_protein_relations = [
            ("disease_protein_TP53", "disease_breast_cancer", RelationType.CAUSES_DISEASE),
            ("disease_protein_TP53", "disease_lung_cancer", RelationType.CAUSES_DISEASE),
            ("disease_protein_MYC", "disease_breast_cancer", RelationType.CAUSES_DISEASE),
            ("disease_protein_RAS", "disease_colorectal_cancer", RelationType.CAUSES_DISEASE),
            ("disease_protein_RAS", "disease_pancreatic_cancer", RelationType.CAUSES_DISEASE),
            ("disease_protein_PTEN", "disease_prostate_cancer", RelationType.CAUSES_DISEASE),
        ]
        
        for source, target, rel_type in disease_protein_relations:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"association_strength": "strong"},
                    mathematical_constraints=[],
                    confidence_score=0.85
                )
                self.kg.add_relationship(rel)
    
    def _add_drug_interaction_relationships(self):
        """Add drug interaction relationships"""
        # Drug-target interactions
        drug_target_relations = [
            ("drug_DB00619", "protein_ABL1", RelationType.COMPETITIVE_INHIBITION),  # Imatinib -> BCR-ABL
            ("drug_DB00530", "receptor_EGFR", RelationType.COMPETITIVE_INHIBITION),  # Erlotinib -> EGFR
            ("drug_DB01268", "receptor_VEGFR2", RelationType.COMPETITIVE_INHIBITION),  # Sunitinib -> VEGFR
            ("drug_DB00641", "enzyme_HMGCR", RelationType.COMPETITIVE_INHIBITION),  # Simvastatin -> HMG-CoA reductase
        ]
        
        for source, target, rel_type in drug_target_relations:
            if self.kg.get_entity(source):
                # Create target if it doesn't exist
                if not self.kg.get_entity(target):
                    entity = BiologicalEntity(
                        id=target,
                        name=target.replace('_', ' '),
                        entity_type="protein",
                        properties={"drug_target": True},
                        confidence_score=0.9
                    )
                    self.kg.add_entity(entity)
                
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"interaction_type": "drug_target"},
                    mathematical_constraints=["competitive_mm"],
                    confidence_score=0.91
                )
                self.kg.add_relationship(rel)
        
        # Drug-drug interactions via CYP
        drug_cyp_relations = [
            ("drug_DB00641", "cyp_CYP3A4", RelationType.SUBSTRATE_OF),  # Simvastatin metabolized by CYP3A4
            ("drug_DB00682", "cyp_CYP2C9", RelationType.SUBSTRATE_OF),  # Warfarin metabolized by CYP2C9
            ("drug_DB00758", "cyp_CYP2C19", RelationType.SUBSTRATE_OF),  # Clopidogrel metabolized by CYP2C19
            ("drug_DB01026", "cyp_CYP3A4", RelationType.COMPETITIVE_INHIBITION),  # Ketoconazole inhibits CYP3A4
            ("drug_DB01045", "cyp_CYP3A4", RelationType.INDUCES),  # Rifampin induces CYP3A4
        ]
        
        for source, target, rel_type in drug_cyp_relations:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"metabolism_pathway": True},
                    mathematical_constraints=["michaelis_menten"] if rel_type == RelationType.SUBSTRATE_OF else ["competitive_mm"],
                    confidence_score=0.88
                )
                self.kg.add_relationship(rel)
        
        # Direct drug-drug interactions
        drug_drug_relations = [
            ("drug_DB00641", "drug_DB01026", RelationType.DRUG_DRUG_INTERACTION),  # Simvastatin-Ketoconazole
            ("drug_DB00682", "drug_DB00472", RelationType.DRUG_DRUG_INTERACTION),  # Warfarin-Fluoxetine
        ]
        
        for source, target, rel_type in drug_drug_relations:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"severity": "major", "mechanism": "cyp_inhibition"},
                    mathematical_constraints=["drug_interaction_competitive"],
                    confidence_score=0.86
                )
                self.kg.add_relationship(rel)
    
    def build_complete_graph(self) -> KnowledgeGraph:
        """Build the complete knowledge graph with all entities and relationships"""
        logger.info("Building complete knowledge graph...")
        
        # Build all entity categories
        self.build_enzyme_kinetics_entities()
        self.build_multiscale_entities()
        self.build_disease_state_entities()
        self.build_drug_interaction_entities()
        
        # Add cross-category relationships
        self._add_cross_category_relationships()
        
        # Calculate final statistics
        total_entities = len(self.kg.entities)
        total_relationships = len(self.kg.relationships)
        
        logger.info(f"Complete knowledge graph built:")
        logger.info(f"  Total entities: {total_entities}")
        logger.info(f"  Total relationships: {total_relationships}")
        
        # Count by type
        entity_types = {}
        for entity in self.kg.entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
        
        logger.info("Entity distribution:")
        for entity_type, count in sorted(entity_types.items()):
            logger.info(f"  {entity_type}: {count}")
        
        return self.kg
    
    def _add_cross_category_relationships(self):
        """Add relationships that connect different experimental categories"""
        # Connect drugs to diseases (treatment relationships)
        treatment_relations = [
            ("drug_DB00619", "disease_breast_cancer", RelationType.TREATS),  # Imatinib for cancer
            ("drug_DB00530", "disease_lung_cancer", RelationType.TREATS),  # Erlotinib for lung cancer
            ("drug_DB00563", "disease_type2_diabetes", RelationType.TREATS),  # Metformin for diabetes
        ]
        
        for source, target, rel_type in treatment_relations:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"approved": True, "indication": "primary"},
                    mathematical_constraints=[],
                    confidence_score=0.93
                )
                self.kg.add_relationship(rel)
        
        # Connect metabolic enzymes to diseases
        enzyme_disease_relations = [
            ("enzyme_HMGCR", "disease_atherosclerosis", RelationType.CAUSES_DISEASE),
            ("enzyme_G6PD", "phenotype_metabolic_reprogramming", RelationType.CAUSES_DISEASE),
        ]
        
        for source, target, rel_type in enzyme_disease_relations:
            if self.kg.get_entity(source) and self.kg.get_entity(target):
                rel = BiologicalRelationship(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    properties={"mechanism": "metabolic_dysfunction"},
                    mathematical_constraints=[],
                    confidence_score=0.84
                )
                self.kg.add_relationship(rel)

def save_knowledge_graph(kg: KnowledgeGraph, filepath: str):
    """Save knowledge graph to file"""
    kg.save(filepath)
    logger.info(f"Knowledge graph saved to {filepath}")

def load_knowledge_graph(filepath: str, config: Dict) -> KnowledgeGraph:
    """Load knowledge graph from file"""
    kg = KnowledgeGraph(config)
    kg.load(filepath)
    logger.info(f"Knowledge graph loaded from {filepath}")
    return kg

def main():
    parser = argparse.ArgumentParser(description='Build comprehensive biological knowledge graph')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file')
    parser.add_argument('--output', type=str, default='kg_cache/knowledge_graph.json', 
                       help='Output file for knowledge graph')
    parser.add_argument('--load-external', action='store_true', 
                       help='Also load from external sources (GO, KEGG, etc.)')
    parser.add_argument('--sources', nargs='+', 
                       help='External sources to load (e.g., GO KEGG DrugBank)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build comprehensive knowledge graph
    builder = ComprehensiveKGBuilder(config)
    kg = builder.build_complete_graph()
    
    # Optionally load from external sources
    if args.load_external and args.sources:
        logger.info(f"Loading from external sources: {args.sources}")
        loader = KnowledgeGraphLoader(config)
        kg = loader.load_from_sources(kg, args.sources, validate=True)
    
    # Save the knowledge graph
    save_knowledge_graph(kg, str(output_path))
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("Knowledge Graph Summary:")
    logger.info("="*60)
    logger.info(f"Total entities: {len(kg.entities)}")
    logger.info(f"Total relationships: {len(kg.relationships)}")
    
    # Verify we have sufficient coverage for experiments
    entity_types = {}
    for entity in kg.entities.values():
        entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
    
    required_counts = {
        'enzyme': 50,
        'substrate': 100,
        'drug': 30,
        'receptor': 30,
        'disease': 20,
        'biomarker': 50
    }
    
    logger.info("\nExperiment coverage check:")
    for req_type, req_count in required_counts.items():
        actual_count = entity_types.get(req_type, 0)
        status = "✓" if actual_count >= req_count else "✗"
        logger.info(f"  {req_type}: {actual_count}/{req_count} {status}")
    
    logger.info("\nKnowledge graph building complete!")
    logger.info(f"Graph saved to: {output_path}")

if __name__ == "__main__":
    main()