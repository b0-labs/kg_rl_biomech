# Biological Knowledge Graph Entities and Relationships Design

## Overview

This document defines the complete entity (node) and relationship (edge) structure for the biological knowledge graph supporting all five experimental categories in the scaled-down methodology (500 systems total).

---

## 1. Core Entity Types (Nodes)

### 1.1 Molecular Entities

#### **Proteins/Enzymes**
- **Attributes**: 
  - `id`: Unique identifier (e.g., "enzyme_HK1", "protein_EGFR")
  - `name`: Common name
  - `type`: {enzyme, receptor, transporter, channel, structural}
  - `ec_number`: Enzyme Commission number (for enzymes)
  - `molecular_weight`: kDa
  - `sequence`: Amino acid sequence (optional)
  - `structure`: PDB ID (if available)
  - `confidence_score`: 0-1

#### **Small Molecules**
- **Subtypes**: Substrates, Products, Metabolites, Cofactors, Inhibitors
- **Attributes**:
  - `id`: Unique identifier (e.g., "metabolite_glucose")
  - `name`: Common name
  - `smiles`: Chemical structure
  - `molecular_weight`: g/mol
  - `logP`: Lipophilicity
  - `solubility`: mg/mL
  - `charge`: At physiological pH

#### **Drugs/Compounds**
- **Attributes**:
  - `id`: ChEMBL/PubChem ID
  - `name`: Generic name
  - `brand_names`: List
  - `mechanism_of_action`: Text description
  - `bioavailability`: 0-1
  - `half_life`: Hours
  - `clearance`: L/h
  - `volume_distribution`: L/kg

#### **Genes**
- **Attributes**:
  - `id`: Gene symbol
  - `name`: Full name
  - `chromosome`: Location
  - `start_position`: Base pair
  - `end_position`: Base pair
  - `strand`: +/-

### 1.2 Biological Process Entities

#### **Pathways**
- **Attributes**:
  - `id`: KEGG/Reactome ID
  - `name`: Pathway name
  - `category`: {metabolic, signaling, regulatory}
  - `organism`: Species
  - `components`: List of member entities

#### **Reactions**
- **Attributes**:
  - `id`: Unique identifier
  - `equation`: Chemical equation
  - `reversible`: Boolean
  - `delta_g`: kcal/mol
  - `ec_number`: If enzyme-catalyzed

#### **Biological Processes**
- **Attributes**:
  - `id`: GO term
  - `name`: Process name
  - `category`: {cellular, molecular, physiological}
  - `level`: {molecular, cellular, tissue, organism}

### 1.3 Organizational Entities

#### **Compartments**
- **Attributes**:
  - `id`: Unique identifier
  - `name`: {cytoplasm, nucleus, mitochondria, extracellular}
  - `volume`: L
  - `pH`: Value
  - `ionic_strength`: M

#### **Cell Types**
- **Attributes**:
  - `id`: Cell ontology ID
  - `name`: Cell type name
  - `tissue`: Parent tissue
  - `markers`: List of marker proteins

#### **Tissues/Organs**
- **Attributes**:
  - `id`: UBERON ID
  - `name`: Tissue name
  - `system`: Body system
  - `volume`: mL
  - `blood_flow`: mL/min

### 1.4 Clinical/Phenotype Entities

#### **Diseases**
- **Attributes**:
  - `id`: MONDO/ICD ID
  - `name`: Disease name
  - `category`: Disease type
  - `prevalence`: Cases per 100,000
  - `severity`: {mild, moderate, severe}

#### **Biomarkers**
- **Attributes**:
  - `id`: Unique identifier
  - `name`: Biomarker name
  - `type`: {protein, metabolite, genetic}
  - `normal_range`: Min-max with units
  - `diagnostic_threshold`: Value

#### **Phenotypes**
- **Attributes**:
  - `id`: HPO term
  - `name`: Phenotype description
  - `severity`: Scale 1-10
  - `frequency`: Percentage in population

---

## 2. Relationship Types (Edges)

### 2.1 Catalytic Relationships

#### **CATALYSIS**
- **From**: Enzyme → **To**: Reaction
- **Properties**:
  - `kcat`: Turnover number (s⁻¹)
  - `km`: Michaelis constant (μM)
  - `vmax`: Maximum velocity (μmol/min/mg)
  - `hill_coefficient`: For cooperative binding
  - `mathematical_constraint`: ["michaelis_menten", "hill_equation"]
  - `confidence_score`: 0-1

#### **SUBSTRATE_OF**
- **From**: Small Molecule → **To**: Enzyme
- **Properties**:
  - `binding_affinity`: Kd (μM)
  - `specificity`: {primary, secondary}
  - `mathematical_constraint`: ["simple_binding"]

#### **PRODUCT_OF**
- **From**: Small Molecule → **To**: Reaction
- **Properties**:
  - `stoichiometry`: Integer
  - `rate_limiting`: Boolean

### 2.2 Inhibition Relationships

#### **COMPETITIVE_INHIBITION**
- **From**: Drug/Inhibitor → **To**: Enzyme
- **Properties**:
  - `ki`: Inhibition constant (μM)
  - `ic50`: Half-maximal inhibition (μM)
  - `reversible`: Boolean
  - `mathematical_constraint`: ["competitive_mm"]

#### **NON_COMPETITIVE_INHIBITION**
- **From**: Drug/Inhibitor → **To**: Enzyme
- **Properties**:
  - `ki`: Inhibition constant
  - `mechanism`: {pure_non_competitive, mixed}
  - `mathematical_constraint`: ["non_competitive_mm"]

#### **ALLOSTERIC_REGULATION**
- **From**: Regulator → **To**: Enzyme
- **Properties**:
  - `type`: {positive, negative}
  - `ka`: Allosteric constant (μM)
  - `alpha`: Allosteric factor
  - `mathematical_constraint`: ["allosteric_hill"]

### 2.3 Binding Relationships

#### **BINDS_TO**
- **From**: Ligand → **To**: Receptor
- **Properties**:
  - `kd`: Dissociation constant (nM)
  - `kon`: Association rate (M⁻¹s⁻¹)
  - `koff`: Dissociation rate (s⁻¹)
  - `stoichiometry`: Ratio
  - `mathematical_constraint`: ["simple_binding", "cooperative_binding"]

#### **TRANSPORTS**
- **From**: Transporter → **To**: Small Molecule
- **Properties**:
  - `direction`: {influx, efflux, bidirectional}
  - `mechanism`: {passive, active, facilitated}
  - `km_transport`: μM
  - `vmax_transport`: nmol/min/mg
  - `mathematical_constraint`: ["facilitated_diffusion", "active_transport"]

### 2.4 Regulatory Relationships

#### **INDUCES**
- **From**: Drug → **To**: Enzyme/Gene
- **Properties**:
  - `fold_induction`: Ratio
  - `ec50`: Half-maximal induction (μM)
  - `time_constant`: Hours
  - `mathematical_constraint`: ["enzyme_induction"]

#### **REPRESSES**
- **From**: Protein → **To**: Gene
- **Properties**:
  - `fold_repression`: Ratio
  - `ic50`: Half-maximal repression (nM)
  - `mathematical_constraint`: ["gene_repression"]

#### **PHOSPHORYLATES**
- **From**: Kinase → **To**: Protein
- **Properties**:
  - `site`: Amino acid position
  - `kcat_phos`: s⁻¹
  - `km_atp`: μM
  - `mathematical_constraint`: ["phosphorylation_kinetics"]

### 2.5 Multi-Scale Relationships

#### **EXPRESSED_IN**
- **From**: Gene → **To**: Cell Type/Tissue
- **Properties**:
  - `expression_level`: TPM/FPKM
  - `regulation`: {constitutive, inducible}
  - `conditions`: List of conditions

#### **LOCATED_IN**
- **From**: Protein/Metabolite → **To**: Compartment
- **Properties**:
  - `concentration`: μM
  - `localization_signal`: Sequence
  - `transport_required`: Boolean

#### **PART_OF**
- **From**: Reaction/Enzyme → **To**: Pathway
- **Properties**:
  - `pathway_position`: Integer
  - `rate_limiting`: Boolean
  - `branch_point`: Boolean

### 2.6 Clinical Relationships

#### **CAUSES_DISEASE**
- **From**: Gene/Protein → **To**: Disease
- **Properties**:
  - `penetrance`: 0-1
  - `onset_age`: Years
  - `severity_correlation`: 0-1

#### **BIOMARKER_FOR**
- **From**: Biomarker → **To**: Disease/Drug Response
- **Properties**:
  - `sensitivity`: 0-1
  - `specificity`: 0-1
  - `predictive_value`: 0-1
  - `threshold`: Value with units

#### **TREATS**
- **From**: Drug → **To**: Disease
- **Properties**:
  - `efficacy`: Percentage
  - `nnt`: Number needed to treat
  - `approved`: Boolean
  - `indication`: {primary, secondary}

### 2.7 Drug Interaction Relationships

#### **DRUG_DRUG_INTERACTION**
- **From**: Drug → **To**: Drug
- **Properties**:
  - `interaction_type`: {pharmacokinetic, pharmacodynamic}
  - `mechanism`: {cyp_inhibition, cyp_induction, competition}
  - `severity`: {minor, moderate, major, contraindicated}
  - `fold_change`: Ratio of effect
  - `mathematical_constraint`: ["drug_interaction_competitive", "drug_interaction_synergistic"]

---

## 3. Mathematical Constraints Library

### 3.1 Enzyme Kinetics Constraints

```python
MATHEMATICAL_CONSTRAINTS = {
    # Basic Michaelis-Menten
    "michaelis_menten": "v = (Vmax * [S]) / (Km + [S])",
    
    # Competitive inhibition
    "competitive_mm": "v = (Vmax * [S]) / (Km * (1 + [I]/Ki) + [S])",
    
    # Non-competitive inhibition
    "non_competitive_mm": "v = (Vmax * [S]) / ((Km + [S]) * (1 + [I]/Ki))",
    
    # Hill equation
    "hill_equation": "v = (Vmax * [S]^n) / (Km^n + [S]^n)",
    
    # Allosteric regulation
    "allosteric_hill": "v = (Vmax * [S]^n) / (Km^n + [S]^n) * ((1 + α*[A]/Ka) / (1 + [A]/Ka))",
    
    # Multi-substrate
    "multi_substrate": "v = (Vmax * [S1] * [S2]) / ((Km1 + [S1]) * (Km2 + [S2]))",
    
    # Product inhibition
    "product_inhibition": "v = (Vmax * [S]) / (Km + [S]) * (1 / (1 + [P]/Kp))"
}
```

### 3.2 Binding Kinetics Constraints

```python
BINDING_CONSTRAINTS = {
    # Simple binding
    "simple_binding": "θ = [L] / (Kd + [L])",
    
    # Cooperative binding
    "cooperative_binding": "θ = [L]^n / (Kd^n + [L]^n)",
    
    # Competitive binding
    "competitive_binding": "θ = [L] / (Kd * (1 + [C]/Kc) + [L])",
    
    # Two-site binding
    "two_site_binding": "B = (Bmax1 * [L]) / (Kd1 + [L]) + (Bmax2 * [L]) / (Kd2 + [L])"
}
```

### 3.3 Transport Constraints

```python
TRANSPORT_CONSTRAINTS = {
    # Facilitated diffusion
    "facilitated_diffusion": "J = (Jmax * ([S]out - [S]in)) / (Km + [S]out + [S]in)",
    
    # Active transport
    "active_transport": "J = (Jmax * [S]) / (Km * (1 + [S]/Km) * (1 + [ATP]/Katp))",
    
    # Ion channel
    "ion_channel": "I = g * P_open * (V - E_rev)"
}
```

### 3.4 Multi-Scale Constraints

```python
MULTISCALE_CONSTRAINTS = {
    # Receptor occupancy to signal
    "receptor_signal": "Signal = Smax * ([DR] / ([DR] + Kd))",
    
    # Signal amplification
    "signal_amplification": "Response = A * Signal^n",
    
    # Tissue response
    "tissue_response": "dE/dt = Emax * ([S]^γ / (EC50^γ + [S]^γ)) - kout * E"
}
```

---

## 4. Experiment-Specific Entity Requirements

### 4.1 Enzyme Kinetics (200 systems)
**Required Entities**:
- 50 unique enzymes
- 100 substrates
- 50 products
- 30 inhibitors (competitive, non-competitive)
- 20 allosteric regulators
- 4 complexity levels of mechanisms

### 4.2 Multi-Scale (100 systems)
**Required Entities**:
- 30 receptors
- 30 ligands
- 20 signaling proteins
- 10 compartments
- 5 cell types
- 3 scale levels (molecular → cellular → tissue)

### 4.3 Disease-State (50 systems)
**Required Entities**:
- 20 diseases
- 50 biomarkers
- 30 disease-associated proteins
- 20 phenotypes
- Switching thresholds and progression patterns

### 4.4 Drug Interactions (50 systems)
**Required Entities**:
- 30 drugs (from ChEMBL/PubChem)
- 20 drug targets
- 15 metabolizing enzymes (CYP family)
- 4 interaction types
- Network topologies (star, chain, fully connected)

### 4.5 Complexity Progression (100 systems)
**Required Entities**:
- Hierarchical enzyme families
- Progressive mechanism components
- 5 complexity levels
- Parent-child relationships

---

## 5. Knowledge Graph Statistics Summary

### Overall Graph Metrics
- **Total Entity Types**: 18 primary types
- **Total Relationship Types**: 20 primary types
- **Mathematical Constraints**: 20+ functional forms
- **Hierarchical Levels**: 6 (molecular → subcellular → cellular → tissue → organ → organism)

### Coverage Requirements
- **Enzyme Kinetics Coverage**: >95% of common mechanisms
- **Drug Interaction Coverage**: Major interaction types
- **Disease Coverage**: Common conditions relevant to drug development
- **Pathway Coverage**: Core metabolic and signaling pathways

### Quality Metrics
- **Confidence Scores**: All entities and relationships have confidence scores (0-1)
- **Mathematical Validation**: All constraints have defined parameter ranges
- **Biological Plausibility**: Parameter ranges from literature/databases
- **Completeness**: Sufficient coverage for 500 experimental systems

---

## 6. Implementation Schema

### Node Storage Format
```json
{
  "id": "enzyme_HK1",
  "labels": ["Enzyme", "Protein"],
  "properties": {
    "name": "Hexokinase 1",
    "ec_number": "2.7.1.1",
    "molecular_weight": 102.5,
    "confidence_score": 0.95
  }
}
```

### Edge Storage Format
```json
{
  "source": "enzyme_HK1",
  "target": "metabolite_glucose",
  "type": "SUBSTRATE_OF",
  "properties": {
    "km": 0.1,
    "kcat": 158,
    "mathematical_constraint": "michaelis_menten",
    "confidence_score": 0.9
  }
}
```

### Hierarchical Organization
```
Organism
├── Organ System
│   ├── Organ/Tissue
│   │   ├── Cell Type
│   │   │   ├── Compartment
│   │   │   │   ├── Pathway
│   │   │   │   │   ├── Reaction
│   │   │   │   │   │   ├── Enzyme
│   │   │   │   │   │   └── Metabolite
```

---

## 7. Data Sources for Population

### Open Access Sources (No Authentication Required)
1. **ChEMBL**: Drug-target interactions, bioactivities
2. **PubChem**: Chemical structures, bioassays
3. **KEGG**: Pathways, reactions (limited free access)
4. **Gene Ontology**: Biological processes, molecular functions
5. **UniProt**: Protein information (open access)
6. **Reactome**: Pathway database (open)
7. **STRING**: Protein-protein interactions
8. **BRENDA**: Enzyme kinetics (academic free)

### Literature-Curated Values
- Michaelis-Menten parameters from reviews
- Drug interaction severities from FDA
- Disease-biomarker associations from clinical studies

---

## 8. Validation Requirements

### Biological Validation
- Parameter values within physiologically relevant ranges
- Stoichiometry conservation in reactions
- Thermodynamic feasibility (ΔG considerations)
- Compartment-specific concentrations

### Graph Validation
- No orphan nodes (all nodes have at least one edge)
- Relationship consistency (no contradictory relationships)
- Hierarchical consistency (proper parent-child relationships)
- Mathematical constraint compatibility

### Experimental Validation
- Sufficient entities for each experiment type
- Complete mechanism representation capability
- Parameter identifiability
- Complexity progression support

---

# Complete Entity Names for Knowledge Graph Experiments

## 4.1 Enzyme Kinetics (200 systems)

### 50 Unique Enzymes

#### Glycolysis Enzymes (10)
1. **HK1** (Hexokinase 1) - EC 2.7.1.1
2. **HK2** (Hexokinase 2) - EC 2.7.1.1
3. **GPI** (Glucose-6-phosphate isomerase) - EC 5.3.1.9
4. **PFK1** (Phosphofructokinase 1) - EC 2.7.1.11
5. **ALDOA** (Aldolase A) - EC 4.1.2.13
6. **TPI1** (Triosephosphate isomerase) - EC 5.3.1.1
7. **GAPDH** (Glyceraldehyde-3-phosphate dehydrogenase) - EC 1.2.1.12
8. **PGK1** (Phosphoglycerate kinase 1) - EC 2.7.2.3
9. **ENO1** (Enolase 1) - EC 4.2.1.11
10. **PKM** (Pyruvate kinase M) - EC 2.7.1.40

#### TCA Cycle Enzymes (8)
11. **CS** (Citrate synthase) - EC 2.3.3.1
12. **ACO2** (Aconitase 2) - EC 4.2.1.3
13. **IDH1** (Isocitrate dehydrogenase 1) - EC 1.1.1.42
14. **IDH2** (Isocitrate dehydrogenase 2) - EC 1.1.1.42
15. **OGDH** (α-Ketoglutarate dehydrogenase) - EC 1.2.4.2
16. **SUCLA2** (Succinate-CoA ligase) - EC 6.2.1.5
17. **SDH** (Succinate dehydrogenase) - EC 1.3.5.1
18. **FH** (Fumarase) - EC 4.2.1.2
19. **MDH2** (Malate dehydrogenase 2) - EC 1.1.1.37

#### Amino Acid Metabolism Enzymes (8)
20. **GOT1** (Glutamic-oxaloacetic transaminase 1) - EC 2.6.1.1
21. **GOT2** (Glutamic-oxaloacetic transaminase 2) - EC 2.6.1.1
22. **GPT** (Glutamic-pyruvic transaminase) - EC 2.6.1.2
23. **GLS** (Glutaminase) - EC 3.5.1.2
24. **GLS2** (Glutaminase 2) - EC 3.5.1.2
25. **GLUD1** (Glutamate dehydrogenase 1) - EC 1.4.1.3
26. **ASS1** (Argininosuccinate synthase 1) - EC 6.3.4.5
27. **ASL** (Argininosuccinate lyase) - EC 4.3.2.1

#### Lipid Metabolism Enzymes (7)
28. **FASN** (Fatty acid synthase) - EC 2.3.1.85
29. **ACC1** (Acetyl-CoA carboxylase 1) - EC 6.4.1.2
30. **HMGCR** (HMG-CoA reductase) - EC 1.1.1.34
31. **CPT1A** (Carnitine palmitoyltransferase 1A) - EC 2.3.1.21
32. **HADHA** (Hydroxyacyl-CoA dehydrogenase) - EC 1.1.1.35
33. **ACACA** (Acetyl-CoA carboxylase alpha) - EC 6.4.1.2
34. **LDLR** (LDL receptor - enzymatic activity)

#### Oxidoreductases (8)
35. **CAT** (Catalase) - EC 1.11.1.6
36. **SOD1** (Superoxide dismutase 1) - EC 1.15.1.1
37. **SOD2** (Superoxide dismutase 2) - EC 1.15.1.1
38. **GPX1** (Glutathione peroxidase 1) - EC 1.11.1.9
39. **PRDX1** (Peroxiredoxin 1) - EC 1.11.1.15
40. **NQO1** (NAD(P)H dehydrogenase quinone 1) - EC 1.6.5.2
41. **G6PD** (Glucose-6-phosphate dehydrogenase) - EC 1.1.1.49
42. **ALDH2** (Aldehyde dehydrogenase 2) - EC 1.2.1.3

#### Transferases (8)
43. **GST** (Glutathione S-transferase) - EC 2.5.1.18
44. **UGT1A1** (UDP glucuronosyltransferase 1A1) - EC 2.4.1.17
45. **COMT** (Catechol-O-methyltransferase) - EC 2.1.1.6
46. **TPMT** (Thiopurine S-methyltransferase) - EC 2.1.1.67
47. **NAT1** (N-acetyltransferase 1) - EC 2.3.1.5
48. **NAT2** (N-acetyltransferase 2) - EC 2.3.1.5
49. **SULT1A1** (Sulfotransferase 1A1) - EC 2.8.2.1
50. **AANAT** (Aralkylamine N-acetyltransferase) - EC 2.3.1.87

### 100 Substrates
1. Glucose
2. Glucose-6-phosphate
3. Fructose-6-phosphate
4. Fructose-1,6-bisphosphate
5. Glyceraldehyde-3-phosphate
6. Dihydroxyacetone phosphate
7. 1,3-Bisphosphoglycerate
8. 3-Phosphoglycerate
9. 2-Phosphoglycerate
10. Phosphoenolpyruvate
11. Pyruvate
12. Acetyl-CoA
13. Citrate
14. Isocitrate
15. α-Ketoglutarate
16. Succinyl-CoA
17. Succinate
18. Fumarate
19. Malate
20. Oxaloacetate
21. Lactate
22. Glutamate
23. Glutamine
24. Aspartate
25. Asparagine
26. Alanine
27. Glycine
28. Serine
29. Threonine
30. Cysteine
31. Methionine
32. Valine
33. Leucine
34. Isoleucine
35. Lysine
36. Arginine
37. Histidine
38. Phenylalanine
39. Tyrosine
40. Tryptophan
41. Proline
42. ATP
43. ADP
44. AMP
45. GTP
46. GDP
47. NAD+
48. NADH
49. NADP+
50. NADPH
51. FAD
52. FADH2
53. Coenzyme A
54. S-Adenosylmethionine
55. Tetrahydrofolate
56. Biotin
57. Thiamine pyrophosphate
58. Pyridoxal phosphate
59. Riboflavin
60. Pantothenic acid
61. Nicotinic acid
62. Folic acid
63. Vitamin B12
64. Ascorbic acid
65. α-Tocopherol
66. Palmitic acid
67. Stearic acid
68. Oleic acid
69. Linoleic acid
70. Arachidonic acid
71. Cholesterol
72. Phosphatidylcholine
73. Phosphatidylserine
74. Sphingomyelin
75. Ceramide
76. Diacylglycerol
77. Inositol-1,4,5-trisphosphate
78. cAMP
79. cGMP
80. Calcium ions
81. Magnesium ions
82. Iron ions
83. Zinc ions
84. Copper ions
85. Manganese ions
86. Dopamine
87. Serotonin
88. GABA
89. Acetylcholine
90. Epinephrine
91. Norepinephrine
92. Histamine
93. Melatonin
94. Prostaglandin E2
95. Leukotriene B4
96. Thromboxane A2
97. Nitric oxide
98. Carbon monoxide
99. Hydrogen sulfide
100. Uric acid

### 50 Products
(Many overlap with substrates due to reversible reactions)
1. Glucose-1-phosphate
2. 6-Phosphogluconate
3. Ribulose-5-phosphate
4. Ribose-5-phosphate
5. Xylulose-5-phosphate
6. Sedoheptulose-7-phosphate
7. Erythrose-4-phosphate
8. 2-Oxoglutarate
9. Hydroxypyruvate
10. Glycerate
11. Phosphohydroxypyruvate
12. Phosphoserine
13. N-Acetylglutamate
14. Citrulline
15. Ornithine
16. Urea
17. Creatine
18. Creatinine
19. Glucuronic acid
20. Gluconic acid
21. Sorbitol
22. Fructose
23. Mannose-6-phosphate
24. GDP-mannose
25. UDP-glucose
26. UDP-galactose
27. UDP-glucuronate
28. CMP-sialic acid
29. Acetoacetate
30. β-Hydroxybutyrate
31. Malonyl-CoA
32. Propionyl-CoA
33. Methylmalonyl-CoA
34. HMG-CoA
35. Mevalonate
36. Isopentenyl pyrophosphate
37. Geranyl pyrophosphate
38. Farnesyl pyrophosphate
39. Squalene
40. Lanosterol
41. 7-Dehydrocholesterol
42. Calcitriol
43. Bile acids
44. Bilirubin
45. Biliverdin
46. Heme
47. Protoporphyrin IX
48. Coproporphyrinogen III
49. Uroporphyrinogen III
50. δ-Aminolevulinic acid

### 30 Inhibitors (Competitive and Non-competitive)
1. **Metformin** (Complex I inhibitor)
2. **Rotenone** (Complex I inhibitor)
3. **Antimycin A** (Complex III inhibitor)
4. **Oligomycin** (ATP synthase inhibitor)
5. **2-Deoxyglucose** (Hexokinase inhibitor)
6. **Lonidamine** (Hexokinase inhibitor)
7. **3-Bromopyruvate** (GAPDH inhibitor)
8. **Dichloroacetate** (PDK inhibitor)
9. **Oxamate** (LDH inhibitor)
10. **FX11** (LDHA inhibitor)
11. **CB-839** (Glutaminase inhibitor)
12. **BPTES** (GLS1 inhibitor)
13. **Aminooxyacetate** (Transaminase inhibitor)
14. **Etomoxir** (CPT1 inhibitor)
15. **Orlistat** (FASN inhibitor)
16. **Statins** (HMGCR inhibitors)
17. **Allopurinol** (Xanthine oxidase inhibitor)
18. **Febuxostat** (Xanthine oxidase inhibitor)
19. **Disulfiram** (ALDH inhibitor)
20. **Valproic acid** (HDAC inhibitor)
21. **Vorinostat** (HDAC inhibitor)
22. **Azacitidine** (DNMT inhibitor)
23. **Decitabine** (DNMT inhibitor)
24. **Tranylcypromine** (MAO inhibitor)
25. **Selegiline** (MAO-B inhibitor)
26. **Entacapone** (COMT inhibitor)
27. **Tolcapone** (COMT inhibitor)
28. **Mycophenolic acid** (IMPDH inhibitor)
29. **Ribavirin** (IMPDH inhibitor)
30. **6-Mercaptopurine** (Purine synthesis inhibitor)

### 20 Allosteric Regulators
1. **AMP** (Activator of PFK, AMPK)
2. **ADP** (Activator of PFK)
3. **ATP** (Inhibitor of PFK, citrate synthase)
4. **Citrate** (Inhibitor of PFK, activator of ACC)
5. **Fructose-2,6-bisphosphate** (Activator of PFK)
6. **Acetyl-CoA** (Activator of pyruvate carboxylase)
7. **Malonyl-CoA** (Inhibitor of CPT1)
8. **Palmitate** (Inhibitor of ACC)
9. **cAMP** (Activator of PKA)
10. **Calcium** (Activator of various enzymes)
11. **Phosphoenolpyruvate** (Inhibitor of PFK)
12. **Glucose-6-phosphate** (Activator of glycogen synthase)
13. **NADH** (Inhibitor of several dehydrogenases)
14. **Succinyl-CoA** (Inhibitor of citrate synthase)
15. **GTP** (Activator of PEP carboxykinase)
16. **IMP** (Feedback inhibitor)
17. **CTP** (Feedback inhibitor of aspartate transcarbamoylase)
18. **UTP** (Activator of carbamoyl phosphate synthetase II)
19. **S-Adenosylhomocysteine** (Inhibitor of methyltransferases)
20. **Coenzyme Q10** (Electron transport modulator)

---

## 4.2 Multi-Scale Systems (100 systems)

### 30 Receptors
1. **EGFR** (ErbB1/HER1) - Epidermal growth factor receptor
2. **HER2** (ErbB2/neu) - Human epidermal growth factor receptor 2
3. **HER3** (ErbB3) - Human epidermal growth factor receptor 3
4. **HER4** (ErbB4) - Human epidermal growth factor receptor 4
5. **VEGFR1** (FLT1) - Vascular endothelial growth factor receptor 1
6. **VEGFR2** (KDR/FLK1) - Vascular endothelial growth factor receptor 2
7. **VEGFR3** (FLT4) - Vascular endothelial growth factor receptor 3
8. **PDGFRA** - Platelet-derived growth factor receptor alpha
9. **PDGFRB** - Platelet-derived growth factor receptor beta
10. **FGFR1** - Fibroblast growth factor receptor 1
11. **FGFR2** - Fibroblast growth factor receptor 2
12. **FGFR3** - Fibroblast growth factor receptor 3
13. **FGFR4** - Fibroblast growth factor receptor 4
14. **IGF1R** - Insulin-like growth factor 1 receptor
15. **INSR** - Insulin receptor
16. **MET** (c-Met) - Hepatocyte growth factor receptor
17. **KIT** (c-Kit) - Stem cell factor receptor
18. **RET** - Rearranged during transfection
19. **ALK** - Anaplastic lymphoma kinase
20. **ROS1** - ROS proto-oncogene 1
21. **NTRK1** (TrkA) - Neurotrophic receptor tyrosine kinase 1
22. **NTRK2** (TrkB) - Neurotrophic receptor tyrosine kinase 2
23. **NTRK3** (TrkC) - Neurotrophic receptor tyrosine kinase 3
24. **CSF1R** - Colony stimulating factor 1 receptor
25. **EPHB2** - Ephrin type-B receptor 2
26. **AXL** - AXL receptor tyrosine kinase
27. **TIE2** (TEK) - Tyrosine kinase with immunoglobulin and EGF homology domains
28. **RON** (MST1R) - Macrophage stimulating 1 receptor
29. **DDR1** - Discoidin domain receptor 1
30. **MUSK** - Muscle-specific kinase

### 30 Ligands
1. **EGF** - Epidermal growth factor
2. **TGFα** - Transforming growth factor alpha
3. **HB-EGF** - Heparin-binding EGF-like growth factor
4. **Amphiregulin** (AREG)
5. **Epiregulin** (EREG)
6. **Betacellulin** (BTC)
7. **Neuregulin-1** (NRG1/Heregulin)
8. **Neuregulin-2** (NRG2)
9. **VEGF-A** - Vascular endothelial growth factor A
10. **VEGF-B** - Vascular endothelial growth factor B
11. **VEGF-C** - Vascular endothelial growth factor C
12. **VEGF-D** - Vascular endothelial growth factor D
13. **PlGF** - Placental growth factor
14. **PDGF-AA** - Platelet-derived growth factor AA
15. **PDGF-BB** - Platelet-derived growth factor BB
16. **PDGF-CC** - Platelet-derived growth factor CC
17. **PDGF-DD** - Platelet-derived growth factor DD
18. **FGF1** - Fibroblast growth factor 1 (acidic)
19. **FGF2** - Fibroblast growth factor 2 (basic)
20. **FGF7** - Fibroblast growth factor 7 (KGF)
21. **FGF10** - Fibroblast growth factor 10
22. **IGF1** - Insulin-like growth factor 1
23. **IGF2** - Insulin-like growth factor 2
24. **Insulin**
25. **HGF** - Hepatocyte growth factor
26. **SCF** - Stem cell factor
27. **NGF** - Nerve growth factor
28. **BDNF** - Brain-derived neurotrophic factor
29. **NT-3** - Neurotrophin-3
30. **GDNF** - Glial cell line-derived neurotrophic factor

### 20 Signaling Proteins
1. **RAS** (HRAS, KRAS, NRAS)
2. **RAF1** (c-Raf)
3. **BRAF**
4. **MEK1** (MAP2K1)
5. **MEK2** (MAP2K2)
6. **ERK1** (MAPK3)
7. **ERK2** (MAPK1)
8. **PI3K** (p110α/PIK3CA)
9. **AKT1** (PKB)
10. **AKT2**
11. **mTOR**
12. **PTEN**
13. **PDK1**
14. **PLCγ1**
15. **PKC** (Multiple isoforms)
16. **JAK1**
17. **JAK2**
18. **STAT3**
19. **STAT5**
20. **SRC**

### 10 Compartments
1. **Extracellular space**
2. **Plasma membrane**
3. **Cytoplasm**
4. **Nucleus**
5. **Mitochondria**
6. **Endoplasmic reticulum**
7. **Golgi apparatus**
8. **Lysosomes**
9. **Peroxisomes**
10. **Endosomes**

### 5 Cell Types
1. **Epithelial cells**
2. **Endothelial cells**
3. **Fibroblasts**
4. **Immune cells (T cells)**
5. **Smooth muscle cells**

---

## 4.3 Disease-State Switching (50 systems)

### 20 Diseases
1. **Breast cancer** (ER+, HER2+, TNBC subtypes)
2. **Lung cancer** (NSCLC, SCLC)
3. **Colorectal cancer**
4. **Pancreatic cancer**
5. **Prostate cancer**
6. **Ovarian cancer**
7. **Glioblastoma**
8. **Melanoma**
9. **Renal cell carcinoma**
10. **Hepatocellular carcinoma**
11. **Type 2 diabetes**
12. **Metabolic syndrome**
13. **Atherosclerosis**
14. **Heart failure**
15. **Chronic kidney disease**
16. **Alzheimer's disease**
17. **Parkinson's disease**
18. **Rheumatoid arthritis**
19. **Inflammatory bowel disease**
20. **Sepsis**

### 50 Biomarkers
1. **CA 15-3** (Breast cancer)
2. **CA 125** (Ovarian cancer)
3. **CEA** (Carcinoembryonic antigen)
4. **PSA** (Prostate-specific antigen)
5. **AFP** (Alpha-fetoprotein)
6. **CA 19-9** (Pancreatic cancer)
7. **HER2/neu**
8. **PD-L1**
9. **BRCA1**
10. **BRCA2**
11. **KRAS mutation**
12. **BRAF mutation**
13. **EGFR mutation**
14. **ALK fusion**
15. **ROS1 fusion**
16. **CRP** (C-reactive protein)
17. **IL-6** (Interleukin-6)
18. **TNF-α** (Tumor necrosis factor alpha)
19. **IL-1β**
20. **HbA1c** (Glycated hemoglobin)
21. **Glucose**
22. **Insulin**
23. **Leptin**
24. **Adiponectin**
25. **LDL cholesterol**
26. **HDL cholesterol**
27. **Triglycerides**
28. **BNP** (Brain natriuretic peptide)
29. **Troponin I**
30. **Troponin T**
31. **Creatinine**
32. **eGFR**
33. **Albumin**
34. **ALT** (Alanine aminotransferase)
35. **AST** (Aspartate aminotransferase)
36. **Bilirubin**
37. **Amyloid-β**
38. **Tau protein**
39. **α-Synuclein**
40. **Neurofilament light chain**
41. **S100B**
42. **GFAP** (Glial fibrillary acidic protein)
43. **MMP-9** (Matrix metalloproteinase-9)
44. **VEGF**
45. **Lactate**
46. **Procalcitonin**
47. **D-dimer**
48. **Fibrinogen**
49. **Ferritin**
50. **Cortisol**

### 30 Disease-Associated Proteins
1. **TP53** (p53)
2. **MYC**
3. **RAS proteins** (HRAS, KRAS, NRAS)
4. **PI3K/PIK3CA**
5. **PTEN**
6. **APC**
7. **VHL**
8. **RB1** (Retinoblastoma)
9. **CDKN2A** (p16)
10. **CDK4**
11. **CDK6**
12. **CCND1** (Cyclin D1)
13. **BCL2**
14. **BCL-XL**
15. **MCL1**
16. **BAX**
17. **BAK**
18. **Caspase-3**
19. **Caspase-9**
20. **NF-κB**
21. **HIF-1α**
22. **HIF-2α**
23. **FOXO3**
24. **SIRT1**
25. **AMPK**
26. **LKB1**
27. **TSC1/TSC2**
28. **RHEB**
29. **S6K1**
30. **4E-BP1**

### 20 Phenotypes
1. **Cell proliferation**
2. **Apoptosis resistance**
3. **Angiogenesis**
4. **Metastasis**
5. **Drug resistance**
6. **Metabolic reprogramming**
7. **Immune evasion**
8. **Senescence**
9. **Autophagy**
10. **EMT** (Epithelial-mesenchymal transition)
11. **Stemness**
12. **Hypoxia response**
13. **DNA damage response**
14. **Oxidative stress**
15. **Inflammation**
16. **Fibrosis**
17. **Necrosis**
18. **Ferroptosis**
19. **Pyroptosis**
20. **Mitochondrial dysfunction**

---

## 4.4 Drug Interactions (50 systems)

### 30 Drugs (from ChEMBL/PubChem)

#### Tyrosine Kinase Inhibitors (10)
1. **Imatinib** (Gleevec) - BCR-ABL, KIT, PDGFR inhibitor
2. **Erlotinib** (Tarceva) - EGFR inhibitor
3. **Gefitinib** (Iressa) - EGFR inhibitor
4. **Lapatinib** (Tykerb) - EGFR/HER2 dual inhibitor
5. **Sunitinib** (Sutent) - Multi-kinase inhibitor
6. **Sorafenib** (Nexavar) - Multi-kinase inhibitor
7. **Pazopanib** (Votrient) - VEGFR/PDGFR/KIT inhibitor
8. **Crizotinib** (Xalkori) - ALK/ROS1/MET inhibitor
9. **Osimertinib** (Tagrisso) - EGFR T790M inhibitor
10. **Cabozantinib** (Cometriq) - MET/VEGFR2 inhibitor

#### Common CYP Substrates/Inhibitors (10)
11. **Simvastatin** - CYP3A4 substrate
12. **Atorvastatin** - CYP3A4 substrate
13. **Warfarin** - CYP2C9 substrate
14. **Clopidogrel** - CYP2C19 substrate
15. **Omeprazole** - CYP2C19 substrate/inhibitor
16. **Metoprolol** - CYP2D6 substrate
17. **Fluoxetine** - CYP2D6 inhibitor
18. **Ketoconazole** - Strong CYP3A4 inhibitor
19. **Clarithromycin** - CYP3A4 inhibitor
20. **Rifampin** - CYP3A4 inducer

#### Chemotherapy Agents (10)
21. **Paclitaxel** - CYP2C8/3A4 substrate
22. **Docetaxel** - CYP3A4 substrate
23. **Cyclophosphamide** - CYP2B6/3A4 substrate
24. **Tamoxifen** - CYP2D6 substrate
25. **Doxorubicin** - Multiple pathways
26. **5-Fluorouracil** - DPD substrate
27. **Methotrexate** - Folate antagonist
28. **Vincristine** - CYP3A4/3A5 substrate
29. **Cisplatin** - Direct DNA binding
30. **Gemcitabine** - Nucleoside analog

### 20 Drug Targets
1. **EGFR**
2. **HER2**
3. **VEGFR2**
4. **PDGFRA/B**
5. **BCR-ABL**
6. **KIT**
7. **ALK**
8. **ROS1**
9. **MET**
10. **BRAF**
11. **MEK1/2**
12. **mTOR**
13. **CDK4/6**
14. **PARP**
15. **Topoisomerase II**
16. **Tubulin**
17. **DHFR** (Dihydrofolate reductase)
18. **Thymidylate synthase**
19. **Aromatase**
20. **Androgen receptor**

### 15 Metabolizing Enzymes (CYP Family)
1. **CYP3A4** - Most abundant, metabolizes ~50% of drugs
2. **CYP3A5** - Polymorphic expression
3. **CYP2D6** - Highly polymorphic
4. **CYP2C9** - Warfarin metabolism
5. **CYP2C19** - Clopidogrel activation
6. **CYP2C8** - Paclitaxel metabolism
7. **CYP1A2** - Caffeine, theophylline
8. **CYP2B6** - Efavirenz, cyclophosphamide
9. **CYP2E1** - Acetaminophen, ethanol
10. **CYP2A6** - Nicotine metabolism
11. **CYP1A1** - PAH metabolism
12. **CYP1B1** - Estrogen metabolism
13. **CYP4A11** - Fatty acid metabolism
14. **CYP11A1** - Steroid synthesis
15. **CYP19A1** (Aromatase) - Estrogen synthesis

### 4 Interaction Types
1. **Competitive inhibition** - Same binding site competition
2. **Non-competitive inhibition** - Different binding site
3. **Enzyme induction** - Increased enzyme expression
4. **Mechanism-based inhibition** - Irreversible binding

---

## 4.5 Complexity Progression (100 systems)

### Hierarchical Enzyme Families

#### Level 1: Simple Binding (20 systems)
- Basic receptor-ligand binding
- Simple Michaelis-Menten kinetics
- No cooperativity or regulation

#### Level 2: Saturable Kinetics (20 systems)
- Michaelis-Menten with competitive inhibition
- Single substrate, single product
- Reversible inhibition

#### Level 3: Cooperative Binding (20 systems)
- Hill equation kinetics
- Allosteric regulation
- Multiple binding sites

#### Level 4: Allosteric Regulation (20 systems)
- Complex allosteric mechanisms
- Multiple regulators
- Feedback inhibition/activation

#### Level 5: Multi-Pathway Networks (20 systems)
- Multiple interacting pathways
- Cross-talk between systems
- Temporal dynamics
- Spatial compartmentalization

### Progressive Mechanism Components

#### Basic Components (Level 1)
- Enzyme (E)
- Substrate (S)
- Product (P)
- Simple ES complex

#### Intermediate Components (Level 2-3)
- Inhibitor (I)
- ESI complex
- Activator (A)
- Multiple substrates (S1, S2)

#### Advanced Components (Level 4-5)
- Regulatory subunits
- Scaffolding proteins
- Post-translational modifications
- Compartment-specific factors
- Time-dependent changes
- Feedback loops

### Parent-Child Relationships
- **Oxidoreductases** → Dehydrogenases → Specific dehydrogenases (LDH, MDH, etc.)
- **Transferases** → Kinases → Tyrosine kinases → Receptor tyrosine kinases
- **Hydrolases** → Proteases → Serine proteases → Specific proteases
- **Lyases** → Decarboxylases → Specific decarboxylases
- **Isomerases** → Racemases → Specific racemases
- **Ligases** → Synthetases → Specific synthetases

---

## Summary Statistics

### Total Unique Entities
- **Enzymes**: 50 unique enzymes across all experiments
- **Substrates/Metabolites**: 100+ unique small molecules
- **Products**: 50+ unique products
- **Inhibitors**: 30 competitive/non-competitive inhibitors
- **Allosteric Regulators**: 20 regulatory molecules
- **Receptors**: 30 cell surface receptors
- **Ligands**: 30 growth factors/cytokines
- **Signaling Proteins**: 20 intracellular signaling molecules
- **Drugs**: 30 FDA-approved drugs
- **Drug Targets**: 20 validated drug targets
- **CYP Enzymes**: 15 drug-metabolizing enzymes
- **Diseases**: 20 disease states
- **Biomarkers**: 50 clinical biomarkers
- **Disease Proteins**: 30 disease-associated proteins
- **Phenotypes**: 20 cellular/clinical phenotypes

### Relationship Types
- Catalysis (enzyme → reaction)
- Substrate binding (substrate → enzyme)
- Product formation (reaction → product)
- Competitive inhibition (inhibitor → enzyme)
- Non-competitive inhibition (inhibitor → enzyme)
- Allosteric regulation (regulator → enzyme)
- Receptor binding (ligand → receptor)
- Signal transduction (receptor → signaling protein)
- Drug-target interaction (drug → target)
- Drug-drug interaction (drug → drug via CYP)
- Disease association (protein → disease)
- Biomarker correlation (biomarker → disease state)

### Mathematical Constraints
- Michaelis-Menten kinetics
- Competitive inhibition equations
- Non-competitive inhibition equations
- Hill equation (cooperativity)
- Allosteric modulation
- Multi-substrate kinetics
- Product inhibition
- Receptor occupancy theory
- Dose-response curves
- Drug interaction models

This comprehensive list provides all the specific, scientifically valid entity names needed for your 500 experimental systems, using only open-access data sources.