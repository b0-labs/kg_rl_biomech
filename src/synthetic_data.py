import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import scipy.integrate as integrate

class SystemType(Enum):
    ENZYME_KINETICS = "enzyme_kinetics"
    MULTI_SCALE = "multi_scale"
    DISEASE_STATE = "disease_state"
    DRUG_INTERACTION = "drug_interaction"
    HIERARCHICAL = "hierarchical"

class NoiseType(Enum):
    ADDITIVE = "additive"
    PROPORTIONAL = "proportional"
    COMBINED = "combined"
    TIME_CORRELATED = "time_correlated"
    HETEROSCEDASTIC = "heteroscedastic"

@dataclass
class SyntheticSystem:
    system_type: SystemType
    mechanism: str
    true_parameters: Dict[str, float]
    data_X: np.ndarray
    data_y: np.ndarray
    noise_type: NoiseType
    complexity_level: int
    metadata: Dict

class SyntheticDataGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.rng = np.random.RandomState(42)
        
    def generate_dataset(self, system_type: SystemType, 
                        num_systems: int = 100) -> List[SyntheticSystem]:
        
        if system_type == SystemType.ENZYME_KINETICS:
            return self._generate_enzyme_kinetics_systems(num_systems)
        elif system_type == SystemType.MULTI_SCALE:
            return self._generate_multi_scale_systems(num_systems)
        elif system_type == SystemType.DISEASE_STATE:
            return self._generate_disease_state_systems(num_systems)
        elif system_type == SystemType.DRUG_INTERACTION:
            return self._generate_drug_interaction_systems(num_systems)
        elif system_type == SystemType.HIERARCHICAL:
            return self._generate_hierarchical_systems(num_systems)
        else:
            raise ValueError(f"Unknown system type: {system_type}")
    
    def _generate_enzyme_kinetics_systems(self, num_systems: int) -> List[SyntheticSystem]:
        systems = []
        
        for i in range(num_systems):
            complexity_level = (i % 4) + 1
            
            if complexity_level == 1:
                system = self._create_basic_michaelis_menten()
            elif complexity_level == 2:
                system = self._create_competitive_inhibition()
            elif complexity_level == 3:
                system = self._create_allosteric_regulation()
            else:
                system = self._create_multi_substrate_inhibition()
            
            systems.append(system)
        
        return systems
    
    def _create_basic_michaelis_menten(self) -> SyntheticSystem:
        v_max = self._sample_log_normal(1.0, 0.5)
        k_m = self._sample_log_normal(0.1, 0.3)
        
        params = {'v_max': v_max, 'k_m': k_m}
        mechanism = f"v_max * S / (k_m + S)"
        
        # Generate substrate values spanning reasonable range around Km
        # From 0.01*Km to 100*Km approximately
        S_values = np.logspace(-3, 1, 50)  # 0.001 to 10
        v_values = v_max * S_values / (k_m + S_values)
        
        v_noisy = self._add_noise(v_values, NoiseType.COMBINED)
        
        return SyntheticSystem(
            system_type=SystemType.ENZYME_KINETICS,
            mechanism=mechanism,
            true_parameters=params,
            data_X=S_values.reshape(-1, 1),
            data_y=v_noisy,
            noise_type=NoiseType.COMBINED,
            complexity_level=1,
            metadata={'substrate': 'S', 'product': 'P'}
        )
    
    def _create_competitive_inhibition(self) -> SyntheticSystem:
        v_max = self._sample_log_normal(1.0, 0.5)
        k_m = self._sample_log_normal(0.1, 0.3)
        k_i = self._sample_log_normal(0.1, 0.3)
        
        params = {'v_max': v_max, 'k_m': k_m, 'k_i': k_i}
        mechanism = f"v_max * S / (k_m * (1 + I/k_i) + S)"
        
        n_points = 50
        # Generate substrate and inhibitor values in reasonable ranges
        S_values = np.logspace(-3, 1, n_points)  # 0.001 to 10
        I_values = np.logspace(-3, 0, n_points)  # 0.001 to 1
        
        X = np.column_stack([S_values, I_values])
        
        v_values = v_max * S_values / (k_m * (1 + I_values/k_i) + S_values)
        v_noisy = self._add_noise(v_values, NoiseType.COMBINED)
        
        return SyntheticSystem(
            system_type=SystemType.ENZYME_KINETICS,
            mechanism=mechanism,
            true_parameters=params,
            data_X=X,
            data_y=v_noisy,
            noise_type=NoiseType.COMBINED,
            complexity_level=2,
            metadata={'substrate': 'S', 'inhibitor': 'I', 'inhibition_type': 'competitive'}
        )
    
    def _create_allosteric_regulation(self) -> SyntheticSystem:
        v_max = self._sample_log_normal(1.0, 0.5)
        k_m = self._sample_log_normal(0.1, 0.3)
        n = self.rng.uniform(1.0, 4.0)
        alpha = self._sample_log_normal(1.0, 1.0)
        k_a = self._sample_log_normal(0.1, 0.3)
        
        params = {'v_max': v_max, 'k_m': k_m, 'n': n, 'alpha': alpha, 'k_a': k_a}
        mechanism = f"v_max * S**n / (k_m**n + S**n) * (1 + alpha * A/k_a) / (1 + A/k_a)"
        
        n_points = 50
        # Generate substrate and allosteric modulator values in reasonable ranges
        S_values = np.logspace(-3, 1, n_points)  # 0.001 to 10
        A_values = np.logspace(-3, 0, n_points)  # 0.001 to 1
        
        X = np.column_stack([S_values, A_values])
        
        v_values = (v_max * S_values**n / (k_m**n + S_values**n) * 
                   (1 + alpha * A_values/k_a) / (1 + A_values/k_a))
        v_noisy = self._add_noise(v_values, NoiseType.PROPORTIONAL)
        
        return SyntheticSystem(
            system_type=SystemType.ENZYME_KINETICS,
            mechanism=mechanism,
            true_parameters=params,
            data_X=X,
            data_y=v_noisy,
            noise_type=NoiseType.PROPORTIONAL,
            complexity_level=3,
            metadata={'substrate': 'S', 'allosteric_modulator': 'A', 'hill_coefficient': n}
        )
    
    def _create_multi_substrate_inhibition(self) -> SyntheticSystem:
        v_max = self._sample_log_normal(1.0, 0.5)
        k_m1 = self._sample_log_normal(0.1, 0.3)
        k_m2 = self._sample_log_normal(0.1, 0.3)
        k_p = self._sample_log_normal(0.05, 0.2)
        
        params = {'v_max': v_max, 'k_m1': k_m1, 'k_m2': k_m2, 'k_p': k_p}
        mechanism = f"v_max * S1 * S2 / ((k_m1 + S1) * (k_m2 + S2) * (1 + P/k_p))"
        
        n_points = 30
        # Generate substrate and product values in reasonable ranges
        S1_values = np.logspace(-3, 1, n_points)  # 0.001 to 10
        S2_values = np.logspace(-3, 1, n_points)  # 0.001 to 10
        P_values = np.logspace(-3, 0, n_points)  # 0.001 to 1
        
        X = np.column_stack([S1_values, S2_values, P_values])
        
        v_values = (v_max * S1_values * S2_values / 
                   ((k_m1 + S1_values) * (k_m2 + S2_values) * (1 + P_values/k_p)))
        v_noisy = self._add_noise(v_values, NoiseType.HETEROSCEDASTIC)
        
        return SyntheticSystem(
            system_type=SystemType.ENZYME_KINETICS,
            mechanism=mechanism,
            true_parameters=params,
            data_X=X,
            data_y=v_noisy,
            noise_type=NoiseType.HETEROSCEDASTIC,
            complexity_level=4,
            metadata={'substrate1': 'S1', 'substrate2': 'S2', 'product': 'P', 
                     'inhibition_type': 'product_inhibition'}
        )
    
    def _generate_multi_scale_systems(self, num_systems: int) -> List[SyntheticSystem]:
        systems = []
        
        for i in range(num_systems):
            k_on = self._sample_log_normal(1e6, 1e5)
            k_off = self._sample_log_normal(1.0, 0.5)
            k_trans = self._sample_log_normal(0.1, 0.05)
            k_deg = self._sample_log_normal(0.01, 0.005)
            E_max = self._sample_log_normal(100.0, 20.0)
            EC50 = self._sample_log_normal(1e-6, 5e-7)
            gamma = self.rng.uniform(1.0, 3.0)
            k_out = self._sample_log_normal(0.1, 0.05)
            
            params = {
                'k_on': k_on, 'k_off': k_off, 'k_trans': k_trans, 'k_deg': k_deg,
                'E_max': E_max, 'EC50': EC50, 'gamma': gamma, 'k_out': k_out
            }
            
            # Solve multi-scale ODEs
            time_points = np.linspace(0, 100, 50)
            # Generate drug concentrations in reasonable range (nM to mM)
            D_values = np.logspace(-9, -3, 50)  # 1 nM to 1 mM
            
            responses = []
            for D in D_values:
                # Solve coupled ODEs for each drug concentration
                response_trajectory = self._solve_multi_scale_odes(
                    D, params, time_points
                )
                # Take steady-state (last time point) response
                responses.append(response_trajectory[-1])
            
            responses = np.array(responses)
            responses_noisy = self._add_noise(responses, NoiseType.COMBINED)
            
            mechanism = "multi_scale_ode_coupling"
            
            systems.append(SyntheticSystem(
                system_type=SystemType.MULTI_SCALE,
                mechanism=mechanism,
                true_parameters=params,
                data_X=D_values.reshape(-1, 1),
                data_y=responses_noisy,
                noise_type=NoiseType.COMBINED,
                complexity_level=2,
                metadata={'scales': ['molecular', 'cellular', 'tissue']}
            ))
        
        return systems
    
    def _solve_multi_scale_odes(self, D: float, params: Dict[str, float], 
                               time_points: np.ndarray) -> np.ndarray:
        """Solve coupled multi-scale ODEs using scipy integrate"""
        from scipy.integrate import odeint
        
        # Initial conditions
        R_total = 1e-6
        y0 = [D, R_total, 0.0, 0.0, 0.0]  # [D, R, DR, S, E]
        
        def multi_scale_dynamics(y, t, p):
            D, R, DR, S, E = y
            
            # Molecular level: drug-receptor binding
            dDR_dt = p['k_on'] * D * R - p['k_off'] * DR
            dR_dt = -dDR_dt
            dD_dt = -dDR_dt
            
            # Cellular level: signal transduction
            f_DR = DR / (DR + p['k_off']/p['k_on'])  # Receptor occupancy
            dS_dt = p['k_trans'] * f_DR - p['k_deg'] * S
            
            # Tissue level: physiological response
            effect = p['E_max'] * S**p['gamma'] / (p['EC50']**p['gamma'] + S**p['gamma'])
            dE_dt = effect - p['k_out'] * E
            
            return [dD_dt, dR_dt, dDR_dt, dS_dt, dE_dt]
        
        # Solve ODEs
        solution = odeint(multi_scale_dynamics, y0, time_points, args=(params,))
        
        # Return tissue-level response (E)
        return solution[:, 4]
    
    def _generate_disease_state_systems(self, num_systems: int) -> List[SyntheticSystem]:
        systems = []
        
        for i in range(num_systems):
            v_max_healthy = self._sample_log_normal(1.0, 0.2)
            v_max_disease = self._sample_log_normal(0.3, 0.1)
            k_m_healthy = self._sample_log_normal(0.1, 0.02)
            k_m_disease = self._sample_log_normal(0.3, 0.05)
            
            biomarker_threshold = self.rng.uniform(0.3, 0.7)
            switching_steepness = self.rng.uniform(5, 20)
            
            params = {
                'v_max_healthy': v_max_healthy, 'v_max_disease': v_max_disease,
                'k_m_healthy': k_m_healthy, 'k_m_disease': k_m_disease,
                'biomarker_threshold': biomarker_threshold,
                'switching_steepness': switching_steepness
            }
            
            n_points = 50
            S_values = np.logspace(-9, -3, n_points)
            biomarker_values = np.linspace(0, 1, n_points)
            
            X = np.column_stack([S_values, biomarker_values])
            
            switch_factor = 1 / (1 + np.exp(-switching_steepness * 
                                           (biomarker_values - biomarker_threshold)))
            
            v_max_effective = v_max_healthy + (v_max_disease - v_max_healthy) * switch_factor
            k_m_effective = k_m_healthy + (k_m_disease - k_m_healthy) * switch_factor
            
            v_values = v_max_effective * S_values / (k_m_effective + S_values)
            v_noisy = self._add_noise(v_values, NoiseType.TIME_CORRELATED)
            
            mechanism = "disease_state_switching"
            
            systems.append(SyntheticSystem(
                system_type=SystemType.DISEASE_STATE,
                mechanism=mechanism,
                true_parameters=params,
                data_X=X,
                data_y=v_noisy,
                noise_type=NoiseType.TIME_CORRELATED,
                complexity_level=3,
                metadata={'biomarker': 'disease_progression', 'states': ['healthy', 'disease']}
            ))
        
        return systems
    
    def _generate_drug_interaction_systems(self, num_systems: int) -> List[SyntheticSystem]:
        systems = []
        
        for i in range(num_systems):
            interaction_type = self.rng.choice(['competitive', 'non_competitive', 'mixed'])
            
            v_max1 = self._sample_log_normal(1.0, 0.3)
            v_max2 = self._sample_log_normal(0.8, 0.3)
            k_m1 = self._sample_log_normal(0.1, 0.05)
            k_m2 = self._sample_log_normal(0.15, 0.05)
            k_interaction = self._sample_log_normal(0.2, 0.1)
            
            params = {
                'v_max1': v_max1, 'v_max2': v_max2,
                'k_m1': k_m1, 'k_m2': k_m2,
                'k_interaction': k_interaction,
                'interaction_type': interaction_type
            }
            
            n_points = 40
            drug1_conc = np.logspace(-9, -4, n_points)
            drug2_conc = np.logspace(-9, -4, n_points)
            
            X = np.column_stack([drug1_conc, drug2_conc])
            
            if interaction_type == 'competitive':
                response = (v_max1 * drug1_conc / (k_m1 * (1 + drug2_conc/k_interaction) + drug1_conc) +
                          v_max2 * drug2_conc / (k_m2 * (1 + drug1_conc/k_interaction) + drug2_conc))
            elif interaction_type == 'non_competitive':
                response = (v_max1 * drug1_conc / ((k_m1 + drug1_conc) * (1 + drug2_conc/k_interaction)) +
                          v_max2 * drug2_conc / ((k_m2 + drug2_conc) * (1 + drug1_conc/k_interaction)))
            else:
                alpha = 0.5
                response = (v_max1 * drug1_conc / (k_m1 * (1 + drug2_conc/k_interaction) + 
                                                  drug1_conc * (1 + alpha * drug2_conc/k_interaction)) +
                          v_max2 * drug2_conc / (k_m2 * (1 + drug1_conc/k_interaction) + 
                                                drug2_conc * (1 + alpha * drug1_conc/k_interaction)))
            
            response_noisy = self._add_noise(response, NoiseType.PROPORTIONAL)
            
            mechanism = f"drug_interaction_{interaction_type}"
            
            systems.append(SyntheticSystem(
                system_type=SystemType.DRUG_INTERACTION,
                mechanism=mechanism,
                true_parameters=params,
                data_X=X,
                data_y=response_noisy,
                noise_type=NoiseType.PROPORTIONAL,
                complexity_level=3,
                metadata={'drug1': 'DrugA', 'drug2': 'DrugB', 'interaction': interaction_type}
            ))
        
        return systems
    
    def _generate_hierarchical_systems(self, num_systems: int) -> List[SyntheticSystem]:
        systems = []
        
        for i in range(num_systems):
            complexity = (i % 5) + 1
            
            if complexity == 1:
                system = self._create_basic_michaelis_menten()
            elif complexity == 2:
                base_system = self._create_basic_michaelis_menten()
                system = self._add_regulation_layer(base_system)
            elif complexity == 3:
                base_system = self._create_competitive_inhibition()
                system = self._add_regulation_layer(base_system)
            elif complexity == 4:
                base_system = self._create_allosteric_regulation()
                system = self._add_feedback_layer(base_system)
            else:
                base_system = self._create_multi_substrate_inhibition()
                system = self._add_feedback_layer(base_system)
            
            system.system_type = SystemType.HIERARCHICAL
            system.complexity_level = complexity
            systems.append(system)
        
        return systems
    
    def _add_regulation_layer(self, base_system: SyntheticSystem) -> SyntheticSystem:
        regulator_k = self._sample_log_normal(0.1, 0.05)
        regulator_n = self.rng.uniform(1.5, 3.0)
        
        base_response = base_system.data_y
        
        if base_system.data_X.shape[1] < 3:
            regulator_conc = np.logspace(-9, -5, len(base_response))
            X_new = np.column_stack([base_system.data_X, regulator_conc])
        else:
            X_new = base_system.data_X
            regulator_conc = X_new[:, -1]
        
        regulation_factor = regulator_conc**regulator_n / (regulator_k**regulator_n + regulator_conc**regulator_n)
        regulated_response = base_response * (1 + 2 * regulation_factor)
        
        new_params = base_system.true_parameters.copy()
        new_params['regulator_k'] = regulator_k
        new_params['regulator_n'] = regulator_n
        
        return SyntheticSystem(
            system_type=base_system.system_type,
            mechanism=base_system.mechanism + "_with_regulation",
            true_parameters=new_params,
            data_X=X_new,
            data_y=regulated_response,
            noise_type=base_system.noise_type,
            complexity_level=base_system.complexity_level + 1,
            metadata={**base_system.metadata, 'regulation': 'positive'}
        )
    
    def _add_feedback_layer(self, base_system: SyntheticSystem) -> SyntheticSystem:
        feedback_strength = self.rng.uniform(0.1, 0.5)
        feedback_delay = self.rng.uniform(1, 5)
        
        base_response = base_system.data_y
        
        feedback_response = np.zeros_like(base_response)
        feedback_response[0] = base_response[0]
        
        for i in range(1, len(base_response)):
            delay_idx = max(0, i - int(feedback_delay))
            feedback_term = feedback_strength * feedback_response[delay_idx]
            feedback_response[i] = base_response[i] / (1 + feedback_term)
        
        new_params = base_system.true_parameters.copy()
        new_params['feedback_strength'] = feedback_strength
        new_params['feedback_delay'] = feedback_delay
        
        return SyntheticSystem(
            system_type=base_system.system_type,
            mechanism=base_system.mechanism + "_with_feedback",
            true_parameters=new_params,
            data_X=base_system.data_X,
            data_y=feedback_response,
            noise_type=base_system.noise_type,
            complexity_level=base_system.complexity_level + 1,
            metadata={**base_system.metadata, 'feedback': 'negative'}
        )
    
    def _solve_binding_equilibrium(self, D: float, R_total: float, 
                                  k_on: float, k_off: float) -> float:
        K_d = k_off / k_on
        DR = (D + R_total + K_d - np.sqrt((D + R_total + K_d)**2 - 4*D*R_total)) / 2
        return DR
    
    def _add_noise(self, values: np.ndarray, noise_type: NoiseType) -> np.ndarray:
        if noise_type == NoiseType.ADDITIVE:
            noise_std = self.config['synthetic_data']['noise_models']['additive_std']
            noise = self.rng.normal(0, noise_std, size=values.shape)
            return values + noise
        
        elif noise_type == NoiseType.PROPORTIONAL:
            noise_std = self.config['synthetic_data']['noise_models']['proportional_std']
            noise = self.rng.normal(0, noise_std, size=values.shape)
            return values * (1 + noise)
        
        elif noise_type == NoiseType.COMBINED:
            add_std = self.config['synthetic_data']['noise_models']['additive_std']
            prop_std = self.config['synthetic_data']['noise_models']['proportional_std']
            add_noise = self.rng.normal(0, add_std, size=values.shape)
            prop_noise = self.rng.normal(0, prop_std, size=values.shape)
            return values * (1 + prop_noise) + add_noise
        
        elif noise_type == NoiseType.TIME_CORRELATED:
            noise_std = self.config['synthetic_data']['noise_models']['additive_std']
            rho = self.config['synthetic_data']['noise_models']['correlation_coeff']
            
            noise = np.zeros_like(values)
            noise[0] = self.rng.normal(0, noise_std)
            
            for i in range(1, len(values)):
                noise[i] = rho * noise[i-1] + np.sqrt(1 - rho**2) * self.rng.normal(0, noise_std)
            
            return values + noise
        
        elif noise_type == NoiseType.HETEROSCEDASTIC:
            base_std = self.config['synthetic_data']['noise_models']['additive_std']
            alpha = self.config['synthetic_data']['noise_models']['heteroscedastic_alpha']
            
            noise_std = base_std + base_std * np.abs(values)**alpha
            noise = self.rng.normal(0, 1, size=values.shape) * noise_std
            
            return values + noise
        
        else:
            return values
    
    def _sample_log_normal(self, mean: float, std: float) -> float:
        return self.rng.lognormal(np.log(mean), std)