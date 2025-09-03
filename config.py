EPS = 1e-12                               # para evitar divisiones por cero

# ============================================================================
# MACHINE LEARNING CONFIGURATION
# ============================================================================

# Default training parameters
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42
EPS_MIN = 1e-12

# Model hyperparameters
HGB_PARAMS = {
    "random_state": DEFAULT_RANDOM_STATE,
    "l2_regularization": 0.00018,
    "learning_rate": 0.019,
    "max_depth": 7,
    "min_samples_leaf": 17,
    "max_iter": 388,
    "min_samples_leaf": 14
}

RF_PARAMS = {
    "n_estimators": 400,
    "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": -1
}

PERMUTATION_IMPORTANCE_PARAMS = {
    "n_repeats": 10,
    "random_state": DEFAULT_RANDOM_STATE
}

# ============================================================================
# STRUCTURAL ANALYSIS CONSTANTS
# ============================================================================

# Section types
SECTION_TYPES = {
    "BAR": "BAR",
    "BOX": "BOX", 
    "ROD": "ROD",
    "TUBE": "TUBE",
    "I": "I"
}

# Support types
SUPPORT_TYPES = {
    "CLAMPED": "clamped",
    "SIMPLE": "simple"
}

# Load types
LOAD_TYPES = {
    "FORCE": "force",
    "MOMENT": "moment"
}

# Load directions
LOAD_DIRECTIONS = {
    "X": "X",
    "Y": "Y", 
    "Z": "Z"
}

# Effective length factors (K-factors) for different boundary conditions
K_FACTORS = {
    "cantilever": 2.0,              # Fixed-free
    "reverse_cantilever": 2.0,      # Free-fixed  
    "simply_supported": 1.0,        # Pinned-pinned
    "fixed_both": 0.5,              # Fixed-fixed
    "fixed_pinned": 0.7,            # Fixed-pinned or pinned-fixed
    "default": 1.0                  # Default to simply supported
}

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Target columns (outputs to predict)
TARGET_COLUMNS = [
    "max_displacement", 
    "max_stress"
]

# Additional columns that might be targets or reaction outputs
OPTIONAL_TARGET_COLUMNS = [
    "left_FX", "left_FY", "left_MZ", 
    "right_FX", "right_FY", "right_MZ", 
    "falla"
]

# All possible target/output columns
ALL_TARGET_COLUMNS = TARGET_COLUMNS + OPTIONAL_TARGET_COLUMNS

# Candidate features for ML training (physics-based selection)
ML_CANDIDATE_FEATURES = [
    # Geometric and mechanical properties
    "L", "A", "Iz", "Wz", "EI", "L_over_rz", "L3_over_EI",
    "L_effective_squared",  # Nueva característica
    "J_over_A",            # Nueva característica (relación de inercia torsional sobre área)
    
    # Enhanced effective length features
    "L_effective_cubed", "L_effective_4th", "L_effective_over_rz",
    "L_effective_3_over_EI", "L_effective_4_over_EI",
    
    # Enhanced stiffness features
    "EI_over_L", "EI_over_L_effective", "EI_over_L2", "EI_over_L_effective_2",
    "rz", "rz_over_L", "rz_over_L_effective", "Iz_over_A2",
    
    # Load aggregations
    "FY_total", "FX_total", "MZ_total_left", "Mmax_sss",
    
    # Physics-based scales
    "delta_scale", "sigma_scale",
    
    # Enhanced displacement scales
    "delta_scale_effective", "delta_scale_effective_4", "load_intensity",
    "load_intensity_effective", "moment_intensity", "sigma_delta_interaction",
    "sigma_delta_effective_interaction", "flexibility_factor", "stiffness_factor",
    "normalized_load",
    
    # Enhanced K_factor features
    "K_factor_squared", "K_factor_cubed", "K_factor_L_interaction", "K_factor_EI_interaction",
    
    # Polynomial interactions for displacement
    "sigma_scale_L_effective", "sigma_scale_L_effective_squared", "delta_scale_L_effective",
    "L_effective_L3_over_EI", "L_effective_squared_L_over_rz", "moment_over_stiffness",
    "load_over_stiffness", "total_moment_L_ratio",
    
    # Log features for non-linear relationships
    "log_L_effective", "log_EI", "log_delta_scale", "log_sigma_scale",
    
    # Load counts by type/direction
    "n_force_X", "n_force_Y", "n_moment_Z",
    
    # Relative positions
    "c1_pos_rel", "c2_pos_rel", "c3_pos_rel",
    
    # Original support features
    "support_L_fixed", "support_L_simple", "support_R_fixed", "support_R_simple", "num_supports",
    
    # Engineering-based support features
    "is_cantilever_reverse", "is_simply_supported", "is_mixed_support",
    "K_factor", "L_effective", "cantilever_stiffness", "simply_supported_stiffness",
    
    # Load configuration
    "num_cargas",
]


# Complete feature list for preprocessing output
feat_cols = [
    # originales clave
    "model_name","L","section_type","dim1","dim2","dim3","dim4","dim5","dim6",
    "material","E","nu","density","yield_strength",
    "support_L","support_R","num_cargas",
    
    # geométricas y mecánicas
    "A","Iz","Wz","J","EI","L_over_rz","L3_over_EI", 
    "L_effective_squared",  # Nueva característica
    "J_over_A",            # Nueva característica
    
    # Enhanced effective length features
    "L_effective_cubed", "L_effective_4th", "L_effective_over_rz",
    "L_effective_3_over_EI", "L_effective_4_over_EI",
    
    # Enhanced stiffness features  
    "EI_over_L", "EI_over_L_effective", "EI_over_L2", "EI_over_L_effective_2",
    "rz", "rz_over_L", "rz_over_L_effective", "Iz_over_A2",
    
    # apoyos (original + engineering features)
    "support_L_fixed","support_L_simple","support_R_fixed","support_R_simple","num_supports",
    "is_cantilever","is_cantilever_reverse","is_simply_supported","is_fixed_both","is_mixed_support",
    "K_factor","L_effective","cantilever_stiffness","simply_supported_stiffness",
    
    # Enhanced K_factor features
    "K_factor_squared", "K_factor_cubed", "K_factor_L_interaction", "K_factor_EI_interaction",
    
    # cargas
    "FY_total","FX_total","FZ_total","MZ_total_left","Mmax_sss",
    "delta_scale","sigma_scale",
    
    # Enhanced displacement scales
    "delta_scale_effective", "delta_scale_effective_4", "load_intensity",
    "load_intensity_effective", "moment_intensity", "sigma_delta_interaction", 
    "sigma_delta_effective_interaction", "flexibility_factor", "stiffness_factor",
    "normalized_load",
    
    # Polynomial interactions for displacement
    "sigma_scale_L_effective", "sigma_scale_L_effective_squared", "delta_scale_L_effective",
    "L_effective_L3_over_EI", "L_effective_squared_L_over_rz", "moment_over_stiffness",
    "load_over_stiffness", "total_moment_L_ratio",
    
    # Log features for non-linear relationships
    "log_L_effective", "log_EI", "log_delta_scale", "log_sigma_scale",
    
    "n_force_X","n_force_Y","n_force_Z","n_moment_Z",
    
    # posiciones relativas
    "c1_pos_rel","c2_pos_rel","c3_pos_rel",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_effective_length_factor(support_L, support_R):
    """
    Returns K factor for effective length calculation Le = K*L
    Centralized logic for structural boundary conditions.
    """
    import pandas as pd
    
    if pd.isna(support_R):  # Cantilever
        return K_FACTORS["cantilever"]
    elif pd.isna(support_L):  # Reverse cantilever
        return K_FACTORS["reverse_cantilever"]
    elif support_L == SUPPORT_TYPES["SIMPLE"] and support_R == SUPPORT_TYPES["SIMPLE"]:
        return K_FACTORS["simply_supported"]
    elif support_L == SUPPORT_TYPES["CLAMPED"] and support_R == SUPPORT_TYPES["CLAMPED"]:
        return K_FACTORS["fixed_both"]
    elif ((support_L == SUPPORT_TYPES["CLAMPED"] and support_R == SUPPORT_TYPES["SIMPLE"]) or 
          (support_L == SUPPORT_TYPES["SIMPLE"] and support_R == SUPPORT_TYPES["CLAMPED"])):
        return K_FACTORS["fixed_pinned"]
    else:
        return K_FACTORS["default"]