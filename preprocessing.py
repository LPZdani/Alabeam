import numpy as np
import pandas as pd
import argparse
from config import (
    EPS, feat_cols, SECTION_TYPES, SUPPORT_TYPES, LOAD_TYPES, LOAD_DIRECTIONS,
    ALL_TARGET_COLUMNS, get_effective_length_factor
)

# ---------- Geometría por tipo de sección ----------
def main_preprocess(IN_CSV, OUT_CSV):
    df = pd.read_csv(IN_CSV)
    
    # ---------- Utilidades ----------
    def safe_div(a, b, eps=EPS):
        return a / np.maximum(b, eps)

    df["A"]  = np.nan
    df["Iz"] = np.nan
    df["Wz"] = np.nan
    df["J"]  = np.nan  # Inercia torsional (nuevo)

    # Máscaras por tipo
    m_BAR  = df["section_type"] == SECTION_TYPES["BAR"]
    m_BOX  = df["section_type"] == SECTION_TYPES["BOX"] 
    m_ROD  = df["section_type"] == SECTION_TYPES["ROD"]
    m_TUBE = df["section_type"] == SECTION_TYPES["TUBE"]
    m_I    = df["section_type"] == SECTION_TYPES["I"]

    # --- BAR ---
    b = df.loc[m_BAR, "dim1"].values
    h = df.loc[m_BAR, "dim2"].values
    A  = b * h
    Iz = b * (h**3) / 12.0
    J = (b * h**3) / 3.0  # Aproximación de inercia torsional para barra rectangular
    Wz = safe_div(Iz, h/2.0)
    df.loc[m_BAR, ["A","Iz","Wz","J"]] = np.c_[A, Iz, Wz, J]

    # --- BOX (rectangular hueca) ---
    b = df.loc[m_BOX, "dim1"].values
    h = df.loc[m_BOX, "dim2"].values
    ty = df.loc[m_BOX, "dim3"].values
    tx = df.loc[m_BOX, "dim4"].values
    bi = np.maximum(b - 2.0*tx, 0.0)
    hi = np.maximum(h - 2.0*ty, 0.0)
    A  = b*h - bi*hi
    Iz = (b*h**3 - bi*hi**3) / 12.0
    J = (b * h**3) / 3.0  # Aproximación de inercia torsional para sección hueca
    Wz = safe_div(Iz, h/2.0)
    df.loc[m_BOX, ["A","Iz","Wz","J"]] = np.c_[A, Iz, Wz, J]

    # --- ROD (circular maciza) ---
    R  = df.loc[m_ROD, "dim1"].values
    D = 2*R 
    A  = np.pi * (D**2) / 4.0
    Iz = np.pi * (D**4) / 64.0
    J = (np.pi * D**4) / 32.0  # Inercia torsional para barra circular
    Wz = safe_div(Iz, D/2.0)
    df.loc[m_ROD, ["A","Iz","Wz","J"]] = np.c_[A, Iz, Wz, J]

    # --- TUBE (tubular circular) ---
    Ro = df.loc[m_TUBE, "dim1"].values
    Do = 2*Ro
    Ri = df.loc[m_TUBE, "dim2"].values
    Di = 2*Ri
    Di = np.minimum(Di, Do - 1e-9) # asegura Di < Do
    A  = (np.pi/4.0)  * (Do**2 - Di**2)
    Iz = (np.pi/64.0) * (Do**4 - Di**4)
    J  = (np.pi/32.0) * (Do**4 - Di**4)  # Inercia torsional para tubo
    Wz = safe_div(Iz, Do/2.0)
    df.loc[m_TUBE, ["A","Iz","Wz","J"]] = np.c_[A, Iz, Wz, J]

    # --- I (perfil en I) ---
    h     = df.loc[m_I, "dim1"].values
    bf_t  = df.loc[m_I, "dim2"].values
    bf_b  = df.loc[m_I, "dim3"].values
    tw    = df.loc[m_I, "dim4"].values
    tf_t  = df.loc[m_I, "dim5"].values
    tf_b  = df.loc[m_I, "dim6"].values
    hw    = np.maximum(h - tf_t - tf_b, 0.0)

    A_top = bf_t * tf_t
    A_bot = bf_b * tf_b
    A_web = tw * hw
    A     = A_top + A_bot + A_web

    Iz_top_cent = (bf_t * tf_t**3) / 12.0
    d_top       = h/2.0 - tf_t/2.0
    Iz_top      = Iz_top_cent + A_top * (d_top**2)

    Iz_bot_cent = (bf_b * tf_b**3) / 12.0
    d_bot       = h/2.0 - tf_b/2.0
    Iz_bot      = Iz_bot_cent + A_bot * (d_bot**2)

    Iz_web = (tw * hw**3) / 12.0
    Iz     = Iz_top + Iz_bot + Iz_web
    J      = (tw * hw**3) / 3.0  # Aproximación de inercia torsional para sección en I
    Wz     = safe_div(Iz, h/2.0)

    df.loc[m_I, ["A","Iz","Wz","J"]] = np.c_[A, Iz, Wz, J]

    # ---------- Rigidez y esbeltez ----------
    df["EI"]         = df["E"] * df["Iz"]
    df["L_over_rz"]  = df["L"] / np.sqrt(safe_div(df["Iz"], df["A"]))
    df["L3_over_EI"] = (df["L"]**3) / np.maximum(df["EI"], EPS)

    # ---------- Calcular L_effective y agregar la columna L_effective_squared ----------
    df["K_factor"] = df.apply(lambda row: get_effective_length_factor(row["support_L"], row["support_R"]), axis=1)
    df["L_effective"] = df["K_factor"] * df["L"]  # Calcular L_effective

    # Añadir el cuadrado de la longitud efectiva
    df["L_effective_squared"] = df["L_effective"] ** 2  # Nueva característica
    
    # ---------- Nuevas características para mejorar predicción de desplazamiento ----------
    # Características basadas en longitud efectiva (más relevantes para desplazamiento)
    df["L_effective_cubed"] = df["L_effective"] ** 3
    df["L_effective_4th"] = df["L_effective"] ** 4
    df["L_effective_over_rz"] = df["L_effective"] / np.sqrt(safe_div(df["Iz"], df["A"]))
    df["L_effective_3_over_EI"] = (df["L_effective"]**3) / np.maximum(df["EI"], EPS)
    df["L_effective_4_over_EI"] = (df["L_effective"]**4) / np.maximum(df["EI"], EPS)
    
    # Características de rigidez mejoradas
    df["EI_over_L"] = safe_div(df["EI"], df["L"])
    df["EI_over_L_effective"] = safe_div(df["EI"], df["L_effective"])
    df["EI_over_L2"] = safe_div(df["EI"], df["L"]**2)
    df["EI_over_L_effective_2"] = safe_div(df["EI"], df["L_effective"]**2)
    
    # Características geométricas adicionales
    df["J_over_A"] = safe_div(df["J"], df["A"])  # Relación de inercia torsional
    df["Iz_over_A2"] = safe_div(df["Iz"], df["A"]**2)  # Momento de inercia normalizado
    df["rz"] = np.sqrt(safe_div(df["Iz"], df["A"]))  # Radio de giro
    df["rz_over_L"] = safe_div(df["rz"], df["L"])
    df["rz_over_L_effective"] = safe_div(df["rz"], df["L_effective"])

    # ---------- Agregados de cargas ----------
    def aggregate_row(row):
        L = row["L"]
        FY = FX = FZ = 0.0
        MZ_left = 0.0
        Mmax_sss = 0.0  # surrogate para momento máx. (viga simplemente apoyada)
        for i in (1,2,3):
            t   = row.get(f"c{i}_type")
            d   = row.get(f"c{i}_dir")
            mag = row.get(f"c{i}_mag", 0.0) or 0.0
            pos = row.get(f"c{i}_pos", np.nan)
            if pd.isna(t) or pd.isna(d) or mag == 0:
                continue
            if t == LOAD_TYPES["FORCE"]:
                if d == LOAD_DIRECTIONS["Y"]:
                    FY += mag
                    if not pd.isna(pos):
                        a = pos; b = max(L - a, 0.0)
                        Mmax_sss += mag * a * b / max(L, EPS)  # Pab/L
                        MZ_left  += mag * a                     # respecto extremo izq.
                elif d == LOAD_DIRECTIONS["X"]:
                    FX += mag
            elif t == LOAD_TYPES["MOMENT"]:
                if d == LOAD_DIRECTIONS["Z"]:
                    MZ_left  += mag
                    Mmax_sss += abs(mag)  # contribución directa
        return pd.Series({"FY_total":FY, "FX_total":FX, "FZ_total":FZ,
                        "MZ_total_left":MZ_left, "Mmax_sss":Mmax_sss})

    loads = df.apply(aggregate_row, axis=1)
    df = pd.concat([df, loads], axis=1)

    # Posiciones normalizadas x/L (si existen)
    for i in (1,2,3):
        if f"c{i}_pos" in df.columns:
            df[f"c{i}_pos_rel"] = df[f"c{i}_pos"] / df["L"]

    # ---------- Escalas físicas útiles para el modelo ----------
    df["delta_scale"] = df["FY_total"] * (df["L"]**3) / np.maximum(df["EI"], EPS)
    df["sigma_scale"] = df["Mmax_sss"] / np.maximum(df["Wz"], EPS)
    
    # ---------- Escalas mejoradas para desplazamiento ----------
    # Escalas basadas en longitud efectiva (más precisas para desplazamiento)
    df["delta_scale_effective"] = df["FY_total"] * (df["L_effective"]**3) / np.maximum(df["EI"], EPS)
    df["delta_scale_effective_4"] = df["FY_total"] * (df["L_effective"]**4) / np.maximum(df["EI"], EPS)
    
    # Escalas de carga normalizadas
    df["load_intensity"] = safe_div(df["FY_total"], df["L"])  # Carga por unidad de longitud
    df["load_intensity_effective"] = safe_div(df["FY_total"], df["L_effective"])
    df["moment_intensity"] = safe_div(df["MZ_total_left"], df["L"])
    
    # Interacciones entre escalas importantes
    df["sigma_delta_interaction"] = df["sigma_scale"] * df["delta_scale"]
    df["sigma_delta_effective_interaction"] = df["sigma_scale"] * df["delta_scale_effective"]
    
    # Características de flexibilidad/rigidez
    df["flexibility_factor"] = (df["L_effective"]**3) / np.maximum(df["EI"], EPS)
    df["stiffness_factor"] = safe_div(df["EI"], df["L_effective"]**3)
    df["normalized_load"] = df["FY_total"] * df["flexibility_factor"]
    
    # Características combinadas K_factor
    df["K_factor_squared"] = df["K_factor"] ** 2
    df["K_factor_cubed"] = df["K_factor"] ** 3
    df["K_factor_L_interaction"] = df["K_factor"] * df["L"]
    df["K_factor_EI_interaction"] = df["K_factor"] * df["EI"]
    
    # ---------- Interacciones polinómicas adicionales para desplazamiento ----------
    # Interacciones entre las características más importantes según feature importance
    df["sigma_scale_L_effective"] = df["sigma_scale"] * df["L_effective"]
    df["sigma_scale_L_effective_squared"] = df["sigma_scale"] * df["L_effective_squared"]
    df["delta_scale_L_effective"] = df["delta_scale"] * df["L_effective"]
    df["L_effective_L3_over_EI"] = df["L_effective"] * df["L3_over_EI"]
    df["L_effective_squared_L_over_rz"] = df["L_effective_squared"] * df["L_over_rz"]
    
    # Ratios y productos útiles para vigas
    df["moment_over_stiffness"] = safe_div(df["Mmax_sss"], df["EI"])
    df["load_over_stiffness"] = safe_div(df["FY_total"], df["EI"])
    df["total_moment_L_ratio"] = safe_div(df["MZ_total_left"], df["L"])
    
    # Características log para capturar relaciones no lineales
    df["log_L_effective"] = np.log1p(df["L_effective"])  # log(1 + x) para evitar log(0)
    df["log_EI"] = np.log1p(df["EI"])
    df["log_delta_scale"] = np.log1p(np.abs(df["delta_scale"]))
    df["log_sigma_scale"] = np.log1p(np.abs(df["sigma_scale"]))
    
    # targets y reacciones si existen
    feat_cols_local = feat_cols.copy()  # Create local copy to avoid modifying the imported list
    for c in ALL_TARGET_COLUMNS:
        if c in df.columns: feat_cols_local.append(c)

    feat_cols_local = [c for c in feat_cols_local if c in df.columns]  # robustez
    df_out = df[feat_cols_local].copy()

    # ---------- Guardado ----------
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Guardado → {OUT_CSV}")
