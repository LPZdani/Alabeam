# app.py
# -------------------------------------------------------------------
# Streamlit app para predicción de vigas con LGBM o Red Neuronal
# Usa tus scripts: preprocessing.py, config.py, predict_neural.py
# -------------------------------------------------------------------

import io
import os
import sys
import tempfile
from pathlib import Path


import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Custom CSS for blue color ---
st.markdown(
    """
    <style>
    /* Header title color */
    .st-emotion-cache-10trblm, .st-emotion-cache-1v0mbdj, h1, h2, h3, h4, h5, h6 {
        color: #214E7A !important;
    }
    /* Selectbox and radio label color */
    label, .st-emotion-cache-1c7y2kd, .st-emotion-cache-1h7ebrz, .st-emotion-cache-1qg05tj {
        color: #214E7A !important;
        font-weight: 600;
    }
    /* Override Streamlit primary color for all widgets */
    :root {
        --primary-color: #214E7A !important;
        --accent-color: #214E7A !important;
        --secondary-background-color: #eaf1f8 !important;
    }
    /* Radio button selected color */
    .stRadio [data-baseweb="radio"] svg {
        color: #214E7A !important;
    }
    .stRadio [data-baseweb="radio"] > div:first-child {
        border-color: #214E7A !important;
    }
    .stRadio [data-baseweb="radio"] [aria-checked="true"] svg {
        color: #214E7A !important;
    }
    .stRadio [data-baseweb="radio"] [aria-checked="true"] > div {
        border-color: #214E7A !important;
    }
    /* Slider bar and thumb color */
    .stSlider > div[data-baseweb="slider"] .css-14g5b3i .css-1gv0vcd,
    .stSlider .css-1gv0vcd,
    .stSlider .css-1eoe787,
    .stSlider .css-1c5h5b6,
    .stSlider .st-c2,
    .stSlider .css-1dp5vir {
        color: #214E7A !important;
        border-color: #214E7A !important;
    }
    .stSlider .css-1c5h5b6 {
        color: #214E7A !important;
    }
    .stSlider .css-1dp5vir {
        color: #214E7A !important;
    }
    /* Slider number color */
    .stSlider .css-1dp5vir, .stSlider .css-1c5h5b6 {
        color: #214E7A !important;
    }
    /* Slider track fill */
    .stSlider .css-1gv0vcd[style*="background: rgb(255, 75, 75)"] {
        background: #214E7A !important;
    }
    /* Slider thumb */
    .stSlider .css-1eoe787[style*="background: rgb(255, 75, 75)"] {
        background: #214E7A !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")  # opcional, silencia warnings de TF en la consola

# Intento flexible de importar tus módulos locales
# (asume que app.py está en la misma carpeta que preprocessing.py, config.py, predict_neural.py)
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from preprocessing import main_preprocess  # tu función de preprocesado
from config import SECTION_TYPES, SUPPORT_TYPES  # para coherencia de valores


# ----------------------------
# Configuración básica app
# ----------------------------

st.set_page_config(
    page_title="AlaBeam",
    page_icon="🔮",
    layout="wide"
)



# Mostrar el logo alineado a la izquierda del header
logo_path = str(HERE / "Alabeam_logo.png")
cols = st.columns([1, 6])
with cols[0]:
    if os.path.exists(logo_path):
        st.image(logo_path, width=500)
with cols[1]:
    st.title("Predicción de vigas | Alabeam")
    st.markdown(
        "**¡Bienvenid@ soy Alabeam!** el genio del cálculo estructural, deja que te ayude a diseñar vigas de manera rápida y eficiente.<br>"
        "Obten predicciones de **desplazamiento máximo** y **tensión máxima** a partir de las características de tu viga.",
        unsafe_allow_html=True
    )

# ----------------------------
# Datos maestros
# ----------------------------
MATERIALS = [
    {"name": "Steel",      "E": 210_000.0, "nu": 0.30, "density": 7.8e-6, "yield_strength": 370.0},
    {"name": "Inox Steel", "E": 210_000.0, "nu": 0.30, "density": 7.9e-6, "yield_strength": 170.0},
    {"name": "Aluminum",   "E":  70_000.0, "nu": 0.32, "density": 2.7e-6, "yield_strength": 270.0},
    {"name": "Titanium",   "E": 120_000.0, "nu": 0.32, "density": 4.5e-6, "yield_strength": 830.0},
]
MAT_INDEX = {m["name"]: m for m in MATERIALS}

# Spanish material names for UI display
MATERIALS_SPANISH = {
    "Acero": "Steel",
    "Acero Inoxidable": "Inox Steel", 
    "Aluminio": "Aluminum",
    "Titanio": "Titanium"
}

# Reverse mapping for recommendations (English to Spanish)
MATERIALS_SPANISH_REVERSE = {v: k for k, v in MATERIALS_SPANISH.items()}

# Mapeos UI → valores que espera tu código
SECTION_LABELS = {
    "Barra rectangular (BAR)": "BAR",
    "Caja/rect. hueca (BOX)": "BOX",
    "Redonda maciza (ROD)": "ROD",
    "Tubo circular (TUBE)": "TUBE",
    "Perfil en I (I)": "I",
}
SUPPORT_LABELS = {
    "Empotrado": "clamped",
    "Simple": "simple",
    "Libre": None,  # se guarda como NaN/None
}
LOAD_TYPES = {"Fuerza": "force", "Momento": "moment"}
LOAD_DIRS_FORCE = {"X": "X", "Y": "Y"}
LOAD_DIRS_MOMENT = {"Z": "Z"}

# ----------------------------
# Cálculo de propiedades geométricas
# ----------------------------
def calculate_section_properties(section_type, dim1, dim2, dim3, dim4, dim5, dim6):
    """
    Calcula el área (A) y momento de inercia (Iz) de la sección transversal.
    """
    EPS = 1e-12  # Para evitar divisiones por cero
    
    if section_type == "ROD":
        # Circular maciza
        R = dim1
        D = 2 * R
        A = np.pi * (D**2) / 4.0
        Iz = np.pi * (D**4) / 64.0
        
    elif section_type == "TUBE":
        # Tubular circular
        Ro = dim1  # Radio externo
        Ri = dim2  # Radio interno
        Do = 2 * Ro
        Di = 2 * Ri
        Di = min(Di, Do - 1e-9)  # Asegura Di < Do
        A = (np.pi/4.0) * (Do**2 - Di**2)
        Iz = (np.pi/64.0) * (Do**4 - Di**4)
        
    elif section_type == "BAR":
        # Barra rectangular
        b = dim1  # ancho
        h = dim2  # alto
        A = b * h
        Iz = b * (h**3) / 12.0
        
    elif section_type == "BOX":
        # Sección rectangular hueca
        b = dim1   # ancho total
        h = dim2   # alto total
        ty = dim3  # espesor en Y
        tx = dim4  # espesor en X
        bi = max(b - 2.0*tx, 0.0)  # ancho interno
        hi = max(h - 2.0*ty, 0.0)  # alto interno
        A = b*h - bi*hi
        Iz = (b*h**3 - bi*hi**3) / 12.0
        
    elif section_type == "I":
        # Perfil en I
        h = dim1      # altura total
        bf_t = dim2   # ancho ala inferior
        bf_b = dim3   # ancho ala superior
        tw = dim4     # espesor alma
        tf_t = dim5   # espesor ala inferior
        tf_b = dim6   # espesor ala superior
        hw = max(h - tf_t - tf_b, 0.0)  # altura del alma
        
        A_top = bf_t * tf_t
        A_bot = bf_b * tf_b
        A_web = tw * hw
        A = A_top + A_bot + A_web
        
        # Momento de inercia usando teorema de ejes paralelos
        Iz_top_cent = (bf_t * tf_t**3) / 12.0
        d_top = h/2.0 - tf_t/2.0
        Iz_top = Iz_top_cent + A_top * (d_top**2)
        
        Iz_bot_cent = (bf_b * tf_b**3) / 12.0
        d_bot = h/2.0 - tf_b/2.0
        Iz_bot = Iz_bot_cent + A_bot * (d_bot**2)
        
        Iz_web = (tw * hw**3) / 12.0
        Iz = Iz_top + Iz_bot + Iz_web
        
    else:
        A = 0.0
        Iz = 0.0
    
    return A, Iz

# ----------------------------
# Dibujo de secciones transversales
# ----------------------------
def draw_cross_section(section_type, dim1, dim2, dim3, dim4, dim5, dim6):
    """
    Dibuja la sección transversal con las dimensiones indicadas usando Plotly.
    Mucho más interactivo y profesional que matplotlib.
    """
    fig = go.Figure()
    
    if section_type == "ROD":
        # Círculo sólido
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = dim1 * np.cos(theta)
        y_circle = dim1 * np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=x_circle, y=y_circle,
            fill='toself',
            fillcolor='rgba(70, 130, 180, 0.4)',
            line=dict(color='navy', width=3),
            name='Sección circular',
            showlegend=False
        ))
        
        # Línea de radio
        fig.add_trace(go.Scatter(
            x=[0, dim1], y=[0, 0],
            mode='lines+markers',
            line=dict(color='red', width=4, dash='dash'),
            marker=dict(size=8, color='red'),
            name=f'Radio = {dim1:.1f} mm',
            showlegend=True
        ))
        
        # Anotación del radio
        fig.add_annotation(
            x=dim1/2, y=dim1*0.1,
            text=f"r = {dim1:.1f} mm",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            bgcolor="white",
            bordercolor="red",
            borderwidth=2,
            font=dict(size=14, color="red")
        )
        
        axis_range = [-dim1*1.3, dim1*1.3]
        
    elif section_type == "TUBE":
        # Círculo exterior
        theta = np.linspace(0, 2*np.pi, 100)
        x_outer = dim1 * np.cos(theta)
        y_outer = dim1 * np.sin(theta)
        x_inner = dim2 * np.cos(theta)
        y_inner = dim2 * np.sin(theta)
        
        # Crear el área del material (entre círculos)
        x_tube = np.concatenate([x_outer, x_inner[::-1], [x_outer[0]]])
        y_tube = np.concatenate([y_outer, y_inner[::-1], [y_outer[0]]])
        
        fig.add_trace(go.Scatter(
            x=x_tube, y=y_tube,
            fill='toself',
            fillcolor='rgba(70, 130, 180, 0.4)',
            line=dict(color='navy', width=3),
            name='Material del tubo',
            showlegend=False
        ))
        
        # Círculo interior (hueco)
        fig.add_trace(go.Scatter(
            x=x_inner, y=y_inner,
            fill='toself',
            fillcolor='white',
            line=dict(color='darkred', width=3),
            name='Hueco interior',
            showlegend=False
        ))
        
        # Líneas de radio separadas para evitar superposición
        # Radio exterior - línea horizontal
        fig.add_trace(go.Scatter(
            x=[0, dim1], y=[0, 0],
            mode='lines+markers',
            line=dict(color='red', width=4, dash='dash'),
            marker=dict(size=8, color='red'),
            name=f'Radio exterior = {dim1:.1f} mm'
        ))
        
        # Radio interior - línea a 45 grados para evitar superposición
        angle_inner = 5*np.pi/6  # 45 grados
        x_inner_line = dim2 * np.cos(angle_inner)
        y_inner_line = dim2 * np.sin(angle_inner)
        fig.add_trace(go.Scatter(
            x=[0, x_inner_line], y=[0, y_inner_line],
            mode='lines+markers',
            line=dict(color='green', width=4, dash='dash'),
            marker=dict(size=8, color='green'),
            name=f'Radio interior = {dim2:.1f} mm'
        ))
        
        # Anotaciones separadas para evitar superposición
        fig.add_annotation(
            x=dim1*0.7, y=dim1*0.15,
            text=f"R = {dim1:.1f} mm",
            bgcolor="white", bordercolor="red", borderwidth=2,
            font=dict(size=14, color="red")
        )
        fig.add_annotation(
            x=x_inner_line*0.7, y=y_inner_line*1.3,
            text=f"r = {dim2:.1f} mm",
            bgcolor="white", bordercolor="green", borderwidth=2,
            font=dict(size=14, color="green")
        )
        
        axis_range = [-dim1*1.3, dim1*1.3]
        
    elif section_type == "BAR":
        # Rectángulo sólido
        x_rect = [-dim1/2, dim1/2, dim1/2, -dim1/2, -dim1/2]
        y_rect = [-dim2/2, -dim2/2, dim2/2, dim2/2, -dim2/2]
        
        fig.add_trace(go.Scatter(
            x=x_rect, y=y_rect,
            fill='toself',
            fillcolor='rgba(70, 130, 180, 0.4)',
            line=dict(color='navy', width=3),
            name='Sección rectangular',
            showlegend=False
        ))
        
        # Líneas de dimensión y anotaciones
        margin = max(dim1, dim2) * 0.15
        
        # Dimensión horizontal (ancho)
        fig.add_shape(
            type="line",
            x0=-dim1/2, y0=-dim2/2-margin,
            x1=dim1/2, y1=-dim2/2-margin,
            line=dict(color="red", width=3)
        )
        fig.add_annotation(
            x=0, y=-dim2/2-margin*1.3,
            text=f"b = {dim1:.1f} mm",
            bgcolor="white", bordercolor="red", borderwidth=2,
            font=dict(size=14, color="red")
        )
        
        # Dimensión vertical (altura)
        fig.add_shape(
            type="line",
            x0=-dim1/2-margin, y0=-dim2/2,
            x1=-dim1/2-margin, y1=dim2/2,
            line=dict(color="green", width=3)
        )
        fig.add_annotation(
            x=-dim1/2-margin*1.3, y=0,
            text=f"h = {dim2:.1f} mm",
            bgcolor="white", bordercolor="green", borderwidth=2,
            font=dict(size=14, color="green"),
            textangle=90
        )
        
        max_dim = max(dim1, dim2)
        axis_range = [-max_dim*0.8, max_dim*0.8]
        
    elif section_type == "BOX":
        # Rectángulo exterior
        x_outer = [-dim1/2, dim1/2, dim1/2, -dim1/2, -dim1/2]
        y_outer = [-dim2/2, -dim2/2, dim2/2, dim2/2, -dim2/2]
        
        # Rectángulo interior (hueco)
        inner_w = dim1 - 2*dim3
        inner_h = dim2 - 2*dim4
        
        if inner_w > 0 and inner_h > 0:
            x_inner = [-inner_w/2, inner_w/2, inner_w/2, -inner_w/2, -inner_w/2]
            y_inner = [-inner_h/2, -inner_h/2, inner_h/2, inner_h/2, -inner_h/2]
            
            # Crear la forma hueca combinando exterior e interior
            x_box = x_outer + x_inner[::-1] + [x_outer[0]]
            y_box = y_outer + y_inner[::-1] + [y_outer[0]]
            
            fig.add_trace(go.Scatter(
                x=x_box, y=y_box,
                fill='toself',
                fillcolor='rgba(70, 130, 180, 0.4)',
                line=dict(color='navy', width=3),
                name='Sección rectangular hueca',
                showlegend=False
            ))
            
            # Línea del hueco interior
            fig.add_trace(go.Scatter(
                x=x_inner, y=y_inner,
                fill='toself',
                fillcolor='white',
                line=dict(color='darkred', width=3),
                name='Hueco interior',
                showlegend=False
            ))
        else:
            # Si no hay hueco válido, dibujar como rectángulo sólido
            fig.add_trace(go.Scatter(
                x=x_outer, y=y_outer,
                fill='toself',
                fillcolor='rgba(70, 130, 180, 0.4)',
                line=dict(color='navy', width=3),
                name='Sección rectangular',
                showlegend=False
            ))
        
        # Dimensiones y anotaciones
        margin = max(dim1, dim2) * 0.15
        
        fig.add_annotation(
            x=0, y=-dim2/2-margin*1.3,
            text=f"b = {dim1:.1f} mm",
            bgcolor="white", bordercolor="red", borderwidth=2,
            font=dict(size=14, color="red")
        )
        fig.add_annotation(
            x=-dim1/2-margin*1.3, y=0,
            text=f"h = {dim2:.1f} mm",
            bgcolor="white", bordercolor="green", borderwidth=2,
            font=dict(size=14, color="green"),
            textangle=90
        )
        
        if dim3 > 0:
            fig.add_annotation(
                x=-dim1/4, y=dim2/2+margin,
                text=f"tx = {dim3:.1f} mm",
                bgcolor="white", bordercolor="orange", borderwidth=2,
                font=dict(size=12, color="orange")
            )
        if dim4 > 0:
            fig.add_annotation(
                x=dim1/2+margin, y=dim2/4,
                text=f"ty = {dim4:.1f} mm",
                bgcolor="white", bordercolor="purple", borderwidth=2,
                font=dict(size=12, color="purple"),
                textangle=90
            )
            
        max_dim = max(dim1, dim2)
        axis_range = [-max_dim*0.8, max_dim*0.8]
        
    elif section_type == "I":
        # Perfil I: ala inferior, alma, ala superior
        h = dim1; b_b = dim2; b_t = dim3; t_w = dim4; t_b = dim5; t_t = dim6
        
        # Crear la forma del perfil I
        x_profile = []
        y_profile = []
        
        # Ala inferior
        x_profile.extend([-b_b/2, b_b/2, b_b/2, t_w/2, t_w/2, b_t/2, b_t/2, -b_t/2, -b_t/2, -t_w/2, -t_w/2, -b_b/2, -b_b/2])
        y_profile.extend([-h/2, -h/2, -h/2+t_b, -h/2+t_b, h/2-t_t, h/2-t_t, h/2, h/2, h/2-t_t, h/2-t_t, -h/2+t_b, -h/2+t_b, -h/2])
        
        fig.add_trace(go.Scatter(
            x=x_profile, y=y_profile,
            fill='toself',
            fillcolor='rgba(70, 130, 180, 0.4)',
            line=dict(color='navy', width=3),
            name='Perfil I',
            showlegend=False
        ))
        
        # Dimensiones con líneas de cota más claras
        margin = h * 0.15
        
        # Línea de cota para altura total (h) - lado izquierdo
        fig.add_shape(
            type="line",
            x0=-h*0.6, y0=-h/2,
            x1=-h*0.6, y1=h/2,
            line=dict(color="red", width=2)
        )
        # Marcas de extremo para altura
        for y_pos in [-h/2, h/2]:
            fig.add_shape(
                type="line",
                x0=-h*0.65, y0=y_pos,
                x1=-h*0.55, y1=y_pos,
                line=dict(color="red", width=2)
            )
        fig.add_annotation(
            x=-h*0.75, y=0,
            text=f"h = {h:.1f} mm",
            bgcolor="white", bordercolor="red", borderwidth=2,
            font=dict(size=14, color="red"),
            textangle=90
        )
        
        # Línea de cota para ala inferior (b_b) - parte inferior
        fig.add_shape(
            type="line",
            x0=-b_b/2, y0=-h/2-margin,
            x1=b_b/2, y1=-h/2-margin,
            line=dict(color="green", width=2)
        )
        # Marcas de extremo para ala inferior
        for x_pos in [-b_b/2, b_b/2]:
            fig.add_shape(
                type="line",
                x0=x_pos, y0=-h/2-margin*1.1,
                x1=x_pos, y1=-h/2-margin*0.9,
                line=dict(color="green", width=2)
            )
        fig.add_annotation(
            x=0, y=-h/2-margin*1.4,
            text=f"b_b = {b_b:.1f} mm",
            bgcolor="white", bordercolor="green", borderwidth=2,
            font=dict(size=12, color="green")
        )
        
        # Línea de cota para ala superior (b_t) - parte superior
        fig.add_shape(
            type="line",
            x0=-b_t/2, y0=h/2+margin,
            x1=b_t/2, y1=h/2+margin,
            line=dict(color="purple", width=2)
        )
        # Marcas de extremo para ala superior
        for x_pos in [-b_t/2, b_t/2]:
            fig.add_shape(
                type="line",
                x0=x_pos, y0=h/2+margin*0.9,
                x1=x_pos, y1=h/2+margin*1.1,
                line=dict(color="purple", width=2)
            )
        fig.add_annotation(
            x=0, y=h/2+margin*1.4,
            text=f"b_t = {b_t:.1f} mm",
            bgcolor="white", bordercolor="purple", borderwidth=2,
            font=dict(size=12, color="purple")
        )
        
        # Dimensión del espesor del alma (tw) - centro
        if t_w > 0:
            fig.add_shape(
                type="line",
                x0=-t_w/2, y0=h*0.25,
                x1=t_w/2, y1=h*0.25,
                line=dict(color="orange", width=2)
            )
            # Marcas de extremo para espesor alma
            for x_pos in [-t_w/2, t_w/2]:
                fig.add_shape(
                    type="line",
                    x0=x_pos, y0=h*0.22,
                    x1=x_pos, y1=h*0.28,
                    line=dict(color="orange", width=2)
                )
            fig.add_annotation(
                x=0, y=h*0.35,
                text=f"tw = {t_w:.1f} mm",
                bgcolor="white", bordercolor="orange", borderwidth=2,
                font=dict(size=10, color="orange")
            )
            
        max_dim = max(h, b_b, b_t)
        axis_range = [-max_dim*0.8, max_dim*0.8]
    
    # Configurar el layout del gráfico
    fig.update_layout(
        title=dict(
            text=f'Sección Transversal: {section_type}',
            x=0.5,
            font=dict(size=18, color='darkblue')
        ),
        xaxis=dict(
            range=axis_range,
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            title='X [mm]'
        ),
        yaxis=dict(
            range=axis_range,
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            title='Y [mm]',
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=500,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# ----------------------------
# Dibujo de la viga y cargas
# ----------------------------
def draw_beam(L, support_L, support_R, cargas):
    """
    Dibuja la viga con cargas y apoyos usando Plotly interactivo.
    Mucho mejor visualización que matplotlib para aplicaciones web.
    """
    fig = go.Figure()
    
    # Dibujar la viga principal - más prominente y profesional
    beam_height = 8  # Altura visual de la viga
    
    # Viga principal como rectángulo relleno
    fig.add_shape(
        type="rect",
        x0=0, y0=-beam_height/2,
        x1=L, y1=beam_height/2,
        fillcolor="rgba(70, 130, 180, 0.7)",
        line=dict(color="darkblue", width=4),
        layer="below"
    )
    
    # Línea central de referencia
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=L, y1=0,
        line=dict(color="navy", width=3, dash="dash")
    )
    
    # Dibujar los apoyos con mayor tamaño y visibilidad
    scale_factor = max(L/1000, 0.5)
    
    # Apoyo izquierdo
    if support_L == "clamped":
        # Empotramiento más grande y visible
        rect_width = L*0.05
        rect_height = 40
        
        fig.add_shape(
            type="rect",
            x0=-rect_width, y0=-rect_height/2,
            x1=0, y1=rect_height/2,
            fillcolor="rgba(128, 128, 128, 0.8)",
            line=dict(color="black", width=3)
        )
        
        # Líneas de hatching para empotrado
        for i in range(-15, 16, 4):
            fig.add_shape(
                type="line",
                x0=-rect_width*0.8, y0=i,
                x1=-rect_width*0.2, y1=i+3,
                line=dict(color="black", width=2)
            )
        
    elif support_L == "simple":
        # Apoyo simple triangular
        triangle_size = 20 * scale_factor
        
        fig.add_trace(go.Scatter(
            x=[0, -triangle_size, triangle_size, 0],
            y=[beam_height/2, -triangle_size, -triangle_size, beam_height/2],
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.8)',
            line=dict(color='darkorange', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Apoyo derecho
    if support_R == "clamped":
        rect_width = L*0.05
        rect_height = 40
        
        fig.add_shape(
            type="rect",
            x0=L, y0=-rect_height/2,
            x1=L+rect_width, y1=rect_height/2,
            fillcolor="rgba(128, 128, 128, 0.8)",
            line=dict(color="black", width=3)
        )
        
        # Líneas de hatching
        for i in range(-15, 16, 4):
            fig.add_shape(
                type="line",
                x0=L+rect_width*0.2, y0=i,
                x1=L+rect_width*0.8, y1=i+3,
                line=dict(color="black", width=2)
            )
        
    elif support_R == "simple":
        triangle_size = 20 * scale_factor
        
        fig.add_trace(go.Scatter(
            x=[L, L-triangle_size, L+triangle_size, L],
            y=[beam_height/2, -triangle_size, -triangle_size, beam_height/2],
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.8)',
            line=dict(color='darkorange', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Dibujar las cargas - Con flechas más claras y visibles
    for i, c in enumerate(cargas):
        x = np.clip(c["pos"], 0, L)
        color = 'crimson' if c["type"] == "force" else 'purple'
        
        if c["type"] == "force":
            if c["dir"] == "Y":
                # Flechas verticales con convención correcta: 
                # Positivo hacia ARRIBA, Negativo hacia ABAJO
                arrow_length = 40 * scale_factor
                arrow_width = 12 * scale_factor
                sgn = 1 if c["mag"] >= 0 else -1  # Positivo = arriba, Negativo = abajo
                
                # Línea del shaft de la flecha
                shaft_start = beam_height/2 + 8
                shaft_end = shaft_start + sgn*arrow_length*0.7
                
                fig.add_shape(
                    type="line",
                    x0=x, y0=shaft_start,
                    x1=x, y1=shaft_end,
                    line=dict(color=color, width=4)
                )
                
                # Cabeza de la flecha como triángulo
                arrow_tip = shaft_start + sgn*arrow_length
                if sgn > 0:  # Flecha hacia arriba (positivo)
                    triangle_x = [x-arrow_width/2, x+arrow_width/2, x, x-arrow_width/2]
                    triangle_y = [shaft_end, shaft_end, arrow_tip, shaft_end]
                else:  # Flecha hacia abajo (negativo)
                    triangle_x = [x-arrow_width/2, x+arrow_width/2, x, x-arrow_width/2]
                    triangle_y = [shaft_end, shaft_end, arrow_tip, shaft_end]
                
                fig.add_trace(go.Scatter(
                    x=triangle_x, y=triangle_y,
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Etiqueta más clara
                label_y = arrow_tip + sgn*8
                fig.add_annotation(
                    x=x, y=label_y,
                    text=f"Fy = {c['mag']:.0f} N",
                    bgcolor="white",
                    bordercolor=color,
                    borderwidth=2,
                    font=dict(size=12, color=color),
                    showarrow=False
                )
                
            else:  # X direction
                # Flechas horizontales con convención correcta:
                # Positivo hacia la DERECHA, Negativo hacia la IZQUIERDA
                arrow_length = 40 * scale_factor
                arrow_width = 8 * scale_factor
                sgn = 1 if c["mag"] >= 0 else -1  # Positivo = derecha, Negativo = izquierda
                
                # Línea del shaft horizontal
                shaft_start = x - sgn*arrow_length*0.7
                
                fig.add_shape(
                    type="line",
                    x0=shaft_start, y0=0,
                    x1=x, y1=0,
                    line=dict(color=color, width=4)
                )
                
                # Cabeza de la flecha como triángulo
                arrow_tip = x + sgn*arrow_length*0.3
                if sgn > 0:  # Flecha hacia la derecha (positivo)
                    triangle_x = [x, arrow_tip, x, x]
                    triangle_y = [arrow_width/2, 0, -arrow_width/2, arrow_width/2]
                else:  # Flecha hacia la izquierda (negativo)
                    triangle_x = [x, arrow_tip, x, x]
                    triangle_y = [arrow_width/2, 0, -arrow_width/2, arrow_width/2]
                
                fig.add_trace(go.Scatter(
                    x=triangle_x, y=triangle_y,
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Etiqueta
                fig.add_annotation(
                    x=x, y=beam_height + 15,
                    text=f"Fx = {c['mag']:.0f} N",
                    bgcolor="white",
                    bordercolor=color,
                    borderwidth=2,
                    font=dict(size=12, color=color),
                    showarrow=False
                )
                
        else:  # moment Z
            # Momento con convención correcta: positivo = counterclockwise
            r = max(L*0.04, 20)
            
            # Determinar dirección del momento
            if c["mag"] >= 0:
                # Momento positivo: counterclockwise (sentido antihorario)
                theta = np.linspace(0, 1.5*np.pi, 50)
            else:
                # Momento negativo: clockwise (sentido horario) 
                theta = np.linspace(0, -1.5*np.pi, 50)
            
            cx, cy = x, 0
            arc_x = cx + r*np.cos(theta)
            arc_y = cy + r*np.sin(theta)
            
            # Dibujar el arco del momento
            fig.add_trace(go.Scatter(
                x=arc_x, y=arc_y,
                mode='lines',
                line=dict(color=color, width=4),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Flecha al final del arco como triángulo pequeño
            arrow_size = 6
            end_x, end_y = arc_x[-1], arc_y[-1]
            prev_x, prev_y = arc_x[-5], arc_y[-5]
            
            # Calcular dirección tangente para la flecha
            dx, dy = end_x - prev_x, end_y - prev_y
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx/length, dy/length
                perp_x, perp_y = -dy, dx
                
                triangle_x = [
                    end_x,
                    end_x - dx*arrow_size + perp_x*arrow_size/2,
                    end_x - dx*arrow_size - perp_x*arrow_size/2,
                    end_x
                ]
                triangle_y = [
                    end_y,
                    end_y - dy*arrow_size + perp_y*arrow_size/2,
                    end_y - dy*arrow_size - perp_y*arrow_size/2,
                    end_y
                ]
                
                fig.add_trace(go.Scatter(
                    x=triangle_x, y=triangle_y,
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Etiqueta del momento con convención clara
            fig.add_annotation(
                x=cx, y=-r-15,
                text=f"Mz = {c['mag']:.0f} N·mm {'⟲' if c['mag'] >= 0 else '⟳'}",
                bgcolor="white",
                bordercolor=color,
                borderwidth=2,
                font=dict(size=12, color=color),
                showarrow=False
            )
    
    # Dimensión horizontal para la longitud total de la viga
    dimension_y = -50  # Posición debajo de la viga
    
    # Línea horizontal principal de dimensión
    fig.add_shape(
        type="line",
        x0=0, y0=dimension_y,
        x1=L, y1=dimension_y,
        line=dict(color="darkblue", width=2)
    )
    
    # Marcas de extremo (ticks) para la dimensión
    tick_height = 8
    for x_pos in [0, L]:
        fig.add_shape(
            type="line",
            x0=x_pos, y0=dimension_y - tick_height/2,
            x1=x_pos, y1=dimension_y + tick_height/2,
            line=dict(color="darkblue", width=2)
        )
    
    # Etiqueta de longitud total centrada
    fig.add_annotation(
        x=L/2, y=dimension_y - 15,
        text=f"L = {L:.0f} mm",
        font=dict(size=14, color="darkblue"),
        bgcolor="white",
        bordercolor="darkblue",
        borderwidth=2,
        showarrow=False
    )
    
    # Configuración del layout con mejor aspecto profesional
    margin = L*0.15
    
    fig.update_layout(
        xaxis=dict(
            range=[-margin, L + margin],
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1,
            title='Posición [mm]',
            tickmode='linear',
            dtick=L/10
        ),
        yaxis=dict(
            range=[-60, 80],
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1,
            title='',
            showticklabels=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900,
        height=400,
        margin=dict(l=60, r=60, t=80, b=60),
        showlegend=False
    )
    
    return fig

# ----------------------------
# Sidebar: modelo y directorios
# ----------------------------
with st.sidebar:
    st.header("⚙️ Configuración de modelos")

    model_choice = st.radio(
        "Modelo a usar",
        ["LGBM", "Red neuronal"],
        index=0,
        horizontal=True
    )

    # Safety factor configuration
    st.subheader("🛡️ Factor de Seguridad")
    safety_factor_percent = st.slider(
        "Factor de seguridad (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=1,
        help="Porcentaje adicional aplicado a las predicciones para mayor seguridad. 10% significa que los resultados se incrementan en un 10%."
    )
    safety_factor = 1 + (safety_factor_percent / 100)

    default_dir = "models" if model_choice == "LGBM" else "models_neural"
    models_dir = st.text_input("Carpeta de modelos", value=default_dir,
                               help="Ruta relativa o absoluta. Ver archivos esperados más abajo.")
    st.caption(
        "**LGBM:** `model_max_displacement_logHGB.joblib`, `model_max_stress_logHGB.joblib`  \n"
        "**Neural:** `model_max_displacement_neuralnet.keras`, `model_max_stress_neuralnet.keras`, "
        "`scaler_displacement_neuralnet.joblib`, `scaler_stress_neuralnet.joblib`, "
        "`feature_columns_neuralnet.csv`"
    )

# ----------------------------
# Entradas: geometría y material
# ----------------------------
colA, colB, colC = st.columns([1, 1, 1])

with colA:
    st.subheader("Geometría global")
    L = st.number_input("L (longitud de la viga) [mm]", min_value=1.0, value=1000.0, step=10.0)

    section_ui = st.selectbox("Tipo de sección transversal", list(SECTION_LABELS.keys()))
    section_type = SECTION_LABELS[section_ui]

    st.markdown("**Dimensiones de la sección (mm):**")
    dim1 = dim2 = dim3 = dim4 = dim5 = dim6 = 0.0

    if section_type == "ROD":
        dim1 = st.number_input("dim1 = radio", min_value=0.0, value=10.0, step=0.5)
    elif section_type == "TUBE":
        dim1 = st.number_input("dim1 = radio externo", min_value=0.0, value=15.0, step=0.5)
        dim2 = st.number_input("dim2 = radio interno", min_value=0.0, value=10.0, step=0.5)
    elif section_type == "BAR":
        dim1 = st.number_input("dim1 = ancho b", min_value=0.0, value=40.0, step=0.5)
        dim2 = st.number_input("dim2 = alto h", min_value=0.0, value=60.0, step=0.5)
    elif section_type == "BOX":
        dim1 = st.number_input("dim1 = ancho b", min_value=0.0, value=60.0, step=0.5)
        dim2 = st.number_input("dim2 = alto h", min_value=0.0, value=80.0, step=0.5)
        dim3 = st.number_input("dim3 = espesor en ancho t_x", min_value=0.0, value=5.0, step=0.5)
        dim4 = st.number_input("dim4 = espesor en alto t_y", min_value=0.0, value=5.0, step=0.5)
    elif section_type == "I":
        dim1 = st.number_input("dim1 = alto total h", min_value=0.0, value=120.0, step=0.5)
        dim2 = st.number_input("dim2 = ancho ala inferior b_b", min_value=0.0, value=60.0, step=0.5)
        dim3 = st.number_input("dim3 = ancho ala superior b_t", min_value=0.0, value=60.0, step=0.5)
        dim4 = st.number_input("dim4 = espesor alma t_w", min_value=0.0, value=6.0, step=0.5)
        dim5 = st.number_input("dim5 = espesor ala inferior t_b", min_value=0.0, value=8.0, step=0.5)
        dim6 = st.number_input("dim6 = espesor ala superior t_t", min_value=0.0, value=8.0, step=0.5)

with colB:
    st.subheader("Material")

    # Display materials in Spanish but store English names
    mat_spanish = st.selectbox("Material", list(MATERIALS_SPANISH.keys()), index=0)
    mat_name = MATERIALS_SPANISH[mat_spanish]  # Convert to English for model

    m = MAT_INDEX[mat_name]
    st.write(
        f"E = **{m['E']:.0f}** MPa  |  ν = **{m['nu']}**  |  ρ = **{m['density']:.2e}** kg/mm³  "
        f"|  fy = **{m['yield_strength']:.0f}** MPa"
    )
    
    # Mostrar propiedades geométricas de la sección
    if any([dim1, dim2, dim3, dim4, dim5, dim6]):
        try:
            area, inertia = calculate_section_properties(section_type, dim1, dim2, dim3, dim4, dim5, dim6)
            st.write("**Propiedades de la sección:**")
            st.write(f"Área (A) = **{area:.2f}** mm²")
            st.write(f"Momento de inercia (Iz) = **{inertia:.2e}** mm⁴")
        except Exception as e:
            st.error(f"Error al calcular propiedades: {e}")
    else:
        st.info("Introduce dimensiones para ver las propiedades de la sección")

with colC:
    st.subheader("Vista de sección")
    # Dibujar la sección transversal en tiempo real
    if any([dim1, dim2, dim3, dim4, dim5, dim6]):
        try:
            cross_fig = draw_cross_section(section_type, dim1, dim2, dim3, dim4, dim5, dim6)
            st.plotly_chart(cross_fig, use_container_width=True, key="cross_section_chart")
        except Exception as e:
            st.error(f"Error al dibujar la sección: {e}")
    else:
        st.info("Introduce las dimensiones para ver la sección")
    
    # Guía contextual para la sección actual
    with st.expander("💡 Ayuda para esta sección"):
        if section_type == "ROD":
            st.markdown("""
            **Sección circular sólida:**
            - `dim1` = radio del círculo
            - Área = π × r²
            - Momento de inercia = π × r⁴/4
            """)
        elif section_type == "TUBE":
            st.markdown("""
            **Tubo circular:**
            - `dim1` = radio externo
            - `dim2` = radio interno
            - Área = π × (R² - r²)
            - El radio interno debe ser menor que el externo
            """)
        elif section_type == "BAR":
            st.markdown("""
            **Barra rectangular sólida:**
            - `dim1` = ancho (b)
            - `dim2` = alto (h)
            - Área = b × h
            - Momento de inercia = b × h³/12
            """)
        elif section_type == "BOX":
            st.markdown("""
            **Sección rectangular hueca:**
            - `dim1` = ancho total (b)
            - `dim2` = alto total (h)
            - `dim3` = espesor en dirección X (tx)
            - `dim4` = espesor en dirección Y (ty)
            - Los espesores deben ser menores que las dimensiones totales
            """)
        elif section_type == "I":
            st.markdown("""
            **Perfil I:**
            - `dim1` = altura total (h)
            - `dim2` = ancho del ala inferior (b_b)
            - `dim3` = ancho del ala superior (b_t)
            - `dim4` = espesor del alma (t_w)
            - `dim5` = espesor del ala inferior (t_b)
            - `dim6` = espesor del ala superior (t_t)
            """)
    
    # Validaciones visuales
    validation_issues = []
    if section_type == "TUBE" and dim2 >= dim1:
        validation_issues.append("⚠️ Radio interno debe ser menor que externo")
    elif section_type == "BOX":
        if dim3 >= dim1/2:
            validation_issues.append("⚠️ Espesor X muy grande")
        if dim4 >= dim2/2:
            validation_issues.append("⚠️ Espesor Y muy grande")
    
    if validation_issues:
        for issue in validation_issues:
            st.warning(issue)

# ----------------------------
# Apoyos y cargas
# ----------------------------
st.subheader("Apoyos y cargas")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    sL_ui = st.selectbox("Apoyo izquierdo", list(SUPPORT_LABELS.keys()), index=0)
    sR_ui = st.selectbox("Apoyo derecho", list(SUPPORT_LABELS.keys()), index=1)
    support_L = SUPPORT_LABELS[sL_ui]
    support_R = SUPPORT_LABELS[sR_ui]

with col2:
    num_cargas = st.slider("Número de cargas", min_value=1, max_value=3, value=1, step=1)

with col3:
    st.info("Unidades: Fuerza en **N**, Momento en **N·mm**, posiciones en **mm**.")

# Entradas por carga
cargas = []
for i in range(1, num_cargas + 1):
    st.markdown(f"**Carga {i}**")
    ccol1, ccol2, ccol3, ccol4 = st.columns([1, 1, 1, 1])
    with ccol1:
        t_ui = st.selectbox(f"Tipo {i}", list(LOAD_TYPES.keys()), key=f"t{i}")
        c_type = LOAD_TYPES[t_ui]
    with ccol2:
        if c_type == "force":
            d_ui = st.selectbox(f"Dirección {i}", list(LOAD_DIRS_FORCE.keys()), key=f"d{i}")
            c_dir = LOAD_DIRS_FORCE[d_ui]
        else:
            d_ui = st.selectbox(f"Dirección {i}", list(LOAD_DIRS_MOMENT.keys()), key=f"d{i}")
            c_dir = LOAD_DIRS_MOMENT[d_ui]
    with ccol3:
        c_mag = st.number_input(f"Magnitud {i}", value=1000.0, step=50.0, key=f"m{i}")
    with ccol4:
        # Para momento la posición no altera el preprocesado agregado, pero la usamos en el dibujo
        c_pos = st.number_input(f"Posición {i} [mm]", min_value=0.0, max_value=float(L),
                                value=float(L/2.0), step=max(1.0, L/100.0), key=f"p{i}")
    cargas.append({"type": c_type, "dir": c_dir, "mag": c_mag, "pos": c_pos})

# ----------------------------
# Vista en vivo de la viga
# ----------------------------
st.subheader("🔧 Vista previa de la viga")

# Crear dos columnas: una para el dibujo y otra para el resumen
col_beam, col_summary = st.columns([2, 1])

with col_beam:
    st.markdown("**Configuración actual de la viga:**")
    # Dibujar la viga en tiempo real con las entradas actuales
    try:
        live_fig = draw_beam(L, support_L, support_R, cargas)
        st.plotly_chart(live_fig, use_container_width=True, key="live_beam_chart")
    except Exception as e:
        st.error(f"Error al dibujar la viga: {e}")

with col_summary:
    st.markdown("**📋 Configuración:**")
    
    # Información básica en una caja de información
    st.info(f"""
    **Geometría:**
    • L = {L:.1f} mm
    • Sección: {section_type}
    • Material: {mat_name}
    
    **Apoyos:**
    • Izq: {sL_ui.split('(')[0].strip()}
    • Dcha: {sR_ui.split('(')[0].strip()}
    """)
    
    # Detalles de cargas
    st.markdown("**Cargas:**")
    for i, c in enumerate(cargas, 1):
        tipo_texto = "F" if c["type"] == "force" else "M"
        unidad = "N" if c["type"] == "force" else "N·mm"
        color = "🔵" if c["type"] == "force" else "🟠"
        st.markdown(f"{color} **{tipo_texto}{c['dir']}** = {c['mag']:.0f} {unidad}")
        st.caption(f"Posición: {c['pos']:.1f} mm")
    
    # Indicadores de estado
    total_forces = sum(abs(c["mag"]) for c in cargas if c["type"] == "force")
    total_moments = sum(abs(c["mag"]) for c in cargas if c["type"] == "moment")

st.divider()

# ----------------------------
# Construcción del DataFrame de entrada cruda
# ----------------------------
def build_raw_row():
    row = {
        "model_name": model_choice,
        "L": float(L),
        "section_type": section_type,
        "dim1": float(dim1), "dim2": float(dim2), "dim3": float(dim3),
        "dim4": float(dim4), "dim5": float(dim5), "dim6": float(dim6),
        "material": mat_name,
        "E": float(m["E"]), "nu": float(m["nu"]),
        "density": float(m["density"]), "yield_strength": float(m["yield_strength"]),
        "support_L": support_L, "support_R": support_R,
        "num_cargas": int(num_cargas),
    }
    # Cargas
    for i, c in enumerate(cargas, start=1):
        row[f"c{i}_type"] = c["type"]
        row[f"c{i}_dir"]  = c["dir"]
        row[f"c{i}_mag"]  = float(c["mag"])
        row[f"c{i}_pos"]  = float(c["pos"])
    # Relleno para cargas no usadas
    for i in range(num_cargas + 1, 4):
        row[f"c{i}_type"] = np.nan
        row[f"c{i}_dir"]  = np.nan
        row[f"c{i}_mag"]  = 0.0
        row[f"c{i}_pos"]  = np.nan
    return pd.DataFrame([row])

# ----------------------------
# Preprocesado (tu función)
# ----------------------------
def run_preprocess_on_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        in_csv  = Path(td) / "raw.csv"
        out_csv = Path(td) / "features.csv"
        df_raw.to_csv(in_csv, index=False)
        # llama a tu main_preprocess
        main_preprocess(str(in_csv), str(out_csv))
        df_feat = pd.read_csv(out_csv)
    return df_feat

# ----------------------------
# Predicción LGBM
# ----------------------------
def predict_lgbm(df_feat: pd.DataFrame, models_path: Path, safety_factor: float = 1.1):
    disp_path = models_path / "model_max_displacement_logHGB.joblib"
    stress_path = models_path / "model_max_stress_logHGB.joblib"
    if not disp_path.exists() or not stress_path.exists():
        raise FileNotFoundError("No se encontraron los ficheros .joblib esperados en la carpeta indicada.")
    # Los joblib guardan un bundle con 'hgb' y 'features'
    bundle_disp = joblib.load(disp_path)
    bundle_strs = joblib.load(stress_path)

    feats_d = bundle_disp["features"]
    feats_s = bundle_strs["features"]

    Xd = df_feat.reindex(columns=feats_d)
    Xs = df_feat.reindex(columns=feats_s)

    y_disp = bundle_disp["hgb"].predict(Xd)
    y_stress = bundle_strs["hgb"].predict(Xs)
    
    # Apply safety factor for conservative predictions
    y_disp_safe = y_disp[0] * safety_factor
    y_stress_safe = y_stress[0] * safety_factor
    
    return float(y_disp_safe), float(y_stress_safe)

# ----------------------------
# Predicción Red Neuronal
# ----------------------------
# def _read_feature_list_csv(path: Path):
#     """
#     Lee feature_columns_neuralnet.csv con tolerancia a formatos:
#     - Caso A: una sola columna llamada 'feature' o similar con los nombres.
#     - Caso B: CSV de una fila con todas las columnas como nombres.
#     Devuelve: lista de nombres de columnas (str).
#     """
#     import pandas as pd
#     df = pd.read_csv(path)
#     # Caso A: columna única con los nombres
#     if df.shape[1] == 1:
#         return df.iloc[:, 0].dropna().astype(str).tolist()
#     # Caso B: los nombres son los headers
#     return list(df.columns)

def predict_neural(df_feat: pd.DataFrame, models_path: Path, safety_factor: float = 1.1):
    """
    Carga modelos + scalers de NN, asegura:
      - Orden EXACTO de columnas
      - Misma selección de columnas que en entrenamiento
      - Escalado con el scaler de entrenamiento
      - Manejo correcto de valores NaN antes del escalado
    """
    from tensorflow.keras.models import load_model
    import joblib
    import numpy as np

    disp_model = models_path / "model_max_displacement_neuralnet.keras"
    stress_model = models_path / "model_max_stress_neuralnet.keras"
    disp_scaler = models_path / "scaler_displacement_neuralnet.joblib"
    stress_scaler = models_path / "scaler_stress_neuralnet.joblib"
    feat_cols = models_path / "feature_columns_neuralnet.csv"

    for p in [disp_model, stress_model, disp_scaler, stress_scaler, feat_cols]:
        if not p.exists():
            raise FileNotFoundError(f"Falta el archivo requerido para NN: {p.name}")

    # 1) Lista de columnas en el orden del entrenamiento
    exclude_cols = ['model_name', 'section_type', 'material', 'support_L', 'support_R']
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols]

    X = df_feat[feature_cols]

    # 2) Reindex estricto con esas columnas (las no presentes → NaN)
    X = df_feat.reindex(columns=feature_cols)

    # Debug: Check which columns are missing
    missing_cols = [col for col in feature_cols if col not in df_feat.columns]
    if missing_cols:
        print(f"Warning: Las siguientes columnas esperadas no están presentes: {missing_cols}")

    missing_analysis = X.isnull().sum()
    missing_features = missing_analysis[missing_analysis > 0]

    if len(missing_features) > 0:
        for feature, count in missing_features.items():
            percentage = (count / len(X)) * 100
            print(f"  {feature}: {count} missing ({percentage:.1f}%)")
            
        # Let's look at some examples to understand the pattern
        # Show first few rows with missing values
        rows_with_missing = X[X.isnull().any(axis=1)].head()
        
        # Engineering-based missing value handling:
        # 1. For load-related features (forces, moments): fill with 0
        # 2. For geometric features: need to understand the context
        # 3. For material properties: use appropriate defaults
        
        load_keywords = ['force', 'load', 'moment', 'pressure', 'distributed']
        geometric_keywords = ['length', 'width', 'height', 'thickness', 'area', 'inertia']
        
        for feature in missing_features.index:
            feature_lower = feature.lower()
            
            # Load-related features should be zero when missing
            if any(keyword in feature_lower for keyword in load_keywords):
                X[feature] = X[feature].fillna(0.0)
                print(f"  Filled {feature} with 0.0 (no load applied)")
                
            # For other features, we need to be more careful
            # Let's use 0 as default for now, but this should be reviewed
            else:
                X[feature] = X[feature].fillna(0.0)
                print(f"  Filled {feature} with 0.0 (default - needs review)")
        X_filled = X
    else:
        X_filled = X

    # 5) Carga escaladores
    sc_d = joblib.load(disp_scaler)
    sc_s = joblib.load(stress_scaler)

    # 6) Transformación - usar datos sin NaN
    Xd_scaled = sc_d.transform(X_filled)
    Xs_scaled = sc_s.transform(X_filled)

    # 7) Verificar que no hay NaN en los datos escalados
    if np.isnan(Xd_scaled).any():
        raise ValueError("Los datos escalados para desplazamiento contienen NaN")
    if np.isnan(Xs_scaled).any():
        raise ValueError("Los datos escalados para tensión contienen NaN")

    # 8) Carga modelos (sin recompilar)
    mdl_d = load_model(str(disp_model), compile=False)
    mdl_s = load_model(str(stress_model), compile=False)

    # 9) Predicción
    yd = mdl_d.predict(Xd_scaled, verbose=0)
    ys = mdl_s.predict(Xs_scaled, verbose=0)

    # Asegurar salida escalar aunque predict devuelva (n,1)
    yd = float(np.ravel(yd)[0])
    ys = float(np.ravel(ys)[0])

    # Apply safety factor for conservative predictions
    yd_safe = yd * safety_factor
    ys_safe = ys * safety_factor

    return yd_safe, ys_safe


# ----------------------------
# Engineering Recommendations
# ----------------------------
def generate_engineering_recommendations(stress_mpa: float, displacement_mm: float, 
                                        current_material: str, current_area: float = None) -> list:
    """
    Generate engineering recommendations based on stress, displacement, and current design.
    
    Returns a list of recommendation dictionaries with 'type', 'message', and 'icon' keys.
    """
    recommendations = []
    current_mat = MAT_INDEX[current_material]
    yield_strength = current_mat["yield_strength"]
    
    # Convert English material name to Spanish for display
    current_material_spanish = MATERIALS_SPANISH_REVERSE.get(current_material, current_material)
    
    # Safety factors for recommendations
    stress_safety_factor = 2.0  # Conservative safety factor for stress
    safe_stress_limit = yield_strength / stress_safety_factor
    
    # Stress-based recommendations
    stress_ratio = stress_mpa / yield_strength
    
    if stress_ratio < 0.3:  # Very low stress utilization
        current_material_spanish = MATERIALS_SPANISH_REVERSE.get(current_material, current_material)
        recommendations.append({
            'type': 'material_downgrade',
            'icon': '💰',
            'message': f"**Optimización de material:** El estrés actual ({stress_mpa:.1f} MPa) es muy bajo comparado con la resistencia del {current_material_spanish} ({yield_strength:.0f} MPa). Considera usar un material más económico:"
        })
        
        # Suggest cheaper alternatives
        if current_material == "Titanium":
            recommendations.append({
                'type': 'suggestion',
                'icon': '➡️',
                'message': "• **Cambiar a Acero** (370 MPa) - Mucho más económico y suficiente para esta aplicación"
            })
            recommendations.append({
                'type': 'suggestion',
                'icon': '➡️',
                'message': "• **Cambiar a Aluminio** (270 MPa) - Más ligero y económico que el titanio"
            })
        elif current_material == "Steel":
            recommendations.append({
                'type': 'suggestion',
                'icon': '➡️',
                'message': "• **Cambiar a Aluminio** (270 MPa) - Más ligero y suficiente para esta carga"
            })

    elif stress_ratio >= 1.0:  # Exceeds yield strength - critical
        recommendations.append({
            'type': 'critical',
            'icon': '❌',
            'message': f"**¡Peligro!** El estrés ({stress_mpa:.1f} MPa) excede el límite del {current_material_spanish} ({yield_strength:.0f} MPa). Factor de utilización: {stress_ratio:.1%}. Se requiere acción inmediata:"
        })

        # Suggest stronger materials or larger sections
        if current_material in ["Aluminum", "Inox Steel"]:
            recommendations.append({
                'type': 'material_upgrade',
                'icon': '🔧',
                'message': "• **Cambiar a Acero** (370 MPa) - Mayor resistencia"
            })
            recommendations.append({
                'type': 'material_upgrade',
                'icon': '🔧',
                'message': "• **Cambiar a Titanio** (830 MPa) - Máxima resistencia (alta resistencia/peso)"
            })
        elif current_material == "Steel":
            recommendations.append({
                'type': 'material_upgrade',
                'icon': '🔧',
                'message': "• **Cambiar a Titanio** (830 MPa) - Resistencia superior"
            })
        recommendations.append({
            'type': 'section_upgrade',
            'icon': '📐',
            'message': "• **Aumentar área de sección** - Incrementar dimensiones para reducir el estrés"
        })

    elif stress_ratio > 0.9 and stress_ratio < 1.0:  # High stress utilization - concerning
        recommendations.append({
            'type': 'warning',
            'icon': '⚠️',
            'message': f"**¡Atención!** El estrés ({stress_mpa:.1f} MPa) está cerca del límite del {current_material_spanish} ({yield_strength:.0f} MPa). Factor de utilización: {stress_ratio:.1%}"
        })
        
        # Suggest stronger materials or larger sections
        if current_material in ["Aluminum", "Inox Steel"]:
            recommendations.append({
                'type': 'material_upgrade',
                'icon': '🔧',
                'message': "• **Cambiar a Acero** (370 MPa) - Mayor resistencia"
            })
            recommendations.append({
                'type': 'material_upgrade',
                'icon': '🔧',
                'message': "• **Cambiar a Titanio** (830 MPa) - Máxima resistencia (alta resistencia/peso)"
            })
        elif current_material == "Steel":
            recommendations.append({
                'type': 'material_upgrade',
                'icon': '🔧',
                'message': "• **Cambiar a Titanio** (830 MPa) - Resistencia superior"
            })
        
        recommendations.append({
            'type': 'section_upgrade',
            'icon': '📐',
            'message': "• **Aumentar área de sección** - Incrementar dimensiones para reducir el estrés"
        })
    
    elif stress_ratio > 0.5:  # Moderate stress utilization
        recommendations.append({
            'type': 'info',
            'icon': '📊',
            'message': f"**Factor de utilización moderado:** {stress_ratio:.1%} del límite elástico. Diseño aceptable con margen de seguridad razonable."
        })
    
    # Displacement-based recommendations
    if displacement_mm > 10.0:  # Large displacement
        recommendations.append({
            'type': 'warning',
            'icon': '📏',
            'message': f"**Desplazamiento elevado:** {displacement_mm:.2f} mm. Considera:"
        })
        recommendations.append({
            'type': 'suggestion',
            'icon': '🔧',
            'message': "• **Aumentar rigidez** - Incrementar momento de inercia (altura de la sección)"
        })
        recommendations.append({
            'type': 'suggestion',
            'icon': '🔧',
            'message': "• **Material más rígido** - Mayor módulo elástico (E)"
        })
        recommendations.append({
            'type': 'suggestion',
            'icon': '🔧',
            'message': "• **Reducir luz libre** - Añadir apoyos intermedios si es posible"
        })
    
    elif displacement_mm < 1.0:  # Very small displacement
        recommendations.append({
            'type': 'info',
            'icon': '✅',
            'message': f"**Desplazamiento aceptable:** {displacement_mm:.3f} mm. La rigidez del diseño es adecuada."
        })
    
    # Combined recommendations
    if stress_ratio > 0.9 and displacement_mm > 5.0:
        recommendations.append({
            'type': 'critical',
            'icon': '🚨',
            'message': "**Revisión crítica necesaria:** Tanto el estrés como el desplazamiento son elevados. Se recomienda rediseño completo."
        })
    
    # Efficiency recommendations
    if stress_ratio < 0.2 and displacement_mm < 2.0:
        recommendations.append({
            'type': 'optimization',
            'icon': '♻️',
            'message': "**Oportunidad de optimización:** El diseño actual es muy conservador. Puedes reducir dimensiones o usar materiales más económicos."
        })
    
    return recommendations

# ----------------------------
# Acción: Predecir
# ----------------------------
st.divider()
btn = st.button("🔮 Predecir")

if btn:
    try:
        df_raw = build_raw_row()
        df_feat = run_preprocess_on_df(df_raw)

        models_path = Path(models_dir)
        if not models_path.exists():
            raise FileNotFoundError(f"La carpeta de modelos no existe: {models_path}")

        if model_choice == "LGBM":
            y_disp, y_stress = predict_lgbm(df_feat, models_path, safety_factor)
        else:
            y_disp, y_stress = predict_neural(df_feat, models_path, safety_factor)

        st.success("Predicción completada.")
        if safety_factor_percent > 0:
            st.info(f"🛡️ **Nota:** Se ha aplicado un factor de seguridad del {safety_factor_percent}% a los resultados predichos.")
        colr1, colr2 = st.columns(2)
        with colr1:
            help_text = f"Incluye factor de seguridad del {safety_factor_percent}%" if safety_factor_percent > 0 else "Sin factor de seguridad aplicado"
            st.metric("Desplazamiento máximo [mm]", f"{y_disp:,.6f}", help=help_text)
        with colr2:
            help_text = f"Incluye factor de seguridad del {safety_factor_percent}%" if safety_factor_percent > 0 else "Sin factor de seguridad aplicado"
            st.metric("Tensión máxima [MPa]", f"{y_stress:,.3f}", help=help_text)

        # Generate and display engineering recommendations
        st.divider()
        st.subheader("🎯 Recomendaciones")
        
        # Get current area for context (if available)
        current_area = None
        try:
            if any([dim1, dim2, dim3, dim4, dim5, dim6]):
                current_area, _ = calculate_section_properties(section_type, dim1, dim2, dim3, dim4, dim5, dim6)
        except:
            pass
        
        recommendations = generate_engineering_recommendations(
            stress_mpa=y_stress, 
            displacement_mm=y_disp, 
            current_material=mat_name,
            current_area=current_area
        )
        
        if recommendations:
            for rec in recommendations:
                if rec['type'] == 'critical':
                    st.error(f"{rec['icon']} {rec['message']}")
                elif rec['type'] == 'warning':
                    st.warning(f"{rec['icon']} {rec['message']}")
                elif rec['type'] in ['material_downgrade', 'optimization']:
                    st.info(f"{rec['icon']} {rec['message']}")
                elif rec['type'] == 'info':
                    st.success(f"{rec['icon']} {rec['message']}")
                else:
                    st.write(f"{rec['icon']} {rec['message']}")
        else:
            st.success("✅ El diseño actual parece adecuado para las cargas aplicadas.")
        
        # Material utilization chart
        current_mat = MAT_INDEX[mat_name]
        stress_ratio = y_stress / current_mat["yield_strength"]
        
        if stress_ratio > 0.1:  # Only show if there's meaningful stress
            with st.expander("📊 Análisis de Utilización del Material"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Create utilization bar chart
                    fig = go.Figure()
                    
                    # Add stress utilization bar
                    fig.add_trace(go.Bar(
                        x=[stress_ratio * 100],
                        y=['Utilización'],
                        orientation='h',
                        marker_color='red' if stress_ratio > 0.7 else 'orange' if stress_ratio > 0.5 else 'green',
                        text=[f'{stress_ratio:.1%}'],
                        textposition='inside',
                        name='Estrés actual'
                    ))
                    
                    # Add safe limit line
                    fig.add_vline(x=50, line_dash="dash", line_color="orange", 
                                 annotation_text="Límite recomendado (50%)")
                    
                    # Add yield strength line
                    fig.add_vline(x=100, line_dash="dash", line_color="red", 
                                 annotation_text="Límite elástico (100%)")
                    
                    fig.update_layout(
                        title=f"Utilización del {MATERIALS_SPANISH_REVERSE.get(mat_name, mat_name)} (fy = {current_mat['yield_strength']:.0f} MPa)",
                        xaxis_title="Porcentaje de utilización (%)",
                        xaxis=dict(range=[0, min(120, max(100, stress_ratio * 120))]),
                        height=200,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Factor de utilización", f"{stress_ratio:.1%}")
                    st.metric("Margen de seguridad", f"{(1-stress_ratio):.1%}")
                    if stress_ratio < 1.0:
                        additional_load = (1.0 - stress_ratio) * 100
                        st.metric("Capacidad adicional", f"{additional_load:.0f}%")

        # Descarga CSV
        out = df_raw.copy()
        if safety_factor_percent > 0:
            out["max_displacement_pred_with_safety"] = y_disp
            out["max_stress_pred_with_safety"] = y_stress
            download_text = f"💾 Descargar CSV con entradas + predicciones (con factor de seguridad {safety_factor_percent}%)"
            filename = f"prediccion_viga_seguridad_{safety_factor_percent}pct.csv"
        else:
            out["max_displacement_pred"] = y_disp
            out["max_stress_pred"] = y_stress
            download_text = "💾 Descargar CSV con entradas + predicciones"
            filename = "prediccion_viga.csv"
        
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(download_text, 
                           data=csv_bytes,
                           file_name=filename, mime="text/csv")

        with st.expander("🔎 Debug: ver fila cruda y features preprocesadas"):
            st.markdown("**Entrada cruda (1 fila):**")
            st.dataframe(df_raw)
            st.markdown("**Features tras `main_preprocess`:**")
            st.dataframe(df_feat)

    except Exception as e:
        st.error(f"⚠️ Error durante la predicción: {e}")

# ----------------------------
# Notas finales
# ----------------------------
st.caption(
    "Creado por Daniel López López - https://www.linkedin.com/in/daniel-lopez-lopez1313/"
)
