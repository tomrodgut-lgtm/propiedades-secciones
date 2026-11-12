"""Propiedades de secciones transversales — versión web (Streamlit)."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import secciones as sc

st.set_page_config(page_title="Propiedades de secciones", page_icon="📐", layout="wide")

st.markdown(
    """
<style>
.banner {background: linear-gradient(135deg,#0b2447,#19376d); padding: 1.4rem 1.8rem;
         border-radius: 12px; margin-bottom: 1rem;}
.banner h1 {color:#fff; margin:0; font-size:2rem;}
.banner p {color:#a5c8e4; margin:.3rem 0 0 0;}
</style>
<div class="banner">
  <h1>Propiedades de secciones transversales</h1>
  <p>Área · centroide · Q · inercias · módulos · Mohr · esfuerzo cortante τ = V·Q/(I·t)</p>
</div>
""",
    unsafe_allow_html=True,
)
st.info("Herramienta educativa, no certificada para diseño.")

# ------------------------------------------------------------ estado inicial
SHAPE_NAMES = list(sc.SHAPES)
st.session_state.setdefault("shape", "Perfil I / H")
st.session_state.setdefault("prof", "")
st.session_state.setdefault("uin", "mm")
st.session_state.setdefault("uout", "mm")
st.session_state.setdefault("theta", 30.0)
st.session_state.setdefault("V", 10.0)
st.session_state.setdefault("use_cut", False)
st.session_state.setdefault("ycut", 50.0)
st.session_state.setdefault("pasos", True)


def al_elegir_perfil():
    """Al elegir un perfil comercial: rellena la sección I/H con sus medidas (mm)."""
    p = st.session_state["prof"]
    if not p:
        return
    h, b, tw, tf = sc.PROFILES[p]
    st.session_state["shape"] = "Perfil I / H"
    st.session_state["uin"] = "mm"
    for nombre, valor in zip(["bf", "tf", "h", "tw"], [b, tf, h, tw]):
        st.session_state[f"dim_Perfil I / H_{nombre}"] = float(valor)


# ------------------------------------------------------------------- entrada
with st.sidebar:
    st.header("Datos de entrada")
    shape = st.selectbox("Sección", SHAPE_NAMES, key="shape")
    st.selectbox("Perfil comercial (mm)", [""] + list(sc.PROFILES), key="prof",
                 on_change=al_elegir_perfil,
                 help="Valores aproximados (sin radio de acuerdo).")
    c1, c2 = st.columns(2)
    uin = c1.selectbox("Unidad de entrada", list(sc.UNITS), key="uin")
    uout = c2.selectbox("Unidad de resultados", list(sc.UNITS), key="uout")

    st.subheader("Dimensiones")
    nombres = sc.SHAPES[shape][0]
    dims = []
    for n in nombres:
        k = f"dim_{shape}_{n}"
        st.session_state.setdefault(k, float(sc.DEF.get(n, 10)))
        dims.append(st.number_input(f"{n} [{uin}]", min_value=0.0, format="%g", key=k))

    st.subheader("Extras")
    theta = st.number_input("Giro de ejes θ (°)", format="%g", key="theta")
    V = st.number_input("Cortante V", format="%g", key="V")
    use_cut = st.checkbox("Calcular Q y τ a una altura", key="use_cut")
    ycut = None
    if use_cut:
        ycut = st.number_input(f"Altura y desde la base [{uin}]", min_value=0.0,
                               format="%g", key="ycut")
    pasos = st.checkbox("Mostrar paso a paso", key="pasos")

# ------------------------------------------------------------------- cálculo
try:
    last = sc.calcular(shape, dims, uin, uout, theta, V, ycut)
except (ValueError, ZeroDivisionError) as e:
    st.error(f"⚠ {e}")
    st.stop()

r = last[1]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(f"Área [{uout}²]", f"{r['A']:.6g}")
m2.metric(f"ȳ [{uout}]", f"{r['yb']:.6g}")
m3.metric(f"Ix [{uout}⁴]", f"{r['Ix']:.6g}")
m4.metric(f"Iy [{uout}⁴]", f"{r['Iy']:.6g}")
m5.metric(f"Sx [{uout}³]", f"{r['Sx']:.6g}")

fig = sc.dibujar(last, uout)
texto = sc.reporte(last, uout, pasos)

tab1, tab2, tab3 = st.tabs(["Resultados", "Paso a paso", "Exportar"])

with tab1:
    st.pyplot(fig)
    df = pd.DataFrame(
        [(n, f"{v:.6g}", u) for n, v, u in sc.filas(last, uout)],
        columns=["Propiedad", "Valor", "Unidad"],
    )
    st.dataframe(df, hide_index=True)

with tab2:
    if pasos:
        st.code(texto, language=None)
    else:
        st.write("Activa **Mostrar paso a paso** en el panel izquierdo.")

with tab3:
    st.download_button("Descargar Excel", sc.excel_bytes(last, uout, shape, dims, uin),
                       file_name="resultados.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Descargar PDF", sc.pdf_bytes(last, uout, fig, texto),
                       file_name="resultados.pdf", mime="application/pdf")

plt.close(fig)
