"""
MOTOR de propiedades de secciones transversales.
La matemática es la del código original (versión Colab); solo se quitaron
ipywidgets / IPython / google.colab para poder usarla en una página web.
"""
import io
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle

UNITS = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}


# ---------------------------------------------------------------- SECCIONES
# Componente rectángulo: ("r", cx, cy, b, h, signo)   círculo: ("c", cx, cy, d, signo)
# Origen en la esquina inferior izquierda de la sección.
def _check(cond, msg):
    if not cond:
        raise ValueError(msg)


def s_rect(b, h):
    return [("r", b/2, h/2, b, h, 1)]


def s_rect_hueco(b, h, bi, hi):
    _check(bi < b and hi < h, "El hueco debe ser menor que el exterior")
    return [("r", b/2, h/2, b, h, 1), ("r", b/2, h/2, bi, hi, -1)]


def s_hss(b, h, t):
    _check(2*t < min(b, h), "El espesor t es demasiado grande")
    return s_rect_hueco(b, h, b-2*t, h-2*t)


def s_circ(d):
    return [("c", d/2, d/2, d, 1)]


def s_tubo(D, t):
    _check(2*t < D, "El espesor t es demasiado grande")
    return [("c", D/2, D/2, D, 1), ("c", D/2, D/2, D-2*t, -1)]


def s_I(bf, tf, h, tw):
    _check(2*tf < h and tw < bf, "Dimensiones inconsistentes")
    return [("r", bf/2, tf/2, bf, tf, 1),
            ("r", bf/2, h/2, tw, h-2*tf, 1),
            ("r", bf/2, h-tf/2, bf, tf, 1)]


def s_T(bf, tf, h, tw):
    _check(tf < h and tw < bf, "Dimensiones inconsistentes")
    return [("r", bf/2, (h-tf)/2, tw, h-tf, 1),
            ("r", bf/2, h-tf/2, bf, tf, 1)]


SHAPES = {
    "Rectángulo sólido":       (["b", "h"], s_rect),
    "Rectángulo hueco":        (["b", "h", "bi", "hi"], s_rect_hueco),
    "Círculo sólido":          (["d"], s_circ),
    "Tubo circular":           (["D", "t"], s_tubo),
    "Perfil I / H":            (["bf", "tf", "h", "tw"], s_I),
    "Perfil T":                (["bf", "tf", "h", "tw"], s_T),
    "Tubo rect./cuad. (HSS)":  (["b", "h", "t"], s_hss),
}

# Base de datos (mm): (h, b, tw, tf). Sin radio de acuerdo -> valores aproximados.
PROFILES = {
    "IPE 100": (100, 55, 4.1, 5.7), "IPE 120": (120, 64, 4.4, 6.3),
    "IPE 140": (140, 73, 4.7, 6.9), "IPE 160": (160, 82, 5.0, 7.4),
    "IPE 180": (180, 91, 5.3, 8.0), "IPE 200": (200, 100, 5.6, 8.5),
    "IPE 220": (220, 110, 5.9, 9.2), "IPE 240": (240, 120, 6.2, 9.8),
    "IPE 270": (270, 135, 6.6, 10.2), "IPE 300": (300, 150, 7.1, 10.7),
    "HEA 100": (96, 100, 5.0, 8.0), "HEA 120": (114, 120, 5.0, 8.0),
    "HEA 140": (133, 140, 5.5, 8.5), "HEA 160": (152, 160, 6.0, 9.0),
    "HEA 180": (171, 180, 6.0, 9.5), "HEA 200": (190, 200, 6.5, 10.0),
    "HEA 240": (230, 240, 7.5, 12.0), "HEA 300": (290, 300, 8.5, 14.0),
    "HEB 100": (100, 100, 6.0, 10.0), "HEB 120": (120, 120, 6.5, 11.0),
    "HEB 140": (140, 140, 7.0, 12.0), "HEB 160": (160, 160, 8.0, 13.0),
    "HEB 180": (180, 180, 8.5, 14.0), "HEB 200": (200, 200, 9.0, 15.0),
    "HEB 240": (240, 240, 10.0, 17.0), "HEB 300": (300, 300, 11.0, 19.0),
}

DEF = {"b": 100, "h": 200, "bi": 80, "hi": 160, "d": 100, "D": 100,
       "t": 8, "bf": 100, "tf": 10, "tw": 6}


# ------------------------------------------------------------------ CÁLCULO
def width(comps, t, axis):
    """Ancho neto de la sección en un corte. axis='y': corte horizontal a y=t."""
    t = np.asarray(t, float)
    w = np.zeros_like(t)
    for c in comps:
        if c[0] == "r":
            _, cx, cy, b, h, s = c
            ctr, half, wd = (cy, h/2, b) if axis == "y" else (cx, b/2, h)
            w += s * wd * (np.abs(t - ctr) <= half)
        else:
            _, cx, cy, d, s = c
            ctr = cy if axis == "y" else cx
            w += s * 2*np.sqrt(np.maximum((d/2)**2 - (t-ctr)**2, 0))
    return w


def area_i(c):
    return c[3]*c[4] if c[0] == "r" else math.pi*c[3]**2/4


def own_I(c):
    if c[0] == "r":
        return c[3]*c[4]**3/12, c[4]*c[3]**3/12
    return math.pi*c[3]**4/64, math.pi*c[3]**4/64


def bbox(comps):
    xs, ys = [], []
    for c in comps:
        if c[-1] > 0:
            wx, wy = (c[3], c[4]) if c[0] == "r" else (c[3], c[3])
            xs += [c[1]-wx/2, c[1]+wx/2]
            ys += [c[2]-wy/2, c[2]+wy/2]
    return min(xs), max(xs), min(ys), max(ys)


def plastic(comps, axis, lo, hi, A, n=6000):
    t = np.linspace(lo, hi, n+1)
    tm = (t[:-1]+t[1:])/2
    dt = (hi-lo)/n
    w = width(comps, tm, axis)
    cum = np.cumsum(w*dt)
    k = min(np.searchsorted(cum, A/2), n-1)
    tp = tm[k]
    return float(np.sum(w*np.abs(tm-tp)*dt)), float(tp)


def compute(comps, theta_deg=0.0, V=None, ycut=None):
    A = sum(c[-1]*area_i(c) for c in comps)
    _check(A > 0, "Área nula o negativa")
    Qx0 = sum(c[-1]*area_i(c)*c[2] for c in comps)     # respecto a la base
    Qy0 = sum(c[-1]*area_i(c)*c[1] for c in comps)     # respecto al lado izq.
    xb, yb = Qy0/A, Qx0/A
    Ix = Iy = Ixy = 0.0
    for c in comps:
        s, a = c[-1], area_i(c)
        i0x, i0y = own_I(c)
        dx, dy = c[1]-xb, c[2]-yb
        Ix += s*(i0x + a*dy**2)
        Iy += s*(i0y + a*dx**2)
        Ixy += s*a*dx*dy
    x0, x1, y0, y1 = bbox(comps)
    Sx = Ix/max(yb-y0, y1-yb)
    Sy = Iy/max(xb-x0, x1-xb)
    Zx, ypl = plastic(comps, "y", y0, y1, A)
    Zy, xpl = plastic(comps, "x", x0, x1, A)
    avg, dif = (Ix+Iy)/2, (Ix-Iy)/2
    R = math.hypot(dif, Ixy)
    thp = 0.5*math.atan2(-2*Ixy, Ix-Iy)
    th = math.radians(theta_deg)
    Iu = avg + dif*math.cos(2*th) - Ixy*math.sin(2*th)
    Iv = avg - dif*math.cos(2*th) + Ixy*math.sin(2*th)
    Iuv = dif*math.sin(2*th) + Ixy*math.cos(2*th)
    r = dict(A=A, xb=xb, yb=yb, Qx0=Qx0, Qy0=Qy0, Ix=Ix, Iy=Iy, Ixy=Ixy,
             rx=math.sqrt(Ix/A), ry=math.sqrt(Iy/A), Sx=Sx, Sy=Sy, Zx=Zx, Zy=Zy,
             ypl=ypl, xpl=xpl, I1=avg+R, I2=avg-R, thp=math.degrees(thp),
             J=Ix+Iy, Iu=Iu, Iv=Iv, Iuv=Iuv, avg=avg, R=R, box=(x0, x1, y0, y1))
    if ycut is not None:                      # Q a una altura (desde la base)
        n = 6000
        t = np.linspace(ycut, y1, n+1)
        tm = (t[:-1]+t[1:])/2
        Q = abs(float(np.sum((tm-yb)*width(comps, tm, "y"))*(y1-ycut)/n))
        tw_ = float(width(comps, [ycut], "y")[0])
        if tw_ <= 1e-12:
            tw_ = float(width(comps, [ycut-1e-6*(y1-y0)], "y")[0])
        r["Q"], r["t"] = Q, tw_
        if V is not None and tw_ > 0:
            r["tau"] = V*Q/(Ix*tw_)
    return r


# ------------------------------------------------------------ ENTRADA / SALIDA
def calcular(shape, dims, uin, uout, theta=0.0, V=10.0, ycut=None):
    """Punto de entrada para la web. dims en la unidad uin; resultados en uout.
    Devuelve la tupla `last` = (comps, resultados, theta, ycut_convertido, V)."""
    k = UNITS[uin] / UNITS[uout]
    dims = [d * k for d in dims]
    _check(all(d > 0 for d in dims), "Todas las dimensiones deben ser > 0")
    comps = SHAPES[shape][1](*dims)
    yc = ycut * k if ycut is not None else None
    if yc is not None:   # validación añadida: el corte debe caer dentro de la sección
        _, _, y0, y1 = bbox(comps)
        _check(y0 <= yc <= y1, "La altura del corte debe estar entre 0 y la altura total de la sección")
    return comps, compute(comps, theta, V, yc), theta, yc, V


def filas(last, u):
    r, th, yc = last[1], last[2], last[3]
    L = [("Área A", r["A"], f"{u}²"), ("Centroide x̄", r["xb"], u), ("Centroide ȳ", r["yb"], u),
         ("Q respecto al eje x base (A·ȳ)", r["Qx0"], f"{u}³"),
         ("Q respecto al eje y base (A·x̄)", r["Qy0"], f"{u}³"),
         ("Ix (centroidal)", r["Ix"], f"{u}⁴"), ("Iy (centroidal)", r["Iy"], f"{u}⁴"),
         ("Ixy (centroidal)", r["Ixy"], f"{u}⁴"),
         ("Radio de giro rx", r["rx"], u), ("Radio de giro ry", r["ry"], u),
         ("Módulo elástico Sx", r["Sx"], f"{u}³"), ("Módulo elástico Sy", r["Sy"], f"{u}³"),
         ("Módulo plástico Zx", r["Zx"], f"{u}³"), ("Módulo plástico Zy", r["Zy"], f"{u}³"),
         ("Inercia principal I1", r["I1"], f"{u}⁴"), ("Inercia principal I2", r["I2"], f"{u}⁴"),
         ("Ángulo ejes principales θp", r["thp"], "°"),
         ("Momento polar J = Ix+Iy", r["J"], f"{u}⁴"),
         (f"Ix' (θ={th:g}°)", r["Iu"], f"{u}⁴"), (f"Iy' (θ={th:g}°)", r["Iv"], f"{u}⁴"),
         (f"Ix'y' (θ={th:g}°)", r["Iuv"], f"{u}⁴")]
    if yc is not None and "Q" in r:
        L += [("Q a la altura y dada", r["Q"], f"{u}³"), ("Ancho t en y", r["t"], u)]
        if "tau" in r:
            L.append(("τ = V·Q/(Ix·t)", r["tau"], f"[V]/{u}²"))
    return L


def reporte(last, u, pasos=True):
    comps, r, th, yc, V = last
    s = [f"{'RESULTADO':<38}{'VALOR':>16}  UNIDAD", "-" * 66]
    s += [f"{n:<38}{v:>16.6g}  {x}" for n, v, x in filas(last, u)]
    if pasos:
        s += ["", "PASO A PASO", "-" * 66,
              "1) Componentes (+ material, − hueco), origen en la esquina inferior izquierda:"]
        for i, c in enumerate(comps, 1):
            s.append(f"   #{i} {'+' if c[-1] > 0 else '-'}  Ai={area_i(c):.5g}  xi={c[1]:.5g}  yi={c[2]:.5g}")
        s += [f"2) A = Σ si·Ai = {r['A']:.6g}",
              f"3) x̄ = Σ si·Ai·xi / A = {r['xb']:.6g}    ȳ = Σ si·Ai·yi / A = {r['yb']:.6g}",
              f"4) Steiner: Ix = Σ si·(Ix_i + Ai·(yi−ȳ)²) = {r['Ix']:.6g}",
              f"            Iy = Σ si·(Iy_i + Ai·(xi−x̄)²) = {r['Iy']:.6g}",
              f"            Ixy = Σ si·Ai·(xi−x̄)(yi−ȳ) = {r['Ixy']:.6g}",
              f"5) rx = √(Ix/A) = {r['rx']:.6g}    ry = √(Iy/A) = {r['ry']:.6g}",
              f"6) Sx = Ix/c_max = {r['Sx']:.6g}    Sy = Iy/c_max = {r['Sy']:.6g}",
              f"7) Z = ∫|y−y_pl| dA, eje plástico (divide A/2): y_pl={r['ypl']:.5g}, x_pl={r['xpl']:.5g}",
              f"8) I1,2 = (Ix+Iy)/2 ± √(((Ix−Iy)/2)²+Ixy²) = {r['I1']:.6g} ; {r['I2']:.6g}",
              f"   tan 2θp = −2Ixy/(Ix−Iy) → θp = {r['thp']:.4f}°",
              f"9) J = Ix + Iy = {r['J']:.6g}",
              "10) Giro: Ix' = (Ix+Iy)/2 + (Ix−Iy)/2·cos2θ − Ixy·sin2θ"]
        if yc is not None:
            s.append("11) Q(y) = ∫ (y'−ȳ)·b(y') dy' desde y hasta el borde;  τ = V·Q/(Ix·t)")
    return "\n".join(s)


def dibujar(last, u):
    comps, r, th, yc, _ = last
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x0, x1, y0, y1 = r["box"]
    L = max(x1 - x0, y1 - y0)
    for c in comps:
        fc = "#9ec5e8" if c[-1] > 0 else "white"
        z = 1 if c[-1] > 0 else 2
        if c[0] == "r":
            ax.add_patch(Rectangle((c[1]-c[3]/2, c[2]-c[4]/2), c[3], c[4], fc=fc, ec="k", zorder=z))
        else:
            ax.add_patch(Circle((c[1], c[2]), c[3]/2, fc=fc, ec="k", zorder=z))
    xb, yb, m = r["xb"], r["yb"], 0.25 * L
    ax.axhline(yb, color="r", lw=1, zorder=3)
    ax.axvline(xb, color="r", lw=1, zorder=3)
    ax.plot(xb, yb, "ro", zorder=4)
    a = math.radians(r["thp"])
    ax.plot([xb - L*math.cos(a), xb + L*math.cos(a)], [yb - L*math.sin(a), yb + L*math.sin(a)],
            "g--", lw=1, zorder=3, label="eje principal 1")
    if yc is not None:
        ax.axhline(yc, color="orange", ls=":", zorder=3, label="corte y")
    ax.set_xlim(x0 - m, x1 + m)
    ax.set_ylim(y0 - m, y1 + m)
    ax.set_aspect("equal")
    ax.set_title(f"Centroide = ({xb:.4g}, {yb:.4g}) [{u}]")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    t = np.linspace(0, 2*np.pi, 200)
    ax2.plot(r["avg"] + r["R"]*np.cos(t), r["R"]*np.sin(t), "b", lw=1)
    ax2.plot([r["Ix"], r["Iy"]], [r["Ixy"], -r["Ixy"]], "ro-", label="Ix, Iy")
    ax2.plot([r["Iu"], r["Iv"]], [r["Iuv"], -r["Iuv"]], "gs--", label=f"θ={th:g}°")
    ax2.axhline(0, color="k", lw=.5)
    ax2.set_aspect("equal", "datalim")
    ax2.set_title("Círculo de Mohr (I, Ixy)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.3)
    fig.tight_layout()
    return fig


def excel_bytes(last, u, shape, dims, uin):
    """Genera el .xlsx en memoria (para el botón de descarga de la web)."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Resultados"
    ws.append(["Sección", shape])
    for n, v in zip(SHAPES[shape][0], dims):
        ws.append([n, v, uin])
    ws.append([])
    ws.append(["Propiedad", "Valor", "Unidad"])
    for row in filas(last, u):
        ws.append(list(row))
    ws.column_dimensions["A"].width = 38
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def pdf_bytes(last, u, fig, texto):
    """Genera el PDF en memoria: figura + reporte de texto."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig)
        fg = Figure(figsize=(8.27, 11.69))
        fg.text(0.05, 0.97, texto, va="top", family="monospace", fontsize=7)
        pdf.savefig(fg)
    return buf.getvalue()
