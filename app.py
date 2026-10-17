# -*- coding: utf-8 -*-
"""
LogoCutter - Rilevamento contorni da foto/PDF per taglio laser (RDWorks)
Avvio:  streamlit run app.py
"""
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="LogoCutter", layout="wide")
st.title("🛥️ LogoCutter — da foto a DXF per taglierina laser")

# ----------------------------- UTILS ---------------------------------

def load_any(file_bytes, name):
    """Carica immagine o PDF (prima pagina renderizzata) -> BGR opencv."""
    data = np.frombuffer(file_bytes, np.uint8)
    if name.lower().endswith(".pdf"):
        import fitz  # PyMuPDF, nessuna dipendenza esterna
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # ~216 DPI
        img = np.frombuffer(pix.samples, np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        doc.close()
        return img
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def detect_contours(img, mode, block, C, thresh, blur, close_iter, min_area_pct):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur > 0:
        k = blur * 2 + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    if mode == "Adattivo (consigliato per foto al telefono)":
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block * 2 + 1, C)
    else:
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    if close_iter > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker, iterations=close_iter)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    h, w = img.shape[:2]
    min_area = (min_area_pct / 100.0) * h * w
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours.sort(key=cv2.contourArea, reverse=True)
    return contours, bw


def smooth_approx(cnt, epsilon_pct):
    eps = (epsilon_pct / 100.0) * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    return approx.astype(np.float64)


def contour_to_mm(pts, target_width_mm):
    xs, ys = pts[:, 0], pts[:, 1]
    w = xs.max() - xs.min()
    scale = target_width_mm / w if w > 0 else 1.0
    # Y ribaltato: il laser lavora in coordinate cartesiane
    out = np.column_stack([(xs - xs.min()) * scale,
                           (ys.max() - ys) * scale])
    return out, scale


def export_dxf(polys_mm):
    import ezdxf
    doc = ezdxf.new("R12")  # massima compatibilità RDWorks
    doc.header["$INSUNITS"] = 4  # millimetri
    msp = doc.modelspace()
    for p in polys_mm:
        msp.add_polyline2d(p, close=True)
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue()


def export_svg(polys_mm, target_w):
    maxw = max((p[:, 0].max() - p[:, 0].min()) for p in polys_mm)
    maxh = max((p[:, 1].max() - p[:, 1].min()) for p in polys_mm)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{maxw:.2f}mm" '
             f'height="{maxh:.2f}mm" viewBox="0 0 {maxw:.2f} {maxh:.2f}">',
             '<g fill="none" stroke="black" stroke-width="0.1">']
    for p in polys_mm:
        d = "M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in p) + " Z"
        parts.append(f'<path d="{d}"/>')
    parts.append("</g></svg>")
    return "\n".join(parts)

# ----------------------------- UI ------------------------------------

with st.sidebar:
    st.header("⚙️ Rilevamento")
    mode = st.selectbox("Tipo soglia", ["Adattivo (consigliato per foto al telefono)", "Fissa"])
    block = st.slider("Finestra adattiva", 3, 51, 25, 2)
    C = st.slider("Sensibilità (C)", 1, 30, 10)
    thresh = st.slider("Soglia fissa", 0, 255, 128)
    blur = st.slider("Sfocatura anti-rumore", 0, 10, 2)
    close_iter = st.slider("Chiusura contorni", 0, 5, 1)
    min_area = st.slider("Area minima (% dell'immagine)", 0.1, 20.0, 1.0, 0.1)
    epsilon = st.slider("Semplificazione contorno (%)", 0.1, 5.0, 1.0, 0.1)
    st.header("📏 Dimensione reale")
    target_mm = st.number_input("Larghezza del taglio in mm", 10, 3000, 500)

files = st.file_uploader("Carica una o più foto / PDF",
                         type=["jpg", "jpeg", "png", "bmp", "pdf"],
                         accept_multiple_files=True)

if files:
    names = [f.name for f in files]
    sel = st.selectbox("File da elaborare", names)
    f = files[names.index(sel)]
    img = load_any(f.getvalue(), f.name)
    st.caption(f"Risoluzione: {img.shape[1]}×{img.shape[0]} px")

    contours, bw = detect_contours(img, mode, block, C, thresh, blur,
                                   close_iter, min_area)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Originale")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Maschera binaria")
        st.image(bw, use_container_width=True)

    if not contours:
        st.warning("Nessun contorno trovato — regola i parametri nella barra laterale.")
        st.stop()

    n_show = st.slider("Quanti contorni esportare", 1, max(1, len(contours)),
                       min(3, len(contours)))
    preview = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255)]
    for i, c in enumerate(contours[:n_show]):
        cv2.drawContours(preview, [c], -1, colors[i % len(colors)], 3)
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(preview, str(i + 1), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, colors[i % len(colors)], 3)
    st.subheader("Anteprima contorni")
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

    polys_mm = []
    for c in contours[:n_show]:
        pts = smooth_approx(c, epsilon)
        mm, _ = contour_to_mm(pts, target_mm)
        polys_mm.append(mm)

    dxf = export_dxf(polys_mm)
    svg = export_svg(polys_mm, target_mm)
    st.subheader("📥 Esportazione per RDWorks")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Scarica DXF (RDWorks)", dxf,
                           file_name=f.name.rsplit(".", 1)[0] + ".dxf",
                           mime="application/dxf")
    with c2:
        st.download_button("Scarica SVG (riserva)", svg,
                           file_name=f.name.rsplit(".", 1)[0] + ".svg",
                           mime="image/svg+xml")
    st.info("In RDWorks: File → Importa DXF → imposta larghezza taglio come da anteprima.")
else:
    st.info("👆 Carica una o più foto del logo (o il PDF) per iniziare.")
