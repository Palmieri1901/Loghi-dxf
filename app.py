# -*- coding: utf-8 -*-
"""
LogoCutter - da foto/PDF di un logo a DXF per taglio laser (RDWorks)
Rileva i contorni, li sovrappone alla foto e permette di correggerli
(spostare / aggiungere / eliminare punti, disegnare o cancellare forme).
Avvio:  streamlit run app.py
"""
import base64
import hashlib
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Supporto foto HEIC/HEIF (iPhone). Se la libreria manca, l'app funziona lo stesso.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

MAX_SIDE = 2200  # le foto del telefono (12-50 MP) vengono ridotte: piu' veloce e stabile
OK_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
          ".heic", ".heif", ".pdf")

MODE_AUTO = "Automatico con correzione ombre (consigliato)"
MODE_ADAPT = "Adattivo (solo per loghi sottili / a linea)"
MODE_FIXED = "Fissa"

# ----------------------------- CORE ---------------------------------


def _shrink(bgr, max_side=MAX_SIDE):
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m > max_side:
        s = max_side / m
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


def load_any(file_bytes, name):
    """Carica immagine o PDF (prima pagina) -> BGR opencv, ridotta e raddrizzata."""
    if name.lower().endswith(".pdf"):
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
        finally:
            doc.close()
        return _shrink(img)

    # Foto: si passa da Pillow (legge anche HEIC e applica la rotazione EXIF)
    try:
        pil = Image.open(io.BytesIO(file_bytes))
        pil = ImageOps.exif_transpose(pil)
        if pil.mode in ("RGBA", "LA", "P"):
            pil = pil.convert("RGBA")
            bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))  # trasparenza -> bianco
            bg.alpha_composite(pil)
            pil = bg
        pil = pil.convert("RGB")
    except Exception as e:
        ext = name.lower().rsplit(".", 1)[-1]
        if ext in ("heic", "heif") and not HEIC_OK:
            raise ValueError("Formato HEIC non supportato su questo server "
                             "(manca pillow-heif). Salva la foto come JPG.") from e
        raise ValueError(f"Impossibile leggere l'immagine: {e}") from e
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return _shrink(img)


def _flat_field(gray):
    """Toglie ombre e sfumature di luce: stima lo sfondo (su copia piccola, veloce)
    e divide. Le zone scure piu' piccole di ~35% dell'immagine non vengono scambiate per ombra."""
    h, w = gray.shape
    s = 260.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    k = int(max(small.shape) * 0.35) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, ker)
    bg = cv2.GaussianBlur(bg, (0, 0), max(small.shape) * 0.08)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray, np.maximum(bg, 1), scale=255)


def detect_shapes(img, mode, block, C, thresh, blur, close_iter,
                  min_area_pct, invert):
    """Ritorna (forme, maschera). Ogni forma = (contorno esterno, [fori interni])."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if invert:                      # logo chiaro su sfondo scuro -> si riconduce al caso normale
        gray = 255 - gray
    if blur > 0:
        k = blur * 2 + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    if mode == MODE_AUTO:
        norm = _flat_field(gray)
        _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif mode == MODE_ADAPT:
        # finestra proporzionale alla foto, cosi' non dipende dalla risoluzione
        win = int(max(gray.shape) * block / 400.0) | 1
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, max(win, 3), C)
    else:
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    if close_iter > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker, iterations=close_iter)

    # RETR_CCOMP: livello 1 = contorni esterni, livello 2 = fori interni
    contours, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return [], bw
    hier = hier[0]
    h, w = img.shape[:2]
    min_area = (min_area_pct / 100.0) * h * w
    min_hole = min_area * 0.1  # i fori possono essere piu' piccoli (es. occhio di una lettera)

    shapes = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1:       # e' un foro: lo gestiamo dal suo padre
            continue
        if cv2.contourArea(c) < min_area:
            continue
        holes = [contours[j] for j in range(len(contours))
                 if hier[j][3] == i and cv2.contourArea(contours[j]) >= min_hole]
        shapes.append((c, holes))
    shapes.sort(key=lambda s: cv2.contourArea(s[0]), reverse=True)
    return shapes, bw


def smooth_approx(cnt, epsilon_pct):
    eps = (epsilon_pct / 100.0) * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float64)


def shapes_to_polys(shapes, epsilon_pct):
    """Forme rilevate -> lista di poligoni modificabili in pixel:
    [{"pts": [[x, y], ...], "hole": bool}, ...]"""
    polys = []
    for outer, holes in shapes:
        po = smooth_approx(outer, epsilon_pct)
        if len(po) >= 3:
            polys.append({"pts": np.round(po, 1).tolist(), "hole": False})
        for hh in holes:
            ph = smooth_approx(hh, epsilon_pct)
            if len(ph) >= 3:
                polys.append({"pts": np.round(ph, 1).tolist(), "hole": True})
    return polys


def polys_to_mm(polys, target_width_mm):
    """Pixel -> mm con UNA scala comune (le proporzioni tra le forme restano giuste).
    La larghezza totale di tutto il disegno diventa target_width_mm."""
    arrs = [np.asarray(p["pts"], dtype=np.float64) for p in polys if len(p["pts"]) >= 3]
    if not arrs:
        return []
    allpts = np.vstack(arrs)
    xmin, xmax = allpts[:, 0].min(), allpts[:, 0].max()
    ymax = allpts[:, 1].max()
    width = xmax - xmin
    scale = target_width_mm / width if width > 0 else 1.0
    # Y ribaltato: il laser lavora in coordinate cartesiane
    return [np.column_stack([(a[:, 0] - xmin) * scale, (ymax - a[:, 1]) * scale]) for a in arrs]


def export_dxf(polys_mm):
    import ezdxf
    doc = ezdxf.new("R12")  # massima compatibilita' RDWorks (unita': mm)
    msp = doc.modelspace()
    for poly in polys_mm:
        msp.add_polyline2d([tuple(pt) for pt in poly], close=True)
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue()


def export_svg(polys_mm):
    allp = np.vstack(polys_mm)
    w = allp[:, 0].max()
    h = allp[:, 1].max()
    d = " ".join("M" + " L".join(f"{x:.3f},{h - y:.3f}" for x, y in poly) + " Z"
                 for poly in polys_mm)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2f}mm" height="{h:.2f}mm" '
            f'viewBox="0 0 {w:.2f} {h:.2f}">'
            f'<path d="{d}" fill="none" stroke="black" stroke-width="0.1" fill-rule="evenodd"/></svg>')


def img_to_data_url(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

# ------------------------- EDITOR (componente) ------------------------

EDITOR_HTML = r'''<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;padding:0;background:transparent;font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#222}
#bar{display:flex;flex-wrap:wrap;gap:6px;padding:2px 0 8px 0;align-items:center}
button{border:1px solid #8a8a8a;background:#f5f5f5;color:#222;border-radius:8px;padding:6px 10px;font-size:15px;min-height:40px;cursor:pointer}
button.on{background:#1f7aec;color:#fff;border-color:#1f7aec}
button:disabled{opacity:.4}
.sep{width:1px;height:26px;background:#999;margin:0 2px}
label{font-size:13px;display:flex;align-items:center;gap:6px}
input[type=range]{width:90px}
#wrap{position:relative;width:100%;border:1px solid #999;border-radius:8px;overflow:hidden;background:#cfcfcf;touch-action:none}
canvas{position:absolute;left:0;top:0;width:100%;height:100%;touch-action:none;display:block}
#st{font-size:13px;padding:6px 2px 2px 2px;min-height:18px}
</style></head>
<body>
<div id="bar">
  <button id="t_move" class="on" title="Trascina i punti; trascina lo sfondo per spostare la vista">✋ Sposta</button>
  <button id="t_add" title="Tocca una linea per aggiungere un punto (poi trascinalo)">➕ Punto</button>
  <button id="t_del" title="Tocca un punto per eliminarlo">➖ Punto</button>
  <button id="t_new" title="Tocca per disegnare una forma nuova">✏️ Nuova forma</button>
  <button id="b_close" style="display:none">✔ Chiudi forma</button>
  <button id="t_delshape" title="Tocca una forma per eliminarla">🗑 Forma</button>
  <span class="sep"></span>
  <button id="b_undo">↶ Annulla</button>
  <span class="sep"></span>
  <button id="b_zin">🔍＋</button><button id="b_zout">🔍－</button><button id="b_fit">Adatta</button>
  <span class="sep"></span>
  <label>Foto <input id="alpha" type="range" min="0" max="100" value="60"></label>
  <button id="b_show" class="on">👁</button>
</div>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="st"></div>

<script>
(function(){
var cv=document.getElementById('cv'), wrap=document.getElementById('wrap'), ctx=cv.getContext('2d');
var stEl=document.getElementById('st');
var img=null,imgW=1,imgH=1,polys=[],dataId=null,tool='move';
var view={s:1,ox:0,oy:0}, undo=[], drawing=[], drag=null, panStart=null;
var pointers=new Map(), pinch=null, downPos=null, moved=false, alpha=0.6, showImg=true;
var W=300,H=300,dpr=1,sendTimer=null,lastSent=null;

/* ---------- protocollo componente Streamlit (senza librerie esterne) ---------- */
function post(msg){ msg.isStreamlitMessage=true; window.parent.postMessage(msg,'*'); }
function setHeight(){ var h=document.getElementById('bar').offsetHeight+16+wrap.offsetHeight+stEl.offsetHeight+16; post({type:'streamlit:setFrameHeight',height:h}); }
function sendValue(){
  var v={data_id:dataId,polys:polys};
  post({type:'streamlit:setComponentValue',value:v,dataType:'json'});
  stEl.textContent=statusText()+'  ·  modifiche inviate ✓';
}
function scheduleSend(){ clearTimeout(sendTimer); sendTimer=setTimeout(sendValue,500); }

/* ---------- utilita' ---------- */
function statusText(){
  var n=0; polys.forEach(function(p){n+=p.pts.length;});
  return 'Forme: '+polys.length+' · Punti: '+n;
}
function toScr(p){ return [p[0]*view.s+view.ox, p[1]*view.s+view.oy]; }
function toImg(x,y){ return [(x-view.ox)/view.s,(y-view.oy)/view.s]; }
function pos(e){ var r=cv.getBoundingClientRect(); return [e.clientX-r.left,e.clientY-r.top]; }
function hitR(e){ return (e&&e.pointerType==='touch')?24:12; }
function clone(o){ return JSON.parse(JSON.stringify(o)); }
function pushUndo(){ undo.push(JSON.stringify(polys)); if(undo.length>80) undo.shift(); }
function projSeg(x,y,a,b){
  var dx=b[0]-a[0],dy=b[1]-a[1],l2=dx*dx+dy*dy,t=l2?((x-a[0])*dx+(y-a[1])*dy)/l2:0;
  t=Math.max(0,Math.min(1,t)); var px=a[0]+t*dx,py=a[1]+t*dy;
  return {t:t,x:px,y:py,d:Math.hypot(x-px,y-py)};
}
function hitNode(x,y,r){
  var best=null,bd=r*r;
  polys.forEach(function(P,pi){ P.pts.forEach(function(p,ni){
    var s=toScr(p),d=(s[0]-x)*(s[0]-x)+(s[1]-y)*(s[1]-y);
    if(d<=bd){bd=d;best={pi:pi,ni:ni};} }); });
  return best;
}
function hitSeg(x,y,r){
  var best=null,bd=r;
  polys.forEach(function(P,pi){ var n=P.pts.length;
    for(var i=0;i<n;i++){
      var q=projSeg(x,y,toScr(P.pts[i]),toScr(P.pts[(i+1)%n]));
      if(q.d<bd){bd=q.d;best={pi:pi,ni:i,x:q.x,y:q.y};}
    } });
  return best;
}
function inPoly(pt,pts){
  var c=false;
  for(var i=0,j=pts.length-1;i<pts.length;j=i++){
    if(((pts[i][1]>pt[1])!==(pts[j][1]>pt[1])) &&
       (pt[0]<(pts[j][0]-pts[i][0])*(pt[1]-pts[i][1])/(pts[j][1]-pts[i][1])+pts[i][0])) c=!c;
  }
  return c;
}
function area(pts){ var a=0; for(var i=0;i<pts.length;i++){var p=pts[i],q=pts[(i+1)%pts.length];a+=p[0]*q[1]-q[0]*p[1];} return Math.abs(a)/2; }

/* ---------- operazioni di modifica ---------- */
function deleteNode(h){
  var P=polys[h.pi];
  if(P.pts.length<=3){ polys.splice(h.pi,1); } else { P.pts.splice(h.ni,1); }
}
function insertNode(h){
  var im=toImg(h.x,h.y);
  polys[h.pi].pts.splice(h.ni+1,0,im);
  return {pi:h.pi,ni:h.ni+1};
}
function deleteShapeAt(x,y){
  var h=hitSeg(x,y,22);
  if(h){ pushUndo(); polys.splice(h.pi,1); return true; }
  var im=toImg(x,y),best=-1,ba=Infinity;
  polys.forEach(function(P,i){ if(inPoly(im,P.pts)){ var a=area(P.pts); if(a<ba){ba=a;best=i;} } });
  if(best>=0){ pushUndo(); polys.splice(best,1); return true; }
  return false;
}
function closeDrawing(){
  if(drawing.length>=3){ pushUndo(); polys.push({pts:drawing.slice(),hole:false}); drawing=[]; commit(); }
  else if(drawing.length>0){ drawing=[]; draw(); }
}
function commit(){ updateUI(); draw(); scheduleSend(); }

/* ---------- vista ---------- */
function resize(){
  dpr=window.devicePixelRatio||1;
  W=wrap.clientWidth||300;
  var h=W<620?440:600; wrap.style.height=h+'px'; H=h;
  cv.width=Math.round(W*dpr); cv.height=Math.round(H*dpr);
}
function fit(){
  view.s=Math.min(W/imgW,H/imgH)*0.97;
  view.ox=(W-imgW*view.s)/2; view.oy=(H-imgH*view.s)/2;
}
function zoomAt(cx,cy,f){
  var s2=Math.max(0.03,Math.min(60,view.s*f)),k=s2/view.s;
  view.ox=cx-(cx-view.ox)*k; view.oy=cy-(cy-view.oy)*k; view.s=s2;
}

/* ---------- disegno ---------- */
function draw(){
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle='#cfcfcf'; ctx.fillRect(0,0,W,H);
  var iw=imgW*view.s, ih=imgH*view.s;
  ctx.fillStyle='#fff'; ctx.fillRect(view.ox,view.oy,iw,ih);
  if(img&&showImg){ ctx.globalAlpha=alpha; ctx.drawImage(img,view.ox,view.oy,iw,ih); ctx.globalAlpha=1; }
  var total=0; polys.forEach(function(p){total+=p.pts.length;});
  var showNodes=total<4000;
  polys.forEach(function(P){
    var col=P.hole?'#e0262d':'#0a9a3c';
    ctx.beginPath();
    P.pts.forEach(function(p,i){ var s=toScr(p); if(i===0)ctx.moveTo(s[0],s[1]); else ctx.lineTo(s[0],s[1]); });
    ctx.closePath(); ctx.lineWidth=2; ctx.strokeStyle=col; ctx.stroke();
    if(showNodes){
      ctx.fillStyle='#fff'; ctx.lineWidth=1.5;
      P.pts.forEach(function(p){ var s=toScr(p); ctx.beginPath(); ctx.arc(s[0],s[1],4,0,6.2832); ctx.fill(); ctx.stroke(); });
    }
  });
  if(drawing.length){
    ctx.strokeStyle='#1f7aec'; ctx.fillStyle='#1f7aec'; ctx.lineWidth=2; ctx.beginPath();
    drawing.forEach(function(p,i){ var s=toScr(p); if(i===0)ctx.moveTo(s[0],s[1]); else ctx.lineTo(s[0],s[1]); });
    ctx.stroke();
    drawing.forEach(function(p,i){ var s=toScr(p); ctx.beginPath(); ctx.arc(s[0],s[1],i===0?7:4,0,6.2832); ctx.fill(); });
  }
}

/* ---------- interfaccia ---------- */
var toolBtns={move:'t_move',add:'t_add',del:'t_del',new:'t_new',delshape:'t_delshape'};
function setTool(t){
  if(tool==='new'&&t!=='new') drawing=[];
  tool=t;
  Object.keys(toolBtns).forEach(function(k){ document.getElementById(toolBtns[k]).className=(k===t?'on':''); });
  document.getElementById('b_close').style.display=(t==='new')?'':'none';
  draw(); setTimeout(setHeight,50);
}
function updateUI(){ stEl.textContent=statusText(); document.getElementById('b_undo').disabled=(undo.length===0&&drawing.length===0); }
Object.keys(toolBtns).forEach(function(k){ document.getElementById(toolBtns[k]).onclick=function(){ setTool(k); }; });
document.getElementById('b_close').onclick=closeDrawing;
document.getElementById('b_undo').onclick=function(){
  if(drawing.length){ drawing.pop(); draw(); updateUI(); return; }
  if(undo.length){ polys=JSON.parse(undo.pop()); commit(); }
};
document.getElementById('b_zin').onclick=function(){ zoomAt(W/2,H/2,1.4); draw(); };
document.getElementById('b_zout').onclick=function(){ zoomAt(W/2,H/2,1/1.4); draw(); };
document.getElementById('b_fit').onclick=function(){ fit(); draw(); };
document.getElementById('alpha').oninput=function(e){ alpha=e.target.value/100; draw(); };
document.getElementById('b_show').onclick=function(e){ showImg=!showImg; e.target.className=showImg?'on':''; draw(); };

/* ---------- puntatore (mouse, touch, penna) ---------- */
cv.addEventListener('pointerdown',function(e){
  try{ cv.setPointerCapture(e.pointerId); }catch(_){}
  var p=pos(e); pointers.set(e.pointerId,p);
  if(pointers.size===2){
    var a=Array.from(pointers.values());
    pinch={d:Math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1]),cx:(a[0][0]+a[1][0])/2,cy:(a[0][1]+a[1][1])/2};
    drag=null; panStart=null; return;
  }
  if(pointers.size>2) return;
  downPos=p; moved=false; drag=null; panStart=null;
  var r=hitR(e);
  if(tool==='move'){
    var h=hitNode(p[0],p[1],r);
    if(h){ pushUndo(); drag=h; } else panStart={x:p[0],y:p[1],ox:view.ox,oy:view.oy};
  } else if(tool==='add'){
    var hn=hitNode(p[0],p[1],r*0.6);
    if(hn){ pushUndo(); drag=hn; }
    else{
      var hs=hitSeg(p[0],p[1],r+6);
      if(hs){ pushUndo(); drag=insertNode(hs); draw(); }
      else panStart={x:p[0],y:p[1],ox:view.ox,oy:view.oy};
    }
  } else if(tool==='del'){
    var hd=hitNode(p[0],p[1],r);
    if(hd){ pushUndo(); deleteNode(hd); commit(); downPos=null; }
    else panStart={x:p[0],y:p[1],ox:view.ox,oy:view.oy};
  } else {
    panStart={x:p[0],y:p[1],ox:view.ox,oy:view.oy};
  }
});
cv.addEventListener('pointermove',function(e){
  if(!pointers.has(e.pointerId)) return;
  var p=pos(e); pointers.set(e.pointerId,p);
  if(pinch&&pointers.size>=2){
    var a=Array.from(pointers.values());
    var d=Math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1]),cx=(a[0][0]+a[1][0])/2,cy=(a[0][1]+a[1][1])/2;
    if(pinch.d>0) zoomAt(cx,cy,d/pinch.d);
    view.ox+=cx-pinch.cx; view.oy+=cy-pinch.cy;
    pinch={d:d,cx:cx,cy:cy}; draw(); return;
  }
  if(downPos&&Math.hypot(p[0]-downPos[0],p[1]-downPos[1])>6) moved=true;
  if(drag){
    polys[drag.pi].pts[drag.ni]=toImg(p[0],p[1]); draw();
  } else if(panStart){
    view.ox=panStart.ox+(p[0]-panStart.x); view.oy=panStart.oy+(p[1]-panStart.y); draw();
  }
});
function endPointer(e){
  var had=pointers.has(e.pointerId); pointers.delete(e.pointerId);
  if(pointers.size<2) pinch=null;
  if(!had) return;
  if(drag){ drag=null; commit(); return; }
  if(!moved&&downPos&&pointers.size===0){
    var p=downPos;
    if(tool==='delshape'){ if(deleteShapeAt(p[0],p[1])) commit(); }
    else if(tool==='new'){
      var im=toImg(p[0],p[1]);
      if(drawing.length>=3){ var f=toScr(drawing[0]); if(Math.hypot(f[0]-p[0],f[1]-p[1])<hitR(e)){ closeDrawing(); panStart=null; downPos=null; return; } }
      drawing.push(im); draw(); updateUI();
    }
  }
  panStart=null; downPos=null;
}
cv.addEventListener('pointerup',endPointer);
cv.addEventListener('pointercancel',endPointer);
cv.addEventListener('wheel',function(e){
  e.preventDefault(); var p=pos(e); zoomAt(p[0],p[1],e.deltaY<0?1.15:1/1.15); draw();
},{passive:false});
window.addEventListener('resize',function(){ resize(); draw(); setHeight(); });

/* ---------- ricezione dati da Streamlit ---------- */
window.addEventListener('message',function(ev){
  var d=ev.data; if(!d||d.type!=='streamlit:render') return;
  var a=d.args||{};
  if(d.theme&&d.theme.textColor) document.body.style.color=d.theme.textColor;
  if(a.data_id!==dataId){
    dataId=a.data_id; polys=clone(a.polys||[]); undo=[]; drawing=[]; drag=null; panStart=null;
    resize();
    var im=new Image();
    im.onload=function(){ img=im; imgW=im.naturalWidth; imgH=im.naturalHeight; fit(); updateUI(); draw(); setHeight(); };
    im.src=a.image;
  }
  setHeight();
});
resize(); updateUI(); draw();
post({type:'streamlit:componentReady',apiVersion:1});
setHeight();
/* per i test nel browser: espone lo stato */
window.__logoEditor={getPolys:function(){return polys;},getView:function(){return view;},hitSeg:hitSeg,hitNode:hitNode,getTool:function(){return tool;}};
})();
</script></body></html>
'''


def _editor_component():
    import streamlit.components.v1 as components
    d = Path(tempfile.gettempdir()) / "logocutter_editor"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "index.html"
    if not f.exists() or f.read_text(encoding="utf-8") != EDITOR_HTML:
        f.write_text(EDITOR_HTML, encoding="utf-8")
    return components.declare_component("logocutter_editor", path=str(d))

# ----------------------------- UI ------------------------------------


def main():
    import streamlit as st

    st.set_page_config(page_title="LogoCutter", layout="wide")
    st.title("🛥️ LogoCutter — da foto a DXF per taglierina laser")

    editor = _editor_component()

    @st.cache_data(show_spinner=False, max_entries=8)
    def cached_load(data, name):
        img = load_any(data, name)
        return img, img_to_data_url(img)

    @st.cache_data(show_spinner=False, max_entries=16)
    def cached_detect(data, name, mode, block, C, thresh, blur, close_iter,
                      min_area, invert):
        img, _ = cached_load(data, name)
        return detect_shapes(img, mode, block, C, thresh, blur, close_iter,
                             min_area, invert)

    with st.sidebar:
        st.header("⚙️ Rilevamento")
        mode = st.selectbox("Tipo soglia", [MODE_AUTO, MODE_ADAPT, MODE_FIXED])
        invert = st.checkbox("Logo chiaro su sfondo scuro", value=False)
        block = st.slider("Finestra adattiva (solo modo adattivo)", 3, 51, 25, 2)
        C = st.slider("Sensibilità (C) (solo modo adattivo)", 1, 30, 10)
        thresh = st.slider("Soglia fissa (solo modo fisso)", 0, 255, 128)
        blur = st.slider("Sfocatura anti-rumore", 0, 10, 2)
        close_iter = st.slider("Chiusura contorni", 0, 5, 1)
        min_area = st.slider("Area minima (% dell'immagine)", 0.1, 20.0, 0.2, 0.1)
        epsilon = st.slider("Semplificazione contorno (%)", 0.1, 5.0, 1.0, 0.1)
        st.caption("Imposta questi valori PRIMA di correggere a mano: "
                   "se li cambi dopo, il rilevamento riparte da zero.")
        st.header("📏 Dimensione reale")
        target_mm = st.number_input("Larghezza totale del taglio in mm", 10, 3000, 500)

    # Nessun filtro "type": su alcuni telefoni blocca la scelta delle foto.
    files = st.file_uploader("Carica una o più foto / PDF", accept_multiple_files=True)
    if not files:
        st.info("👆 Carica una o più foto del logo (o il PDF) per iniziare.")
        return

    names = [f.name for f in files]
    sel = st.selectbox("File da elaborare", names)
    f = files[names.index(sel)]
    if not f.name.lower().endswith(OK_EXT):
        st.error(f"Formato non supportato: {f.name}. Usa JPG, PNG, WEBP, HEIC, BMP, TIFF o PDF.")
        return

    data = f.getvalue()
    try:
        with st.spinner("Carico l'immagine..."):
            img, data_url = cached_load(data, f.name)
        shapes, bw = cached_detect(data, f.name, mode, block, C, thresh, blur,
                                   close_iter, min_area, invert)
    except Exception as e:
        st.error(f"Non riesco a elaborare il file: {e}")
        return
    st.caption(f"Immagine elaborata a {img.shape[1]}×{img.shape[0]} px")

    if not shapes:
        st.warning("Nessun contorno trovato — regola i parametri nella barra laterale "
                   "(prova ad abbassare «Area minima» o a spuntare «Logo chiaro su sfondo scuro»).")
        with st.expander("Maschera binaria"):
            st.image(bw)
        return

    # ---- stato delle modifiche, per file e per impostazioni di rilevamento ----
    if "edits" not in st.session_state:
        st.session_state["edits"] = {}
    if "reset_n" not in st.session_state:
        st.session_state["reset_n"] = 0
    fhash = hashlib.md5(data).hexdigest()[:10]
    data_id = "|".join(map(str, [fhash, mode, block, C, thresh, blur, close_iter,
                                 min_area, invert, epsilon, st.session_state["reset_n"]]))
    detected = shapes_to_polys(shapes, epsilon)
    start = st.session_state["edits"].get(data_id, detected)

    st.markdown("**Correggi il vettore sopra la foto** — verde = contorno esterno, rosso = foro. "
                "Le modifiche si salvano da sole.")
    val = editor(image=data_url, polys=start, data_id=data_id,
                 key="logo_editor", default=None)
    if isinstance(val, dict) and val.get("data_id") == data_id and val.get("polys") is not None:
        st.session_state["edits"][data_id] = val["polys"]
        current = val["polys"]
    else:
        current = start

    cols = st.columns([1, 1, 3])
    with cols[0]:
        if st.button("↺ Ripristina rilevamento"):
            st.session_state["reset_n"] += 1
            st.rerun()

    polys_mm = polys_to_mm(current, target_mm)
    if not polys_mm:
        st.warning("Non c'è nessuna forma da esportare.")
        return

    allp = np.vstack(polys_mm)
    st.subheader("📥 Esportazione per RDWorks")
    st.caption(f"{len(polys_mm)} contorni · {allp[:, 0].max():.1f} × {allp[:, 1].max():.1f} mm")
    base = f.name.rsplit(".", 1)[0]
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Scarica DXF (RDWorks)", export_dxf(polys_mm),
                           file_name=base + ".dxf", mime="application/dxf")
    with c2:
        st.download_button("Scarica SVG (riserva)", export_svg(polys_mm),
                           file_name=base + ".svg", mime="image/svg+xml")
    st.info("In RDWorks: File → Importa → scegli il DXF. Le misure sono in millimetri.")

    with st.expander("Maschera binaria (per capire se la soglia è giusta)"):
        st.image(bw)


if __name__ == "__main__":
    main()
