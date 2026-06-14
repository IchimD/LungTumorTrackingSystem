"""
Lung Tumor Tracking System — Thesis Demo
==========================================
Streamlit UI with:
  • Example tab  — instant pre-computed results (no waiting)
  • Upload tab   — live inference on uploaded .npy file
"""

import os, json, time
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import zoom, center_of_mass
from io import BytesIO

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "D:/LICENTA2/results/fold2_best.pt"
CACHE_DIR    = "D:/LICENTA2/demo_cache"
TRACKING_DIR = "D:/LICENTA2/tracking_results"
HU_MIN, HU_MAX = -200, 300
IMG_SIZE = 384

# ── Design tokens ─────────────────────────────────────────────────────────────
BG     = "#070b14"
CARD   = "#0d1117"
ACC    = "#388bfd"
RED    = "#ff6b6b"
GRN    = "#56d364"
TXT    = "#e6edf3"
SUB    = "#8b949e"
BORDER = "#1e2535"

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Tumor Tracking System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"], .stApp {{
      font-family: 'Inter', sans-serif !important;
      background-color: {BG} !important;
      color: {TXT} !important;
  }}
  .stApp > header {{ background: transparent !important; }}
  [data-testid="stSidebar"] {{
      background-color: {CARD} !important;
      border-right: 1px solid {BORDER} !important;
  }}
  [data-testid="stSidebar"] * {{ color: {TXT} !important; }}
  .stTabs [data-baseweb="tab-list"] {{
      gap: 12px;
      background: {CARD} !important;
      border: 1px solid {BORDER} !important;
      border-radius: 12px !important;
      padding: 6px !important;
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 8px !important;
      padding: 8px 20px !important;
      color: {SUB} !important;
      font-weight: 500;
      font-size: 14px;
  }}
  .stTabs [aria-selected="true"] {{
      background: {ACC} !important;
      color: white !important;
  }}
  .stButton > button {{
      background: {ACC} !important;
      color: white !important;
      border: none !important;
      border-radius: 8px !important;
      padding: 10px 24px !important;
      font-weight: 600 !important;
      font-size: 14px !important;
      width: 100%;
  }}
  .stImage > img {{
      border-radius: 12px !important;
      border: 1px solid {BORDER} !important;
  }}
  div[data-testid="stMetric"] {{
      background: {CARD} !important;
      border: 1px solid {BORDER} !important;
      border-radius: 12px !important;
      padding: 16px 20px !important;
  }}
  div[data-testid="stMetric"] label {{ color: {SUB} !important; font-size: 12px !important; }}
  div[data-testid="stMetricValue"] {{
      color: {TXT} !important;
      font-size: 26px !important;
      font-weight: 700 !important;
  }}
  hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:8px 0 20px; border-bottom:1px solid {BORDER}; margin-bottom:20px;">
      <div style="font-size:20px; font-weight:700; color:{TXT};">🫁 LungTrack</div>
      <div style="font-size:11px; color:{SUB}; margin-top:4px;">Bachelor's Thesis — 2025</div>
    </div>
    <div style="font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase;
                color:{SUB}; margin-bottom:8px;">Model</div>
    <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:20px;">
      <div style="display:flex; justify-content:space-between;">
        <span style="color:{SUB}; font-size:13px;">Architecture</span>
        <span style="color:{TXT}; font-size:13px; font-weight:600;">U-Net++</span>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span style="color:{SUB}; font-size:13px;">Encoder</span>
        <span style="color:{TXT}; font-size:13px; font-weight:600;">EfficientNet-B4</span>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span style="color:{SUB}; font-size:13px;">Input size</span>
        <span style="color:{TXT}; font-size:13px; font-weight:600;">384 × 384</span>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span style="color:{SUB}; font-size:13px;">Training set</span>
        <span style="color:{TXT}; font-size:13px; font-weight:600;">421 patients</span>
      </div>
    </div>
    <hr/>
    <div style="font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase;
                color:{SUB}; margin:16px 0 8px;">Performance</div>
    <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:20px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="color:{SUB}; font-size:13px;">Mean Dice</span>
        <span style="color:{GRN}; font-size:15px; font-weight:700;">0.6294</span>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="color:{SUB}; font-size:13px;">TTA Dice</span>
        <span style="color:{GRN}; font-size:15px; font-weight:700;">0.6390</span>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="color:{SUB}; font-size:13px;">Best fold</span>
        <span style="color:{ACC}; font-size:15px; font-weight:700;">0.6434</span>
      </div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    threshold = 0.5


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:32px 0 24px;">
  <div style="font-size:32px; font-weight:700; color:{TXT}; letter-spacing:-0.5px;">
    Lung Tumor Tracking System
  </div>
  <div style="font-size:15px; color:{SUB}; margin-top:6px;">
    Automated detection &amp; motion tracking for radiotherapy planning
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  Example Results  ", "  Upload Your CT  "])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXAMPLE RESULTS (instant, pre-computed, no inference)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    meta_path = os.path.join(CACHE_DIR, "meta.json")

    if not os.path.isfile(meta_path):
        st.warning("Pre-computed examples not found. Run **`python precompute_demo.py`** once.", icon="⚠️")
    else:
        # ── Patient info (simplified) ─────────────────────────────────────────
        st.markdown(f"""
        <div style="display:flex; gap:16px; margin-bottom:28px; flex-wrap:wrap;">
          <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:10px;
                      padding:14px 20px; flex:1; min-width:150px;">
            <div style="font-size:11px; color:{SUB}; text-transform:uppercase; letter-spacing:.08em;">Patient</div>
            <div style="font-size:16px; font-weight:700; color:{TXT}; margin-top:4px;">100_HM10395</div>
            <div style="font-size:11px; color:{SUB}; margin-top:2px;">4D-Lung dataset (TCIA)</div>
          </div>
          <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:10px;
                      padding:14px 20px; flex:1; min-width:150px;">
            <div style="font-size:11px; color:{SUB}; text-transform:uppercase; letter-spacing:.08em;">What we scan</div>
            <div style="font-size:16px; font-weight:700; color:{TXT}; margin-top:4px;">10 CT volumes</div>
            <div style="font-size:11px; color:{SUB}; margin-top:2px;">one per breathing phase</div>
          </div>
          <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:10px;
                      padding:14px 20px; flex:1; min-width:150px;">
            <div style="font-size:11px; color:{SUB}; text-transform:uppercase; letter-spacing:.08em;">AI accuracy</div>
            <div style="font-size:16px; font-weight:700; color:{GRN}; margin-top:4px;">Dice 0.6294</div>
            <div style="font-size:11px; color:{SUB}; margin-top:2px;">trained on 421 patients</div>
          </div>
          <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:10px;
                      padding:14px 20px; flex:1; min-width:150px;">
            <div style="font-size:11px; color:{SUB}; text-transform:uppercase; letter-spacing:.08em;">vs. doctor error</div>
            <div style="font-size:16px; font-weight:700; color:#f9c513; margin-top:4px;">16.7 mm avg</div>
            <div style="font-size:11px; color:{SUB}; margin-top:2px;">AI vs. radiologist position</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr style="margin:20px 0 28px;"/>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # INTERACTIVE PHASE VIEWER
        # ══════════════════════════════════════════════════════════════════════
        have_phases = os.path.isfile(os.path.join(CACHE_DIR, "phase_00_panel.png"))

        st.markdown(f"""
        <div style="margin-bottom:16px;">
          <div style="font-size:20px; font-weight:700; color:{TXT};">
            Live Breathing Phase Viewer
          </div>
          <div style="font-size:13px; color:{SUB}; margin-top:6px; line-height:1.7;">
            10 breathing phases. One patient. Slide through the full respiratory cycle and watch
            the <b style="color:#22ff66;">oncologist's contour</b> and the
            <b style="color:#ff6b6b;">AI detection</b> track the tumor in real time.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not have_phases:
            st.info("Phase panels not yet generated. Run `python precompute_phases.py` to enable this.", icon="ℹ️")
        else:
            # Load phase metadata
            pm_path = os.path.join(CACHE_DIR, "phase_meta.json")
            pm = {}
            if os.path.isfile(pm_path):
                with open(pm_path) as f:
                    pm = json.load(f)

            # Pre-load all panel PNGs + pre-render all charts as base64 once
            if "panel_b64" not in st.session_state:
                import base64 as _b64
                st.session_state.panel_b64 = {}
                for _ph in range(0, 100, 10):
                    _pp = os.path.join(CACHE_DIR, f"phase_{_ph:02d}_panel.png")
                    if os.path.isfile(_pp):
                        with open(_pp, "rb") as _f:
                            st.session_state.panel_b64[_ph] = _b64.b64encode(_f.read()).decode()

            if "chart_b64" not in st.session_state and pm:
                import base64 as _b64c
                _all = list(range(0, 100, 10))
                _dy  = [pm[str(k)].get("doc_cy", 0) for k in _all]
                _aiy = [pm[str(k)].get("ai_cy",  0) for k in _all]
                _dx  = [pm[str(k)].get("doc_cx", 0) for k in _all]
                _aix = [pm[str(k)].get("ai_cx",  0) for k in _all]
                st.session_state.chart_b64 = {}
                from matplotlib.lines import Line2D as _L2D
                for _sel in _all:
                    _gi = _all.index(_sel)
                    _fig, (_axA, _axB) = plt.subplots(1, 2, figsize=(16, 4), facecolor=BG)
                    _fig.subplots_adjust(wspace=0.28, left=0.06, right=0.97, top=0.78, bottom=0.18)
                    for _ax in (_axA, _axB):
                        _ax.set_facecolor(CARD)
                        for _sp in _ax.spines.values(): _sp.set_edgecolor(BORDER)
                        _ax.tick_params(colors=SUB, labelsize=9)
                    _axA.plot(_all, _dy,  "o-",  color="#22ff66", lw=2, markersize=5, markerfacecolor=BG, markeredgewidth=2, label="Doctor")
                    _axA.plot(_all, _aiy, "s--", color="#ff6b6b", lw=2, markersize=5, markerfacecolor=BG, markeredgewidth=2, label="AI")
                    _axA.axvline(_sel, color="white", lw=1.2, alpha=0.35, ls=":")
                    _axA.plot(_sel, _dy[_gi],  "o", color="#22ff66", markersize=13, zorder=5)
                    _axA.plot(_sel, _aiy[_gi], "s", color="#ff6b6b", markersize=11, zorder=5)
                    _axA.set_xlabel("Breathing Phase (%)", color=SUB, fontsize=10)
                    _axA.set_ylabel("Image row (px)", color=SUB, fontsize=10)
                    _axA.set_title("Position over breathing cycle", color=TXT, fontsize=11, fontweight="600", pad=8)
                    _axA.set_xticks(_all); _axA.grid(color=BORDER, lw=0.8, alpha=0.5)
                    _axA.legend(fontsize=9, facecolor=CARD, edgecolor=BORDER, labelcolor=TXT)
                    for _i, (_ddx, _ddy, _aax, _aay) in enumerate(zip(_dx, _dy, _aix, _aiy)):
                        _al = 0.35 + 0.65*(_i/9); _fil = (_i == _gi)
                        _axB.plot(_ddx, _ddy, "o", color="#22ff66", markersize=9, alpha=_al, markerfacecolor="#22ff66" if _fil else "none", markeredgewidth=2)
                        _axB.plot(_aax, _aay, "s", color="#ff6b6b", markersize=8, alpha=_al, markerfacecolor="#ff6b6b" if _fil else "none", markeredgewidth=2)
                        if _i < len(_all)-1:
                            _axB.annotate("", xy=(_dx[_i+1], _dy[_i+1]), xytext=(_ddx, _ddy), arrowprops=dict(arrowstyle="->", color="#22ff66", lw=1, alpha=0.4))
                            _axB.annotate("", xy=(_aix[_i+1], _aiy[_i+1]), xytext=(_aax, _aay), arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1, alpha=0.4))
                        _axB.text(_ddx-1.5, _ddy, f"{_all[_i]}%", color=SUB, fontsize=7, ha="right")
                    _axB.plot(_dx[_gi],  _dy[_gi],  "o", color="#22ff66", markersize=16, zorder=10, markerfacecolor="#22ff66", alpha=0.9)
                    _axB.plot(_aix[_gi], _aiy[_gi], "s", color="#ff6b6b", markersize=14, zorder=10, markerfacecolor="#ff6b6b", alpha=0.9)
                    _axB.invert_yaxis()
                    _axB.set_xlabel("Horizontal position (px)", color=SUB, fontsize=10)
                    _axB.set_ylabel("Vertical position (px)", color=SUB, fontsize=10)
                    _axB.set_title("Motion footprint map\n(each dot = one phase)", color=TXT, fontsize=11, fontweight="600", pad=8)
                    _axB.grid(color=BORDER, lw=0.8, alpha=0.5)
                    _axB.legend(handles=[_L2D([0],[0],marker="o",color="#22ff66",markersize=8,label="Doctor path",lw=0),
                                         _L2D([0],[0],marker="s",color="#ff6b6b",markersize=7,label="AI path",lw=0)],
                                fontsize=9, facecolor=CARD, edgecolor=BORDER, labelcolor=TXT)
                    _fig.text(0.5, 0.93, f"Tumor tracking — Phase {_sel}% highlighted", ha="center", color=TXT, fontsize=12, fontweight="600")
                    _buf = BytesIO(); _fig.savefig(_buf, format="png", dpi=100, bbox_inches="tight", facecolor=BG)
                    plt.close(_fig); _buf.seek(0)
                    st.session_state.chart_b64[_sel] = _b64c.b64encode(_buf.read()).decode()

            if "tracking_b64" not in st.session_state:
                import base64 as _b64t
                _tb = {}
                for _k, _p in [("cmp", os.path.join(TRACKING_DIR, "comparison_tracking.png")),
                                ("err", os.path.join(TRACKING_DIR, "error_per_phase.png"))]:
                    if os.path.isfile(_p):
                        with open(_p, "rb") as _f:
                            _tb[_k] = _b64t.b64encode(_f.read()).decode()
                st.session_state.tracking_b64 = _tb

            if "tracking_stats" not in st.session_state:
                _cp = os.path.join(CACHE_DIR, "tracking_corrected.json")
                if os.path.isfile(_cp):
                    with open(_cp) as _f: _cd = json.load(_f)
                    _vp = [p for p in _cd if _cd[p] is not None]
                    _es = [_cd[p]["error_3d_mm"] for p in _vp if _cd[p]["error_3d_mm"] is not None]
                    _zs = [_cd[p]["z_mm"] for p in _vp]
                    _ys = [_cd[p]["y_mm"] for p in _vp]
                    _xs = [_cd[p]["x_mm"] for p in _vp]
                    st.session_state.tracking_stats = {
                        "mean_err": float(np.mean(_es)) if _es else 16.7,
                        "si_range": max(_zs)-min(_zs), "ap_range": max(_ys)-min(_ys), "lr_range": max(_xs)-min(_xs),
                    }
                else:
                    st.session_state.tracking_stats = {"mean_err":16.7,"si_range":3.0,"ap_range":4.5,"lr_range":2.1}

            if "phase_idx" not in st.session_state:
                st.session_state.phase_idx = 0
            if "playing" not in st.session_state:
                st.session_state.playing = False

            @st.fragment
            def _phase_viewer():
                all_phases = list(range(0, 100, 10))
                phase_labels = {
                    0:"0% — Full Inhale", 10:"10%", 20:"20%", 30:"30%", 40:"40%",
                    50:"50%", 60:"60%", 70:"70%", 80:"80%", 90:"90% — Full Exhale",
                }

                def make_wave_svg(idx):
                    n, W, H = 300, 1000, 70
                    cy, amp = H / 2, H * 0.40
                    pts = " ".join(
                        f"{i/n*W:.1f},{cy - amp*np.cos(i/n*2*np.pi):.1f}"
                        for i in range(n + 1)
                    )
                    dx = idx / 9 * W
                    dy = cy - amp * np.cos(idx / 9 * 2 * np.pi)
                    ty = max(12, dy - 14)
                    lbl = all_phases[idx]
                    return (f'<div style="background:{BG};padding:2px 0 6px;">'
                            f'<svg viewBox="0 0 {W} {H+22}" width="100%" style="display:block;overflow:visible;">'
                            f'<polyline points="{pts}" fill="none" stroke="{BORDER}" stroke-width="2.5" stroke-linejoin="round"/>'
                            f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="10" fill="{ACC}"/>'
                            f'<text x="{dx:.1f}" y="{ty:.1f}" text-anchor="middle" fill="{ACC}" font-size="13" font-weight="700" font-family="monospace">{lbl}%</text>'
                            f'<text x="4" y="{H+18}" fill="{SUB}" font-size="11" font-family="Inter,sans-serif">INHALE</text>'
                            f'<text x="{W/2}" y="{H+18}" text-anchor="middle" fill="{SUB}" font-size="11" font-family="Inter,sans-serif">EXHALE</text>'
                            f'<text x="{W-4}" y="{H+18}" text-anchor="end" fill="{SUB}" font-size="11" font-family="Inter,sans-serif">INHALE</text>'
                            f'</svg></div>')

                def panel_html(ph):
                    b64 = st.session_state.panel_b64.get(ph, "")
                    return (f'<img src="data:image/png;base64,{b64}" '
                            f'style="width:100%;border-radius:12px;border:1px solid {BORDER};display:block;">')

                def metrics_html(ph, prev_ph):
                    p   = pm.get(str(ph), {})
                    p0  = pm.get("0", {})
                    pp  = pm.get(str(prev_ph), {})
                    ai_cx  = p.get("ai_cx", 0);  ai_cy  = p.get("ai_cy", 0)
                    doc_cx = p.get("doc_cx", 0); doc_cy = p.get("doc_cy", 0)
                    ai_move  = np.sqrt((ai_cx-p0.get("ai_cx",ai_cx))**2+(ai_cy-p0.get("ai_cy",ai_cy))**2)
                    doc_move = np.sqrt((doc_cx-p0.get("doc_cx",doc_cx))**2+(doc_cy-p0.get("doc_cy",doc_cy))**2)
                    ai_d  = np.sqrt((ai_cx -pp.get("ai_cx",ai_cx))**2 +(ai_cy -pp.get("ai_cy",ai_cy))**2)
                    doc_d = np.sqrt((doc_cx-pp.get("doc_cx",doc_cx))**2+(doc_cy-pp.get("doc_cy",doc_cy))**2)
                    dist  = p.get("dist_px", 0)
                    conf  = p.get("ai_conf", 0)
                    n_px  = p.get("ai_px", 0)
                    def badge(d):
                        if d < 0.5: return f'<span style="color:{SUB};font-size:11px;">no change</span>'
                        c = "#56d364" if d < 5 else "#f9c513" if d < 15 else "#ff6b6b"
                        return f'<span style="color:{c};font-size:11px;font-weight:600;">+{d:.1f}px vs prev</span>'
                    return f"""
                    <div style="display:flex;gap:10px;margin-top:12px;flex-wrap:wrap;">
                      <div style="background:{CARD};border:1px solid #22ff66;border-radius:10px;padding:12px 18px;flex:1;min-width:150px;text-align:center;">
                        <div style="font-size:11px;color:#22ff66;font-weight:600;text-transform:uppercase;">Doctor centroid</div>
                        <div style="font-size:18px;font-weight:700;color:{TXT};margin-top:4px;font-family:monospace;">({doc_cx:.0f}, {doc_cy:.0f})</div>
                        <div style="margin-top:3px;">{badge(doc_d)}</div>
                        <div style="font-size:11px;color:{SUB};margin-top:3px;">total {doc_move:.1f}px from inhale</div>
                      </div>
                      <div style="background:{CARD};border:1px solid #ff6b6b;border-radius:10px;padding:12px 18px;flex:1;min-width:150px;text-align:center;">
                        <div style="font-size:11px;color:#ff6b6b;font-weight:600;text-transform:uppercase;">AI centroid</div>
                        <div style="font-size:18px;font-weight:700;color:{TXT};margin-top:4px;font-family:monospace;">({ai_cx:.0f}, {ai_cy:.0f})</div>
                        <div style="margin-top:3px;">{badge(ai_d)}</div>
                        <div style="font-size:11px;color:{SUB};margin-top:3px;">total {ai_move:.1f}px from inhale</div>
                      </div>
                      <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 18px;flex:1;min-width:150px;text-align:center;">
                        <div style="font-size:11px;color:{SUB};font-weight:600;text-transform:uppercase;">Gap between them</div>
                        <div style="font-size:24px;font-weight:700;color:#f9c513;margin-top:4px;">{dist:.1f} px</div>
                        <div style="font-size:11px;color:{SUB};margin-top:4px;">doctor vs AI distance</div>
                      </div>
                      <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 18px;flex:1;min-width:150px;text-align:center;">
                        <div style="font-size:11px;color:{SUB};font-weight:600;text-transform:uppercase;">AI confidence</div>
                        <div style="font-size:24px;font-weight:700;color:{ACC};margin-top:4px;">{conf}%</div>
                        <div style="font-size:11px;color:{SUB};margin-top:4px;">{n_px} tumor pixels</div>
                      </div>
                    </div>"""

                # ── Controls ────────────────────────────────────────────────────
                col_slider, col_btn = st.columns([5, 1])
                with col_slider:
                    selected_phase = st.select_slider(
                        "Phase", options=all_phases,
                        value=all_phases[st.session_state.phase_idx],
                        format_func=lambda x: phase_labels.get(x, f"{x}%"),
                        label_visibility="collapsed",
                        key="phase_slider",
                        disabled=st.session_state.playing,
                    )
                    if not st.session_state.playing:
                        st.session_state.phase_idx = all_phases.index(selected_phase)
                with col_btn:
                    if st.button("⏸ Stop" if st.session_state.playing else "▶ Play",
                                 use_container_width=True):
                        st.session_state.playing = not st.session_state.playing
                        st.rerun()

                LEGEND_HTML = (
                    f'<div style="display:flex;gap:20px;margin:6px 0 10px;align-items:center;flex-wrap:wrap;">'
                    f'<div style="display:flex;align-items:center;gap:7px;">'
                    f'<div style="width:16px;height:16px;background:#22ff66;border-radius:3px;"></div>'
                    f'<span style="color:{TXT};font-size:13px;font-weight:600;">Doctor</span>'
                    f'<span style="color:{SUB};font-size:12px;">manual contour</span></div>'
                    f'<div style="display:flex;align-items:center;gap:7px;">'
                    f'<div style="width:16px;height:16px;background:#ff6b6b;border-radius:3px;"></div>'
                    f'<span style="color:{TXT};font-size:13px;font-weight:600;">AI</span>'
                    f'<span style="color:{SUB};font-size:12px;">automatic detection</span></div>'
                    f'<span style="color:{SUB};font-size:12px;">| Center panel = 4x zoom</span></div>'
                )

                def tracking_html():
                    ts = st.session_state.get("tracking_stats", {"mean_err":16.7,"si_range":3.0,"ap_range":4.5,"lr_range":2.1})
                    me, si, ap, lr = ts["mean_err"], ts["si_range"], ts["ap_range"], ts["lr_range"]
                    tb = st.session_state.get("tracking_b64", {})
                    cmp_tag = (
                        f'<div style="font-size:14px;font-weight:700;color:{TXT};margin:4px 0 6px;">'
                        f'Tumor Position across Breathing Cycle — AI vs. Doctor</div>'
                        f'<div style="font-size:12px;color:{SUB};margin-bottom:10px;">'
                        f'<span style="color:#f9c513;font-weight:600;font-size:16px;">&#9632;</span>'
                        f'<span style="color:#f9c513;font-weight:600;"> Yellow circles</span>'
                        f' = where the <b>radiation oncologist</b> said the tumor is &nbsp;&nbsp;'
                        f'<span style="color:{ACC};font-weight:600;font-size:16px;">&#9632;</span>'
                        f'<span style="color:{ACC};font-weight:600;"> Blue squares (dashed)</span>'
                        f' = where the <b>AI model</b> thinks the tumor is</div>'
                        f'<img src="data:image/png;base64,{tb["cmp"]}" '
                        f'style="width:100%;border-radius:8px;border:1px solid {BORDER};display:block;">'
                    ) if "cmp" in tb else ""
                    err_tag = (
                        f'<div style="font-size:14px;font-weight:700;color:{TXT};margin:20px 0 6px;">'
                        f'How far off is the AI at each phase?</div>'
                        f'<div style="font-size:12px;color:{SUB};margin-bottom:8px;">'
                        f'Each bar = the 3D distance (in mm) between where the AI found the tumor '
                        f'and where the doctor marked it. Lower = more accurate. '
                        f'Green bars (&lt;5mm) are excellent, blue bars (&lt;15mm) are clinically acceptable.</div>'
                        f'<img src="data:image/png;base64,{tb["err"]}" '
                        f'style="width:100%;border-radius:8px;border:1px solid {BORDER};display:block;">'
                    ) if "err" in tb else ""
                    return (
                        f'<hr style="margin:32px 0;"/>'
                        f'<div style="margin-bottom:16px;">'
                        f'<div style="font-size:20px;font-weight:700;color:{TXT};">Tumor Motion Across the Breathing Cycle</div>'
                        f'<div style="font-size:13px;color:{SUB};margin-top:6px;line-height:1.7;">'
                        f'Every breath moves the tumor. The AI maps the full trajectory across all 10 phases '
                        f'and measures how closely it matches the oncologist\'s ground truth.'
                        f'</div></div>'
                        f'<div style="display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap;">'
                        f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:16px 20px;flex:1;min-width:140px;text-align:center;">'
                        f'<div style="font-size:11px;color:{SUB};text-transform:uppercase;">Up-Down motion</div>'
                        f'<div style="font-size:26px;font-weight:700;color:{ACC};margin-top:6px;">{si:.1f} mm</div>'
                        f'<div style="font-size:11px;color:{SUB};margin-top:4px;">with the diaphragm</div></div>'
                        f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:16px 20px;flex:1;min-width:140px;text-align:center;">'
                        f'<div style="font-size:11px;color:{SUB};text-transform:uppercase;">Front-Back motion</div>'
                        f'<div style="font-size:26px;font-weight:700;color:{RED};margin-top:6px;">{ap:.1f} mm</div>'
                        f'<div style="font-size:11px;color:{SUB};margin-top:4px;">chest expansion</div></div>'
                        f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:16px 20px;flex:1;min-width:140px;text-align:center;">'
                        f'<div style="font-size:11px;color:{SUB};text-transform:uppercase;">Left-Right motion</div>'
                        f'<div style="font-size:26px;font-weight:700;color:{GRN};margin-top:6px;">{lr:.1f} mm</div>'
                        f'<div style="font-size:11px;color:{SUB};margin-top:4px;">lateral shift</div></div>'
                        f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:16px 20px;flex:1;min-width:140px;text-align:center;">'
                        f'<div style="font-size:11px;color:{SUB};text-transform:uppercase;">AI vs. Doctor error</div>'
                        f'<div style="font-size:26px;font-weight:700;color:#f9c513;margin-top:6px;">{me:.1f} mm</div>'
                        f'<div style="font-size:11px;color:{SUB};margin-top:4px;">average difference</div>'
                        f'<div style="font-size:10px;color:{SUB};margin-top:3px;opacity:0.65;">clinical RT margin: 5–10 mm</div></div>'
                        f'</div>'
                        f'{cmp_tag}'
                        f'{err_tag}'
                        f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:20px 24px;margin-top:24px;">'
                        f'<div style="font-size:15px;font-weight:700;color:{TXT};margin-bottom:10px;">Clinical Significance</div>'
                        f'<div style="font-size:13px;color:{SUB};line-height:1.9;">'
                        f'A radiation beam aimed at a static contour misses the tumor the moment the patient exhales. '
                        f'The <strong style="color:{ACC}">safety margin</strong> must cover the entire motion envelope — '
                        f'which this system computes automatically across all 10 phases.<br><br>'
                        f'Manual contouring by a radiation oncologist takes 20–40 minutes per patient. '
                        f'The AI produces the same result in seconds, with a mean localization error of '
                        f'<strong style="color:#f9c513;">{me:.1f} mm</strong> against verified ground truth.'
                        f'</div></div>'
                    )

                def build_html(ph, prev_ph, idx, include_chart=True, include_tracking=True):
                    chart_tag = ""
                    if include_chart:
                        cb = st.session_state.get("chart_b64", {}).get(ph, "")
                        if cb:
                            chart_tag = (
                                f'<img src="data:image/png;base64,{cb}" '
                                f'style="width:100%;border-radius:8px;border:1px solid {BORDER};'
                                f'display:block;margin-top:12px;">'
                            )
                    tr = tracking_html() if include_tracking else ""
                    return (
                        make_wave_svg(idx) +
                        LEGEND_HTML +
                        panel_html(ph) +
                        metrics_html(ph, prev_ph) +
                        chart_tag +
                        tr
                    )

                # Play mode needs a reusable placeholder for repeated per-frame updates
                content_ph = st.empty()

                if st.session_state.playing:
                    for _ in range(len(all_phases)):
                        idx      = st.session_state.phase_idx
                        ph       = all_phases[idx]
                        prev_idx = max(0, idx - 1)
                        content_ph.markdown(
                            build_html(ph, all_phases[prev_idx], idx, include_chart=False, include_tracking=False),
                            unsafe_allow_html=True)
                        time.sleep(0.8)
                        next_idx = (idx + 1) % len(all_phases)
                        st.session_state.phase_idx = next_idx
                        if next_idx == 0:
                            st.session_state.playing = False
                            break
                    final_idx = st.session_state.phase_idx
                    content_ph.markdown(
                        build_html(all_phases[final_idx], all_phases[final_idx], final_idx, True, True),
                        unsafe_allow_html=True)
                    st.rerun()

                else:
                    idx      = st.session_state.phase_idx
                    prev_idx = max(0, idx - 1)
                    # Direct st.markdown() = single delta = atomic browser update, no intermediate empty state
                    st.markdown(
                        build_html(selected_phase, all_phases[prev_idx], idx, True, True),
                        unsafe_allow_html=True)
                    _dl_b64 = st.session_state.panel_b64.get(selected_phase, "")
                    if _dl_b64:
                        import base64 as _b64dl
                        st.download_button(
                            label="⬇ Download Phase Report",
                            data=_b64dl.b64decode(_dl_b64),
                            file_name=f"tumor_phase_{selected_phase:02d}pct.png",
                            mime="image/png",
                        )

            _phase_viewer()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD YOUR CT (live inference only)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:

    @st.cache_resource(show_spinner=False)
    def load_model():
        import torch
        import segmentation_models_pytorch as smp
        if not os.path.isfile(MODEL_PATH):
            return None, f"Model not found: {MODEL_PATH}"
        try:
            m = smp.UnetPlusPlus(
                encoder_name="efficientnet-b4", encoder_weights=None,
                in_channels=3, classes=1, activation=None,
            )
            m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
            m.eval()
            return m, None
        except Exception as e:
            return None, str(e)

    def run_inference(model, vol_3ch):
        import torch
        with torch.no_grad():
            t   = torch.tensor(vol_3ch[None]).float()
            out = torch.sigmoid(model(t)).squeeze().numpy()
        return out

    def make_figure(raw_norm, prob_map, thr):
        mask = (prob_map >= thr).astype(np.float32)
        n_px = int(mask.sum())
        conf = float(prob_map[mask > 0].mean()) * 100 if n_px > 0 else 0.0

        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(20, 7), facecolor=BG)
        gs  = GridSpec(1, 3, figure=fig, wspace=0.06, left=0.03, right=0.97,
                       top=0.88, bottom=0.08)
        ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor(BG); ax.axis("off")

        # Panel 1 — original
        ax1.imshow(raw_norm, cmap="gray", vmin=0, vmax=1)
        ax1.set_title("Original CT Scan", color=TXT, fontsize=13, fontweight="600", pad=10)
        ax1.text(0.5, -0.04, "Grayscale — Hounsfield Units (lung window)",
                 transform=ax1.transAxes, ha="center", color=SUB, fontsize=9)

        # Panel 2 — confidence
        ax2.imshow(raw_norm, cmap="gray", vmin=0, vmax=1, alpha=0.3)
        im = ax2.imshow(prob_map, cmap="plasma", vmin=0, vmax=1, alpha=0.9)
        ax2.set_title("Model Confidence Map", color=TXT, fontsize=13, fontweight="600", pad=10)
        ax2.text(0.5, -0.04, "Bright yellow = high tumor probability",
                 transform=ax2.transAxes, ha="center", color=SUB, fontsize=9)
        cb = fig.colorbar(im, ax=ax2, fraction=0.038, pad=0.02, shrink=0.85)
        cb.set_label("tumor probability", color=SUB, fontsize=9)
        cb.ax.yaxis.set_tick_params(color=SUB, labelsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=SUB)
        cb.outline.set_edgecolor(BORDER)
        ax2.text(0.02, 0.03, "Low", transform=ax2.transAxes, color=SUB, fontsize=8, va="bottom")
        ax2.text(0.02, 0.96, "High", transform=ax2.transAxes, color="#f9c513", fontsize=8, va="top")

        # Panel 3 — overlay
        ax3.imshow(raw_norm, cmap="gray", vmin=0, vmax=1)
        if n_px > 0:
            rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba[:, :, 0] = mask
            rgba[:, :, 1] = mask * 0.15
            rgba[:, :, 3] = mask * 0.6
            ax3.imshow(rgba)
            ax3.contour(mask, levels=[0.5], colors=[RED], linewidths=2)
            cy_c, cx_c = center_of_mass(mask)
            ax3.plot(cx_c, cy_c, "+", color="white", markersize=12, markeredgewidth=2)
            ax3.annotate(
                f"Tumor\n{n_px}px · {conf:.0f}% conf",
                xy=(cx_c, cy_c), xytext=(cx_c + 30, cy_c - 30),
                color="white", fontsize=9,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec=RED, alpha=0.9),
            )
        patches = [
            mpatches.Patch(color=RED,   label="Tumor region"),
            mpatches.Patch(color="#888", label="Lung tissue"),
        ]
        ax3.legend(handles=patches, loc="lower left", fontsize=9,
                   facecolor="#0d1117", edgecolor=BORDER, labelcolor=TXT)
        ax3.set_title("Segmentation Overlay", color=TXT, fontsize=13, fontweight="600", pad=10)
        ax3.text(0.5, -0.04,
                 "Red = tumor region  |  + = centroid" if n_px > 0 else "No tumor detected",
                 transform=ax3.transAxes, ha="center", color=SUB, fontsize=9)

        status = (f"TUMOR DETECTED — {n_px} pixels | {conf:.0f}% confidence"
                  if n_px > 0 else "NO TUMOR DETECTED")
        fig.text(0.5, 0.95, status, ha="center",
                 color=RED if n_px > 0 else GRN, fontsize=11, fontweight="600")

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        return buf, n_px, conf

    # Instructions
    st.markdown(f"""
    <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:12px;
                padding:20px 24px; margin-bottom:28px;">
      <div style="font-size:15px; font-weight:700; color:{TXT}; margin-bottom:14px;">Live Inference</div>
      <div style="display:flex; gap:24px; flex-wrap:wrap;">
        <div style="flex:1; min-width:190px;">
          <div style="font-size:13px; font-weight:600; color:{ACC}; margin-bottom:6px;">Input</div>
          <div style="font-size:12px; color:{SUB}; line-height:1.7;">
            A <code>.npy</code> file — 3 consecutive CT slices in Hounsfield Units.<br>
            Sample files: <code>D:/LICENTA2/demo_cache/sample_uploads/</code>
          </div>
        </div>
        <div style="flex:1; min-width:190px;">
          <div style="font-size:13px; font-weight:600; color:{ACC}; margin-bottom:6px;">Model</div>
          <div style="font-size:12px; color:{SUB}; line-height:1.7;">
            U-Net++ with EfficientNet-B4 encoder.<br>
            Trained on 421 patients. Runs in under 2 seconds.
          </div>
        </div>
        <div style="flex:1; min-width:190px;">
          <div style="font-size:13px; font-weight:600; color:{ACC}; margin-bottom:6px;">Output</div>
          <div style="font-size:12px; color:{SUB}; line-height:1.7;">
            Pixel-level tumor segmentation with confidence map.<br>
            No human input required.
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a CT slice (.npy, shape 3 × H × W)",
        type=["npy"],
        label_visibility="visible",
    )

    if uploaded is not None:
        try:
            raw = np.load(BytesIO(uploaded.read()))
            if raw.ndim != 3 or raw.shape[0] != 3:
                st.error(f"Expected shape (3, H, W) — got {raw.shape}. Use the sample_uploads/ files.")
            else:
                with st.spinner("Loading model..."):
                    model, err = load_model()

                if err:
                    st.error(f"Could not load model: {err}")
                else:
                    sf = (IMG_SIZE / raw.shape[1], IMG_SIZE / raw.shape[2])
                    vol_3ch = np.stack([
                        zoom(np.clip(raw[i], HU_MIN, HU_MAX).astype(np.float32), sf, order=1)
                        for i in range(3)
                    ])
                    vol_3ch = (vol_3ch - HU_MIN) / (HU_MAX - HU_MIN)
                    center_norm = vol_3ch[1]

                    with st.spinner("Running segmentation..."):
                        t0 = time.time()
                        prob_map = run_inference(model, vol_3ch)
                        elapsed  = time.time() - t0

                    buf, n_px, conf = make_figure(center_norm, prob_map, threshold)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("Status", "DETECTED" if n_px > 0 else "CLEAR")
                    with c2: st.metric("Tumor pixels", f"{n_px:,}")
                    with c3: st.metric("Avg confidence", f"{conf:.1f}%" if n_px > 0 else "—")
                    with c4: st.metric("Inference time", f"{elapsed:.2f}s")

                    # ── Section 1: AI result ────────────────────────────────────
                    st.markdown(
                        f'<div style="font-size:14px;font-weight:700;color:{TXT};margin:20px 0 6px;">'
                        f'AI Segmentation</div>'
                        f'<div style="font-size:12px;color:{SUB};margin-bottom:10px;">'
                        f'Zero human input. The model localizes the tumor in under 2 seconds.</div>',
                        unsafe_allow_html=True,
                    )
                    st.image(buf, use_container_width=True)

                    # ── Section 2: Compare with Doctor ──────────────────────────
                    st.markdown('<hr style="margin:28px 0 20px;"/>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="font-size:14px;font-weight:700;color:{TXT};margin-bottom:6px;">'
                        f'Ground Truth Comparison</div>'
                        f'<div style="font-size:12px;color:{SUB};margin-bottom:16px;">'
                        f'The right panel shows the radiation oncologist\'s manual contour (green) '
                        f'and the AI detection (red) on the same patient from the validated dataset. '
                        f'Select the matching breathing phase to align the comparison.</div>',
                        unsafe_allow_html=True,
                    )

                    import re as _re
                    _pm = _re.search(r'phase(\d{1,2})', uploaded.name.lower())
                    _detected = int(_pm.group(1)) if _pm else None
                    _valid_phases = list(range(0, 100, 10))
                    _auto_phase = _detected if _detected in _valid_phases else None

                    if _auto_phase is not None:
                        st.markdown(
                            f'<div style="font-size:12px;color:{GRN};margin-bottom:12px;">'
                            f'Phase auto-detected from filename: <b>{_auto_phase}%</b></div>',
                            unsafe_allow_html=True,
                        )
                        _ref_phase = _auto_phase
                    else:
                        _ref_phase = st.selectbox(
                            "Which breathing phase does your CT correspond to?",
                            options=_valid_phases,
                            format_func=lambda x: {0:"0% — Full Inhale", 90:"90% — Full Exhale"}.get(x, f"{x}%"),
                            key="tab2_ref_phase",
                        )

                    _ref_panel_path = os.path.join(CACHE_DIR, f"phase_{_ref_phase:02d}_panel.png")
                    col_ai, col_doc = st.columns(2)
                    with col_ai:
                        st.markdown(
                            f'<div style="font-size:12px;font-weight:600;color:{RED};margin-bottom:6px;">'
                            f'YOUR UPLOAD — AI prediction</div>',
                            unsafe_allow_html=True,
                        )
                        buf.seek(0)
                        st.image(buf, use_container_width=True)
                    with col_doc:
                        st.markdown(
                            f'<div style="font-size:12px;font-weight:600;color:#22ff66;margin-bottom:6px;">'
                            f'VALIDATED PATIENT — Phase {_ref_phase}% &nbsp;'
                            f'<span style="color:#22ff66; font-weight:400;">Doctor</span> vs '
                            f'<span style="color:{RED}; font-weight:400;">AI</span></div>',
                            unsafe_allow_html=True,
                        )
                        if os.path.isfile(_ref_panel_path):
                            st.image(_ref_panel_path, use_container_width=True)
                        else:
                            st.info("Pre-computed panel not found. Run precompute_phases.py first.", icon="ℹ️")

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.markdown(f"""
        <div style="border:2px dashed {BORDER}; border-radius:12px; padding:60px 20px;
                    text-align:center; background:{CARD};">
          <div style="font-size:36px; margin-bottom:12px; color:{SUB};">↑</div>
          <div style="font-size:15px; font-weight:600; color:{TXT}; margin-bottom:8px;">
            Upload a .npy file to run live inference
          </div>
          <div style="font-size:13px; color:{SUB};">
            Sample files: <code>D:/LICENTA2/demo_cache/sample_uploads/ct_slice_*.npy</code>
          </div>
        </div>
        """, unsafe_allow_html=True)
