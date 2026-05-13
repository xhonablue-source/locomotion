"""
CognitiveCloud.ai — Human Locomotion Profiler
Shoulder Pivot · Bicep Flexion · Fist Clasp
Human ↔ Machine Synergy | Sensor | Calibration | XP | AI Robot Drawing
Pedagogy: Crusader Vision — attempt-tracked quizzes, matching, reflection, AI art prompt
"""

import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from datetime import datetime
import anthropic
import io, time

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Locomotion Profiler · CognitiveCloud.ai",
    page_icon="🦾", layout="wide",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STYLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Base reset — force dark text on all elements on all devices ── */
html, body { background:#f5f5f3 !important; color:#111827 !important; }
[class*="css"], .main, .block-container,
section[data-testid="stSidebar"],
div[data-testid="stAppViewContainer"] {
  background:#f5f5f3 !important;
  color:#111827 !important;
}

/* ── Force ALL text elements dark ── */
p, span, div, li, label, small, caption,
h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown span, .stMarkdown div,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
  color: #111827 !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Streamlit widget labels ── */
label, .stSelectbox label, .stRadio label,
.stNumberInput label, .stTextInput label,
.stTextArea label, .stToggle label {
  color: #111827 !important;
  font-weight: 600 !important;
}

/* ── Input fields ── */
input, textarea, select,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
  color: #111827 !important;
  background: #ffffff !important;
  border: 1.5px solid #d1d5db !important;
}

/* ── Info / success / warning boxes ── */
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
.stAlert p { color: #111827 !important; }

/* ── Code blocks ── */
code, pre { color: #1e40af !important; background:#eff6ff !important; }

/* ── Dataframe text ── */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th { color: #111827 !important; }

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
  background: #f0ede8 !important;
  border-right: 1px solid #d1d5db;
}
div[data-testid="stSidebar"] p,
div[data-testid="stSidebar"] span,
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] div { color: #111827 !important; }

/* ── Expander ── */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] p { color: #111827 !important; }

.main,.block-container{ background:#f5f5f3 !important; }
.dev-credit{background:linear-gradient(135deg,#0f3460 0%,#16213e 100%);
  padding:20px;border-radius:10px;margin-bottom:20px;box-shadow:0 4px 6px rgba(0,0,0,.1);}
.dev-credit h2{margin:0;color:#ffffff !important;font-size:1.3rem;}
.dev-credit p{margin:5px 0;color:rgba(255,255,255,.9) !important;font-size:.9rem;}
.dev-credit span{color:rgba(255,255,255,.9) !important;}
.dev-credit div{color:rgba(255,255,255,.9) !important;}
.dev-credit hr{border:1px solid rgba(255,255,255,.3);margin:10px 0;}
.dev-credit a{color:#00d4ff !important;text-decoration:underline;}
.pill{display:inline-block;font-size:.62rem;font-weight:700;letter-spacing:.14em;
  text-transform:uppercase;padding:3px 12px;border-radius:20px;margin:22px 0 14px;color:#fff;}
.p-s{background:#7c3aed;}.p-b{background:#0891b2;}.p-f{background:#059669;}
.p-x{background:#d97706;}.p-y{background:#e11d48;}.p-z{background:#374151;}
.p-g{background:linear-gradient(90deg,#f59e0b,#ef4444);}
.p-ai{background:linear-gradient(90deg,#7c3aed,#0891b2);}
.mcard{background:#ffffff;border:2px solid #d1d5db;border-radius:12px;padding:16px 20px;margin-bottom:10px;}
.mcard-t{font-size:.7rem;color:#374151 !important;font-weight:600;margin:0 0 2px;}
.mcard-v{font-size:1.9rem;font-weight:700;font-family:'DM Mono',monospace;color:#111827 !important;}
.mcard-u{font-size:.72rem;color:#6b7280 !important;}
.xp-chip{display:inline-block;background:#fef3c7;border:1px solid #fbbf24;
  color:#92400e;font-size:.65rem;font-weight:700;padding:2px 8px;border-radius:99px;margin-top:4px;}
.syn-box{border-radius:12px;padding:20px 14px;text-align:center;margin-bottom:10px;}
.syn-n{font-size:2.8rem;font-weight:700;font-family:'DM Mono',monospace;line-height:1;}
.syn-l{font-size:.6rem;letter-spacing:.16em;text-transform:uppercase;color:#374151 !important;margin-top:4px;font-weight:600;}
.irow{display:flex;justify-content:space-between;align-items:center;
  padding:9px 13px;background:#e8ecf0;border-radius:8px;margin-bottom:5px;border:1px solid #d1d5db;}
.ik{font-size:.76rem;color:#374151 !important;font-weight:600;}
.iv{font-size:.8rem;font-family:'DM Mono',monospace;font-weight:700;color:#111827 !important;}
.ok{color:#059669;}.warn{color:#d97706;}.bad{color:#e11d48;}
.xp-bar-wrap{background:#e2e8f0;border-radius:99px;height:14px;width:100%;margin:6px 0 2px;}
.xp-bar{height:14px;border-radius:99px;background:linear-gradient(90deg,#f59e0b,#ef4444);}
.xp-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;}
.xp-label{font-size:.72rem;font-weight:700;color:#111827 !important;}
.xp-num{font-family:'DM Mono',monospace;font-size:.8rem;font-weight:700;color:#ef4444 !important;}
.badge-row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;}
.badge{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;
  border-radius:20px;font-size:.72rem;font-weight:600;}
.badge-earned{background:#f0fdf4;border:1px solid #86efac;color:#15803d;}
.badge-locked{background:#f8fafc;border:1px solid #e2e8f0;color:#94a3b8;}
.celebrate-box{background:linear-gradient(135deg,#fef3c7,#fde68a 50%,#fed7aa);
  border:2px solid #f59e0b;border-radius:16px;padding:28px 32px;text-align:center;margin:20px 0;}
.celebrate-title{font-size:1.6rem;font-weight:700;color:#111827 !important;margin-bottom:6px;}
.celebrate-sub{font-size:.9rem;color:#92400e !important;}
.drawing-prompt{background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
  border:2px solid #0891b2;border-radius:14px;padding:24px 28px;margin:16px 0;}
.drawing-prompt h3{color:#0369a1 !important;margin:0 0 10px;}
.drawing-prompt p{color:#0c4a6e !important;font-size:.9rem;line-height:1.7;}
.drawing-prompt strong{color:#0c4a6e !important;}
.robot-desc{background:#ffffff;border-left:4px solid #7c3aed;
  border-radius:0 12px 12px 0;padding:20px 24px;margin:16px 0;font-size:.9rem;line-height:1.8;
  color:#111827 !important;}
div[data-testid="stSidebar"]{background:#f8fafc;border-right:1px solid #e2e8f0;}
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MACHINE OPTIMAL ENVELOPES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPT = {
    "shoulder_abduction" : (180, 90,  15),
    "shoulder_flexion"   : (180, 90,  15),
    "shoulder_int_rot"   : ( 90, 60,  10),
    "shoulder_ext_rot"   : ( 90, 60,  10),
    "elbow_flexion"      : (145, 90,  12),
    "forearm_supination" : ( 90, 75,  10),
    "forearm_pronation"  : ( 90, 75,  10),
    "wrist_flexion"      : ( 80, 50,  10),
    "wrist_extension"    : ( 70, 45,  10),
    "arm_flexion"        : (145, 90,  12),
    "arm_extension"      : ( 30, 15,  8),
    "grip_closure"       : (100, 80,  10),
    "finger_flexion"     : ( 90, 70,  10),
}
LABELS = {
    "shoulder_abduction":"Shoulder Abduction","shoulder_flexion":"Shoulder Flexion",
    "shoulder_int_rot":"Internal Rotation","shoulder_ext_rot":"External Rotation",
    "elbow_flexion":"Elbow Flexion","forearm_supination":"Forearm Supination",
    "forearm_pronation":"Forearm Pronation","wrist_flexion":"Wrist Flexion",
    "wrist_extension":"Wrist Extension","grip_closure":"Grip Closure","finger_flexion":"Finger Flexion",
    "arm_flexion":"Arm Flexion","arm_extension":"Arm Extension",
}
UNITS = {k: "%" if k == "grip_closure" else "°" for k in OPT}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# XP & LEVELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEVELS = [
    (0,"Rookie Pilot","🪖"),(100,"Apprentice Operator","🔧"),
    (250,"Certified Operator","⚙️"),(450,"Skilled Navigator","🧭"),
    (700,"Advanced Pilot","🚀"),(1000,"Elite Biomech","🦾"),(1400,"Master Operator","🏆"),
]
BADGES = [
    ("shoulder_ace","Shoulder Ace","💜","All shoulder synergy ≥ 80%"),
    ("bicep_beast","Bicep Beast","💙","All bicep/forearm synergy ≥ 80%"),
    ("iron_grip","Iron Grip","💚","Both grip scores ≥ 80%"),
    ("perfect_run","Perfect Run","⭐","Overall synergy ≥ 85%"),
    ("full_coverage","Full Coverage","🎯","All 11 axes above tolerance"),
    ("sensor_smooth","Sensor Smooth","📡","All 3 sensor RMSE < 8"),
    ("calibrated","Machine Ready","🤖","All control signals > 0.7"),
    ("first_session","First Session","🌱","Complete your first profile"),
]

def xp_for_score(s):
    return 20 if s>=90 else 15 if s>=80 else 10 if s>=65 else 5 if s>=50 else 2

def get_level(xp):
    li=0
    for i,(t,n,ic) in enumerate(LEVELS):
        if xp>=t: li=i
    return li, LEVELS[li]

def xp_to_next(xp):
    for i,(t,n,ic) in enumerate(LEVELS):
        if xp<t:
            prev=LEVELS[i-1][0] if i>0 else 0
            return t, xp-prev, t-prev
    return None,0,1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE LOCOMOTION FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def synergy_score(h,o,hm):
    if hm==0: return 0.0
    return round(max(0.0,1.0-abs(h/hm-o/hm))*100,1)

def utilisation(h,hm):
    if hm==0: return 0.0
    return round(min(h/hm*100,100.0),1)

def machine_coverage(h,mo):
    if mo==0: return 0.0
    return round(min(h/mo*100,100.0),1)

def deviation(h,o):
    return round(abs(h-o),1)

def normalise_to_machine(h,hm,mo):
    if hm==0: return 0.0
    return round((h/hm)*mo,2)

def sensor_rmse(series,opt):
    return round(float(np.sqrt(np.mean((np.array(series)-opt)**2))),2)

def sensor_lag(series,opt,dt=0.05):
    sig=np.array(series)-np.mean(series)
    ref=np.full(len(series),float(opt))-float(opt)
    corr=np.correlate(sig,ref+0.001,mode='full')
    return round((int(np.argmax(corr))-(len(series)-1))*dt,3)

def deviation_status(dev,tol):
    if dev<=tol*0.5: return "✅ Optimal","ok"
    if dev<=tol:     return "⚠️ Acceptable","warn"
    return                  "🔴 Recalibrate","bad"

def score_color(s):
    return "#059669" if s>=80 else "#d97706" if s>=55 else "#e11d48"

def score_bg(s):
    return "#f0fdf4" if s>=80 else "#fffbeb" if s>=55 else "#fff1f2"

def hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


def simulate_sensor(base,hmax,n=120,noise=2.5,drift=0.04):
    t=np.linspace(0,6,n)
    sig=(base+noise*np.sin(2*np.pi*0.4*t)+np.random.normal(0,noise*0.35,n)-drift*base*t/6)
    return np.clip(sig,0,hmax),t

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPHIC DIAGRAM FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def shoulder_diagram(abd, flex, ir, er):
    """Top-down shoulder pivot diagram with four arcs."""
    fig, ax = plt.subplots(figsize=(5,5), facecolor='#f8fafc')
    ax.set_facecolor('#f8fafc')
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("Shoulder Pivot — Top-Down View", fontsize=10,
                 fontweight='bold', color='#374151', pad=10)

    # Joint circle
    joint = plt.Circle((0,0), 0.22, color='#1a1a2e', zorder=5)
    ax.add_patch(joint)

    def draw_arc(angle_deg, start_angle, color, label, r=1.5):
        if angle_deg <= 0: return
        end_a = start_angle + angle_deg
        theta = np.linspace(np.radians(start_angle), np.radians(end_a), 60)
        x = r * np.cos(theta); y = r * np.sin(theta)
        ax.fill_between(x, y, alpha=0.18, color=color)
        ax.plot(x, y, color=color, linewidth=2.5)
        ax.plot([0, x[0]], [0, y[0]], color=color, linewidth=1.5, linestyle='--', alpha=0.5)
        ax.plot([0, x[-1]], [0, y[-1]], color=color, linewidth=2)
        mid = len(theta)//2
        ax.annotate(f"{label}\n{angle_deg}°",
            xy=(x[mid]*0.72, y[mid]*0.72), fontsize=7.5, ha='center', va='center',
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.85))

    draw_arc(abd,  90,  '#7c3aed', 'Abduction', 1.7)
    draw_arc(flex, 0,   '#0891b2', 'Flexion',   1.4)
    draw_arc(ir,   200, '#059669', 'Int Rot',   1.1)
    draw_arc(er,   270, '#d97706', 'Ext Rot',   1.1)

    ax.text(0, -2.05, "← Posterior   Anterior →",
            ha='center', va='bottom', fontsize=7, color='#94a3b8')
    fig.tight_layout()
    return fig

def arm_diagram(elbow, sup, pro, wf, we):
    """Side-view arm showing elbow flexion and forearm rotation."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='#f8fafc')
    fig.suptitle("Bicep & Forearm — Side View", fontsize=10,
                 fontweight='bold', color='#374151')

    # Left panel — elbow flexion arc
    ax = axes[0]; ax.set_facecolor('#f8fafc')
    ax.set_xlim(-0.3, 2.5); ax.set_ylim(-0.3, 2.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"Elbow Flexion: {elbow}°", fontsize=9, color='#0891b2')

    # Upper arm
    ax.plot([0, 0], [0, 1.8], color='#475569', linewidth=12, solid_capstyle='round')
    # Elbow joint
    elbow_circle = plt.Circle((0, 0), 0.15, color='#1a1a2e', zorder=5)
    ax.add_patch(elbow_circle)
    # Forearm at angle
    angle_rad = np.radians(180 - elbow)
    fa_x = 1.5 * np.cos(angle_rad); fa_y = 1.5 * np.sin(angle_rad)
    ax.plot([0, fa_x], [0, fa_y], color='#0891b2', linewidth=10, solid_capstyle='round')
    # Arc
    theta = np.linspace(np.radians(90), np.radians(180-elbow), 40)
    ax.plot(np.cos(theta)*0.9, np.sin(theta)*0.9,
            color='#0891b2', linewidth=2, linestyle='--', alpha=0.7)
    ax.annotate(f"{elbow}°", xy=(0.45, 0.55), fontsize=9, color='#0891b2', fontweight='bold')
    # Machine optimal line
    opt_rad = np.radians(180 - 90)
    ax.plot([0, 1.3*np.cos(opt_rad)], [0, 1.3*np.sin(opt_rad)],
            color='#f97316', linewidth=1.5, linestyle=':', alpha=0.8)
    ax.text(1.35*np.cos(opt_rad), 1.35*np.sin(opt_rad), 'Opt',
            fontsize=7, color='#f97316', ha='center')

    # Right panel — forearm rotation gauge
    ax2 = axes[1]; ax2.set_facecolor('#f8fafc')
    ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title(f"Sup {sup}°  |  Pro {pro}°", fontsize=9, color='#059669')

    bg = plt.Circle((0,0), 1.1, color='#e2e8f0', zorder=0)
    ax2.add_patch(bg)
    # Supination arc (left, green)
    ts = np.linspace(np.radians(90), np.radians(90+sup), 40)
    ax2.fill_between(np.cos(ts)*1.0, np.sin(ts)*1.0, alpha=0.2, color='#059669')
    ax2.plot(np.cos(ts)*1.0, np.sin(ts)*1.0, color='#059669', linewidth=2.5)
    # Pronation arc (right, purple)
    tp = np.linspace(np.radians(90-pro), np.radians(90), 40)
    ax2.fill_between(np.cos(tp)*1.0, np.sin(tp)*1.0, alpha=0.2, color='#7c3aed')
    ax2.plot(np.cos(tp)*1.0, np.sin(tp)*1.0, color='#7c3aed', linewidth=2.5)

    ax2.plot([0,0],[0,1.1], color='#1a1a2e', linewidth=3)
    ax2.text(-0.55, 0.35, f'SUP\n{sup}°', fontsize=8, color='#059669',
             ha='center', fontweight='bold')
    ax2.text(0.55, 0.35, f'PRO\n{pro}°', fontsize=8, color='#7c3aed',
             ha='center', fontweight='bold')
    ax2.text(0, -1.35, "Forearm Rotation (palm view)", ha='center',
             fontsize=7.5, color='#94a3b8')

    fig.tight_layout()
    return fig

def finger_curl_diagram(grip, finger_flex):
    """Original anatomical curling finger diagram — the one you liked!"""
    grip_col   = score_color(synergy_score(grip, 80, 100))
    finger_col = score_color(synergy_score(finger_flex, 70, 90))
    closure    = grip / 100.0

    fig, ax = plt.subplots(figsize=(5, 6), facecolor='#f8fafc')
    ax.set_facecolor('#f8fafc')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.8, 4.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Grip: {grip}%  |  Finger Flexion: {finger_flex}°",
                 fontsize=10, fontweight='bold', color='#374151', pad=10)

    # Palm
    palm = patches.FancyBboxPatch((-0.7, 0), 1.4, 1.1,
        boxstyle="round,pad=0.1", linewidth=2,
        edgecolor='#475569', facecolor='#cbd5e1', zorder=2)
    ax.add_patch(palm)
    ax.text(0, 0.55, "PALM", ha='center', va='center',
            fontsize=7, color='#64748b', fontweight='bold')

    # Thumb
    thumb_angle = np.radians(150 - closure * 70)
    tx = -0.75 + 0.55 * np.cos(thumb_angle)
    ty = 0.4   + 0.55 * np.sin(thumb_angle)
    ax.plot([-0.75, tx], [0.4, ty], color='#94a3b8',
            linewidth=13, solid_capstyle='round', zorder=3)
    ax.plot([-0.75, tx], [0.4, ty], color='#e2e8f0',
            linewidth=9, solid_capstyle='round', zorder=4)

    # Four fingers
    finger_x    = [-0.48, -0.16,  0.16,  0.46]
    base_lens   = [  0.55,  0.65,  0.62,  0.50]
    mid_lens    = [  0.48,  0.55,  0.52,  0.42]
    tip_lens    = [  0.32,  0.38,  0.36,  0.28]
    knuckle_colors = ['#94a3b8'] * 4

    for fx, bl, ml, tl, kc in zip(finger_x, base_lens, mid_lens, tip_lens, knuckle_colors):

        # ── Base phalanx — always roughly vertical ──
        base_angle = np.radians(88)
        bx2 = fx  + bl * np.cos(base_angle)
        by2 = 1.1 + bl * np.sin(base_angle)

        ax.plot([fx, bx2], [1.1, by2], color='#64748b',
                linewidth=11, solid_capstyle='round', zorder=3)
        ax.plot([fx, bx2], [1.1, by2], color='#cbd5e1',
                linewidth=7,  solid_capstyle='round', zorder=4)

        # Knuckle dot
        ax.plot(bx2, by2, 'o', color=kc, markersize=6, zorder=5)

        # ── Mid phalanx — curls with finger_flex ──
        flex_rad  = np.radians(finger_flex * 0.65)
        mid_angle = base_angle - flex_rad
        mx2 = bx2 + ml * np.cos(mid_angle)
        my2 = by2 + ml * np.sin(mid_angle)

        ax.plot([bx2, mx2], [by2, my2], color=finger_col,
                linewidth=10, solid_capstyle='round', zorder=3)
        ax.plot([bx2, mx2], [by2, my2], color='#f0fdf4' if finger_col=='#059669' else '#fff7ed' if finger_col=='#d97706' else '#fff1f2',
                linewidth=6,  solid_capstyle='round', zorder=4)

        ax.plot(mx2, my2, 'o', color=finger_col, markersize=5, zorder=5)

        # ── Tip phalanx — curls further with closure ──
        tip_angle = mid_angle - np.radians(closure * 55)
        tx2 = mx2 + tl * np.cos(tip_angle)
        ty2 = my2 + tl * np.sin(tip_angle)

        ax.plot([mx2, tx2], [my2, ty2], color=grip_col,
                linewidth=8, solid_capstyle='round', zorder=3)
        ax.plot([mx2, tx2], [my2, ty2], color='#f0fdf4' if grip_col=='#059669' else '#fff7ed' if grip_col=='#d97706' else '#fff1f2',
                linewidth=4,  solid_capstyle='round', zorder=4)

    # Grip closure bar at bottom
    bar_y = -0.5
    ax.add_patch(patches.Rectangle((-1.5, bar_y), 3.0, 0.2,
        color='#e2e8f0', zorder=2))
    ax.add_patch(patches.Rectangle((-1.5, bar_y), 3.0*(grip/100), 0.2,
        color=grip_col, zorder=3))
    # Optimal marker
    opt_x = -1.5 + 3.0*(80/100)
    ax.plot([opt_x, opt_x], [bar_y-0.05, bar_y+0.25],
            color='#f97316', linewidth=2, linestyle='--', zorder=4)
    ax.text(opt_x, bar_y+0.32, 'Opt 80%',
            ha='center', fontsize=7, color='#f97316', fontweight='bold')
    ax.text(0, bar_y+0.1, f"Grip Closure: {grip}%",
            ha='center', va='center', fontsize=8,
            fontweight='bold', color='white' if grip>15 else '#374151', zorder=5)

    # Legend
    ax.text(-1.8, 3.7, "■ Base phalanx",   fontsize=7, color='#94a3b8')
    ax.text(-1.8, 3.5, "■ Mid phalanx",    fontsize=7, color=finger_col)
    ax.text(-1.8, 3.3, "■ Tip phalanx",    fontsize=7, color=grip_col)
    ax.text( 0.2, 3.7, f"Flex: {finger_flex}°", fontsize=7, color=finger_col, fontweight='bold')
    ax.text( 0.2, 3.5, f"Grip: {grip}%",        fontsize=7, color=grip_col,   fontweight='bold')

    fig.tight_layout()
    return fig


def hand_diagram(grip, finger_flex):
    """Front-view hand — grip bar + finger arc dial."""
    grip_col   = score_color(synergy_score(grip, 80, 100))
    finger_col = score_color(synergy_score(finger_flex, 70, 90))
    closure    = grip / 100.0

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.5), facecolor='#f8fafc')
    fig.suptitle(f"Grip Closure: {grip}%  |  Finger Flexion: {finger_flex}°",
                 fontsize=9, fontweight='bold', color='#374151')

    # ── Left: Grip closure bar chart ──
    ax = axes[0]; ax.set_facecolor('#f8fafc'); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Grip Closure", fontsize=8, color='#64748b')

    # Background bar
    ax.barh(0.5, 1.0, height=0.25, color='#e2e8f0', left=0)
    # Fill bar
    ax.barh(0.5, closure, height=0.25, color=grip_col, left=0)
    # Optimal marker
    ax.axvline(x=0.80, color='#f97316', linewidth=2, linestyle='--', alpha=0.8)
    ax.text(0.80, 0.78, 'Opt 80%', ha='center', fontsize=7,
            color='#f97316', fontweight='bold')
    ax.text(closure/2, 0.5, f"{grip}%", ha='center', va='center',
            fontsize=11, fontweight='bold', color='white' if grip > 20 else '#374151')
    ax.text(0.5, 0.2, "← Open        Closed →", ha='center',
            fontsize=7, color='#94a3b8')

    # Finger silhouettes (simple rect bars)
    for i, (fx, fh) in enumerate(zip([0.12,0.32,0.52,0.72,0.88],
                                      [0.55,0.65,0.63,0.58,0.40])):
        curl = fh * closure * (finger_flex / 90)
        bar_h = max(fh - curl * 0.4, 0.05)
        ax.bar(fx, bar_h, width=0.12, bottom=0.82,
               color=finger_col, alpha=0.85, linewidth=0)
    ax.text(0.5, 0.72, "↑ Fingers", ha='center', fontsize=7, color='#94a3b8')

    # ── Right: Finger flexion arc ──
    ax2 = axes[1]; ax2.set_facecolor('#f8fafc'); ax2.axis('off')
    ax2.set_xlim(-1.3, 1.3); ax2.set_ylim(-1.3, 1.3)
    ax2.set_aspect('equal')
    ax2.set_title("Finger Flexion", fontsize=8, color='#64748b')

    # Background arc (max 90°)
    t_max = np.linspace(np.radians(90), np.radians(90+90), 60)
    ax2.fill_between(np.cos(t_max), np.sin(t_max), alpha=0.08, color='#cbd5e1')
    ax2.plot(np.cos(t_max), np.sin(t_max), color='#e2e8f0', linewidth=6)

    # Optimal arc (70°)
    t_opt = np.linspace(np.radians(90), np.radians(90+70), 60)
    ax2.plot(np.cos(t_opt)*0.95, np.sin(t_opt)*0.95,
             color='#f97316', linewidth=2, linestyle='--')

    # Student arc
    t_stu = np.linspace(np.radians(90), np.radians(90+finger_flex), 60)
    ax2.fill_between(np.cos(t_stu)*0.85, np.sin(t_stu)*0.85, alpha=0.25,
                     color=finger_col)
    ax2.plot(np.cos(t_stu)*0.85, np.sin(t_stu)*0.85,
             color=finger_col, linewidth=3)

    ax2.plot([0,0],[0,1], color='#1a1a2e', linewidth=3)
    ax2.text(0, -1.2, f"{finger_flex}° measured  |  Opt: 70°",
             ha='center', fontsize=7, color='#64748b')
    ax2.text(0.25, 0.25, f"{finger_flex}°", fontsize=10,
             color=finger_col, fontweight='bold')

    fig.tight_layout()
    return fig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOTLY HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG="#ffffff"; GRD="#e2e8f0"

def arc_chart(human,optimal,human_max,label,unit):
    angles=np.linspace(0,1,100)
    def to_xy(av,r=1.0): rad=np.radians(av); return r*np.cos(rad),r*np.sin(rad)
    fig=go.Figure()
    xm,ym=to_xy(angles*human_max)
    fig.add_trace(go.Scatter(x=np.concatenate([[0],xm,[0]]),y=np.concatenate([[0],ym,[0]]),
        fill='toself',fillcolor='#f1f5f9',line=dict(color='#e2e8f0',width=1),
        name=f"Human max ({human_max}{unit})"))
    xo,yo=to_xy(angles*optimal,r=0.90)
    fig.add_trace(go.Scatter(x=np.concatenate([[0],xo,[0]]),y=np.concatenate([[0],yo,[0]]),
        fill='toself',fillcolor='#fff7ed',line=dict(color='#f97316',width=1.5,dash='dot'),
        name=f"Optimal ({optimal}{unit})"))
    col=score_color(synergy_score(human,optimal,human_max))
    xh,yh=to_xy(angles*human,r=0.78)
    fig.add_trace(go.Scatter(x=np.concatenate([[0],xh,[0]]),y=np.concatenate([[0],yh,[0]]),
        fill='toself',fillcolor=hex_to_rgba(col,0.2),line=dict(color=col,width=2.5),
        name=f"Measured ({human}{unit})"))
    fig.update_layout(
        title=dict(text=label,font=dict(family='DM Sans',size=11,color='#374151')),
        paper_bgcolor=BG,plot_bgcolor=BG,
        xaxis=dict(visible=False,range=[-1.2,1.2]),
        yaxis=dict(visible=False,range=[-1.2,1.2],scaleanchor='x'),
        legend=dict(font=dict(size=8,color='#64748b'),bgcolor=BG,
                    orientation='h',x=0.5,y=-0.1,xanchor='center'),
        margin=dict(l=10,r=10,t=34,b=50),height=230)
    return fig

def radar_chart(values_dict):
    cats=[LABELS[k] for k in values_dict]; scores=[values_dict[k] for k in values_dict]
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=[100]*len(cats)+[100],theta=cats+[cats[0]],
        fill='toself',fillcolor='#f1f5f9',line=dict(color='#e2e8f0',width=1),name='Machine Optimal'))
    fig.add_trace(go.Scatterpolar(r=scores+[scores[0]],theta=cats+[cats[0]],
        fill='toself',fillcolor='rgba(59,130,246,0.2)',line=dict(color='#3b82f6',width=2.5),name='Student Synergy'))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True,range=[0,100],
                tickfont=dict(size=8,color='#94a3b8'),gridcolor=GRD,linecolor=GRD),
            angularaxis=dict(tickfont=dict(size=9,color='#475569'),gridcolor=GRD,linecolor=GRD),
            bgcolor=BG),
        paper_bgcolor=BG,legend=dict(font=dict(size=10,color='#475569'),bgcolor=BG),
        margin=dict(l=40,r=40,t=20,b=20),height=380)
    return fig

def sensor_chart(series,t,optimal,label,unit):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t,y=[optimal]*len(t),mode='lines',name='Machine Optimal',
        line=dict(color='#f97316',width=1.5,dash='dash')))
    fig.add_trace(go.Scatter(x=t,y=series,mode='lines',name='Sensor Reading',
        line=dict(color='#3b82f6',width=2),fill='tonexty',fillcolor='rgba(59,130,246,0.04)'))
    fig.update_layout(
        title=dict(text=label,font=dict(family='DM Sans',size=11,color='#374151')),
        paper_bgcolor=BG,plot_bgcolor=BG,
        xaxis=dict(title='Time (s)',tickfont=dict(size=9),gridcolor=GRD,linecolor=GRD,
                   title_font=dict(size=9,color='#94a3b8')),
        yaxis=dict(title=unit,tickfont=dict(size=9),gridcolor=GRD,linecolor=GRD,
                   title_font=dict(size=9,color='#94a3b8')),
        legend=dict(font=dict(size=9,color='#64748b'),bgcolor=BG),
        margin=dict(l=44,r=16,t=34,b=36),height=200)
    return fig

def xp_bar_html(xp_now,xp_in_level,xp_level_total,level_name,level_icon,level_idx):
    pct=min(int(xp_in_level/max(xp_level_total,1)*100),100)
    next_lvl=LEVELS[level_idx+1][1] if level_idx+1<len(LEVELS) else "MAX"
    return f"""
    <div class="xp-header"><span class="xp-label">{level_icon} {level_name}</span>
      <span class="xp-num">⚡ {xp_now} XP</span></div>
    <div class="xp-bar-wrap"><div class="xp-bar" style="width:{pct}%;"></div></div>
    <div style="display:flex;justify-content:space-between;font-size:.65rem;color:#94a3b8;">
      <span>{xp_in_level}/{xp_level_total} XP to next</span><span>→ {next_lvl}</span></div>"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AI ROBOT DESCRIPTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_robot_description(name, all_vals, all_scores, overall,
                                 sh_avg, bic_avg, fist_avg, level_name):
    """Call Claude to generate a personalised robotic suit description."""
    client = anthropic.Anthropic()

    strongest = max(all_scores, key=all_scores.get)
    weakest   = min(all_scores, key=all_scores.get)

    measurements_text = "\n".join(
        f"  - {LABELS[k]}: {all_vals[k]}{UNITS[k]} → synergy {all_scores[k]}%"
        for k in all_vals
    )

    prompt = f"""You are a creative robotics engineer and educator at CognitiveCloud.ai.
A student named {name or 'the student'} has just completed their human locomotion profile.
Their body measurements and machine synergy scores are:

{measurements_text}

Summary:
- Shoulder group synergy: {sh_avg}%
- Bicep/forearm group synergy: {bic_avg}%
- Grip/fist group synergy: {fist_avg}%
- Overall synergy: {overall}%
- Pilot level: {level_name}
- Strongest axis: {LABELS[strongest]} ({all_scores[strongest]}%)
- Weakest axis: {LABELS[weakest]} ({all_scores[weakest]}%)

Write a vivid, exciting 3-paragraph description of what THEIR personalised robotic suit would look like
and how it would move, based DIRECTLY on these math results. Be specific — reference actual degree values
and synergy scores to explain why certain joints are reinforced, more flexible, or need assistance.

Paragraph 1: Overall appearance and design philosophy based on their synergy profile.
Paragraph 2: How each of the 3 joint groups (shoulder, arm, grip) is engineered specifically for their measurements — use the actual numbers.
Paragraph 3: What it feels like to pilot this machine, and one specific improvement challenge (based on their weakest axis) they should train toward.

Keep it inspiring, educational, and grounded in the mathematics. Write for a high school student.
Do NOT use bullet points — flowing descriptive prose only."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role":"user","content":prompt}]
    )
    return message.content[0].text

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for k in ['quiz_answers','quiz_attempts','check_answers','check_attempts']:
    if k not in st.session_state: st.session_state[k]={}
if 'robot_desc' not in st.session_state: st.session_state.robot_desc=None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEV CREDIT BANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="dev-credit">
  <h2>🦾 Human Locomotion Profiler</h2>
  <p>CognitiveCloud.ai · Applied Robotics &amp; Biomechanics</p><hr>
  <p>💻 Powered by <strong><a href='https://www.cognitivecloud.ai' target='_blank'>
  www.cognitivecloud.ai</a></strong> | Developed by Xavier Honablue M.Ed</p>
</div>""", unsafe_allow_html=True)
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("### 🧑‍🎓 Student Profile")
    student_name = st.text_input("Name", placeholder="e.g. Alex Rivera")
    student_id   = st.text_input("Student ID", placeholder="e.g. CC-2024-001")
    side         = st.radio("Dominant Side",["Right","Left"],horizontal=True)
    session_note = st.text_area("Session Notes",placeholder="Observations…",height=70)
    st.divider()
    st.markdown("### ⚙️ Display Options")
    show_sensor = st.toggle("Sensor Simulation",  value=True)
    show_arcs   = st.toggle("Motion Arc Charts",  value=True)
    show_calib  = st.toggle("Robot Calibration",  value=True)
    st.divider()
    st.caption("CognitiveCloud.ai · Locomotion Profiler v1.0")
    st.caption(f"Session: {datetime.now().strftime('%d %b %Y · %H:%M')}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TITLE & INTRO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.title("🦾 Human Locomotion Profiler")

# ── LESSON HOOK ──────────────────────────────────────────────
st.markdown("""
### 🎬 Imagine This...
You step into a robotic suit. It weighs 80 pounds of titanium and steel.
Every time **you** move your arm, the robot moves its arm.
Every time **you** close your fist, the robot closes its hand.

But here's the problem — **your body and the machine speak different languages.**

Your arm moves in *degrees*. The robot understands *electrical signals*.

**Math is the translator.** That's what this lesson is about.
""")

st.info("📚 **8th Grade Standards:** NGSS MS-PS2 · Michigan Merit Curriculum · "
        "8.F.A.1 Functions · 8.EE.B.5 Proportional Relationships · CSTA 2-AP-14")

st.markdown("#### 📋 Common Core & NGSS Standards Covered in This Lesson")
standard = st.selectbox("Select a standard to highlight:", [
    "8.F.A.1 — A function assigns exactly one output to each input (core of every locomotion function)",
    "8.F.A.2 — Compare functions: human range vs machine optimal across 13 joint axes",
    "8.F.B.4 — Construct a linear function: f(θ) = (θ ÷ human_max) × machine_optimal",
    "8.EE.B.5 — Proportional relationships: control signal scales proportionally with measurement",
    "8.EE.C.7 — Solve linear equations: find θ when control signal = target value",
    "NGSS MS-PS2-2 — Apply Newton's Third Law: robot exerts equal/opposite force on pilot",
    "NGSS MS-PS3-1 — Kinetic energy of robot joints relates to angular velocity and mass",
    "CSTA 2-AP-14 — Functions used to organise calibration, synergy, and sensor calculations",
    "CSTA 2-DA-08 — Collect and analyse data (sensor readings) to refine a computational model",
])

std_tips = {
    "8.F.A.1": "Every input box in this app IS a function — one measurement maps to exactly one control signal.",
    "8.F.A.2": "Compare your shoulder synergy vs grip synergy — two functions, two different outputs from similar inputs.",
    "8.F.B.4": "The normalisation formula f(θ)=(θ÷max)×optimal is a linear function. Its slope = optimal÷max.",
    "8.EE.B.5": "If your arm moves twice as far, the robot gets twice the signal — that is a proportional relationship.",
    "8.EE.C.7": "Challenge: if the robot needs a 60° signal and your max is 180°, what must you measure?",
    "NGSS MS-PS2-2": "When your arm pushes the exoskeleton joint, the joint pushes back on your arm with equal force.",
    "NGSS MS-PS3-1": "Faster joint rotation = more kinetic energy. The robot must safely absorb and redirect that energy.",
    "CSTA 2-AP-14": "synergy_score(), normalise_to_machine(), and sensor_rmse() are all reusable functions in this app.",
    "CSTA 2-DA-08": "The sensor simulation generates data — RMSE and lag are the analysis that refines the robot model.",
}
for key, tip in std_tips.items():
    if key in standard:
        st.success(f"💡 **Connection to this lesson:** {tip}")
        break


st.markdown("---")

# ── VOCABULARY ────────────────────────────────────────────────
st.header("📖 Vocabulary — Know These First")
v1,v2,v3 = st.columns(3)
with v1:
    st.markdown("""
    **🔵 Function**
    A rule that turns an INPUT into exactly one OUTPUT.
    > Press a button on a vending machine (input) → get one snack (output).
    No button gives you two random snacks. That's a function.

    **🔵 Input**
    The value you put IN.
    In this app: the degrees you can move your arm.
    """)
with v2:
    st.markdown("""
    **🟠 Output**
    The value that comes OUT.
    In this app: the signal sent to the robot joint.

    **🟠 Range of Motion**
    How far a joint can move, measured in degrees (°).
    A full circle = 360°. Your elbow bends about 145°.
    """)
with v3:
    st.markdown("""
    **🟢 Synergy Score**
    How closely YOUR motion matches what the MACHINE needs.
    100% = perfect match. Below 55% = the robot needs help.

    **🟢 Control Signal**
    The number sent to the robot motor.
    It's calculated by a function using YOUR body measurements.
    """)

st.markdown("---")

# ── CLASSIC EXPLANATION ───────────────────────────────────────
st.header("📘 The Big Idea — What Is a Locomotion Function?")
st.markdown("""
A **locomotion function** does one job:

> **It takes how far YOU can move a body part (input)**
> **and turns it into a command for the robot (output).**

Here's the formula — it's simpler than it looks:

```
f(θ) = (your measurement ÷ your maximum) × machine optimal
```

**Example with real numbers:**
- You can raise your arm out to the side **120°** (that's your measurement)
- A human shoulder can go up to **180°** maximum
- The robot shoulder works best at **90°**

```
f(120) = (120 ÷ 180) × 90
f(120) = 0.667 × 90
f(120) = 60°  ← this signal goes to the robot
```

The robot gets **60°** of movement command. Not too much. Not too little.
That's your math keeping the machine safe and smooth.
""")

with st.expander("🤔 Why not just send the robot your exact measurement?"):
    st.markdown("""
    Great question! Because **humans and robots have different limits.**

    If your arm goes to 120° but the robot's optimal is only 90°,
    sending 120° could:
    - Overstress the robot joint
    - Make the movement jerky and unsafe
    - Damage the actuator (the robot muscle)

    The function **scales** your movement to fit the machine perfectly.
    This is called **normalisation** — and it's used in robotics, music production,
    video game controllers, and medical devices every day.
    """)

with st.expander("📐 What does ° (degrees) actually mean?"):
    st.markdown("""
    Degrees measure how far something rotates.
    - 0° = no movement (arm straight down)
    - 90° = a right angle (arm straight out to the side)
    - 180° = arm straight up above your head

    When we say your shoulder abduction is **120°**, we mean you can raise
    your arm 120 degrees away from your body — two thirds of the way to straight up.
    """)

st.markdown("---")

# ── OBJECTIVES ────────────────────────────────────────────────
st.subheader("🎯 By the End of This Lesson You Will:")
st.markdown("""
- ✅ Explain what a locomotion function does in your own words
- ✅ Measure your body's range of motion across 3 joint groups
- ✅ Calculate control signals using the normalisation formula
- ✅ Read synergy scores and explain what they mean for the robot
- ✅ Design and describe your personalised robotic suit using your math
""")
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MATCHING TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.header("🎯 Locomotion Function Matching Challenge")
st.markdown("**Match each body movement to its locomotion function before exploring the measurements!**")
df_match=pd.DataFrame({
    "Movement":list(LABELS.values()),
    "Function f(θ)":["f(θ)=θ/180×90","f(θ)=θ/180×90","f(θ)=θ/90×60","f(θ)=θ/90×60",
                     "f(θ)=θ/145×90","f(θ)=θ/90×75","f(θ)=θ/90×75",
                     "f(θ)=θ/80×50","f(θ)=θ/70×45","f(p)=p/100×80","f(θ)=θ/90×70",
                     "f(θ)=θ/145×90","f(θ)=θ/30×15"],
    "Human Max":["180°","180°","90°","90°","145°","90°","90°","80°","70°","100%","90°",
                 "145°","30°"],
    "Machine Optimal":["90°","90°","60°","60°","90°","75°","75°","50°","45°","80%","70°",
                       "90°","15°"],
    "Joint Group":["Shoulder"]*4+["Bicep/Forearm"]*5+["Fist/Grip"]*2+["Bicep/Forearm"]*2,
    "Real-World Action":[
        "🙋 Raise arm laterally","🤚 Raise arm forward","🔄 Rotate arm inward",
        "↩️ Rotate arm outward","💪 Curl forearm up","🤲 Palm up (holding bowl)",
        "🔩 Palm down (turning screwdriver)","🙏 Bend wrist inward",
        "🖐️ Bend wrist backward","✊ Close fist","🤜 Curl fingers at knuckle",
        "🦾 Full arm curl toward shoulder","↔️ Arm opens past straight",
    ]
})
st.dataframe(df_match,use_container_width=True,hide_index=True)
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUNCTION TYPE DESCRIPTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.header("📊 The 4 Math Functions Powering Your Robot")
st.markdown("Each of these functions does a specific job. Together they let a robot suit respond to YOUR body.")

col1,col2=st.columns(2)
with col1:
    st.markdown("""
    ### 📐 1. Normalisation Function
    **Plain English:** *Translate your degrees into robot language*

    Your body max and the robot's optimal are different numbers.
    This function scales them to match.

    | You measure | Formula | Robot receives |
    |---|---|---|
    | 120° shoulder | (120÷180)×90 | **60°** |
    | 100° elbow | (100÷145)×90 | **62°** |
    | 75% grip | (75÷100)×80 | **60%** |

    **f(θ) = (your measurement ÷ your max) × machine optimal**
    """)

    st.markdown("""
    ### 🎯 2. Synergy Function
    **Plain English:** *How well does your body match the machine?*

    100% = perfect — you and the robot move as one.
    Below 55% = the robot is struggling to follow you.

    - ✅ 80–100% = Optimal
    - ⚠️ 55–79% = Acceptable
    - 🔴 0–54% = Needs work

    **Score = (1 − gap between you and optimal) × 100**
    """)
with col2:
    st.markdown("""
    ### 📡 3. Sensor Function
    **Plain English:** *How smooth and accurate is the signal?*

    Sensors on your body read movement 24 times per second.
    We measure how far off those readings are from perfect.

    - **RMSE** = average error per reading (lower = smoother)
    - **Lag** = how many seconds behind the robot is

    Think of it like ping in a video game — lower is better.

    **RMSE = √( average of all the errors squared )**
    """)

    st.markdown("""
    ### ⚙️ 4. Calibration Function
    **Plain English:** *Set the robot to fit YOUR body*

    Every student is different. A 5'2" student and a 6'4" student
    need completely different robot settings.

    Calibration runs all 3 functions above for all 13 joints
    and saves a personal profile for your suit.

    **One function per joint × 13 joints = your full robot profile**
    """)
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUIZ 1 — Attempt-tracked
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("📝 Quick Matching Quiz")
quiz_questions={
    "Which function correctly normalises shoulder abduction (measured 120°, human max 180°, machine optimal 90°)?": {
        "options":["f(120)=120/180×90=60°","f(120)=120×90=10,800","f(120)=180/120×90=135°","f(120)=90/180=0.5"],
        "correct":"f(120)=120/180×90=60°",
        "hint":"💡 Divide your measured value by your human max, then multiply by machine optimal: (measured÷max)×optimal.",
        "explanation":"Normalisation: (120/180)×90 = 0.667×90 = 60°. The robot receives a 60° command signal.",
    },
    "A student's elbow synergy score is 45%. What does this mean for the robot?": {
        "options":["Perfectly aligned with the machine","Significantly misaligned — recalibration needed",
                   "The robot ignores the elbow joint","The student has 45° elbow flexion"],
        "correct":"Significantly misaligned — recalibration needed",
        "hint":"💡 Synergy 0–100%. Below 55% = 🔴 Recalibrate. Human motion and machine optimal are far apart.",
        "explanation":"45% synergy = large deviation. The robot's movements won't match the student's intent until recalibrated.",
    },
    "Which sensor metric tells you how far student motion deviates from the machine setpoint on average?": {
        "options":["Synergy Score","RMSE","Normalised Control Signal","Human Max"],
        "correct":"RMSE",
        "hint":"💡 Root Mean Square Error — measures average distance of sensor readings from a target. Lower = more accurate.",
        "explanation":"RMSE = √(Σ(sensor_i − optimal)²/n). Lower RMSE = smoother human-machine alignment.",
    },
}
for q_idx,(question,data) in enumerate(quiz_questions.items()):
    st.write(f"**{question}**")
    user_answer=st.radio("Select your answer:",data["options"],key=f"quiz_{q_idx}")
    if q_idx not in st.session_state.quiz_attempts: st.session_state.quiz_attempts[q_idx]=0
    if st.button("Check Answer",key=f"quiz_check_{q_idx}"):
        if user_answer==data["correct"]:
            st.success(f"✅ Correct! {data['explanation']}")
            st.session_state.quiz_answers[q_idx]=True
            st.session_state.quiz_attempts[q_idx]=0
        else:
            st.session_state.quiz_attempts[q_idx]+=1
            if st.session_state.quiz_attempts[q_idx]==1:
                st.warning("❌ Not quite! Try again with this hint:"); st.info(data["hint"])
                st.session_state.quiz_answers[q_idx]=False
            else:
                st.error(f"❌ {data['explanation']}")
                st.warning(f"💡 **Correct answer: {data['correct']}**")
                st.info("Review the function type descriptions above and try once more!")
                st.session_state.quiz_answers[q_idx]=False

if (len(st.session_state.quiz_answers)==len(quiz_questions)
        and all(st.session_state.quiz_answers.values())):
    st.success("🎉 Perfect score on the matching quiz!")
    st.balloons()
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — SHOULDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<span class="pill p-s">📐 Section 1 — Shoulder Pivot Rotation</span>',
            unsafe_allow_html=True)
st.markdown("Measure your maximum shoulder range across four axes — "
            "the primary pivot joint driving upper-body reach in the robotic suit.")

# Graphic sample first
gc1,gc2=st.columns([1,2])
with gc1:
    st.markdown("#### 🖼️ Shoulder Anatomy Reference")
    st.markdown("""
    <div style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center;">
      <div style="font-size:3rem;">🦴</div>
      <div style="font-weight:700;color:#7c3aed;margin:6px 0 4px;">Glenohumeral Joint</div>
      <div style="font-size:.78rem;color:#64748b;line-height:1.7;">
        The shoulder is a <strong>ball-and-socket joint</strong>.<br>
        The ball (humeral head) rotates inside the socket (glenoid fossa).<br>
        This allows <strong>4 axes of motion</strong>:<br>
        Abduction · Flexion · Internal Rotation · External Rotation<br><br>
        <em>Normal range: 0–180° abduction, 0–90° rotation</em>
      </div>
    </div>""", unsafe_allow_html=True)
with gc2:
    st.markdown("#### 📐 Your Shoulder Arc Diagram")
    st.markdown("Set your sliders below — the diagram updates live.")
    c1,c2,c3,c4=st.columns(4)
    with c1: sh_abd  = st.number_input("Abduction (°)",        min_value=0, max_value=180, value=120, step=1, key="sh_abd")
    with c2: sh_flex = st.number_input("Flexion (°)",           min_value=0, max_value=180, value=110, step=1, key="sh_flex")
    with c3: sh_ir   = st.number_input("Internal Rotation (°)", min_value=0, max_value=90,  value=55,  step=1, key="sh_ir")
    with c4: sh_er   = st.number_input("External Rotation (°)", min_value=0, max_value=90,  value=50,  step=1, key="sh_er")
    st.pyplot(shoulder_diagram(sh_abd,sh_flex,sh_ir,sh_er))

sh_vals={"shoulder_abduction":sh_abd,"shoulder_flexion":sh_flex,
          "shoulder_int_rot":sh_ir,"shoulder_ext_rot":sh_er}
cols=st.columns(4)
for i,(key,val) in enumerate(sh_vals.items()):
    hmax,opt,tol=OPT[key]; syn=synergy_score(val,opt,hmax); dev=deviation(val,opt)
    status,css=deviation_status(dev,tol); xp=xp_for_score(syn)
    with cols[i]:
        st.markdown(f"""
        <div class="mcard">
          <div class="mcard-t">{LABELS[key]}</div>
          <div class="mcard-v">{val}<span class="mcard-u"> °</span></div>
          <span class="xp-chip">+{xp} XP</span>
          <div class="irow" style="margin-top:8px;">
            <span class="ik">Synergy</span>
            <span class="iv" style="color:{score_color(syn)}">{syn}%</span></div>
          <div class="irow"><span class="ik">Control Signal</span>
            <span class="iv">{normalise_to_machine(val,hmax,opt)} °</span></div>
          <div class="irow"><span class="ik">Status</span>
            <span class="iv {css}">{status}</span></div>
        </div>""", unsafe_allow_html=True)
if show_arcs:
    ac=st.columns(4)
    for i,(key,val) in enumerate(sh_vals.items()):
        hmax,opt,tol=OPT[key]
        with ac[i]: st.plotly_chart(arc_chart(val,opt,hmax,LABELS[key],"°"),use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — BICEP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<span class="pill p-b">💪 Section 2 — Bicep Flexion &amp; Forearm</span>',
            unsafe_allow_html=True)
st.markdown("Elbow flexion drives the primary lifting motion. "
            "Supination and pronation control forearm orientation for precision grasping.")

gc1,gc2=st.columns([1,2])
with gc1:
    st.markdown("#### 🖼️ Elbow & Forearm Reference")
    st.markdown("""
    <div style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center;">
      <div style="font-size:3rem;">💪</div>
      <div style="font-weight:700;color:#0891b2;margin:6px 0 4px;">Elbow &amp; Forearm Bones</div>
      <div style="font-size:.78rem;color:#64748b;line-height:1.7;">
        <strong>Humerus</strong> (upper arm) connects at the elbow<br>
        to the <strong>Radius &amp; Ulna</strong> (forearm bones).<br><br>
        Elbow flexion: <strong>0–145°</strong><br>
        Supination (palm up): <strong>0–90°</strong><br>
        Pronation (palm down): <strong>0–90°</strong><br>
        Wrist flexion/extension: <strong>0–80° / 0–70°</strong>
      </div>
    </div>""", unsafe_allow_html=True)
with gc2:
    st.markdown("#### 💪 Your Arm Motion Diagram")
    c1,c2,c3,c4,c5=st.columns(5)
    with c1: el_flex = st.number_input("Elbow Flexion (°)",      min_value=0, max_value=145, value=100, step=1, key="el_flex")
    with c2: f_sup   = st.number_input("Forearm Supination (°)", min_value=0, max_value=90,  value=70,  step=1, key="f_sup")
    with c3: f_pro   = st.number_input("Forearm Pronation (°)",  min_value=0, max_value=90,  value=65,  step=1, key="f_pro")
    with c4: w_flex  = st.number_input("Wrist Flexion (°)",      min_value=0, max_value=80,  value=55,  step=1, key="w_flex")
    with c5: w_ext   = st.number_input("Wrist Extension (°)",    min_value=0, max_value=70,  value=45,  step=1, key="w_ext")
    st.pyplot(arm_diagram(el_flex,f_sup,f_pro,w_flex,w_ext))

st.markdown("##### Arm Flexion & Extension")
ac1,ac2=st.columns(2)
with ac1: arm_flex=st.number_input("Arm Flexion (°)", min_value=0, max_value=145, value=100, step=1, key="arm_flex",
    help="Full elbow-to-shoulder curl — how far you can flex the arm toward you.")
with ac2: arm_ext =st.number_input("Arm Extension (°)", min_value=0, max_value=30, value=10, step=1, key="arm_ext",
    help="Hyperextension past neutral — how far the arm opens beyond straight.")

bic_vals={"elbow_flexion":el_flex,"forearm_supination":f_sup,
           "forearm_pronation":f_pro,"wrist_flexion":w_flex,"wrist_extension":w_ext,
           "arm_flexion":arm_flex,"arm_extension":arm_ext}
bic_cols=st.columns(4)
bic_items=list(bic_vals.items())
for i,(key,val) in enumerate(bic_items):
    hmax,opt,tol=OPT[key]; syn=synergy_score(val,opt,hmax); dev=deviation(val,opt)
    status,css=deviation_status(dev,tol); xp=xp_for_score(syn)
    with bic_cols[i%4]:
        st.markdown(f"""
        <div class="mcard">
          <div class="mcard-t">{LABELS[key]}</div>
          <div class="mcard-v">{val}<span class="mcard-u"> °</span></div>
          <span class="xp-chip">+{xp} XP</span>
          <div class="irow" style="margin-top:8px;">
            <span class="ik">Synergy</span>
            <span class="iv" style="color:{score_color(syn)}">{syn}%</span></div>
          <div class="irow"><span class="ik">Control Signal</span>
            <span class="iv">{normalise_to_machine(val,hmax,opt)} °</span></div>
          <div class="irow"><span class="ik">Status</span>
            <span class="iv {css}">{status}</span></div>
        </div>""", unsafe_allow_html=True)
if show_arcs:
    ac=st.columns(4)
    for i,(key,val) in enumerate(bic_items):
        hmax,opt,tol=OPT[key]
        with ac[i%4]: st.plotly_chart(arc_chart(val,opt,hmax,LABELS[key],"°"),use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — FIST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<span class="pill p-f">✊ Section 3 — Fist Clasp &amp; Grip</span>',
            unsafe_allow_html=True)
st.markdown("Grip closure and finger flexion control the robotic end-effector — "
            "the hand's ability to grasp, hold and release objects.")

gc1,gc2=st.columns([1,2])
with gc1:
    st.markdown("#### 🖼️ Hand Anatomy Reference")
    st.markdown("""
    <div style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center;">
      <div style="font-size:3rem;">✋</div>
      <div style="font-weight:700;color:#059669;margin:6px 0 4px;">Hand Bones</div>
      <div style="font-size:.78rem;color:#64748b;line-height:1.7;">
        <strong>Metacarpals</strong> (palm bones) connect to<br>
        <strong>Proximal → Middle → Distal Phalanges</strong> (finger bones).<br><br>
        MCP Joint (knuckle) flexion: <strong>0–90°</strong><br>
        Grip closure (all fingers): <strong>0–100%</strong><br><br>
        <em>The robotic hand replicates all 3 phalanx segments per finger</em>
      </div>
    </div>""", unsafe_allow_html=True)
with gc2:
    st.markdown("#### ✊ Your Grip Diagram")
    c1,c2=st.columns(2)
    with c1: grip     = st.number_input("Grip Closure (%)",      min_value=0, max_value=100, value=75,  step=1, key="grip")
    with c2: fin_flex = st.number_input("Finger Flexion (°)",    min_value=0, max_value=90,  value=65,  step=1, key="fin_flex")
    st.pyplot(hand_diagram(grip,fin_flex))

fist_vals={"grip_closure":grip,"finger_flexion":fin_flex}
cols=st.columns(2)
for i,(key,val) in enumerate(fist_vals.items()):
    hmax,opt,tol=OPT[key]; syn=synergy_score(val,opt,hmax)
    util=utilisation(val,hmax); cov=machine_coverage(val,opt)
    dev=deviation(val,opt); status,css=deviation_status(dev,tol)
    xp=xp_for_score(syn); unit=UNITS[key]
    with cols[i]:
        st.markdown(f"""
        <div class="mcard">
          <div class="mcard-t">{LABELS[key]}</div>
          <div class="mcard-v">{val}<span class="mcard-u"> {unit}</span></div>
          <span class="xp-chip">+{xp} XP</span>
          <div class="irow" style="margin-top:8px;">
            <span class="ik">Synergy</span>
            <span class="iv" style="color:{score_color(syn)}">{syn}%</span></div>
          <div class="irow"><span class="ik">Utilisation</span>
            <span class="iv">{util}%</span></div>
          <div class="irow"><span class="ik">Machine coverage</span>
            <span class="iv">{cov}%</span></div>
          <div class="irow"><span class="ik">Control Signal</span>
            <span class="iv">{normalise_to_machine(val,hmax,opt)} {unit}</span></div>
          <div class="irow"><span class="ik">Status</span>
            <span class="iv {css}">{status}</span></div>
        </div>""", unsafe_allow_html=True)
if show_arcs:
    ac=st.columns(2)
    for i,(key,val) in enumerate(fist_vals.items()):
        hmax,opt,tol=OPT[key]
        with ac[i]: st.plotly_chart(arc_chart(val,opt,hmax,LABELS[key],UNITS[key]),use_container_width=True)

# ── Curling finger diagram ──────────────────────────────────────
st.markdown("#### ✊ Live Finger Flexion Visualiser")
st.markdown("Watch your fingers curl in real time as you adjust grip closure and finger flexion.")
st.pyplot(finger_curl_diagram(grip, fin_flex))

# Combined
all_vals  ={**sh_vals,**bic_vals,**fist_vals}
all_scores={k:synergy_score(all_vals[k],OPT[k][1],OPT[k][0]) for k in all_vals}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SENSOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sensor_rmse_vals={}
if show_sensor:
    st.markdown('<span class="pill p-x">📡 Section 4 — Sensor Stream vs Machine Optimal</span>',
                unsafe_allow_html=True)
    sensor_keys=["shoulder_abduction","elbow_flexion","grip_closure"]
    sensor_labels=["Shoulder Abduction","Elbow Flexion","Grip Closure"]
    sc1,sc2,sc3=st.columns(3)
    for col,key,lbl in zip([sc1,sc2,sc3],sensor_keys,sensor_labels):
        hmax,opt,tol=OPT[key]; series,t=simulate_sensor(all_vals[key],hmax)
        rmse=sensor_rmse(series,opt); lag=sensor_lag(series,opt)
        syn=all_scores[key]; sensor_rmse_vals[key]=rmse; xp_bonus=5 if rmse<8 else 0
        with col:
            st.plotly_chart(sensor_chart(series,t,opt,lbl,UNITS[key]),use_container_width=True)
            st.markdown(f"""
            <div class="irow"><span class="ik">RMSE vs Optimal</span>
              <span class="iv">{rmse} {UNITS[key]}</span></div>
            <div class="irow"><span class="ik">Estimated Lag</span>
              <span class="iv">{lag} s</span></div>
            <div class="irow"><span class="ik">Synergy</span>
              <span class="iv" style="color:{score_color(syn)}">{syn}%</span></div>
            {"<span class='xp-chip'>+5 XP Smooth Sensor!</span>" if xp_bonus else ""}
            """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYNERGY DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<span class="pill p-y">⚡ Synergy Dashboard</span>', unsafe_allow_html=True)
sh_avg  =round(np.mean([all_scores[k] for k in sh_vals]),  1)
bic_avg =round(np.mean([all_scores[k] for k in bic_vals]), 1)
fist_avg=round(np.mean([all_scores[k] for k in fist_vals]),1)
overall =round(0.35*sh_avg+0.35*bic_avg+0.30*fist_avg,    1)

d1,d2,d3,d4=st.columns(4)
for col,label,score,border in [
    (d1,"Shoulder Synergy",sh_avg,""),
    (d2,"Bicep Synergy",   bic_avg,""),
    (d3,"Grip Synergy",    fist_avg,""),
    (d4,"Overall Synergy", overall,"border:2px solid #3b82f6;"),
]:
    col.markdown(f"""
    <div class="syn-box" style="background:{score_bg(score)};{border}">
      <div class="syn-n" style="color:{score_color(score)}">{score}</div>
      <div class="syn-l">{label}</div>
    </div>""", unsafe_allow_html=True)
st.plotly_chart(radar_chart(all_scores),use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CALIBRATION TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
calib_rows=[]
for key,val in all_vals.items():
    hmax,opt,tol=OPT[key]; unit=UNITS[key]
    syn=synergy_score(val,opt,hmax); dev=deviation(val,opt)
    norm=normalise_to_machine(val,hmax,opt)
    status,_=deviation_status(dev,tol)
    calib_rows.append({"Joint/Axis":LABELS[key],"Student Max":f"{val} {unit}",
        "Machine Optimal":f"{opt} {unit}","Deviation":f"{dev} {unit}",
        "Synergy %":syn,"Utilisation %":utilisation(val,hmax),
        "Coverage %":machine_coverage(val,opt),"Control Signal":norm,"Status":status})
df_calib=pd.DataFrame(calib_rows)

if show_calib:
    st.markdown('<span class="pill p-z">🤖 Robot Calibration Parameters</span>',
                unsafe_allow_html=True)
    def colour_syn(val): return f"color:{score_color(val)};font-weight:700"
    st.dataframe(df_calib.style.map(colour_syn,subset=["Synergy %"]),
                 use_container_width=True,hide_index=True)
    csv_buf=io.StringIO(); df_calib.to_csv(csv_buf,index=False)
    st.download_button("⬇️ Download Calibration CSV",data=csv_buf.getvalue(),
        file_name=f"locomotion_{student_id or 'student'}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv")
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUIZ 2 — Understanding Check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.header("🎲 Understanding Check")
st.markdown("Answer based on **your session results** above:")
check_questions=[
    {"question":"Looking at your synergy dashboard — which joint group has your LOWEST synergy score?",
     "options":["Shoulder","Bicep / Forearm","Fist / Grip","They are all equal"],
     "correct_fn":lambda s,b,f: "Shoulder" if s<=b and s<=f else "Bicep / Forearm" if b<=s and b<=f else "Fist / Grip",
     "hint":"💡 Look at the three coloured synergy boxes above. Which number is smallest?",
     "explanation":"The lowest synergy group deviates most from machine optimal — that's where calibration adjustment is most needed."},
    {"question":"If your grip closure control signal is 0.64, what does that mean for the robot?",
     "options":["Robot hand opens to 64% of its range",
                "Robot hand receives 64% of its optimal grip command",
                "Student can grip 64° of movement","Sensor RMSE is 0.64"],
     "correct_fn":lambda *_:"Robot hand receives 64% of its optimal grip command",
     "hint":"💡 Control signal is a 0–1 scalar. 0.64 = 64% of the optimal command sent to the robot controller.",
     "explanation":"signal=(measured/human_max)×machine_optimal, normalised to 0–1. 0.64 means 64% of optimal grip command."},
    {"question":"A student measures 180° shoulder abduction (full human max). What happens to synergy?",
     "options":["Synergy=100% — perfect!","Synergy drops — over-extension penalised",
                "Synergy is undefined","Synergy=50%"],
     "correct_fn":lambda *_:"Synergy drops — over-extension penalised",
     "hint":"💡 Machine optimal for shoulder abduction is 90°, not 180°. Synergy penalises deviation in BOTH directions.",
     "explanation":"At 180° (norm=1.0) vs optimal 90° (norm=0.5): deviation=0.5, synergy=50%. Excess range doesn't help the robot."},
]
for i,q in enumerate(check_questions):
    st.write(f"**Question {i+1}:** {q['question']}")
    answer=st.radio(f"Select your answer for Q{i+1}:",q["options"],key=f"check_{i}")
    if i not in st.session_state.check_attempts: st.session_state.check_attempts[i]=0
    correct_answer=q["correct_fn"](sh_avg,bic_avg,fist_avg)
    if st.button(f"Check Answer {i+1}",key=f"check_btn_{i}"):
        if answer==correct_answer:
            st.success(f"✅ Correct! {q['explanation']}")
            st.session_state.check_answers[i]=True; st.session_state.check_attempts[i]=0
        else:
            st.session_state.check_attempts[i]+=1
            if st.session_state.check_attempts[i]==1:
                st.warning("❌ Not quite! Try again with this hint:"); st.info(q["hint"])
                st.session_state.check_answers[i]=False
            else:
                st.error(f"❌ {q['explanation']}")
                st.warning(f"💡 **Correct answer: {correct_answer}**")
                st.info("Review your synergy dashboard above and try once more!")
                st.session_state.check_answers[i]=False
if (len(st.session_state.check_answers)==len(check_questions)
        and all(st.session_state.check_answers.values())):
    st.success("🏆 Outstanding! Complete understanding of locomotion functions!")
    st.balloons()
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# XP REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<span class="pill p-g">🏆 XP &amp; Achievement Report</span>', unsafe_allow_html=True)
axis_xp     =sum(xp_for_score(all_scores[k]) for k in all_scores)
sensor_bonus=sum(5 for k,r in sensor_rmse_vals.items() if r<8)
quiz_bonus  =sum(10 for v in list(st.session_state.quiz_answers.values())+
                            list(st.session_state.check_answers.values()) if v)
session_xp  =axis_xp+sensor_bonus+quiz_bonus+10

all_statuses={k:deviation_status(deviation(all_vals[k],OPT[k][1]),OPT[k][2]) for k in all_vals}
control_sigs={k:normalise_to_machine(all_vals[k],OPT[k][0],OPT[k][1]) for k in all_vals}
earned_badges={"first_session"}
if all(all_scores[k]>=80 for k in sh_vals):           earned_badges.add("shoulder_ace")
if all(all_scores[k]>=80 for k in bic_vals):          earned_badges.add("bicep_beast")
if all(all_scores[k]>=80 for k in fist_vals):         earned_badges.add("iron_grip")
if overall>=85:                                         earned_badges.add("perfect_run")
if all(s[1]!="bad" for s in all_statuses.values()):    earned_badges.add("full_coverage")
if all(r<8 for r in sensor_rmse_vals.values()) and sensor_rmse_vals:
                                                        earned_badges.add("sensor_smooth")
if all(v>0.7 for v in control_sigs.values()):          earned_badges.add("calibrated")

xp_col1,xp_col2=st.columns([2,3])
with xp_col1:
    level_idx,(_,level_name,level_icon)=get_level(session_xp)
    next_thresh,xp_in_lvl,lvl_total=xp_to_next(session_xp)
    st.markdown(xp_bar_html(session_xp,xp_in_lvl,lvl_total,level_name,level_icon,level_idx),
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-top:14px;">
      <div class="irow"><span class="ik">📐 Axis XP</span><span class="iv">+{axis_xp}</span></div>
      <div class="irow"><span class="ik">📡 Sensor Bonus</span><span class="iv">+{sensor_bonus}</span></div>
      <div class="irow"><span class="ik">📝 Quiz Bonus</span><span class="iv">+{quiz_bonus}</span></div>
      <div class="irow"><span class="ik">✅ Session Complete</span><span class="iv">+10</span></div>
      <div class="irow" style="background:#fef3c7;border:1px solid #fbbf24;">
        <span class="ik" style="font-weight:700;">⚡ Total</span>
        <span class="iv" style="color:#ef4444;font-size:1rem;">{session_xp} XP</span></div>
    </div>""", unsafe_allow_html=True)
with xp_col2:
    st.markdown("**Achievements**")
    bh='<div class="badge-row">'
    for bid,bname,bicon,bdesc in BADGES:
        if bid in earned_badges:
            bh+=f'<span class="badge badge-earned" title="{bdesc}">{bicon} {bname}</span>'
        else:
            bh+=f'<span class="badge badge-locked" title="{bdesc}">🔒 {bname}</span>'
    bh+='</div>'
    st.markdown(bh,unsafe_allow_html=True)
    locked=[(bid,bname,bdesc) for bid,bname,bicon,bdesc in BADGES if bid not in earned_badges]
    if locked:
        st.markdown("**💡 Earn more XP:**")
        for bid,bname,bdesc in locked[:3]: st.markdown(f"- **{bname}**: {bdesc}")

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# END OF LESSON — COMMON CORE DRAWING ACTIVITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<span class="pill p-ai">🎨 End-of-Lesson · Design Your Robot</span>',
            unsafe_allow_html=True)
st.header("🤖 Design Your Robotic Suit — Math-Based Drawing Activity")

strongest = max(all_scores, key=all_scores.get)
weakest   = min(all_scores, key=all_scores.get)

st.markdown(f"""
<div class="drawing-prompt">
  <h3>🖊️ Common Core Drawing Prompt</h3>
  <p>
    Using the <strong>locomotion functions you just learned</strong>, design a sketch of YOUR personalised robotic suit.
    Your drawing must mathematically reflect your session data:
  </p>
  <p>
    <strong>1. Shoulder Joint:</strong>
    Draw the shoulder pivot with a clearly labelled arc showing your abduction range of
    <strong>{sh_abd}°</strong> (human max 180°, machine optimal 90°).
    Shade the arc in proportion — your synergy is <strong>{all_scores['shoulder_abduction']}%</strong>.
  </p>
  <p>
    <strong>2. Elbow &amp; Forearm:</strong>
    Draw the upper arm and forearm at your measured elbow flexion angle of
    <strong>{el_flex}°</strong>. Label the control signal value
    (<strong>{normalise_to_machine(el_flex,145,90)}°</strong>) sent to the robot joint.
  </p>
  <p>
    <strong>3. Robotic Hand:</strong>
    Draw the hand at <strong>{grip}% grip closure</strong>, with fingers flexed to
    <strong>{fin_flex}°</strong>. Use the formula
    f({grip}) = ({grip}/100) × 80 = <strong>{normalise_to_machine(grip,100,80)}%</strong>
    and label it on your drawing.
  </p>
  <p>
    <strong>4. Reinforcement Zones:</strong>
    Your strongest axis is <strong>{LABELS[strongest]}</strong> ({all_scores[strongest]}% synergy) —
    draw this joint as sleek and unobstructed.
    Your weakest axis is <strong>{LABELS[weakest]}</strong> ({all_scores[weakest]}% synergy) —
    draw this joint with <em>visible actuator support</em> (extra armour, a servo, or a brace).
  </p>
  <p>
    <strong>📐 Math Requirement:</strong>
    Label at least <strong>3 measurements</strong> directly on your drawing
    with their normalisation function and control signal value.
  </p>
</div>
""", unsafe_allow_html=True)

# Reference diagram for drawing
st.markdown("#### 📊 Your Measurement Summary for Drawing Reference")
draw_notes = {
    "shoulder_abduction" : "Draw shoulder arc at this angle",
    "shoulder_flexion"   : "Draw shoulder arc at this angle",
    "shoulder_int_rot"   : "Label internal rotation zone",
    "shoulder_ext_rot"   : "Label external rotation zone",
    "elbow_flexion"      : "Draw forearm at this angle",
    "forearm_supination" : "Show palm-up (supination) range",
    "forearm_pronation"  : "Show palm-down (pronation) range",
    "wrist_flexion"      : "Label wrist bend inward",
    "wrist_extension"    : "Label wrist bend backward",
    "grip_closure"       : "Fill hand grip bar to this %",
    "finger_flexion"     : "Show finger curl angle",
    "arm_flexion"        : "Draw full arm curl arc",
    "arm_extension"      : "Label arm hyperextension zone",
}
draw_ref = pd.DataFrame({
    "Joint / Axis"    : [LABELS[k] for k in all_vals],
    "Your Measurement": [f"{all_vals[k]} {UNITS[k]}" for k in all_vals],
    "Control Signal"  : [f"{normalise_to_machine(all_vals[k],OPT[k][0],OPT[k][1])} {UNITS[k]}" for k in all_vals],
    "Synergy %"       : [all_scores[k] for k in all_vals],
    "Draw Note"       : [draw_notes.get(k, "Label this joint") for k in all_vals],
})
st.dataframe(draw_ref,use_container_width=True,hide_index=True)

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AI ROBOT DESCRIPTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("#### 🤖 AI-Generated: What Your Machine Would Look Like")
st.markdown("Click below to have Claude describe your personalised robotic suit based on your exact math results:")

if st.button("✨ Generate My Robot Description", use_container_width=True):
    with st.spinner("🦾 Claude is designing your robot based on your locomotion data…"):
        try:
            desc = generate_robot_description(
                student_name, all_vals, all_scores, overall,
                sh_avg, bic_avg, fist_avg, level_name
            )
            st.session_state.robot_desc = desc
        except Exception as e:
            st.session_state.robot_desc = (
                f"Could not connect to AI generator ({e}). "
                "Please ensure your ANTHROPIC_API_KEY is set in Streamlit secrets."
            )

if st.session_state.robot_desc:
    st.markdown(f'<div class="robot-desc">{st.session_state.robot_desc}</div>',
                unsafe_allow_html=True)

st.markdown("---")

# Student drawing description text area
st.markdown("#### ✏️ Describe Your Drawing")
st.markdown("After sketching your robot, describe it here. "
            "Reference at least 3 specific measurements and their functions:")

drawing_desc = st.text_area(
    "My robot looks like…",
    placeholder=(
        f"e.g. My robotic suit has a reinforced shoulder pivot that allows {sh_abd}° "
        f"of abduction, sending a {normalise_to_machine(sh_abd,180,90)}° control signal. "
        f"The elbow joint is locked at {el_flex}° flexion with a synergy of "
        f"{all_scores['elbow_flexion']}%, which means…"
    ),
    height=160,
)

if st.button("Submit My Robot Design"):
    if drawing_desc.strip():
        # Check for at least 3 numbers/measurements mentioned
        import re
        numbers_found = re.findall(r'\d+\.?\d*[°%]?', drawing_desc)
        if len(numbers_found) >= 3:
            st.success("✅ Excellent mathematical design thinking! "
                       "Your robot reflects real biomechanical data — that's engineering!")
            st.balloons()
        else:
            st.warning("⚠️ Great start! Try including at least 3 specific measurement values "
                       "(degrees or %) from your session data in your description.")
    else:
        st.warning("Please describe your robot design to complete the activity.")

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REFLECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.header("🧾 Reflection")
st.markdown("Describe **one joint** where your synergy score surprised you. "
            "Why do you think your measurement differed from the machine optimal? "
            "What would you train to improve?")
reflection=st.text_area("Your reflection:",
    placeholder="e.g. My wrist extension synergy was only 55%…",height=100)
if st.button("Submit Reflection"):
    if reflection.strip():
        st.success("✅ Excellent biomechanical thinking! "
                   "Understanding your limits is the first step to mastering the machine.")
        st.balloons()
    else:
        st.warning("Please share your thoughts to complete the reflection.")
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if st.button("🚀 Complete Session & Generate Final Report", use_container_width=True):
    with st.spinner("Compiling your locomotion profile…"): time.sleep(1.2)
    grade=("🏆 ELITE" if overall>=85 else "⭐ GREAT" if overall>=70
           else "✅ GOOD" if overall>=55 else "📈 DEVELOPING")
    name_display=student_name if student_name else "Student"
    st.markdown(f"""
    <div class="celebrate-box">
      <div class="celebrate-title">{grade} — Session Complete!</div>
      <div class="celebrate-sub">
        {name_display} · Overall Synergy: <strong>{overall}%</strong> ·
        Level: <strong>{level_icon} {level_name}</strong> ·
        XP: <strong>⚡ {session_xp}</strong> ·
        Badges: <strong>{len(earned_badges)}/{len(BADGES)}</strong>
      </div>
    </div>""", unsafe_allow_html=True)
    if overall>=65: st.balloons()
    if overall>=85:
        st.success(f"🎉 Outstanding, {name_display}! All systems calibrated. "
                   "You are ready to pilot the robotic suit!")
    elif overall>=65:
        st.success(f"Great work, {name_display}! Review 🔴 axes before piloting.")
    else:
        st.info(f"Keep training, {name_display}! Focus on 🔴 Recalibrate axes.")

    st.markdown("#### 📋 Final Session Summary")
    summary=[{"Axis":LABELS[k],"Measured":f"{all_vals[k]} {UNITS[k]}",
              "Synergy %":all_scores[k],"XP":xp_for_score(all_scores[k]),
              "Control Signal":normalise_to_machine(all_vals[k],OPT[k][0],OPT[k][1])}
             for k in all_vals]
    st.dataframe(pd.DataFrame(summary),use_container_width=True,hide_index=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHAT YOU LEARNED
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.header("🎓 What You've Learned")
st.markdown(f"""
**Congratulations!** In this session you:
- ✅ **Matched 11 locomotion functions** to real body movements and machine control signals
- ✅ **Measured 3 joint groups** — shoulder, bicep/forearm, and fist — with live arc diagrams
- ✅ **Interpreted sensor streams** using RMSE and lag
- ✅ **Generated robot calibration parameters** mapping your biology to machine control signals
- ✅ **Designed your personalised robotic suit** grounded in your own measurement data
- ✅ **Earned ⚡ {session_xp} XP** and **{len(earned_badges)} badges** this session

**Remember:** Every function connects your body's input to the machine's output.
The closer your motion aligns with the machine optimal — the smoother, safer,
and more powerful the robot becomes. *You are the function.*
""")
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RESOURCES — IXL, KHAN ACADEMY, TOOLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.header("📚 Practice Resources & Related Lessons")
st.markdown("Use these to go deeper on the math behind the locomotion functions.")

res_strand = st.selectbox("Filter resources by standard:", [
    "8.F.A.1 — Functions (one input → one output)",
    "8.F.B.4 — Constructing linear functions",
    "8.EE.B.5 — Proportional relationships & slope",
    "NGSS MS-PS2 — Forces & Motion",
    "All Resources",
], key="res_strand")

# IXL
st.subheader("🎯 IXL Practice Lessons")
ixl_lessons = {
    "Grade 8 — Functions": [
        ("8.F.A.1", "Identify functions from tables and graphs",           "https://www.ixl.com/math/grade-8/identify-functions"),
        ("8.F.A.1", "Evaluate a function — find f(x) for a given input",  "https://www.ixl.com/math/grade-8/evaluate-a-function"),
        ("8.F.A.2", "Compare linear functions: tables, graphs, equations", "https://www.ixl.com/math/grade-8/compare-linear-functions"),
        ("8.F.B.4", "Write a linear function from a word problem",         "https://www.ixl.com/math/grade-8/write-a-linear-function-word-problems"),
        ("8.F.B.4", "Interpret the slope and y-intercept of a function",   "https://www.ixl.com/math/grade-8/slope-intercept-form"),
    ],
    "Grade 8 — Equations & Proportions": [
        ("8.EE.B.5", "Graph a proportional relationship",                  "https://www.ixl.com/math/grade-8/graph-a-proportional-relationship"),
        ("8.EE.B.5", "Find the constant of proportionality from a graph",  "https://www.ixl.com/math/grade-8/constant-of-proportionality"),
        ("8.EE.C.7", "Solve linear equations with variables on both sides","https://www.ixl.com/math/grade-8/solve-linear-equations"),
    ],
    "Algebra 1 — Functions (Enrichment)": [
        ("HSA.F.IF.1", "Domain and range of a function",                   "https://www.ixl.com/math/algebra-1/domain-and-range"),
        ("HSA.F.IF.2", "Evaluate a function for a given input",            "https://www.ixl.com/math/algebra-1/evaluate-functions"),
        ("HSA.F.BF.1", "Write a function from a real-world situation",     "https://www.ixl.com/math/algebra-1/write-a-linear-function-word-problems"),
        ("HSA.F.IF.4", "Interpret key features of a function graph",       "https://www.ixl.com/math/algebra-1/interpret-a-graph"),
    ],
}

for topic, lessons in ixl_lessons.items():
    with st.expander(f"📖 {topic}"):
        for std, name, url in lessons:
            st.markdown(f"[🔗 **{name}**]({url})  `{std}`")

st.markdown("---")

# Khan Academy
st.subheader("🎓 Khan Academy Videos & Exercises")
khan = {
    "What is a Function?": (
        "The foundation of this entire lesson — one input, one output.",
        "https://www.khanacademy.org/math/cc-eighth-grade-math/cc-8th-linear-equations-functions/8th-functions-and-function-notation/v/what-is-a-function"
    ),
    "Evaluating Functions": (
        "Practice plugging in values — exactly what f(θ)=(θ÷max)×optimal does.",
        "https://www.khanacademy.org/math/cc-eighth-grade-math/cc-8th-linear-equations-functions/8th-functions-and-function-notation/e/evaluating-functions"
    ),
    "Graphing Linear Functions": (
        "Visualise how your control signal grows linearly with your measurement.",
        "https://www.khanacademy.org/math/cc-eighth-grade-math/cc-8th-linear-equations-functions/8th-slope-intercept-form/v/slope-intercept-form"
    ),
    "Proportional Relationships": (
        "Understand why doubling your arm movement doubles the robot signal.",
        "https://www.khanacademy.org/math/cc-seventh-grade-math/cc-7th-ratio-proportion/cc-7th-proportional-rel/v/proportional-relationships"
    ),
    "Introduction to Robotics (NGSS)": (
        "Forces, motion, and Newton's 3rd Law — the physics behind the exoskeleton.",
        "https://www.khanacademy.org/science/ms-physics/x1baed5db7e7a5ec5:forces-and-newtons-laws-of-motion"
    ),
}

kc1, kc2 = st.columns(2)
for i, (title, (desc, url)) in enumerate(khan.items()):
    with (kc1 if i % 2 == 0 else kc2):
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;margin-bottom:10px;">
          <div style="font-size:.8rem;font-weight:700;color:#1a1a2e;margin-bottom:4px;">🎓 {title}</div>
          <div style="font-size:.74rem;color:#64748b;margin-bottom:8px;">{desc}</div>
          <a href="{url}" target="_blank" style="font-size:.7rem;font-weight:700;color:#1d4ed8;text-decoration:none;">
            Watch / Practice on Khan Academy →</a>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# Interactive Tools
st.subheader("🛠️ Interactive Tools")
tools = {
    "📺 Video Tutorials": [
        ("3Blue1Brown — What is a Function?",      "https://www.youtube.com/watch?v=kvGsIo1TmsM",      "Visual, intuitive explanation of functions and mappings"),
        ("Crash Course — Intro to Robotics",       "https://www.youtube.com/watch?v=AZSiqj7HDQY",      "How robots use sensors, actuators, and control functions"),
        ("Khan Academy — Functions Playlist",      "https://www.khanacademy.org/math/cc-eighth-grade-math/cc-8th-linear-equations-functions","Complete 8th grade functions video series"),
    ],
    "💻 Interactive Calculators": [
        ("Desmos — Graph Your Locomotion Function","https://www.desmos.com/calculator",               "Type f(x)=(x/180)*90 and see your shoulder function graphed live"),
        ("GeoGebra — Function Explorer",          "https://www.geogebra.org/graphing",               "Drag sliders to see how changing max range shifts the function"),
        ("Wolfram Alpha — Function Solver",        "https://www.wolframalpha.com/",                   "Solve: given control signal = 60, human_max = 180, what is θ?"),
    ],
    "📖 Reading & Reference": [
        ("Math Is Fun — Functions",                "https://www.mathsisfun.com/sets/function.html",    "Clear, visual explanation of functions with worked examples"),
        ("NASA — Robotics & Math",                 "https://www.nasa.gov/stem/nextgenstem/robotics/",  "How NASA engineers use functions to program robot arms"),
        ("Paul's Online Math Notes — Functions",  "https://tutorial.math.lamar.edu/Classes/Alg/Functions.aspx","Detailed notes with practice problems"),
    ],
    "🎮 Games & Practice": [
        ("Function Machine — Math Playground",    "https://www.mathplayground.com/functionmachine.html","Interactive: guess the function rule from inputs and outputs"),
        ("Manga High — Algebra Games",            "https://www.mangahigh.com/",                        "Gamified algebra and function practice"),
        ("Prodigy Math — Grade 8",               "https://www.prodigygame.com/",                      "Adaptive math game covering 8th grade standards"),
    ],
}

tool_tabs = st.tabs(list(tools.keys()))
for tab, (category, items) in zip(tool_tabs, tools.items()):
    with tab:
        for name, url, desc in items:
            st.markdown(f"**[{name}]({url})**")
            st.caption(desc)
            st.write("---")

st.markdown("---")

# Study Plan
st.subheader("📅 Personalised Study Plan")
level = st.selectbox("Your current comfort with functions:", [
    "🌱 Beginner — I'm not sure what a function is yet",
    "📈 Developing — I understand input/output but need practice",
    "✅ Proficient — I can write and evaluate functions",
    "🚀 Advanced — I want to go beyond 8th grade",
], key="study_level")
time_avail = st.selectbox("Time per week to study:", ["1–2 hours","3–4 hours","5+ hours"], key="study_time")

if st.button("Generate My Study Plan", key="study_plan_btn"):
    st.success("🎯 Your Personalised Locomotion + Functions Study Plan:")
    if "Beginner" in level:
        st.markdown("""
        **Week 1 — Build the Foundation**
        - Watch Khan Academy: *What is a Function?*
        - Play Function Machine on Math Playground (10 min/day)
        - IXL: *Identify functions from tables and graphs* (8.F.A.1)
        - In this app: focus on the shoulder section — one slider, watch the control signal change

        **Week 2 — Connect to the Robot**
        - IXL: *Evaluate a function — find f(x) for a given input* (8.F.A.1)
        - Practice the formula: f(θ) = (θ ÷ 180) × 90 with different shoulder values
        - Draw your robot arm at 3 different angles and label the control signal each time
        """)
    elif "Developing" in level:
        st.markdown("""
        **Week 1 — Strengthen Function Skills**
        - IXL: *Write a linear function from a word problem* (8.F.B.4)
        - Desmos: graph f(x) = (x/145)*90 — your elbow function
        - In this app: record your measurements and calculate control signals by hand, then check

        **Week 2 — Proportional Relationships**
        - IXL: *Graph a proportional relationship* (8.EE.B.5)
        - Khan Academy: *Proportional Relationships* video
        - Challenge: if your grip doubles from 40% to 80%, does the control signal also double?
        """)
    elif "Proficient" in level:
        st.markdown("""
        **Week 1 — Function Composition**
        - IXL: *Domain and range of a function* (HSA.F.IF.1)
        - Explore: what happens when you compose two locomotion functions?
        - GeoGebra: build a slider model of all 3 joint groups

        **Week 2 — Real Engineering Applications**
        - Research: how do actual exoskeleton companies (Ekso, ReWalk) calibrate their suits?
        - Write your own Python function for synergy_score() from scratch
        - IXL: *Interpret key features of a function graph* (HSA.F.IF.4)
        """)
    else:
        st.markdown("""
        **Ongoing Advanced Challenge**
        - Study inverse functions: given a control signal, solve for the human measurement needed
        - Research RMSE in machine learning — it's the same formula used in AI training
        - Build your own sensor simulation: add Gaussian noise to a signal and calculate drift
        - Explore Algebra 2: piecewise functions to model human joint limits with injury thresholds
        - Khan Academy: *Introduction to Calculus* — how derivatives model joint velocity
        """)

    tips = {
        "1–2 hours": "💡 **Tip:** 15 minutes daily beats 2 hours on one day. One IXL skill + 5 minutes in this app each day.",
        "3–4 hours": "💡 **Tip:** Split time 50/50 — half watching/reading, half hands-on practice in this app and Desmos.",
        "5+ hours":  "💡 **Tip:** Teach it to someone else. Explaining synergy scores or the normalisation formula to a classmate is the deepest learning.",
    }
    for t, tip in tips.items():
        if t in time_avail:
            st.info(tip)


st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODERN WEB RESOURCES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("🌐 Meet the Machines — Real Robotics in the Wild")
st.markdown("These are the actual companies, labs, and tools that use the math you just learned.")

modern_categories = {
    "🦾 Real Exoskeleton Companies": [
        {
            "name"  : "Ekso Bionics",
            "url"   : "https://eksobionics.com",
            "desc"  : "Medical exoskeleton suits helping people walk again after spinal injury. Their calibration system runs locomotion functions for every patient — exactly what you built today.",
            "tag"   : "Industry Leader · Medical Robotics",
            "color" : "#0891b2",
        },
        {
            "name"  : "ReWalk Robotics",
            "url"   : "https://rewalk.com",
            "desc"  : "FDA-cleared wearable robotic exoskeletons for people with lower limb disabilities. Uses sensor data and motion functions to synchronise human and machine movement.",
            "tag"   : "FDA Approved · Wearable Tech",
            "color" : "#7c3aed",
        },
        {
            "name"  : "Sarcos Technology",
            "url"   : "https://sarcos.com",
            "desc"  : "Full-body industrial exoskeleton (Guardian XO) that amplifies human strength up to 20x. The control system maps human force to machine output — a real-world version of your synergy function.",
            "tag"   : "Industrial · Strength Amplification",
            "color" : "#059669",
        },
        {
            "name"  : "Hyundai / Boston Dynamics",
            "url"   : "https://www.bostondynamics.com",
            "desc"  : "Atlas, Spot, and Stretch — the world's most advanced robots. Their motion planning uses the same normalisation and sensor feedback principles covered in this lesson.",
            "tag"   : "Cutting Edge · Humanoid Robots",
            "color" : "#dc2626",
        },
    ],
    "🔬 Research Labs & Science": [
        {
            "name"  : "MIT CSAIL — Robotics",
            "url"   : "https://www.csail.mit.edu/research/robotics",
            "desc"  : "MIT's Computer Science & AI Lab publishes open research on robot locomotion, sensor fusion, and human-robot interaction. This is where the math you used gets invented.",
            "tag"   : "MIT · Open Research",
            "color" : "#0f3460",
        },
        {
            "name"  : "NASA JPL — Robotics",
            "url"   : "https://www-robotics.jpl.nasa.gov",
            "desc"  : "JPL engineers built the Mars rovers using the same locomotion functions — mapping terrain sensor data to wheel control signals. Your calibration table is their rover software.",
            "tag"   : "NASA · Space Robotics",
            "color" : "#1d4ed8",
        },
        {
            "name"  : "Stanford Human-Robot Interaction Lab",
            "url"   : "https://hri.stanford.edu",
            "desc"  : "Researches how robots and humans move together safely — the science behind synergy scores. Publishes student-accessible papers.",
            "tag"   : "Stanford · Human-Robot Synergy",
            "color" : "#b91c1c",
        },
        {
            "name"  : "OpenStax — College Physics",
            "url"   : "https://openstax.org/books/college-physics-2e/pages/1-introduction-to-science-and-the-realm-of-physics-physical-quantities-and-units",
            "desc"  : "Free, peer-reviewed physics textbook covering forces, motion, and angular measurement — the physics behind every degree you measured today.",
            "tag"   : "Free Textbook · NGSS Aligned",
            "color" : "#d97706",
        },
    ],
    "💻 Code & Build": [
        {
            "name"  : "Tinkercad Circuits (Autodesk)",
            "url"   : "https://www.tinkercad.com",
            "desc"  : "Free browser-based tool to design and simulate robot circuits. Build the sensor circuit that would read the joint angles you measured today — no hardware needed.",
            "tag"   : "Free · Browser-Based · Arduino",
            "color" : "#f97316",
        },
        {
            "name"  : "Scratch — MIT (Robotics Projects)",
            "url"   : "https://scratch.mit.edu/studios/1243295",
            "desc"  : "Community of robotics simulation projects built in Scratch. See how other students have built function machines and motion simulators.",
            "tag"   : "MIT · Beginner Friendly",
            "color" : "#eab308",
        },
        {
            "name"  : "Python.org — Getting Started",
            "url"   : "https://www.python.org/about/gettingstarted/",
            "desc"  : "This entire app was written in Python. The functions synergy_score() and normalise_to_machine() you used today are real Python code. Start here to write your own.",
            "tag"   : "Python · Free · Industry Standard",
            "color" : "#3b82f6",
        },
        {
            "name"  : "Streamlit — Build Your Own App",
            "url"   : "https://streamlit.io",
            "desc"  : "The platform this app runs on. Free, Python-based, deploys in minutes. Build your own locomotion profiler or math tool and share it with anyone.",
            "tag"   : "Free · Deploy Instantly",
            "color" : "#e11d48",
        },
    ],
    "📰 Stay Current — Robotics News": [
        {
            "name"  : "IEEE Spectrum — Robotics",
            "url"   : "https://spectrum.ieee.org/robotics",
            "desc"  : "The world's leading engineering publication. Covers the latest exoskeleton breakthroughs, AI-driven motion systems, and human augmentation technology.",
            "tag"   : "Weekly Updates · Engineer-Written",
            "color" : "#0891b2",
        },
        {
            "name"  : "TechCrunch — Robotics",
            "url"   : "https://techcrunch.com/category/robotics",
            "desc"  : "Startup and industry news on the latest robotic suits, prosthetics, and human augmentation companies raising funding and shipping products.",
            "tag"   : "Industry News · Startups",
            "color" : "#16a34a",
        },
        {
            "name"  : "Science News for Students",
            "url"   : "https://www.snexplores.org/topic/technology-engineering",
            "desc"  : "Peer-reviewed science news written specifically for middle and high school students. Covers robotics, biomechanics, and engineering at your level.",
            "tag"   : "8th Grade Level · Peer Reviewed",
            "color" : "#7c3aed",
        },
        {
            "name"  : "FIRST Robotics Competition",
            "url"   : "https://www.firstinspires.org/robotics/frc",
            "desc"  : "The premier high school robotics competition. Teams build full robots using the exact same function-based control systems. Sign up starts in September.",
            "tag"   : "Competition · Scholarships Available",
            "color" : "#dc2626",
        },
    ],
    "🎓 Career Pathways": [
        {
            "name"  : "Bureau of Labor Statistics — Robotics Engineers",
            "url"   : "https://www.bls.gov/ooh/architecture-and-engineering/mechanical-engineers.htm",
            "desc"  : "Median salary: $99,510/year. Job growth: 10% (faster than average). The math you learned today is required in every robotics engineering program.",
            "tag"   : "Career Data · Official US Stats",
            "color" : "#0f3460",
        },
        {
            "name"  : "Code.org — AP Computer Science",
            "url"   : "https://code.org/teach/csp",
            "desc"  : "Free AP Computer Science Principles curriculum. The functions unit directly connects to normalise_to_machine() and synergy_score() from today's lesson.",
            "tag"   : "Free · AP Credit · 9–12",
            "color" : "#1d4ed8",
        },
        {
            "name"  : "Michigan Works! — STEM Careers",
            "url"   : "https://www.michiganworks.org",
            "desc"  : "Michigan-specific career resources. Detroit's auto industry is the largest employer of robotics engineers in the US — right in your backyard.",
            "tag"   : "Michigan · Local Careers",
            "color" : "#059669",
        },
    ],
}

for category, links in modern_categories.items():
    st.markdown(f"### {category}")
    link_cols = st.columns(2)
    for i, link in enumerate(links):
        with link_cols[i % 2]:
            st.markdown(f"""
            <a href="{link['url']}" target="_blank" style="text-decoration:none;">
              <div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid {link['color']};
                border-radius:0 10px 10px 0;padding:14px 16px;margin-bottom:12px;
                transition:background 0.2s;">
                <div style="font-size:.82rem;font-weight:700;color:#1a1a2e;margin-bottom:3px;">
                  {link['name']} ↗</div>
                <div style="font-size:.75rem;color:#475569;line-height:1.6;margin-bottom:6px;">
                  {link['desc']}</div>
                <span style="font-size:.62rem;font-weight:700;color:{link['color']};
                  background:{link['color']}15;padding:2px 8px;border-radius:99px;">
                  {link['tag']}</span>
              </div>
            </a>""", unsafe_allow_html=True)
    st.markdown("")


st.markdown("---")
st.markdown(f"**🎯 Standard Focus:** {standard}")
st.markdown("**📍 Michigan Merit Curriculum & Common Core:** This lesson supports 8th grade function standards, NGSS motion & forces, and applied mathematics through robotics.")

st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:.72rem;color:#94a3b8;padding:16px 0 8px;">
  CognitiveCloud.ai · Human Locomotion Profiler v1.0 · Developed by Xavier Honablue M.Ed ·
  <a href="https://cognitivecloud-launcher.streamlit.app" style="color:#3b82f6;">← Back to Launcher</a>
</div>""", unsafe_allow_html=True)
