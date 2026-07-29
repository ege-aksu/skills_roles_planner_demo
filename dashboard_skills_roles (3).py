"""Skills + Roles hiring planner — dual-angle Streamlit app.

Flow: company → **Browse** (radar list) → click a skill/role → **Plan**
(reskilling / mobility → scenario → cost). Back returns to the list.

Design conventions: skill adjacency on the skills angle; roles use
activity-overlap pathways.

Run:  streamlit run dashboard_skills_roles.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import skills_engine as se
import skills_roles_engine as sre

st.set_page_config(
    page_title="Skills + Roles Planner", layout="wide", page_icon="🧭")

# ---------------------------------------------------------------- branding (match dashboard_app.py)
ASSETS = Path(__file__).parent / "assets"
FONT_DIR = ASSETS / "fonts"
LOGO = ASSETS / "revelio-labs-dark.svg"

_FONT_WEIGHTS = {
    250: ("TWKLausannePan-250.otf", "TWKLausannePan-250Italic.otf"),
    400: ("TWKLausannePan-400.otf", "TWKLausannePan-400Italic.otf"),
    600: ("TWKLausannePan-600.otf", "TWKLausannePan-600Italic.otf"),
    800: ("TWKLausannePan-800.otf", "TWKLausannePan-800Italic.otf"),
}


@st.cache_data(show_spinner=False)
def _brand_css(_mtime_key):
    faces = []
    for weight, (reg, ital) in _FONT_WEIGHTS.items():
        for fname, style in ((reg, "normal"), (ital, "italic")):
            p = FONT_DIR / fname
            if not p.exists():
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            faces.append(
                f"@font-face {{ font-family: 'TWK Lausanne Pan'; "
                f"src: url(data:font/otf;base64,{b64}) format('opentype'); "
                f"font-weight: {weight}; font-style: {style}; "
                f"font-display: swap; }}")
    return f"""
    <style>
    {''.join(faces)}
    html, body, [class^="st-"], [class*=" st-"],
    [data-testid="stAppViewContainer"] * {{
        font-family: 'TWK Lausanne Pan', -apple-system, 'Segoe UI',
                     Roboto, Helvetica, Arial, sans-serif;
    }}
    [data-testid="stIconMaterial"],
    span[data-testid="stIconMaterial"],
    .material-symbols-rounded, .material-symbols-outlined {{
        font-family: 'Material Symbols Rounded' !important;
    }}
    h1, h2, h3 {{ font-weight: 600; letter-spacing: -0.01em; }}
    [data-testid="stMetricValue"] {{ font-weight: 600; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #EDF4FB 0%, #F7FAFD 55%,
                                    #F2F8F3 100%);
        border-right: 3px solid #92E47E;
    }}
    [data-testid="stSidebar"] h1 {{ font-size: 1.15rem; font-weight: 800; }}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        font-weight: 600; color: #25282D;
    }}
    [data-testid="stSidebar"] hr {{ border-color: #A9CCEA55; }}
    div[data-baseweb="popover"] ul[role="listbox"] {{
        max-height: 40vh !important;
        overflow-y: auto !important;
    }}
    </style>
    """


_font_mtimes = tuple(sorted(
    (f.name, f.stat().st_mtime) for f in FONT_DIR.glob("*.otf"))) \
    if FONT_DIR.exists() else ()
st.markdown(_brand_css(_font_mtimes), unsafe_allow_html=True)

if FONT_DIR.exists():
    for _w, (_reg, _ital) in _FONT_WEIGHTS.items():
        _p = FONT_DIR / _reg
        if _p.exists():
            fm.fontManager.addfont(str(_p))
    if (FONT_DIR / "TWKLausannePan-400.otf").exists():
        plt.rcParams["font.family"] = "TWK Lausanne Pan"

if LOGO.exists():
    try:
        st.logo(str(LOGO), size="large")
    except Exception:
        st.sidebar.image(str(LOGO), width=150)

# Roster lives in the engine so the batch snapshot script shares it.
COMPANIES = dict(sre.DEMO_COMPANIES)

CRS = [
    "#63A2D9", "#7FBF85", "#E2D68B", "#B493BA", "#E2AC8B", "#6D73B1",
    "#D38C9E", "#977B72", "#6BB3AE", "#B16D98", "#B1705C", "#7B8E4B",
    "#84848F",
]
TEXT = "#25282D"


def _rgba(color, alpha):
    """Palette hex at reduced opacity. CSS and Plotly both accept the result."""
    h = str(color).lstrip("#")
    if len(h) != 6:
        return color
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha:g})"


# Softer fills are the one palette at lower opacity, so a color edit carries
# through everywhere instead of drifting against a second hand-picked list.
CELL_ALPHA = 0.30   # table column fills
PILL_ALPHA = 0.20   # legend pills
C_BLUE, C_GREEN, C_PURPLE, C_CORAL = CRS[0], CRS[1], CRS[3], CRS[4]
# Composition bar: one color per way of closing the gap.
C_RETAIN, C_BUILD, C_BUY = CRS[5], CRS[1], CRS[4]
BUCKET_COLORS = {
    "emerging": CRS[1], "nascent": CRS[2], "growing": CRS[5],
    "core": CRS[0], "declining": CRS[4],
}
CAT_COLORS = {"expanding": CRS[1], "transforming": CRS[2],
              "at-risk": CRS[4], "stable": CRS[12]}
# Pathway feasibility reads high → low, so it keeps the green/sand/coral run
# the categories already use for "good / watch / caution".
FEAS_COLORS = {"high": CRS[1], "med": CRS[2], "low": CRS[4]}
plt.rcParams.update({
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.edgecolor": "#C8C8CA", "font.size": 10,
})


def _tint(df, *pairs):
    """Wash cells in their value's color. Takes ``(column, colors)`` pairs."""
    sty = df.style
    for col, colors in pairs:
        if col not in df.columns:
            continue

        def paint(v, _c=colors):
            hexc = _c.get(str(v))
            return f"background-color: {_rgba(hexc, CELL_ALPHA)}" if hexc else ""
        sty = sty.map(paint, subset=[col])
    return sty


def _i(label):
    """Mark a column whose header carries the explanation on hover."""
    return f"{label} ⓘ"


def _legend_pills(items):
    """Colored pills for a bucket/category legend (label + short gloss).

    The pill is washed at the same low opacity the table cells get, and the dot
    stays at full strength so the swatch is still legible.
    """
    html = ['<div style="display:flex;flex-wrap:wrap;gap:8px;'
            'margin:2px 0 12px 0">']
    for label, color, desc in items:
        wash, dot = _rgba(color, PILL_ALPHA), color
        html.append(
            f'<span style="display:inline-flex;align-items:center;gap:7px;'
            f'background:{wash};border:1px solid {_rgba(color, 0.45)};'
            f'border-radius:999px;padding:4px 12px;font-size:13px;'
            f'line-height:1.4">'
            f'<span style="width:9px;height:9px;border-radius:50%;'
            f'background:{dot};display:inline-block;flex:none"></span>'
            f'<b>{label}</b>'
            f'<span style="color:#5A5F66">{desc}</span></span>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _pick_from_table(event, view, name_col, extra_key, snapshots=None):
    """Row click on the browse radar → open Plan for that target.

    Snapshot: switch to that frozen pull (must already be saved). Live: stash
    the pick for the plan-page target control, then flip to Plan view.
    """
    rows = getattr(getattr(event, "selection", None), "rows", None) or []
    if not rows or name_col not in getattr(view, "columns", []):
        return
    try:
        picked = str(view.iloc[rows[0]][name_col])
    except Exception:
        return
    if not picked:
        return
    if snapshots is not None:
        if picked in snapshots:
            st.session_state[_SNAPKEY] = picked
            st.session_state[_VIEWKEY] = "plan"
            st.rerun()
        st.info(
            f"No saved snapshot for **{picked}**, so its pathways and "
            "headcount aren't frozen. Switch **Data source** to live to "
            "plan for it, or save a snapshot from there.")
        return
    st.session_state[extra_key] = picked
    st.session_state[f"_apply_{_TKEY}"] = picked
    st.session_state[_VIEWKEY] = "plan"
    st.rerun()


def _apply_pending_target(options, default):
    """Apply an Explore-row click to the target selectbox *before* it renders."""
    pending = st.session_state.pop(f"_apply_{_TKEY}", None)
    if pending and pending in options:
        st.session_state[_TKEY] = pending
    elif (_TKEY not in st.session_state
          or st.session_state[_TKEY] not in options):
        st.session_state[_TKEY] = default


def _keep_saved_rows(shown, full, name_col):
    """Add back rows that have their own snapshot but fell outside the trim.

    ``present_radar`` shows the top rows per bucket, which is fewer than the
    number of targets the batch save covers — so snapshots existed for rows
    that were invisible here, and therefore unclickable. Saved targets are
    pinned to the top of their bucket so the one you're planning for is never
    buried under higher-momentum peers.
    """
    if not snapshots or name_col not in getattr(full, "columns", []):
        return shown
    have = set(shown[name_col].astype(str)) if len(shown) else set()
    extra = full[full[name_col].astype(str).isin(set(snapshots) - have)]
    if extra.empty and not (len(shown) and set(snapshots) & have):
        return shown
    out = (pd.concat([shown, extra], ignore_index=True)
           if not extra.empty else shown.copy())
    # Explicit bucket/category order — never alphabetical (that put core first).
    _bucket_ord = {
        "emerging": 0, "nascent": 1, "growing": 2, "core": 3, "declining": 4}
    _cat_ord = {
        "expanding": 0, "transforming": 1, "at-risk": 2, "stable": 3}
    group = "bucket" if "bucket" in out.columns else (
        "category" if "category" in out.columns else None)
    out["_saved"] = out[name_col].astype(str).isin(set(snapshots)).astype(int)
    sort_cols, asc = [], []
    if group == "bucket":
        out["_g"] = out["bucket"].map(_bucket_ord).fillna(9)
        sort_cols.append("_g"); asc.append(True)
    elif group == "category":
        out["_g"] = out["category"].map(_cat_ord).fillna(9)
        sort_cols.append("_g"); asc.append(True)
    sort_cols.append("_saved"); asc.append(False)
    if "momentum" in out.columns:
        sort_cols.append("momentum"); asc.append(False)
    out = out.sort_values(sort_cols, ascending=asc, kind="stable")
    return out.drop(columns=[c for c in ("_saved", "_g") if c in out.columns]
                    ).reset_index(drop=True)


def _with_badges(frame, name_col, cfg):
    """Put the under-index badge on the radar table itself.

    Share math, so every row can carry it. Buildable supply used to sit beside
    it, but it costs a pathway query per candidate and so could only ever be
    filled for the shortlist — four rows in five came back blank. The supply
    check now fires on the target you actually pick, which covers rows the
    shortlist never evaluated.
    """
    out = frame.copy()
    if out.empty:
        return out
    if {"peer_share", "company_share"} <= set(out.columns):
        out["behind_peers"] = [
            "✓" if b else "✗" for b in se._under_indexed_mask(out, cfg)]
    else:
        out["behind_peers"] = ""
    return out


_BEHIND_HELP = (
    "✓ = the company sits meaningfully below the peer workforce share "
    "(under-indexed), so the gap is genuinely yours to close. ✗ = at or "
    "near parity with peers."
)


def _composition_bar(retained, reskilled, hired, baseline, build_label):
    """One bar showing how the baseline gap gets closed.

    The three segments sum exactly to the pre-retention net need, so the bar
    is the whole plan at a glance: what you avoid, build, and buy.

    The bar carries proportion only; every segment is then read out in an
    identical block beneath it. Writing text inside the segments meant the
    treatment changed with segment width — a wide Hired got its name and share,
    a 5% Retained got nothing — so no two runs looked alike. Equal blocks are
    the same at any mix.
    """
    segs = [("Retained", float(retained or 0), C_RETAIN,
             "attrition you avoid"),
            (build_label, float(reskilled or 0), C_BUILD,
             "internal pathways"),
            ("Hired", float(hired or 0), C_BUY,
             "external hiring")]
    total = sum(v for _, v, _, _ in segs)
    if total <= 0:
        return
    parts, blocks = [], []
    for label, val, color, gloss in segs:
        if val <= 0:
            continue
        w = val / total * 100
        parts.append(
            f'<div title="{label}: {val:,.0f} ({w:.0f}%)" '
            f'style="width:{w:.4f}%;background:{color}"></div>')
        blocks.append(
            f'<div style="flex:1;border-top:3px solid {color};padding:7px 2px 0 0">'
            f'<div style="font-size:12.5px;font-weight:600;color:{TEXT}">'
            f'{label}</div>'
            f'<div style="font-size:16px;font-weight:600;color:{TEXT};'
            f'line-height:1.35">{val:,.0f}'
            f'<span style="font-size:12.5px;font-weight:400;color:#5A5F66">'
            f'&nbsp;· {w:.0f}%</span></div>'
            f'<div style="font-size:12px;color:#5A5F66">{gloss}</div></div>')
    st.markdown(
        '<div style="display:flex;height:38px;border-radius:8px;'
        'overflow:hidden;margin:6px 0 10px 0">' + "".join(parts) + "</div>"
        '<div style="display:flex;gap:18px;margin:0 0 4px 0">'
        + "".join(blocks) + "</div>",
        unsafe_allow_html=True)
    st.caption(f"Closing the {baseline:,.0f}-position gap")


pct = st.column_config.NumberColumn

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Skills + Roles Planner")

st.sidebar.markdown("**Setup**")
angle = st.sidebar.radio(
    "Hiring approach",
    ["Skills-based hiring", "Role-based hiring"],
    help="Same scenario and cost math; different target and pathway story. "
         "Skills = build a capability. Roles = staff a job family.")
is_skills = angle.startswith("Skills")

# Naming the build lever per angle. A skill target teaches a capability to
# people who stay in their job; a role target fills the job by moving someone
# into it. Calling both "reskill" contradicted the Internal mobility tab.
_BUILD_V = "Reskill" if is_skills else "Redeploy"
_BUILD_PAST = "Reskilled" if is_skills else "Redeployed"
_BUILD_N = "reskilling" if is_skills else "redeployment"

_angle_key = "skills" if is_skills else "roles"
# Browse = radar list; Plan = pathways / scenario / cost for one target.
_VIEWKEY = "app_view"
if st.session_state.get(_VIEWKEY) not in ("browse", "plan"):
    st.session_state[_VIEWKEY] = "browse"
# Keyed so a row click in Explore can drive the snapshot picker.
_SNAPKEY = f"snap_sel_{_angle_key}"
_TKEY = "target_skill_sel" if is_skills else "target_role_sel"
snapshots = {}
snap_companies = sre.list_snapshot_companies(_angle_key)
modes = ["snapshot", "live"] if snap_companies else ["live"]
data_mode = st.sidebar.radio(
    "Data source", modes, index=0, horizontal=True,
    help="snapshot = saved Snowflake pulls, instant (use on stage). "
         "live = query Snowflake now (slow; use at your desk, then save).")

# Company in sidebar; radar browse / plan on the main page
if data_mode == "snapshot":
    company_rcid = None
    company = st.sidebar.selectbox(
        "Company", snap_companies,
        help="Companies with snapshots saved for this angle.")
    snapshots = sre.list_snapshots(_angle_key, company=company)
    if not snapshots:
        st.sidebar.error(f"No {_angle_key} snapshots for {company}.")
        st.stop()
    if st.session_state.get(_SNAPKEY) not in snapshots:
        st.session_state.pop(_SNAPKEY, None)   # company changed under the key
    st.sidebar.caption(
        f"{len(snapshots)} saved {'skill' if is_skills else 'role'} "
        f"snapshot{'s' if len(snapshots) != 1 else ''} — click a row to plan.")
    snap_target = st.session_state.get(_SNAPKEY)
    if snap_target not in snapshots:
        snap_target = next(iter(snapshots))
        st.session_state[_SNAPKEY] = snap_target
else:
    pick = st.sidebar.selectbox("Company", list(COMPANIES) + ["Custom…"])
    if pick == "Custom…":
        company = st.sidebar.text_input("Company name", "Company X")
        rcid_in = st.sidebar.text_input(
            "rcid", "", help="Ultimate-parent rcid — required to pull live.")
        company_rcid = int(rcid_in) if rcid_in.strip().isdigit() else None
    else:
        company, company_rcid = pick, COMPANIES[pick]
    if company_rcid is None:
        st.sidebar.error(
            "No rcid for this company — add it to COMPANIES or use Custom…. "
            "Live can't run without one.")
        st.stop()

# Company / angle / data-mode change → back to Browse.
_ctx = f"{_angle_key}|{data_mode}|{company}"
if st.session_state.get("_nav_ctx") != _ctx:
    st.session_state[_VIEWKEY] = "browse"
    st.session_state["_nav_ctx"] = _ctx

st.sidebar.divider()
# Plan levers matter on the Plan page; keep them visible so Browse→Plan
# doesn't reset the room's assumptions.
st.sidebar.markdown("**Plan levers**")
st.sidebar.caption(
    "Used on the Plan page — every number there recomputes as you move these."
    if st.session_state.get(_VIEWKEY) == "browse" else
    "Every number on the page recomputes as you move these.")
growth_basis = st.sidebar.radio(
    "Growth basis", ["fixed", "gap"], horizontal=True,
    help="fixed = client ask (grow by X%). gap = heads to reach peer-average "
         "workforce share (skills or roles).")
# Gap basis has a growth rate too — it's just derived from the peer gap instead
# of typed in. Filled once the scenario knows the target's headcount.
_gap_rate_slot = st.sidebar.empty()
horizon = st.sidebar.slider(
    "Horizon (years)", 1, 5, int(sre.CONFIG["horizon_years"]))

# Three named stances instead of three loose sliders: one decision in the room,
# and the numbers behind it are printed so it stays a stated assumption rather
# than a black box. Custom… still exposes the sliders for "what if 40%?".
STANCES = {
    "Conservative": {
        "growth": 0.15, "persistence": 0.75, "retention": 10,
        "blurb": f"modest ask · no credit for a {_BUILD_N} programme · "
                 "small retention win",
    },
    "Base case": {
        "growth": float(sre.CONFIG["growth_target"]), "persistence": 1.0,
        "retention": int(round(
            float(sre.CONFIG.get("retention_improvement", 0.15)) * 100)),
        "blurb": "client ask · mobility as observed · typical retention win",
    },
    "Aggressive": {
        "growth": 0.50, "persistence": 1.25, "retention": 25,
        "blurb": "stretch ask · program-supported mobility · "
                 "top-of-range retention win",
    },
}
stance = st.sidebar.radio(
    "Plan stance", [*STANCES, "Custom…"], index=1,
    help="Sets growth target, conversion persistence and assumed attrition "
         "improvement together. Conservative → smaller plan; Aggressive → "
         "bigger ask, and more credit for internal levers delivering it.")

if stance == "Custom…":
    growth_target = st.sidebar.slider(
        "Growth target (fixed)", 0.0, 1.0,
        float(sre.CONFIG["growth_target"]), 0.05,
        disabled=growth_basis == "gap",
        help="Gap basis sizes the ask from the peer gap, so this is off there."
             if growth_basis == "gap" else
             "Client ask: grow target headcount by this much over the horizon.")
    persistence = st.sidebar.slider(
        "Conversion persistence", 0.25, 1.5, 1.0, 0.05,
        help="Scales observed mobility forward: <1 conservative, >1 with "
             f"{_BUILD_N}-programme support.")
    retention_improv_pct = st.sidebar.slider(
        "Assumed attrition improvement", 0, 40,
        int(round(float(sre.CONFIG.get("retention_improvement", 0.15)) * 100)),
        1, format="%d%%",
        help="What-if: share of attrition a retention program cuts. Shrinks "
             "replacement / net need. Not 'should you invest here?'")
    st.sidebar.caption(
        "A strong retention program typically cuts attrition 10–25%; "
        "beyond that is aggressive.")
else:
    _st = STANCES[stance]
    growth_target = float(_st["growth"])
    persistence = float(_st["persistence"])
    retention_improv_pct = int(_st["retention"])
    if growth_basis == "gap":
        # On gap basis the ask is set by the peer gap, so the stance only moves
        # the internal levers — aggressive means *fewer* external hires, not
        # more. Say it, or the dial looks like it runs backwards.
        _lead = ("gap basis sets the ask — stance only changes how much credit "
                 "internal levers get, so aggressive means fewer hires")
        _nums = (f"persistence {persistence:.2f} · "
                 f"attrition −{retention_improv_pct}%")
    else:
        _lead = _st["blurb"]
        _nums = (f"growth +{int(growth_target * 100)}% · "
                 f"persistence {persistence:.2f} · "
                 f"attrition −{retention_improv_pct}%")
    st.sidebar.caption(f"{_lead}\n\n{_nums}")


@st.cache_data(show_spinner=False)
def _role_k10_options(use_sf, rcid, snapshot_families=()):
    # Snapshot radars carry their own families — prefer those so the filter
    # options match the rows you can actually narrow to.
    if snapshot_families:
        return list(snapshot_families)
    c = dict(sre.CONFIG)
    c["use_snowflake"] = bool(use_sf)
    c["company_rcid"] = rcid
    return sre.list_role_k10_families(c)


# ---------------------------------------------------------------- main: company title + role_k10 filter
# Resolve company early for snapshot so the title is right
_snap_fam_opts = None
if data_mode == "snapshot":
    _snap_peek = sre.load_snapshot(snapshots[snap_target])
    company = _snap_peek.get("company") or "Company"
    # Families available on this frozen pull (skills or roles).
    _rf = (_snap_peek.get("radar") if is_skills
           else _snap_peek.get("role_radar"))
    if (_rf is not None and not getattr(_rf, "empty", True)
            and "role_k10" in getattr(_rf, "columns", [])):
        _snap_fam_opts = sorted(
            {str(x) for x in _rf["role_k10"].dropna().astype(str)
             if str(x).strip() and str(x) not in ("unknown", "empty", "None")})

st.title(company or "Company")
_view = st.session_state[_VIEWKEY]
st.caption(
    ("**Browse** · pick a "
     f"{'skill' if is_skills else 'role'} to open its plan · {data_mode} data"
     if _view == "browse" else
     f"**Plan** · {data_mode} data")
    + (f" · rcid {company_rcid}"
       if data_mode == "live" and company_rcid else ""))

# Families always; target control only on Plan (Browse picks via row click).
strip = st.container()
with strip:
    if _view == "plan":
        _s0, _s1, _s2 = st.columns([0.8, 1.2, 1.5])
        with _s0:
            if st.button("← Browse", use_container_width=True,
                         help="Back to the skill/role list"):
                st.session_state[_VIEWKEY] = "browse"
                st.rerun()
        _fam_col, _tgt_col = _s1, _s2
    else:
        _fam_col = strip
        _tgt_col = None

with _fam_col:
    _fam_off = bool(data_mode == "snapshot" and not _snap_fam_opts)
    families = st.multiselect(
        "Occupation families",
        [] if _fam_off else _role_k10_options(
            data_mode == "live", company_rcid,
            tuple(_snap_fam_opts or ())),
        default=[], disabled=_fam_off,
        help="This snapshot was saved without family tags on the radar, so "
             "the filter can't narrow it — re-save from live to enable."
             if _fam_off else
             "Narrow the list. Skills: keeps skills concentrated in these "
             "families. Roles: filters the role list. Leave empty to search "
             "everything.")

# Snapshot Plan page: target = which frozen pull. Browse peeks one for radar.
if data_mode == "snapshot":
    if _view == "plan" and _tgt_col is not None:
        with _tgt_col:
            snap_target = st.selectbox(
                "Target", list(snapshots), key=_SNAPKEY,
                help="Saved pulls for this company — or go back to Browse "
                     "and click a row.")
    if snap_target != (_snap_peek.get("target") if _snap_peek else None):
        _snap_peek = sre.load_snapshot(snapshots[snap_target])
        company = _snap_peek.get("company") or company
        _rf = (_snap_peek.get("radar") if is_skills
               else _snap_peek.get("role_radar"))
        if (_rf is not None and not getattr(_rf, "empty", True)
                and "role_k10" in getattr(_rf, "columns", [])):
            _snap_fam_opts = sorted(
                {str(x) for x in _rf["role_k10"].dropna().astype(str)
                 if str(x).strip()
                 and str(x) not in ("unknown", "empty", "None")})

cfg = dict(sre.CONFIG)
cfg.update({
    "use_snowflake": data_mode == "live",
    "growth_mode": "both",
    "growth_primary": growth_basis,
    "growth_target": growth_target,
    "horizon_years": horizon,
    "conversion_persistence": persistence,
    "retention_improvement": retention_improv_pct / 100.0,
    "role_k10_filter": families or None,
    "deck_lead": "swp",
    "build_term": _BUILD_N,   # narrative says reskilling / redeployment
})
if data_mode == "live":
    cfg["company"] = company
    cfg["company_rcid"] = int(company_rcid)
elif data_mode == "snapshot":
    cfg["company"] = company
    cfg["use_snowflake"] = False
    cfg["company_rcid"] = None

_load_key = json.dumps(
    {k: cfg.get(k) for k in
     ["use_snowflake", "company_rcid", "peer_set", "peer_limit", "batchtime",
      "country", "skill_level", "activity_level", "role_k10_filter",
      "share_floor", "emerging_growth_percentile", "emerging_growth_threshold",
      "declining_growth_percentile", "min_skill_headcount", "min_peer_postings",
      "lift_floor_percentile", "require_specialized", "max_signal_growth",
      "require_dual_positive_growth", "exclude_contingent",
      "role_k10_skill_share",
      "role_category_top_n", "pathway_years", "pathway_min_pool",
      "pathway_candidate_n",
      "max_index_ratio", "min_under_index", "min_internal_supply",
      "min_feeder_roles"]},
    sort_keys=True, default=str)


@st.cache_data(show_spinner="Loading skill radar…", persist="disk")
def _skill_radar(key):
    return sre.build_skill_radar(cfg)


@st.cache_data(show_spinner="Loading role radar…", persist="disk")
def _role_radar(key):
    return sre.build_role_radar(cfg)


@st.cache_data(show_spinner="Loading adjacency…", persist="disk")
def _adjacency_data(key, target):
    """Skill adjacency only — roles use activity-overlap pathways instead."""
    return sre.build_adjacent_skills(cfg, target)


@st.cache_data(show_spinner=False, persist="disk")
def _skill_time_to_report(key, target, _v="ttr_overall1"):
    return sre.load_skill_time_to_report(cfg, target)


def _safe_adjacency(target, saved=None):
    """Skill adjacency for the target; saved snapshot wins, errors degrade."""
    if saved is not None and not getattr(saved, "empty", True):
        return saved
    try:
        return _adjacency_data(_load_key, target)
    except Exception:
        return None


def _safe_skill_time(target):
    try:
        return _skill_time_to_report(_load_key, target)
    except Exception:
        return None


@st.cache_data(show_spinner="Loading skill pathways…", persist="disk")
def _skill_paths(key, target):
    raw = se.load_pathways(cfg, target)
    sources = (
        raw["source_role"].dropna().astype(str).unique().tolist()
        if len(raw) and "source_role" in raw.columns else [])
    roles = se.classify_roles(cfg, include_roles=sources)
    return sre.build_skill_pathways(cfg, target, role_categories=roles), roles


@st.cache_data(show_spinner="Loading role pathways…", persist="disk")
def _role_paths(key, target):
    return sre.build_role_pathways(cfg, target)


@st.cache_data(show_spinner="Loading metros + outflows…", persist="disk")
def _external(key, target):
    metros, tight, avail = se.read_metros(cfg, target)
    outflows = se.load_competitor_outflows(cfg, target)
    return metros, tight, avail, outflows


@st.cache_data(show_spinner="Loading role outflows…", persist="disk")
def _role_outflows(key, target):
    return sre.load_role_competitor_outflows(cfg, target)


@st.cache_data(show_spinner="Loading skill population…")
def _skill_pop(key, target):
    return se.load_target_population(cfg, target)


@st.cache_data(show_spinner="Loading role population…", persist="disk")
def _role_pop(key, target):
    return sre.load_role_population(cfg, target)


# ---------------------------------------------------------------- load radar (before target pick)
tight, avail = ["—"], ["—"]
metros, outflows = pd.DataFrame(), pd.DataFrame()
roles_ctx = pd.DataFrame()
funnel = pd.DataFrame()
paths = pd.DataFrame()
adjacency = None
skill_time = None
snap = None
auto = None

if data_mode == "snapshot":
    snap = _snap_peek
    company = snap.get("company") or company or "Company"
    cfg["company"] = company
    if snap.get("company_rcid") is not None:
        cfg["company_rcid"] = int(snap["company_rcid"])
    # Snapshot freezes the pull; main-page role_k10 still notes the freeze filter
    if is_skills:
        radar = snap.get("radar", pd.DataFrame())
        funnel = snap.get("funnel", pd.DataFrame())
        paths = snap.get("paths", pd.DataFrame())
        roles_ctx = snap.get("roles", pd.DataFrame())
        metros = snap.get("metros", pd.DataFrame())
        outflows = snap.get("outflows", pd.DataFrame())
        tight = snap.get("tight") or ["—"]
        avail = snap.get("avail") or ["—"]
    else:
        radar = snap.get("role_radar", pd.DataFrame())
        funnel = snap.get("funnel", pd.DataFrame())
        paths = snap.get("paths", pd.DataFrame())
        outflows = snap.get("outflows", pd.DataFrame())
        if (len(radar) and "peer_share" not in radar.columns):
            radar = sre._attach_role_shares(radar, cfg)
        radar = se.drop_excluded_roles(radar, cfg)
        paths = se.drop_excluded_roles(paths, cfg, col="source_role")
        funnel = se.drop_excluded_roles(funnel, cfg)
else:
    if is_skills:
        radar = _skill_radar(_load_key)
        auto, funnel = sre.select_top_emerging_skill(cfg, radar)
    else:
        radar = _role_radar(_load_key)
        auto, funnel = sre.select_top_emerging_role(cfg, radar)

# ---------------------------------------------------------------- 2) pick target (Plan only)
# Browse stops after the radar list — no pathways / scenario until a row click.
_view = st.session_state[_VIEWKEY]
target = None
pop = (0.0, 0.0)

if _view == "browse":
    pass  # radar rendered below; no target load yet
elif data_mode == "snapshot":
    snap = _snap_peek
    target = snap_target
    adjacency = (_safe_adjacency(target, snap.get("adjacency"))
                 if is_skills else None)
    if is_skills:
        pop = (snap["population"], snap["attrition"])
        if snap.get("hiring_rate") is not None:
            pop = (*pop, snap["hiring_rate"])
        skill_time = snap.get("skill_time")
    else:
        pop = (snap["population"], snap["attrition"])
        if snap.get("hiring_rate") is not None:
            pop = (*pop, snap["hiring_rate"])
else:
    if is_skills:
        emerg = (
            funnel["skill"].tolist()
            if len(funnel) else
            (radar.loc[radar["bucket"] == "emerging", "skill"].astype(str).tolist()
             if "bucket" in radar.columns else []))
        if not emerg and len(radar):
            emerg = radar.sort_values(
                "momentum", ascending=False)["skill"].astype(str).tolist()[:8]
        if not emerg:
            st.error("No skills match. Try clearing the occupation families "
                     "filter, or loosen the entry gates in CONFIG.")
            st.stop()
        _extra = st.session_state.get("extra_target_skill")
        if _extra and _extra not in emerg and _extra in set(
                radar.get("skill", pd.Series(dtype=str)).astype(str)):
            emerg = [_extra] + emerg
        # Prefer the clicked row; else keep prior plan target if still valid.
        _apply_pending_target(
            emerg, (st.session_state.get(_TKEY)
                    if st.session_state.get(_TKEY) in emerg
                    else (auto if auto in emerg else emerg[0])))
        if _tgt_col is not None:
            with _tgt_col:
                target = st.selectbox(
                    "Target skill", emerg, key=_TKEY,
                    help="Or go back to Browse and click a row.")
                st.caption(
                    f"Suggested: **{auto}**" if target != auto
                    else "Suggested by momentum")
        else:
            target = st.session_state.get(_TKEY) or emerg[0]
        paths, roles_ctx = _skill_paths(_load_key, target)
        pop = _skill_pop(_load_key, target)
        adjacency = _safe_adjacency(target)
        skill_time = _safe_skill_time(target)
        if data_mode == "live":
            metros, tight, avail, outflows = _external(_load_key, target)
        if data_mode == "live" and st.sidebar.button(
                "💾 Save skill snapshot", type="primary"):
            sre.save_skill_snapshot(dict(
                radar=radar, funnel=funnel, paths=paths, roles=roles_ctx,
                adjacency=adjacency,
                metros=metros, outflows=outflows,
                target=target, population=pop[0], attrition=pop[1],
                hiring_rate=(pop[2] if len(pop) > 2 else None),
                company=company, company_rcid=cfg.get("company_rcid"),
                role_k10_filter=families or None,
                tight=list(tight), avail=list(avail),
                skill_time=skill_time))
            st.sidebar.success(f"Saved skill snapshot '{target}'.")
    else:
        role_opts = (
            funnel["role"].tolist()
            if len(funnel) else
            (radar.loc[radar["is_hot"] == True, "role"].astype(str).tolist()  # noqa: E712
             if "is_hot" in radar.columns else []))
        if not role_opts and len(radar):
            role_opts = radar.sort_values(
                "momentum", ascending=False)["role"].astype(str).tolist()[:8]
        if not role_opts:
            st.error("No roles match. Try clearing the occupation families "
                     "filter.")
            st.stop()
        _extra = st.session_state.get("extra_target_role")
        if _extra and _extra not in role_opts and _extra in set(
                radar.get("role", pd.Series(dtype=str)).astype(str)):
            role_opts = [_extra] + role_opts
        _apply_pending_target(
            role_opts, (st.session_state.get(_TKEY)
                        if st.session_state.get(_TKEY) in role_opts
                        else (auto if auto in role_opts else role_opts[0])))
        if _tgt_col is not None:
            with _tgt_col:
                target = st.selectbox(
                    "Target role", role_opts, key=_TKEY,
                    help="Or go back to Browse and click a row.")
                st.caption(
                    f"Suggested: **{auto}**" if target != auto
                    else "Suggested by momentum")
        else:
            target = st.session_state.get(_TKEY) or role_opts[0]
        paths = _role_paths(_load_key, target)
        pop = _role_pop(_load_key, target)
        adjacency = None
        if data_mode == "live":
            outflows = _role_outflows(_load_key, target)
        if data_mode == "live" and st.sidebar.button(
                "💾 Save role snapshot", type="primary"):
            sre.save_role_snapshot(dict(
                role_radar=radar, funnel=funnel, paths=paths,
                roles=radar, outflows=outflows, target=target,
                population=pop[0], attrition=pop[1],
                hiring_rate=(pop[2] if len(pop) > 2 else None),
                company=company, company_rcid=cfg.get("company_rcid"),
                role_k10_filter=families or None))
            st.sidebar.success(f"Saved role snapshot '{target}'.")

if len(funnel) and "behind_peers" not in funnel.columns:
    funnel = sre.enrich_funnel_badges(
        cfg, funnel, radar, angle="skills" if is_skills else "roles")

_supply_tab = "Reskilling" if is_skills else "Internal mobility"

# ---------------------------------------------------------------- Browse: radar only
if _view == "browse":
    st.divider()
    st.subheader("Skill radar" if is_skills else "Role radar")
    st.caption(
        f"Click a row to open the plan for that "
        f"{'skill' if is_skills else 'role'}.")
    if data_mode == "snapshot" and len(radar) <= 1:
        st.caption(
            "This snapshot froze only the target row — re-save from live to "
            "browse the whole radar here.")
    if is_skills:
        st.markdown(
            "Which skills are gaining ground across the peer group"
            + (", within the selected occupation families." if families
               else ", ranked by how fast."))
        _legend_pills([
            ("emerging", BUCKET_COLORS["emerging"], "rising fast, already sizable"),
            ("nascent", BUCKET_COLORS["nascent"], "rising fast, still small"),
            ("growing", BUCKET_COLORS["growing"], "gaining, not a breakout yet"),
            ("core", BUCKET_COLORS["core"], "sizable and steady"),
            ("declining", BUCKET_COLORS["declining"], "genuinely shrinking"),
        ])
        with st.expander("How these are measured"):
            st.markdown(
                "Every skill the peer group actually uses enters, as long as "
                "it clears three floors: enough people to measure, enough "
                "posting demand to be real, and distinctive to this peer "
                "group rather than generic. Movement is the change in each "
                "skill's **share** of peer hiring and postings over the last "
                "year versus the year before — a share measure, so it is "
                "not flattered by a hiring boom or punished by a freeze. The "
                "bands separating the buckets come from this run's own "
                "spread, not a fixed cutoff.\n\n"
                "**Buckets**\n\n"
                "- **Emerging** — blended growth at/above the hot bar "
                "(≈ P95 of this universe, floored at 0) **and** peer share "
                "at/above the materiality floor\n"
                "- **Nascent** — same hot growth, but still below the "
                "materiality floor (watch list)\n"
                "- **Growing** — material share, positive growth, but not "
                "a breakout (below the hot bar)\n"
                "- **Core** — material share, steady (not hot, not "
                "declining, not growing)\n"
                "- **Declining** — material share and blended growth at/"
                "below the decline bar (≈ P10 of this universe, capped at 0)"
            )
        show = se.present_radar(radar, cfg) if len(radar) else radar
        show = _keep_saved_rows(show, radar, "skill")
        if families and "role_k10" in show.columns:
            show = show[show["role_k10"].astype(str).isin(families)]
            if show.empty:
                st.caption("No skills from these families in this snapshot.")
        show = _with_badges(show, "skill", cfg)
        # Always present emerging → nascent → growing → core → declining.
        if "bucket" in show.columns:
            _b_ord = {"emerging": 0, "nascent": 1, "growing": 2,
                      "core": 3, "declining": 4}
            show = show.copy()
            show["_b"] = show["bucket"].map(_b_ord).fillna(9)
            _b_sort = ["_b"] + (["momentum"] if "momentum" in show.columns
                                else [])
            show = show.sort_values(
                _b_sort,
                ascending=[True] + [False] * (len(_b_sort) - 1),
                kind="stable",
            ).drop(columns="_b")
        cols = [c for c in ["skill", "bucket", "blended_growth",
                            "behind_peers"]
                if c in show.columns]
        view = show[cols] if cols else show
        _ev = st.dataframe(
            _tint(view, ("bucket", BUCKET_COLORS)) if "bucket" in view.columns
            else view,
            width="stretch", hide_index=True, key="radar_tbl_skills_browse",
            on_select="rerun", selection_mode="single-row",
            column_config={
                "bucket": st.column_config.TextColumn(
                    _i("bucket"),
                    help="Where the skill sits on the momentum/size grid — "
                         "see the legend above."),
                "blended_growth": pct(
                    _i("blended growth"), format="percent",
                    help="Blend of how fast this skill's share of peer "
                         "postings and peer hires has grown recently "
                         "(postings-weighted). Composition change — not "
                         "raw headcount growth."),
                "behind_peers": st.column_config.TextColumn(
                    _i("behind peers"), help=_BEHIND_HELP),
            })
        st.caption("Click any row to open its plan.")
        _pick_from_table(_ev, view, "skill", "extra_target_skill",
                         snapshots=snapshots if data_mode == "snapshot"
                         else None)
    else:
        st.markdown(
            "How each role is changing, and who may be freed up to move"
            + (", within the selected occupation families." if families
               else "."))
        _legend_pills([
            ("expanding", CAT_COLORS["expanding"], "hiring up, AI skill mix rising"),
            ("transforming", CAT_COLORS["transforming"], "AI-exposed, AI skill mix rising"),
            ("at-risk", CAT_COLORS["at-risk"], "AI-exposed, hiring falling"),
            ("stable", CAT_COLORS["stable"], "none of the above"),
        ])
        with st.expander("How these are measured"):
            st.markdown(
                "Categories come from four role-level signals — AI "
                "exposure, how fast the role's **AI-tagged skill share** is "
                "shifting (not total skill count), hiring growth, and "
                "attrition. They are measured independently of the skill "
                "radar, so the two views act as a second opinion on each "
                "other rather than the same number twice. Cutoffs are "
                "**relative to this company's roles** in the run "
                "(percentiles / medians of that cohort).\n\n"
                "**Categories** (first match wins, in this order: "
                "transforming → at-risk → expanding → stable)\n\n"
                "- **Transforming** — AI exposure ≥ P70 **and** AI skill-mix "
                "change ≥ P70\n"
                "- **At-risk** — AI exposure ≥ median **and** hiring growth "
                "≤ P30\n"
                "- **Expanding** — hiring growth ≥ P70 **and** AI skill-mix "
                "change ≥ median (and hiring growth must be > 0 if any role "
                "in the cohort is growing)\n"
                "- **Stable** — everything else"
            )
        show = _keep_saved_rows(radar, radar, "role")
        if families and "role_k10" in show.columns:
            show = show[show["role_k10"].astype(str).isin(families)]
            if show.empty:
                st.caption("No roles from these families in this snapshot.")
        show = _with_badges(show, "role", cfg)
        if "category" in show.columns:
            _cat_order = {"expanding": 0, "transforming": 1, "at-risk": 2, "stable": 3}
            show = show.copy()
            show["_c"] = show["category"].map(_cat_order).fillna(9)
            _rextra = [c for c in ["headcount"] if c in show.columns]
            show = show.sort_values(
                ["_c"] + _rextra,
                ascending=[True] + [False] * len(_rextra)).drop(columns="_c")
        cols = [c for c in ["role", "category", "ai_exposure",
                            "skill_mix_change", "behind_peers"]
                if c in show.columns]
        view = show[cols] if cols else show
        _ev = st.dataframe(
            _tint(view, ("category", CAT_COLORS)) if "category" in view.columns
            else view,
            width="stretch", hide_index=True, key="radar_tbl_roles_browse",
            on_select="rerun", selection_mode="single-row",
            column_config={
                "category": st.column_config.TextColumn(
                    _i("category"),
                    help="How the role is changing — see the legend above."),
                "ai_exposure": pct(
                    _i("AI exposure"), format="percent",
                    help="Mean AI exposure on positions in this role."),
                "skill_mix_change": pct(
                    _i("AI skill-mix Δ"), format="percent",
                    help="Change in the share of AI-tagged skills among "
                         "people entering this role (recent vs prior "
                         "starters) — not overall skill intensity."),
                "behind_peers": st.column_config.TextColumn(
                    _i("behind peers"), help=_BEHIND_HELP),
            })
        st.caption("Click any row to open its plan.")
        _pick_from_table(_ev, view, "role", "extra_target_role",
                         snapshots=snapshots if data_mode == "snapshot"
                         else None)
    st.stop()

# ---------------------------------------------------------------- Plan: scenario for chosen target
if not target:
    st.session_state[_VIEWKEY] = "browse"
    st.rerun()

# Feasibility clock feeds reskill ramp cost (see compute_plan_cost).
if is_skills and skill_time:
    cfg["skill_time"] = skill_time
else:
    cfg.pop("skill_time", None)

if is_skills:
    sc = sre.run_skill_scenario(cfg, target, paths, radar, population=pop)
    cost, naive, close = se.recommend_swp(
        cfg, target, sc, ["—"], ["—"], paths)
    path_blurb = sre.skill_pathway_blurb(target, paths)
    path_overlap_col = "skill_overlap"
    path_overlap_help = "Jaccard on skills vs target-skill holders"
else:
    sc = sre.run_role_scenario(
        cfg, target, paths, role_radar=radar, population=pop)
    cost, naive, close = se.recommend_swp(
        cfg, target, sc, ["—"], ["—"], paths)
    path_blurb = sre.role_pathway_blurb(target, paths)
    path_overlap_col = "activity_overlap"
    path_overlap_help = "Overlap with activities held by people in the target role"

_attr0 = float(sc.get("attrition", 0) or 0)
_attr1 = float(sc.get("attrition_after_retention", _attr0) or 0)
_improv = float(sc.get("retention_improvement", 0) or 0)
_base_net = float(sc.get("net_need_baseline", sc["net_need"]))

# Growth rate the peer gap works out to, as a % of today's headcount. Uses the
# post-cap head count so a capped gap doesn't advertise a rate it isn't using.
_gap_heads = float(sc.get("growth_need_gap", 0) or 0)
_cur_hc = float(sc.get("current", 0) or 0)
gap_growth_pct = (_gap_heads / _cur_hc) if _cur_hc > 0 else None

if growth_basis == "gap":
    if gap_growth_pct:
        _slot_txt = (f"Gap works out to **+{gap_growth_pct * 100:.0f}%** "
                     f"— ≈{round(_gap_heads):,} heads to parity on "
                     f"{round(_cur_hc):,} today")
        if sc.get("gap_capped"):
            _slot_txt += (f", capped at "
                          f"{float(cfg.get('max_gap_multiple', 3.0)):g}× current")
        _gap_rate_slot.caption(_slot_txt)
    elif _cur_hc > 0:
        _gap_rate_slot.caption(
            "At or above peer share — the gap sizes no growth heads, so net "
            "need here is replacement only.")

st.divider()
if is_skills and skill_time and skill_time.get("median_months") is not None:
    _ph, _pm = st.columns([4, 1])
    with _ph:
        st.markdown(f"#### Plan for {target}")
    with _pm:
        _m = float(skill_time["median_months"])
        st.metric(
            "Typical time to show skill",
            f"{_m:.0f} mo",
            help=(
                "Company-wide median months from entering a role that "
                f"needs {target} to first reporting it"
                + (f" (n={skill_time['n']:,})"
                   if skill_time.get("n") else "")
                + ". Floor on programme length; also feeds reskill "
                "opportunity cost below."
            ),
        )
else:
    st.markdown(f"#### Plan for {target}")
if growth_basis == "gap":
    _basis_note = ("closing the peer gap"
                   + (f" (+{gap_growth_pct * 100:.0f}% headcount)"
                      if gap_growth_pct else ""))
else:
    _basis_note = f"growing {int(growth_target * 100)}%"
st.caption(f"{_basis_note} · {horizon}-year horizon")

g1, g2, g3 = st.columns([2, 3, 1.5])
with g1:
    st.caption("THE NEED")
    _a, _b = st.columns(2)
    _a.metric("Current", f"{round(sc['current']):,}",
              help=f"{company} headcount for this target today")
    _b.metric("Net need", f"{round(sc['net_need']):,}",
              help=f"After retention (baseline {_base_net:,.0f})")
with g2:
    st.caption("HOW YOU FILL IT")
    _c, _d, _e = st.columns(3)
    _reskill_help = "applied to the net need from observed feeder transitions"
    if sc.get("pathways_cover_full_need"):
        _avail = sc.get("internal_supply_available", sc["internal_supply"])
        _reskill_help = (
            f"pathways cover the full need "
            f"(~{round(_avail):,} available; "
            f"applied {round(sc['internal_supply']):,})")
    _c.metric(_BUILD_V, f"{round(sc['internal_supply']):,}",
              help=_reskill_help)
    _d.metric("Hire", f"{round(sc['external_need']):,}",
              help=f"residual after internal {_BUILD_N}")
    _e.metric("Retain", f"{round(sc['heads_saved_by_retention']):,}",
              help=f"Assumed {_improv*100:.0f}% cut: "
                   f"{_attr0*100:.1f}% → {_attr1*100:.1f}%")
with g3:
    st.caption("COST")
    st.metric("Plan", f"${cost/1e6:,.1f}M",
              delta=f"-${(naive-cost)/1e6:,.1f}M vs buy-all",
              delta_color="inverse",
              help=f"{_BUILD_V.lower()} + hire + retention vs hiring the "
                   "pre-retention net need externally")

# --- signature visual: how the baseline hole gets closed -----------------
_composition_bar(
    retained=float(sc.get("heads_saved_by_retention", 0) or 0),
    reskilled=float(sc.get("internal_supply", 0) or 0),
    hired=float(sc.get("external_need", 0) or 0),
    baseline=_base_net, build_label=_BUILD_PAST)
if sc.get("pathways_cover_full_need"):
    st.caption(
        f"Internal pathways cover the full net need "
        f"(~{round(sc.get('internal_supply_available', sc['internal_supply'])):,} "
        f"available; plan applies {round(sc['internal_supply']):,}).")
else:
    # The counterpart check: a target can be a real gap and still have no one
    # to build from, which is a reskilling story the data can't support. Warn
    # on the target actually chosen, including ones picked off the radar that
    # the momentum shortlist never scored.
    _avail = float(sc.get("internal_supply_available",
                          sc.get("internal_supply", 0)) or 0)
    _need = float(sc.get("net_need", 0) or 0)
    _floor = float(cfg.get("min_internal_supply", 50))
    if _need > 0 and (_avail < _floor or _avail < 0.1 * _need):
        st.warning(
            f"Thin internal supply: observed feeder pathways can only build "
            f"~**{_avail:,.0f}** of the {_need:,.0f}-position need "
            f"({_avail / _need * 100:.0f}%), so this target is close to a "
            f"hire-everything plan. Check the **{_supply_tab}** tab before "
            "committing to a build story here.", icon="⚠️")
if (sc.get("growth_basis") == "gap"
        and sc.get("index_ratio") is not None
        and pd.notna(sc.get("index_ratio"))):
    st.caption(
        f"Gap basis: company is at **{sc['index_ratio']*100:.0f}%** of peer "
        f"workforce share "
        f"({sc.get('company_share', 0)*100:.2f}% vs "
        f"{sc.get('peer_share', 0)*100:.2f}% peer).")

# ---------------------------------------------------------------- detail tabs (Plan page — list lives on Browse)
tab_names = ["Scenario", _supply_tab, "Plan"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

with tabs[_supply_tab]:
    _noun = "skills" if is_skills else "activities"

    # ---- part 1: adjacent skills (skills angle only) -------------------
    # Role angle skips this: activity Jaccard on pathways already answers
    # "what sits next to this role."
    if is_skills:
        st.subheader(f"Skills adjacent to {target}")
        if adjacency is None or getattr(adjacency, "empty", True):
            st.caption("No adjacency computed for this skill yet.")
        else:
            st.markdown(
                f"What separates **{target}** people from everyone else — the "
                f"curriculum a programme has to close. *Lift* is how much more "
                f"common each skill is among {target} people than across the "
                "rest of the workforce, so a high number means genuinely "
                "nearby rather than generally popular.")
            _adj_cols = [c for c in ["name", "lift", "share_among_target",
                                     "share_outside", "holders_company"]
                         if c in adjacency.columns]
            st.dataframe(
                adjacency[_adj_cols], width="stretch", hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("skill"),
                    "lift": pct("lift", format="%.1f×",
                                help="how many times more common among "
                                     f"{target} people"),
                    "share_among_target": pct(
                        _i("holders have it"), format="percent",
                        help=f"share of {target} people who hold this skill"),
                    "share_outside": pct(
                        _i("everyone else"), format="percent",
                        help="share of the rest of the workforce who hold it — "
                             "low means it really is what sets holders apart"),
                    "holders_company": pct("people", format="%.0f",
                                           help="hold this today"),
                })
            st.caption(
                "People counts overlap (one person holds many skills), so they "
                "describe reach — they are not summed into supply. Headcount "
                "comes from the roles below.")
        st.divider()

    # ---- pathways: where those people sit ------------------------------
    # Time-to-show sits on the Plan header; keep the pathways title simple here.
    st.subheader(
        "Where those people sit today" if is_skills
        else f"Internal pathways into {target}")
    st.info(path_blurb)

    _rank_label = st.radio(
        "Rank feeders by",
        ["Conversion rate", "Move volume"],
        horizontal=True,
        key=f"pathway_rank_{_angle_key}",
        help=("Conversion rate = observed moves ÷ current headcount in the "
              "feeder (how common the path is). Move volume = absolute "
              "weighted moves (where the heads actually came from). The plan "
              "uses the full candidate set either way."),
    )
    cfg["pathway_rank_by"] = (
        "volume" if _rank_label == "Move volume" else "conversion")

    # ---- mobility vs peers: the story the ✓ column is making ------------
    # Same pools, higher conversion — so this is the cheapest supply on the
    # page, and worth stating in words before the table asserts it per row.
    _pk = ("peer_conversion_rate" if "peer_conversion_rate" in paths.columns
           else None)
    _has_peer = bool(_pk) and float(paths[_pk].fillna(0).abs().sum()) > 0
    if _has_peer:
        _pool = paths["feeder_pool"].astype(float)
        _yours = float((_pool * paths["conversion_rate"].astype(float)).sum())
        _peers = float((_pool * paths[_pk].astype(float)).sum())
        _tot = float(_pool.sum()) or 1.0
        _behind = int((paths["mobility_gap"].astype(float) > 0).sum())
        _mine_heads = se._horizon_pathway_supply(cfg, paths)
        _peer_heads = se._horizon_pathway_supply(
            cfg, paths.assign(conversion_rate=paths[_pk]))
        _line = (
            f"**Internal mobility vs peers.** Weighted across these feeders you "
            f"move **{_yours / _tot * 100:.1f}%** of people into {target} "
            f"against a peer rate of **{_peers / _tot * 100:.1f}%** — "
            f"{_behind} of {len(paths)} feeder roles sit behind peers.")
        if _peer_heads > _mine_heads:
            _line += (
                f" Converting at peer rates would put **~{_peer_heads:,.0f}** "
                f"heads in reach over {horizon} years instead of "
                f"~{_mine_heads:,.0f} — the same people, moving more often, so "
                "it is the cheapest supply on the page.")
        else:
            _line += (" You already convert at or above the peer rate here, so "
                      "treat this supply as earned rather than headroom.")
        st.markdown(_line)
    elif _pk:
        st.caption(
            "Peer mobility rates are missing from this pull, so the "
            "**behind peers** column is blank — re-save the snapshot from live "
            "to fill it.")

    if "feasibility" in paths.columns:
        _legend_pills([
            ("high", FEAS_COLORS["high"], "frequent move, close overlap"),
            ("med", FEAS_COLORS["med"], "workable with a programme"),
            ("low", FEAS_COLORS["low"], "far, or a real pay cut"),
        ])

    # Same annotation the scenario sizes on, so supply heads here are the heads
    # in the plan — run_scenario works on its own copy, which left the column
    # missing from this table entirely.
    _pv = se.annotate_pathway_supply(cfg, paths)
    # Recover move counts on older snapshots that only froze conversion.
    if "transition_wt" not in _pv.columns and {
            "feeder_pool", "conversion_rate"}.issubset(_pv.columns):
        _pv = _pv.copy()
        _pv["transition_wt"] = (
            _pv["feeder_pool"].astype(float)
            * _pv["conversion_rate"].astype(float))
    # One verdict column instead of your-rate / peer-rate / gap side by side;
    # the paragraph above carries the underlying shares.
    if "mobility_gap" in _pv.columns and _has_peer:
        _gap = _pv["mobility_gap"].astype(float)
        _pv["behind_peers"] = [
            ("" if pd.isna(g) else
             (f"✓ +{g * 100:.1f}pp" if g > 0 else f"✗ −{abs(g) * 100:.1f}pp"))
            for g in _gap]
    pcols = [c for c in [
        "source_role", "feasibility", path_overlap_col, "wage_gap",
        "behind_peers", "supply_heads", "category", "feeder_pool"]
        if c in _pv.columns]
    # Volume mode: surface absolute moves next to supply.
    if (cfg.get("pathway_rank_by") == "volume"
            and "transition_wt" in _pv.columns):
        pcols = [c for c in pcols]
        if "supply_heads" in pcols:
            i = pcols.index("supply_heads")
            pcols.insert(i, "transition_wt")
        else:
            pcols.append("transition_wt")
    col_cfg = {
        "feasibility": st.column_config.TextColumn(
            _i("feasibility"),
            help="How movable this pathway is: blends how often the move "
                 "already happens, how much the work overlaps, and whether it "
                 "means a pay cut."),
        "source_role": st.column_config.TextColumn("role"),
        "feeder_pool": pct("people", format="%.0f",
                           help="headcount in this role today"),
        "behind_peers": st.column_config.TextColumn(
            _i("behind peers"),
            help="✓ = peers move this role into the target more often than you "
                 "do, so the shortfall is yours to close. ✗ = you already match "
                 "or beat them. Figure is the gap in percentage points."),
        "wage_gap": pct(
            _i("wage gap"), format="percent",
            help="Median pay in the feeder role vs the destination "
                 "(source ÷ target − 1). Positive = people would take a "
                 "pay cut to move — harder to sell, and feasibility "
                 "penalizes it. Negative = a raise into the target — "
                 "easier. Built from median total compensation on "
                 "current positions in each role."),
        "transition_wt": pct(
            _i("moves"), format="%.1f",
            help="Weighted observed moves into the target over the pathway "
                 "window (same count behind conversion rate)."),
        "supply_heads": pct("supply heads", format="%.0f",
                            help="pool × conversion — this is what "
                                 "feeds the plan"),
    }
    if (path_overlap_col in _pv.columns
            and pd.api.types.is_numeric_dtype(_pv[path_overlap_col])):
        col_cfg[path_overlap_col] = pct(
            "overlap", format="percent", help=path_overlap_help)
    _disp_n = int(cfg.get("pathway_display_n", 15))
    _pv = sre.sort_pathways(_pv, cfg, display_n=_disp_n)
    _pv = _pv[[c for c in pcols if c in _pv.columns]] if pcols else _pv
    st.dataframe(
        _tint(_pv, ("feasibility", FEAS_COLORS), ("category", CAT_COLORS)),
        width="stretch", hide_index=True, column_config=col_cfg)
    _rank_cap = (
        "Selected by move volume (absolute observed moves)"
        if cfg.get("pathway_rank_by") == "volume" else
        "Selected by conversion rate (moves ÷ current feeder headcount)")
    st.caption(
        f"{_rank_cap}, then ordered feasibility high → low. "
        f"Showing top {_disp_n}. "
        "Plan supply uses the full candidate set (top by rate ∪ top by "
        "volume). "
        f"Roles are counted, not {_noun}: each person sits in exactly one "
        "role, so these add up cleanly into the plan. **category** is how the "
        "feeder role itself is changing — transforming and at-risk people are "
        "the most available to move.")

with tabs["Scenario"]:
    # "Capability" is skills language — a role target is a staffing question.
    st.subheader(f"{'Capability gap' if is_skills else 'Staffing gap'} — {target}")
    st.markdown(
        "**net need = growth + replacement (after retention).** Growth is the "
        "target you set (fixed %) or the heads to reach peer-average share "
        "(gap). Replacement uses the assumed-improved attrition rate — "
        f"retention shrinks the hole before build/buy. Internal supply comes "
        f"from **{_supply_tab}** (capped at net need); what's left is external "
        "hiring.")
    l, r = st.columns([3, 2])
    with l:
        _g = float(sc["growth_need"])
        _rep = float(sc["replacement_need"])
        _int = float(sc["internal_supply"])
        _ext = float(sc["external_need"])
        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total",
                     "relative", "relative", "total"],
            x=["Growth", "Replacement", "Net need",
               _BUILD_V, "Hire", "Filled"],
            y=[_g, _rep, None, -_int, -_ext, None],
            text=[f"+{_g:,.0f}", f"+{_rep:,.0f}", f"{_g + _rep:,.0f}",
                  f"−{_int:,.0f}", f"−{_ext:,.0f}", "0"],
            textposition="outside",
            connector=dict(line=dict(color="#C8CDD3", width=1)),
            increasing=dict(marker=dict(color=C_CORAL)),
            decreasing=dict(marker=dict(color=C_GREEN)),
            totals=dict(marker=dict(color=C_BLUE))))
        wf.update_layout(
            height=330, margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            yaxis=dict(title="positions", gridcolor="#EEF1F4",
                       zerolinecolor="#C8CDD3"),
            xaxis=dict(tickfont=dict(size=12)),
            font=dict(family="TWK Lausanne Pan, Segoe UI, sans-serif",
                      size=12, color=TEXT),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(wf, width="stretch",
                        config={"displayModeBar": False})
        st.caption(
            f"Retention already shrank this: at "
            f"{_improv*100:.0f}% lower attrition the hole is "
            f"{round(sc['net_need']):,} rather than {_base_net:,.0f}. "
            f"The {round(_int):,} {_BUILD_PAST.lower()} come from the "
            f"**{_supply_tab}** tab.")
    with r:
        st.markdown(
            f"**Current attrition:** {_attr0*100:.1f}%/yr\n\n"
            f"**Assumed-improved rate:** {_attr1*100:.1f}%/yr "
            f"(−{_improv*100:.0f}%)\n\n"
            f"**Heads retained:** ~{round(sc['heads_saved_by_retention']):,} "
            f"over {horizon}y\n\n"
            f"**Growth need:** {round(sc['growth_need']):,}\n\n"
            f"**Replacement:** {round(sc['replacement_need']):,} "
            f"(at {_attr1*100:.1f}% × {horizon}y)\n\n"
            f"**Net need:** {round(sc['net_need']):,} "
            f"(baseline {_base_net:,.0f})")
        st.caption(
            f"A program cutting attrition by {_improv*100:.0f}% "
            f"(to {_attr1*100:.1f}%) retains "
            f"~{round(sc['heads_saved_by_retention']):,} and shrinks net need "
            f"from {_base_net:,.0f} to {round(sc['net_need']):,}."
        )

    st.subheader("Competitive outflows")
    st.markdown(
        f"Where {company}'s departing **{target}** talent goes "
        f"(last {cfg.get('outflow_years', 2)}y). Thicker ribbons = more of "
        "your people landing there — the retention what-if is aimed at the "
        "rivals doing the pulling.")
    of = outflows.copy() if outflows is not None else pd.DataFrame()
    if "dest_company" in of.columns and len(of):
        of = of.sort_values("outflow_wt", ascending=False).head(10)
        dests = of["dest_company"].tolist()
        vals = of["outflow_wt"].tolist()
        n = len(dests)
        sankey = go.Figure(go.Sankey(
            arrangement="snap",
            textfont=dict(
                family="TWK Lausanne Pan, Segoe UI, sans-serif",
                size=13, color=TEXT),
            node=dict(
                label=[company] + dests,
                color=[C_BLUE] * (n + 1),
                pad=16, thickness=18,
                line=dict(color="#FFFFFF", width=1),
                x=[0.001] + [0.999] * n,
                align="left"),
            link=dict(
                source=[0] * n,
                target=list(range(1, n + 1)),
                value=vals,
                color="rgba(99, 162, 217, 0.35)")))
        sankey.update_layout(
            height=max(360, 60 + 34 * n),
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(family="TWK Lausanne Pan, Segoe UI, sans-serif",
                      size=13, color=TEXT),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(sankey, width="stretch",
                        config={"displayModeBar": False})
    else:
        st.caption(
            "No outflow data for this target yet — available live, and in "
            "snapshots that were saved with outflows.")

with tabs["Plan"]:
    st.subheader("Integrated plan")
    st.caption(
        "Priority by return: retain → "
        f"{_BUILD_V.lower()} → hire.")

    # Structured readout instead of one dense paragraph of numbers.
    _g_need = round(float(sc.get("growth_need", 0) or 0))
    _r_need = round(float(sc.get("replacement_need", 0) or 0))
    _n_need = round(float(sc.get("net_need", 0) or 0))
    _int = round(float(sc.get("internal_supply", 0) or 0))
    _ext = round(float(sc.get("external_need", 0) or 0))
    _ret = round(float(sc.get("heads_saved_by_retention", 0) or 0))
    _basis = sc.get("growth_basis", "fixed")
    if _basis == "gap":
        _ask = (f"Close the peer gap on **{target}**"
                + (f" (you're at {sc['index_ratio']*100:.0f}% of peer share)"
                   if sc.get("index_ratio") is not None
                   and pd.notna(sc.get("index_ratio")) else ""))
    else:
        _ask = (f"Grow **{target}** by "
                f"**{int(sc.get('growth_target_pct', growth_target)*100)}%**")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("**1 · The ask**")
        st.markdown(_ask)
        st.markdown(
            f"- Growth: **{_g_need:,}** heads  \n"
            f"- Replacement after retention: **{_r_need:,}**  \n"
            f"- Net need: **{_n_need:,}** over {horizon} years")
    with p2:
        st.markdown(f"**2 · How you fill it**")
        st.markdown(
            f"- Retain: **{_ret:,}** (attrition you avoid)  \n"
            f"- {_BUILD_V}: **{_int:,}** from internal pathways  \n"
            f"- Hire: **{_ext:,}** externally")
        if _ext and (tight not in (None, ["—"]) or avail not in (None, ["—"])):
            _t = ", ".join(x for x in (tight or [])[:2] if x and x != "—") or "—"
            _a = ", ".join(x for x in (avail or [])[:2] if x and x != "—") or "—"
            st.caption(f"Talent map: lean into {_a}; expect competition in {_t}.")
    with p3:
        st.markdown("**3 · Cost**")
        st.markdown(
            f"- Plan: **${cost/1e6:,.1f}M**  \n"
            f"- Buy-everything: **${naive/1e6:,.1f}M**  \n"
            f"- Save: **${(naive-cost)/1e6:,.1f}M**")
        _cost_cap = f"Basis: {sc.get('cost_basis', 'flat')}"
        if sc.get("target_median_comp"):
            _cost_cap += f" · target median ${sc['target_median_comp']:,.0f}"
        if is_skills and sc.get("reskill_ramp_cost"):
            _rm = sc.get("skill_ramp_months")
            _cost_cap += (
                f" · reskill includes {_rm:.0f}mo ramp opportunity "
                f"(${sc['reskill_ramp_cost']/1e6:,.2f}M)"
                if _rm is not None else
                f" · reskill includes ramp "
                f"(${sc['reskill_ramp_cost']/1e6:,.2f}M)")
        st.caption(_cost_cap)

    st.divider()
    c = st.columns(4)
    _reskill_help = None
    if is_skills and sc.get("reskill_ramp_cost"):
        _reskill_help = (
            f"Direct ${sc.get('reskill_direct_cost', 0)/1e6:,.2f}M + "
            f"ramp ${sc['reskill_ramp_cost']/1e6:,.2f}M"
            + (f" ({sc['skill_ramp_months']:.0f} mo × "
               f"{cfg.get('cost_reskill_ramp_pct', 0.25):.0%} of monthly "
               f"comp × supply)"
               if sc.get("skill_ramp_months") is not None else ""))
    c[0].metric(_BUILD_V, f"${sc.get('reskill_cost', 0)/1e6:,.1f}M",
                help=_reskill_help)
    c[1].metric("External hire", f"${sc.get('hire_cost', 0)/1e6:,.1f}M")
    c[2].metric("Retention", f"${sc.get('retain_cost', 0)/1e6:,.1f}M")
    c[3].metric("vs buy-everything", f"${naive/1e6:,.1f}M",
                delta=f"save ${(naive-cost)/1e6:,.1f}M")
    with st.expander("Full narrative"):
        st.write(close)
