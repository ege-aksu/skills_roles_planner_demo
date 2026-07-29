"""Skills + Roles hiring planner — simplified dual-angle engine.

Wraps ``skills_engine`` for the shared scenario / cost math, and adds:
- ``role_k10`` multi-select filter on the skill radar
- top-emerging shortlist (no hard under-index / supply gates; badges instead)
- role peer/company workforce shares for a real gap basis
- skill adjacency (bridge / teach) for the skills angle
- role pathways via **activity** Jaccard (same shape as skill overlap)
- role population + scenario entry points

Tables (activities mirror skills):
- ``service_pipelines.output_current.individual_activities``
- ``service_pipelines.global_ref.custom_activity_taxonomy_v3_overall_latest``
"""

from __future__ import annotations

import json
import os
import re
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

import skills_engine as se

# Re-export common pieces so the app/notebook can import one module.
CONFIG = dict(se.CONFIG)
CONFIG.update({
    "activity_level": "activity_k1500",
    "role_k10_filter": None,  # None / [] = all families; else list of role_k10
    # Live skill filter: fraction of skill HC that must sit in selected families.
    "role_k10_skill_share": 0.30,
    "top_emerging_n": 8,      # shortlist shown in UI
    # Skill adjacency (skills angle only).
    "adjacency_min_holders": 25,
    "adjacency_min_lift": 1.2,
    "adjacency_top_n": 12,
})

# Demo roster shared by the app and the batch snapshot script.
# name → ultimate-parent rcid (None = name only, synthetic mode).
DEMO_COMPANIES = {
    "Lockheed Martin": int(se.CONFIG.get("company_rcid") or 20921805),
    "JPMorgan Chase": 543448,
    "New Balance": 8027943,
    "Autodesk": 1374849,  # US mid-size design software — strong skill/role demo
    "Wayfair": 7969203,   # mid-size e-commerce; richer mobility than New Balance
    "General Dynamics": None,
    "RTX": None,
    "Northrop Grumman": None,
    "Boeing": None,
}

ROLE_K10_FAMILIES = [
    "Education",
    "Engineering",
    "Equipment Operator",
    "Financial Services",
    "Healthcare",
    "Hospitality Staff",
    "Information Technology",
    "Marketing",
    "Miscellaneous",
    "Office Support",
    "Sales",
]


def _company_rcid(cfg):
    """Normalize company_rcid to int for SQL; None if unset."""
    raw = cfg.get("company_rcid")
    if raw is None or raw == "":
        return None
    return int(raw)


def list_role_k10_families(cfg=None):
    """Occupation families for the multi-select (live from taxonomy when possible)."""
    cfg = cfg or CONFIG
    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        return list(ROLE_K10_FAMILIES)
    q = """
    SELECT DISTINCT role_k10
    FROM model_jobembedding.v3_reference.role_taxonomy_current
    WHERE taxonomy_id = 0
      AND role_k10 IS NOT NULL
      AND LOWER(role_k10) NOT IN ('unknown', 'empty', 'retired')
    ORDER BY 1
    """
    try:
        df = se._lower_cols(se._sf(cfg).load_df(q))
        vals = [str(x) for x in df["role_k10"].tolist() if str(x).strip()]
        return vals or list(ROLE_K10_FAMILIES)
    except Exception:
        return list(ROLE_K10_FAMILIES)


def load_role_k10_map(cfg):
    """role_k1500 → role_k10 from Snowflake taxonomy (live) or synthetic map."""
    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        return _synthetic_role_k10_map()
    q = """
    SELECT DISTINCT role_k1500 AS role, role_k10
    FROM model_jobembedding.v3_reference.role_taxonomy_current
    WHERE taxonomy_id = 0
      AND role_k1500 IS NOT NULL
      AND LOWER(role_k1500) NOT IN ('unknown', 'retired', 'on leave', 'empty')
      AND role_k10 IS NOT NULL
    """
    df = se._lower_cols(se._sf(cfg).load_df(q))
    return dict(zip(df["role"].astype(str), df["role_k10"].astype(str)))


def _synthetic_role_k10_map():
    return {
        "IT Engineer": "Information Technology",
        "Software Engineering": "Information Technology",
        "Engineering Managers": "Engineering",
        "Aerospace Engineer": "Engineering",
        "Senior Engineer": "Engineering",
        "Domain Software Engineer": "Information Technology",
        "Mechanical Engineering": "Engineering",
        "Industrial Quality Engineer": "Engineering",
        "Manufacturing Engineer": "Engineering",
        "Senior Program Manager": "Office Support",
        "Project Engineer": "Engineering",
        "Aircraft Mechanic": "Equipment Operator",
        "Electrical Power Engineer": "Engineering",
        "Finance Manager": "Financial Services",
        "Systems Analyst": "Information Technology",
        "Optical Engineer": "Engineering",
        "Test Engineer": "Engineering",
        "Production Planner": "Office Support",
        "Security Architect": "Information Technology",
        "Assembly Technician": "Equipment Operator",
        "Hardware Engineer": "Engineering",
        "Administrative Assistant": "Office Support",
        "System Administrator": "Information Technology",
        "Project Planner": "Office Support",
        "Aerospace Manager": "Engineering",
        "Technical Reporting Analyst": "Engineering",
        "Process Engineer": "Engineering",
        "Site Engineer": "Engineering",
        "Systems Design Engineers": "Engineering",
        "Category Manager": "Office Support",
        "Operations Engineering Specialist": "Engineering",
        "Thermal Engineer": "Engineering",
        "Product Design Engineer": "Engineering",
        "Quality Analyst": "Engineering",
        "Engineering Technician": "Engineering",
        "Solutions Engineer": "Engineering",
        "Manufacturing Excellence Engineer": "Engineering",
        "Data Analyst": "Information Technology",
        "Software Engineer": "Information Technology",
        "Systems Engineer": "Engineering",
        "Quality Engineer": "Engineering",
        "Project Manager": "Office Support",
        "Operations Manager": "Office Support",
        "Financial Analyst": "Financial Services",
    }

# Override with SKILLS_ROLES_SNAP_ROOT to demo from an alternate snapshot set.
SNAP_ROOT = Path(os.environ.get("SKILLS_ROLES_SNAP_ROOT")
                 or Path(__file__).parent / "demo_snapshot_skills_roles")

_FREQ = se._FREQ if hasattr(se, "_FREQ") else {
    "low": 0.2, "low-med": 0.45, "med": 0.65, "med-high": 0.8, "high": 1.0}
_OVL = se._OVL if hasattr(se, "_OVL") else {
    "low": 0.2, "med": 0.5, "med-high": 0.75, "high": 1.0}


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:80]


# ---------------------------------------------------------------- taxonomy SQL

def _activity_taxonomy(cfg):
    level = cfg.get("activity_level", "activity_k1500")
    col = f"{level}_name"
    return f"""
    SELECT activity_v3_id, {col} AS activity
    FROM service_pipelines.global_ref.custom_activity_taxonomy_v3_overall_latest
    WHERE taxonomy_name = 'default'
      AND {col} IS NOT NULL
"""


def _role_taxonomy_with_k10():
    return """
    SELECT role_v3_id, role_k1500 AS role, role_k10
    FROM model_jobembedding.v3_reference.role_taxonomy_current
    WHERE taxonomy_id = 0
      AND role_k1500 IS NOT NULL
      AND LOWER(role_k1500) NOT IN ('unknown', 'retired', 'on leave', 'empty')
"""


def _norm_role_k10_filter(cfg):
    raw = cfg.get("role_k10_filter")
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    out = [str(x).strip() for x in raw if str(x).strip()
           and str(x).strip().lower() not in ("unknown", "all")]
    return out or None


# ---------------------------------------------------------------- skill angle

def build_skill_radar(cfg):
    """Full skill radar, optionally filtered to selected role_k10 families."""
    radar = se.build_radar(cfg)
    return filter_radar_by_role_k10(cfg, radar)


def filter_radar_by_role_k10(cfg, radar):
    """Keep skills concentrated in selected role_k10 families.

    Not "anyone in this family ever holds the skill" (that keeps almost
    everything). Live: skill HC in the selected families must clear a floor
    **and** be ≥ ``role_k10_skill_share`` of that skill's total HC.
    Synthetic: each skill is assigned to one primary family (plus a small
    explicit map for demo staples).
    """
    families = _norm_role_k10_filter(cfg)
    if not families or radar is None or getattr(radar, "empty", True):
        return radar

    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        return _filter_radar_by_role_k10_synthetic(radar, families)

    peers = se._resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    peer_sql = se._sql_quote_list(peers["peer_rcids"]) or str(company_rcid)
    fam_sql = se._sql_quote_list_str(families)
    country = cfg.get("country", "United States")
    min_hc = max(20, int(cfg.get("min_skill_headcount", 500)) // 10)
    min_share = float(cfg.get("role_k10_skill_share", 0.30))

    q = f"""
    WITH skill_names AS ({se._skill_taxonomy(cfg)}),
    roles AS ({_role_taxonomy_with_k10()}),
    peer_rcids AS (
        SELECT {company_rcid} AS rcid
        UNION ALL
        SELECT value::INT AS rcid FROM TABLE(SPLIT_TO_TABLE('{peer_sql}', ','))
    ),
    skill_family AS (
        SELECT
            sn.skill,
            r.role_k10,
            SUM(COALESCE(p.weight_v2_1, 1)) AS wt
        FROM service_pipelines.output_current.individual_position p
        JOIN peer_rcids pr ON p.ultimate_parent_rcid = pr.rcid
        JOIN roles r ON p.role_v3_id = r.role_v3_id
        JOIN service_pipelines.output_current.individual_skills sk
          ON p.user_id = sk.user_id
        JOIN skill_names sn ON sk.skill_v3_id = sn.skill_v3_id
        WHERE p.country = '{country}'
          AND p.enddate_primary IS NULL
          AND {se._pos_filter(cfg)}
          AND r.role_k10 IS NOT NULL
          AND LOWER(r.role_k10) NOT IN ('unknown', 'empty')
        GROUP BY 1, 2
    ),
    skill_tot AS (
        SELECT skill, SUM(wt) AS tot FROM skill_family GROUP BY 1
    ),
    keep AS (
        SELECT
            sf.skill,
            SUM(sf.wt) AS fam_wt,
            MAX(st.tot) AS tot
        FROM skill_family sf
        JOIN skill_tot st ON sf.skill = st.skill
        WHERE sf.role_k10 IN ({fam_sql})
        GROUP BY sf.skill
        HAVING SUM(sf.wt) >= {min_hc}
           AND SUM(sf.wt) / NULLIF(MAX(st.tot), 0) >= {min_share}
    )
    SELECT skill FROM keep
    """
    try:
        keep = set(se._lower_cols(se._sf(cfg).load_df(q))["skill"].astype(str))
    except Exception:
        return radar
    if not keep:
        return radar.iloc[0:0].copy()
    out = radar[radar["skill"].astype(str).isin(keep)].copy()
    return out.reset_index(drop=True)


def _skill_primary_role_k10(skill: str) -> str:
    """Deterministic primary family for offline demos (covers full radar)."""
    explicit = {
        "Data Analysis": "Information Technology",
        "Energy Economics": "Financial Services",
        "Process": "Engineering",
        "Lean Quality": "Engineering",
        "Engineering Project Delivery": "Engineering",
        "Production Engineering": "Engineering",
        "Manufacturing": "Engineering",
        "Production": "Engineering",
        "Engineering Simulation": "Engineering",
        "Windows Systems": "Information Technology",
        "Graphics Programming": "Information Technology",
        "Embedded Systems Engineering": "Engineering",
        "Electrical Design": "Engineering",
        "Electrical Engineering": "Engineering",
        "Vehicle Engineering": "Engineering",
        "CAD Drafting": "Engineering",
        "Software Testing": "Information Technology",
        "Artificial Intelligence": "Information Technology",
        "Language Technology": "Information Technology",
        "Data Modeling": "Information Technology",
    }
    if skill in explicit:
        return explicit[skill]
    fams = ROLE_K10_FAMILIES
    return fams[abs(hash(skill)) % len(fams)]


def _filter_radar_by_role_k10_synthetic(radar, families):
    """Offline: keep skills whose primary role_k10 is in the selection."""
    fam = set(families)
    if "role_k10" in radar.columns:
        return radar[radar["role_k10"].astype(str).isin(fam)].reset_index(drop=True)
    keep = [
        sk for sk in radar["skill"].astype(str)
        if _skill_primary_role_k10(sk) in fam
    ]
    return radar[radar["skill"].astype(str).isin(keep)].reset_index(drop=True)


def annotate_skill_radar_families(cfg, radar):
    """Attach each skill's primary ``role_k10`` (highest peer HC share).

    Frozen onto snapshots so the occupation-family filter works offline later.
    """
    if radar is None or getattr(radar, "empty", True):
        return radar
    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        out = radar.copy()
        if "role_k10" not in out.columns:
            out["role_k10"] = out["skill"].astype(str).map(_skill_primary_role_k10)
        return out
    peers = se._resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    peer_sql = se._sql_quote_list(peers["peer_rcids"]) or str(company_rcid)
    country = cfg.get("country", "United States")
    q = f"""
    WITH skill_names AS ({se._skill_taxonomy(cfg)}),
    roles AS ({_role_taxonomy_with_k10()}),
    peer_rcids AS (
        SELECT {company_rcid} AS rcid
        UNION ALL
        SELECT value::INT AS rcid FROM TABLE(SPLIT_TO_TABLE('{peer_sql}', ','))
    ),
    skill_family AS (
        SELECT sn.skill, r.role_k10, SUM(COALESCE(p.weight_v2_1, 1)) AS wt
        FROM service_pipelines.output_current.individual_position p
        JOIN peer_rcids pr ON p.ultimate_parent_rcid = pr.rcid
        JOIN roles r ON p.role_v3_id = r.role_v3_id
        JOIN service_pipelines.output_current.individual_skills sk
          ON p.user_id = sk.user_id
        JOIN skill_names sn ON sk.skill_v3_id = sn.skill_v3_id
        WHERE p.country = '{country}'
          AND p.enddate_primary IS NULL
          AND {se._pos_filter(cfg)}
          AND r.role_k10 IS NOT NULL
          AND LOWER(r.role_k10) NOT IN ('unknown', 'empty')
        GROUP BY 1, 2
    ),
    ranked AS (
        SELECT skill, role_k10,
               ROW_NUMBER() OVER (PARTITION BY skill ORDER BY wt DESC) AS rn
        FROM skill_family
    )
    SELECT skill, role_k10 FROM ranked WHERE rn = 1
    """
    try:
        fam = se._lower_cols(se._sf(cfg).load_df(q))
    except Exception:
        out = radar.copy()
        if "role_k10" not in out.columns:
            out["role_k10"] = out["skill"].astype(str).map(_skill_primary_role_k10)
        return out
    fmap = dict(zip(fam["skill"].astype(str), fam["role_k10"].astype(str)))
    out = radar.copy()
    out["role_k10"] = out["skill"].astype(str).map(fmap).fillna("Miscellaneous")
    return out


def _badge_mark(ok):
    return "✓" if bool(ok) else "✗"


def role_pathway_supply_summary(cfg, target_role):
    """Feeder count + horizon-scaled supply for role shortlist badges."""
    paths = load_role_activity_pathways(cfg, target_role)
    if paths is None or getattr(paths, "empty", True):
        return {"feeder_roles": 0, "supply_heads": 0.0, "pathways": pd.DataFrame()}
    # Keep low-freq (same mid-size rationale as skill pathways / build_role_pathways).
    usable = paths.copy()
    return {
        "feeder_roles": int(len(usable)),
        "supply_heads": float(se._horizon_pathway_supply(cfg, usable)),
        "pathways": usable,
    }


def enrich_funnel_badges(cfg, funnel, radar, angle="skills"):
    """Add behind-peers / buildable-supply badges (info, not hard filters).

    Soft-sorts rows that clear both checks first so the shortlist still
    surfaces the defensible picks without hiding the rest.
    """
    if funnel is None or getattr(funnel, "empty", True):
        return funnel
    out = funnel.copy()
    name_col = "skill" if angle == "skills" else "role"
    if name_col not in out.columns:
        return out

    min_feeders = int(cfg.get("min_feeder_roles", 1))
    min_supply = float(cfg.get("min_internal_supply", 50))
    behind = {}
    if radar is not None and not getattr(radar, "empty", True):
        rdf = radar.copy()
        if {"peer_share", "company_share"} <= set(rdf.columns):
            if "under_index" not in rdf.columns:
                rdf["under_index"] = (
                    rdf["peer_share"].astype(float)
                    - rdf["company_share"].astype(float))
            if "index_ratio" not in rdf.columns:
                rdf["index_ratio"] = [
                    se._index_ratio(c, p)
                    for c, p in zip(rdf["company_share"], rdf["peer_share"])]
            mask = se._under_indexed_mask(rdf, cfg)
            key = "skill" if "skill" in rdf.columns else "role"
            if key in rdf.columns:
                behind = dict(zip(rdf[key].astype(str), mask.astype(bool)))

    behind_list, build_list, supply_list = [], [], []
    for name in out[name_col].astype(str):
        is_behind = bool(behind.get(name, False))
        if angle == "skills":
            summary = se.pathway_supply_summary(cfg, name)
        else:
            summary = role_pathway_supply_summary(cfg, name)
        n_feeders = int(summary["feeder_roles"])
        supply = float(summary["supply_heads"])
        is_buildable = (n_feeders >= min_feeders) and (supply >= min_supply)
        behind_list.append(_badge_mark(is_behind))
        build_list.append(_badge_mark(is_buildable))
        supply_list.append(round(supply, 1))

    out["behind_peers"] = behind_list
    out["buildable_supply"] = build_list
    out["supply_heads"] = supply_list
    # Soft-sort: both ✓ first, then behind-only, then rest — keep relative order.
    score = (
        (out["behind_peers"] == "✓").astype(int) * 2
        + (out["buildable_supply"] == "✓").astype(int))
    out = out.assign(_badge_score=score).sort_values(
        ["_badge_score", "rank"] if "rank" in out.columns else ["_badge_score"],
        ascending=[False, True] if "rank" in out.columns else [False],
    ).drop(columns="_badge_score").reset_index(drop=True)
    if "rank" in out.columns:
        out["rank"] = np.arange(1, len(out) + 1)
    return out


def select_top_emerging_skill(cfg, radar):
    """Default target = highest-momentum emerging skill (badges, not hard gates)."""
    n = int(cfg.get("top_emerging_n", 8))
    empty = pd.DataFrame(columns=[
        "skill", "momentum", "bucket", "rank",
        "behind_peers", "buildable_supply", "supply_heads"])
    if radar is None or getattr(radar, "empty", True):
        return cfg.get("force_target_skill"), empty

    emerg = radar[radar["bucket"] == "emerging"].copy()
    if emerg.empty:
        # fall back to nascent, then highest momentum overall
        emerg = radar[radar["bucket"] == "nascent"].copy()
    if emerg.empty:
        emerg = radar.copy()

    emerg = emerg.sort_values("momentum", ascending=False).head(n).reset_index(drop=True)
    emerg["rank"] = np.arange(1, len(emerg) + 1)
    funnel = emerg[["skill", "momentum", "bucket", "rank"]].copy()
    funnel = enrich_funnel_badges(cfg, funnel, radar, angle="skills")
    forced = cfg.get("force_target_skill")
    if forced:
        chosen = forced
    elif (not cfg.get("use_snowflake") or _company_rcid(cfg) is None) and len(funnel):
        # Synthetic pathways are frozen on Data Analysis — prefer it when present.
        if "Data Analysis" in set(funnel["skill"].astype(str)):
            chosen = "Data Analysis"
        else:
            chosen = str(funnel.iloc[0]["skill"])
    else:
        chosen = str(funnel.iloc[0]["skill"]) if len(funnel) else None
    return chosen, funnel


def build_skill_pathways(cfg, target_skill, role_categories=None):
    """Same observed role pathways as the main planner; UI reframes as skill→skill.

    Kept local (not a bare ``se.build_pathways`` delegate) so Streamlit picks up
    the mid-size keep-low behavior even when ``skills_engine`` is a stale import.
    """
    df = se.load_pathways(cfg, target_skill).copy()
    if df is None or getattr(df, "empty", True):
        return df
    df = se.drop_excluded_roles(df, cfg, col="source_role")
    if df is None or getattr(df, "empty", True):
        return df
    # Keep transition_freq == "low" — feasibility haircuts thin-n (see
    # build_role_pathways / se.build_pathways).

    sources = (
        df["source_role"].dropna().astype(str).unique().tolist()
        if "source_role" in df.columns and len(df) else []
    )
    if role_categories is not None and not getattr(role_categories, "empty", True):
        have = set(role_categories["role"].astype(str))
        missing = [r for r in sources if r not in have]
        if missing:
            role_categories = se.classify_roles(
                cfg, include_roles=list(have) + missing)
    else:
        role_categories = se.classify_roles(cfg, include_roles=sources)

    if role_categories is not None and not role_categories.empty:
        cats = role_categories[["role", "category"]].drop_duplicates("role")
        df = df.merge(cats, left_on="source_role", right_on="role", how="left")
        df["category"] = df["category"].fillna("stable")
        df = df.drop(columns=["role"], errors="ignore")
    else:
        df["category"] = "stable"

    boosts = cfg.get("pathway_category_boost", se._DEFAULT_CATEGORY_BOOST)
    cat_boost = df["category"].map(boosts).fillna(0)
    f = df["transition_freq"].map(_FREQ)
    o = df["skill_overlap"].map(_OVL)
    wage_pen = 1 - (df["wage_gap"].clip(0, 0.4) / 0.4) * 0.4
    score = (0.45 * f + 0.4 * o + 0.15 * wage_pen + cat_boost).clip(0, 1.2)
    df["feasibility_score"] = score
    df["feasibility"] = np.where(
        score >= 0.75, "high", np.where(score >= 0.55, "med", "low"))
    return sort_pathways(df, cfg)


def sort_pathways(pathways, cfg, display_n=None):
    """Pick top feeders by conversion/volume, then order high→low feasibility.

    Kept local (not delegated to ``skills_engine``) so Streamlit picks up the
    display order even when ``skills_engine`` is still a stale import.
    """
    if pathways is None or getattr(pathways, "empty", True):
        return pathways
    out = pathways.copy()
    rank = str(cfg.get("pathway_rank_by", "conversion") or "conversion").lower()
    by_volume = (rank in ("volume", "moves", "transition_wt")
                 and "transition_wt" in out.columns)
    if by_volume:
        sel = ["transition_wt"] + (
            ["conversion_rate"] if "conversion_rate" in out.columns else [])
    else:
        sel = [c for c in ("conversion_rate", "feasibility_score", "transition_wt")
               if c in out.columns]
    if sel:
        out = out.sort_values(sel, ascending=[False] * len(sel))
    if display_n is not None:
        out = out.head(int(display_n))
    if "feasibility" in out.columns:
        _fo = {"high": 0, "med": 1, "low": 2}
        out = out.copy()
        out["_feas_ord"] = out["feasibility"].map(_fo).fillna(9)
        tie = (["transition_wt", "conversion_rate"] if by_volume
               else ["conversion_rate", "transition_wt"])
        tie = [c for c in tie if c in out.columns]
        out = out.sort_values(
            ["_feas_ord"] + tie,
            ascending=[True] + [False] * len(tie),
        ).drop(columns="_feas_ord")
    return out.reset_index(drop=True)


def run_skill_scenario(cfg, target_skill, pathways, radar=None, population=None):
    return se.run_scenario(cfg, target_skill, pathways, radar=radar,
                           population=population)


# ---------------------------------------------------------------- role angle

def load_role_workforce_shares(cfg, roles=None):
    """Role HC ÷ workforce HC for company vs peers (the role gap basis).

    Same share definition as the skill radar, keyed by ``role_k1500``.
    """
    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        return _load_role_workforce_shares_synthetic(cfg, roles=roles)

    peers = se._resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    peer_sql = se._sql_quote_list(peers["peer_rcids"]) or str(company_rcid)
    country = cfg.get("country", "United States")
    role_filter = ""
    if roles:
        role_sql = se._sql_quote_list_str([str(r) for r in roles if r])
        if role_sql:
            role_filter = f"AND r.role IN ({role_sql})"

    q = f"""
    WITH roles AS ({_role_taxonomy_with_k10()}),
    peer_rcids AS (
        SELECT {company_rcid} AS rcid, 'company' AS peer_type
        UNION ALL
        SELECT rcid, 'peer' AS peer_type
        FROM (SELECT value::INT AS rcid FROM TABLE(SPLIT_TO_TABLE('{peer_sql}', ',')))
    ),
    headcount AS (
        SELECT peer_type, role, SUM(wt) AS headcount
        FROM (
            SELECT
                pr.peer_type,
                r.role,
                p.user_id,
                COALESCE(p.weight_v2_1, 1) AS wt,
                {se._rn_latest_position("pr.peer_type, r.role, p.user_id")} AS rn
            FROM service_pipelines.output_current.individual_position p
            JOIN peer_rcids pr ON p.ultimate_parent_rcid = pr.rcid
            JOIN roles r ON p.role_v3_id = r.role_v3_id
            WHERE p.country = '{country}'
              AND {se._pos_filter(cfg)}
              AND p.enddate_primary IS NULL
              {role_filter}
        ) x
        WHERE rn = 1
        GROUP BY 1, 2
    ),
    workforce AS (
        SELECT peer_type, SUM(wt) AS total_hc
        FROM (
            SELECT
                pr.peer_type,
                p.user_id,
                COALESCE(p.weight_v2_1, 1) AS wt,
                {se._rn_latest_position("pr.peer_type, p.user_id")} AS rn
            FROM service_pipelines.output_current.individual_position p
            JOIN peer_rcids pr ON p.ultimate_parent_rcid = pr.rcid
            WHERE p.country = '{country}'
              AND {se._pos_filter(cfg)}
              AND p.enddate_primary IS NULL
        ) x
        WHERE rn = 1
        GROUP BY 1
    )
    SELECT
        h.role,
        SUM(CASE WHEN h.peer_type = 'peer' THEN h.headcount ELSE 0 END) AS peer_headcount,
        SUM(CASE WHEN h.peer_type = 'company' THEN h.headcount ELSE 0 END) AS company_headcount,
        SUM(CASE WHEN h.peer_type = 'peer' THEN h.headcount ELSE 0 END)::FLOAT
            / NULLIF((SELECT total_hc FROM workforce WHERE peer_type = 'peer'), 0) AS peer_share,
        SUM(CASE WHEN h.peer_type = 'company' THEN h.headcount ELSE 0 END)::FLOAT
            / NULLIF((SELECT total_hc FROM workforce WHERE peer_type = 'company'), 0) AS company_share
    FROM headcount h
    GROUP BY h.role
    """
    return se._lower_cols(se._sf(cfg).load_df(q))


def _load_role_workforce_shares_synthetic(cfg, roles=None):
    """Offline shares from classified role HC; peers slightly ahead on hot roles."""
    cats = se.classify_roles(cfg, include_roles=list(roles) if roles else None)
    if cats is None or getattr(cats, "empty", True):
        return pd.DataFrame(columns=[
            "role", "peer_headcount", "company_headcount",
            "peer_share", "company_share"])
    df = cats[["role", "headcount"]].copy()
    if roles:
        want = set(str(r) for r in roles)
        df = df[df["role"].astype(str).isin(want)]
    total = float(df["headcount"].sum()) or 1.0
    rows = []
    for _, r in df.iterrows():
        name = str(r["role"])
        co_hc = float(r["headcount"])
        co_share = co_hc / total
        # Deterministic peer premium: hot / expanding-ish names slightly leaner at company.
        h = abs(hash(name)) % 1000
        peer_mult = 1.05 + (h % 40) / 100.0  # 1.05–1.44× company share
        # A few large roles sit at/above parity so badges aren't all ✓.
        if h % 7 == 0:
            peer_mult = 0.92 + (h % 10) / 100.0
        peer_share = min(co_share * peer_mult, 0.35)
        rows.append({
            "role": name,
            "company_headcount": co_hc,
            "peer_headcount": co_hc * peer_mult,
            "company_share": co_share,
            "peer_share": peer_share,
        })
    return pd.DataFrame(rows)


def load_role_market_demand(cfg, roles=None):
    """YoY change in each role's share of *peer* job postings.

    External demand — parallel to skill ``peer_postings_share_growth``. Kept
    separate from ``hiring_growth`` (the company's own inflow into the role)
    so Explore can show both without blending them into one ambiguous
    "growth" number.
    """
    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        return _load_role_market_demand_synthetic(roles)
    peers = se._resolve_peer_rcids(cfg)
    peer_sql = se._sql_quote_list(peers["peer_rcids"])
    if not peer_sql:
        return _load_role_market_demand_synthetic(roles)
    country = cfg.get("country", "United States")
    recent_m = int(cfg.get("recent_months", 12))
    prior_m = int(cfg.get("prior_months", 12))
    role_filter = ""
    if roles:
        role_sql = se._sql_quote_list_str([str(r) for r in roles])
        if role_sql:
            role_filter = f"AND r.role_k1500 IN ({role_sql})"
    q = f"""
    WITH peer_rcids AS (
        SELECT value::INT AS rcid
        FROM TABLE(SPLIT_TO_TABLE('{peer_sql}', ','))
    ),
    roles AS (
        SELECT DISTINCT role_v3_id, role_k1500 AS role
        FROM model_jobembedding.v3_reference.role_taxonomy_current
        WHERE taxonomy_id = 0
          AND role_k1500 IS NOT NULL
          AND LOWER(role_k1500) NOT IN ('unknown', 'retired', 'on leave', 'empty')
          {role_filter}
    ),
    postings AS (
        SELECT
            r.role,
            CASE
                WHEN p.post_date >= DATEADD('month', -{recent_m}, CURRENT_DATE())
                    THEN 'recent'
                WHEN p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
                    THEN 'prior'
            END AS period,
            COUNT(DISTINCT p.job_id) AS n_jobs
        FROM service_pipelines.output_current.postings_unique_unified p
        JOIN peer_rcids pr ON COALESCE(p.ult_par_rcid, p.rcid) = pr.rcid
        JOIN roles r ON p.role_v3_id = r.role_v3_id
        WHERE p.country_v3 = '{country}'
          AND p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
        GROUP BY 1, 2
    ),
    period_tot AS (
        SELECT
            CASE
                WHEN p.post_date >= DATEADD('month', -{recent_m}, CURRENT_DATE())
                    THEN 'recent'
                WHEN p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
                    THEN 'prior'
            END AS period,
            COUNT(DISTINCT p.job_id) AS tot_jobs
        FROM service_pipelines.output_current.postings_unique_unified p
        JOIN peer_rcids pr ON COALESCE(p.ult_par_rcid, p.rcid) = pr.rcid
        WHERE p.country_v3 = '{country}'
          AND p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
        GROUP BY 1
    ),
    shares AS (
        SELECT
            po.role,
            po.period,
            po.n_jobs / NULLIF(t.tot_jobs, 0) AS share
        FROM postings po
        JOIN period_tot t ON po.period = t.period
        WHERE po.period IS NOT NULL
    )
    SELECT
        r.role,
        (recent.share / NULLIF(prior.share, 0)) - 1 AS postings_growth
    FROM (SELECT DISTINCT role FROM shares) r
    LEFT JOIN shares recent ON r.role = recent.role AND recent.period = 'recent'
    LEFT JOIN shares prior  ON r.role = prior.role  AND prior.period = 'prior'
    """
    try:
        return se._lower_cols(se._sf(cfg).load_df(q))
    except Exception:
        return _load_role_market_demand_synthetic(roles)


def _load_role_market_demand_synthetic(roles=None):
    """Offline stand-in: mild postings growth correlated with the role name."""
    names = list(roles) if roles is not None else [
        "Hardware Engineer", "Manufacturing Engineer", "Software Engineer",
        "Systems Analyst", "Project Engineer"]
    rows = []
    for i, role in enumerate(names):
        h = abs(hash(str(role))) % 40
        rows.append({"role": str(role), "postings_growth": (h - 10) / 100.0})
    return pd.DataFrame(rows)


def _attach_role_market_demand(roles, cfg):
    """Add ``postings_growth`` (market demand) onto a role radar frame."""
    df = roles.copy()
    if df.empty:
        return df
    if "postings_growth" in df.columns and df["postings_growth"].notna().any():
        df["postings_growth"] = pd.to_numeric(df["postings_growth"], errors="coerce")
        return df
    demand = load_role_market_demand(cfg, roles=df["role"].astype(str).tolist())
    if demand is None or getattr(demand, "empty", True):
        df["postings_growth"] = np.nan
        return df
    df = df.drop(columns=["postings_growth"], errors="ignore")
    df = df.merge(demand[["role", "postings_growth"]], on="role", how="left")
    df["postings_growth"] = pd.to_numeric(df["postings_growth"], errors="coerce")
    return df


def _attach_role_shares(roles, cfg):
    """Merge peer/company workforce shares onto a role radar frame."""
    df = roles.copy()
    if df.empty:
        return df
    if {"peer_share", "company_share"} <= set(df.columns):
        shares = None
    else:
        shares = load_role_workforce_shares(cfg, roles=df["role"].astype(str).tolist())
        if shares is not None and not shares.empty:
            keep = [c for c in [
                "role", "peer_share", "company_share",
                "peer_headcount", "company_headcount"] if c in shares.columns]
            df = df.drop(
                columns=[c for c in keep if c != "role" and c in df.columns],
                errors="ignore")
            df = df.merge(shares[keep], on="role", how="left")
    for col in ("peer_share", "company_share"):
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["under_index"] = df["peer_share"] - df["company_share"]
    df["index_ratio"] = [
        se._index_ratio(c, p) for c, p in zip(df["company_share"], df["peer_share"])]
    return df


def build_role_radar(cfg):
    """Company roles with disruption categories + workforce shares.

    Live: ``classify_roles`` from Snowflake + ``role_k10`` from taxonomy +
    peer vs company role shares. Offline: synthetic roles with static map.
    """
    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        return _build_role_radar_synthetic(cfg)
    roles = se.classify_roles(cfg)
    roles = se.drop_excluded_roles(roles, cfg)
    k10 = load_role_k10_map(cfg)
    roles = roles.copy()
    roles["role_k10"] = roles["role"].astype(str).map(k10).fillna("Miscellaneous")
    return _annotate_role_radar(roles, cfg)


def _annotate_role_radar(roles, cfg):
    df = roles.copy()
    if df.empty:
        return df
    # Map categories to a demo-friendly "emerging-like" flag for auto-pick.
    df["is_hot"] = df["category"].isin(["expanding", "transforming"])
    # Momentum proxy: hiring growth + skill-mix (normed within cohort).
    hg = se._norm(pd.to_numeric(df.get("hiring_growth", 0), errors="coerce").fillna(0))
    sm = se._norm(pd.to_numeric(df.get("skill_mix_change", 0), errors="coerce").fillna(0))
    df["momentum"] = 0.6 * hg + 0.4 * sm
    fam = _norm_role_k10_filter(cfg)
    if fam and "role_k10" in df.columns:
        df = df[df["role_k10"].isin(fam)].copy()
    df = _attach_role_shares(df, cfg)
    df = _attach_role_market_demand(df, cfg)
    return df.reset_index(drop=True)


def _build_role_radar_synthetic(cfg):
    roles = se.classify_roles(cfg)
    roles = se.drop_excluded_roles(roles, cfg)
    roles = roles.copy()
    roles["role_k10"] = (
        roles["role"].astype(str).map(_synthetic_role_k10_map()).fillna("Miscellaneous")
    )
    return _annotate_role_radar(roles, cfg)


def select_top_emerging_role(cfg, role_radar):
    """Default = hottest expanding/transforming role by momentum (badges, not gates)."""
    n = int(cfg.get("top_emerging_n", 8))
    empty = pd.DataFrame(columns=[
        "role", "momentum", "category", "rank",
        "behind_peers", "buildable_supply", "supply_heads"])
    if role_radar is None or getattr(role_radar, "empty", True):
        return cfg.get("force_target_role"), empty

    hot = role_radar[role_radar.get("is_hot", False) == True].copy()  # noqa: E712
    if hot.empty:
        hot = role_radar.copy()
    hot = hot.sort_values("momentum", ascending=False).head(n).reset_index(drop=True)
    hot["rank"] = np.arange(1, len(hot) + 1)
    funnel = hot[["role", "momentum", "category", "rank"]].copy()
    funnel = enrich_funnel_badges(cfg, funnel, role_radar, angle="roles")
    chosen = cfg.get("force_target_role") or (
        str(funnel.iloc[0]["role"]) if len(funnel) else None)
    return chosen, funnel


def load_role_population(cfg, target_role):
    """(current_hc, attrition[, hiring_rate]) for a destination role."""
    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        return _load_role_population_synthetic(cfg, target_role)

    peers = se._resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    country = cfg.get("country", "United States")
    recent_m = int(cfg.get("recent_months", 12))
    role_sql = str(target_role).replace("'", "''")

    q = f"""
    WITH roles AS ({_role_taxonomy_with_k10()}),
    months AS (
        SELECT
            DATEADD('month', -seq, DATE_TRUNC('month', CURRENT_DATE())) AS month_start,
            LAST_DAY(DATEADD('month', -seq, DATE_TRUNC('month', CURRENT_DATE()))) AS month_end
        FROM (
            SELECT ROW_NUMBER() OVER (ORDER BY SEQ4()) - 1 AS seq
            FROM TABLE(GENERATOR(ROWCOUNT => {recent_m}))
        )
    ),
    pos AS (
        SELECT
            p.user_id, p.position_id, p.startdate, p.enddate, p.enddate_primary,
            r.role, COALESCE(p.weight_v2_1, 1) AS wt
        FROM service_pipelines.output_current.individual_position p
        JOIN roles r ON p.role_v3_id = r.role_v3_id
        WHERE p.ultimate_parent_rcid = {company_rcid}
          AND p.country = '{country}'
          AND {se._pos_filter(cfg)}
          AND r.role = '{role_sql}'
          AND p.startdate IS NOT NULL
    ),
    current_hc AS (
        SELECT COALESCE(SUM(wt), 0) AS hc
        FROM (
            SELECT user_id, wt,
                   ROW_NUMBER() OVER (
                       PARTITION BY user_id
                       ORDER BY startdate DESC NULLS LAST, position_id DESC
                   ) AS rn
            FROM pos
            WHERE enddate_primary IS NULL
        ) x WHERE rn = 1
    ),
    month_hc AS (
        SELECT m.month_start, COALESCE(SUM(p.wt), 0) AS hc
        FROM months m
        LEFT JOIN pos p
          ON p.startdate <= m.month_end
         AND (p.enddate_primary IS NULL OR p.enddate_primary >= m.month_start)
        GROUP BY 1
    ),
    avg_hc AS (
        SELECT NULLIF(AVG(hc), 0) AS avg_hc FROM month_hc
    ),
    outflows AS (
        SELECT COALESCE(SUM(p.wt), 0) AS outflow_wt
        FROM pos p
        WHERE p.enddate_primary IS NOT NULL
          AND p.enddate_primary >= DATEADD('month', -{recent_m}, CURRENT_DATE())
          AND NOT EXISTS (
              SELECT 1 FROM service_pipelines.output_current.individual_position p2
              JOIN roles r2 ON p2.role_v3_id = r2.role_v3_id
              WHERE p2.user_id = p.user_id
                AND p2.ultimate_parent_rcid = {company_rcid}
                AND p2.startdate > p.enddate_primary
                AND p2.startdate <= DATEADD('day', 180, p.enddate_primary)
          )
    ),
    inflows AS (
        SELECT COALESCE(SUM(wt), 0) AS inflow_wt
        FROM pos
        WHERE startdate >= DATEADD('month', -{recent_m}, CURRENT_DATE())
    )
    SELECT
        (SELECT hc FROM current_hc) AS current_hc,
        (SELECT outflow_wt FROM outflows) / (SELECT avg_hc FROM avg_hc) AS attrition,
        (SELECT inflow_wt FROM inflows) / (SELECT avg_hc FROM avg_hc) AS hiring_rate
    """
    df = se._lower_cols(se._sf(cfg).load_df(q))
    if df.empty:
        return (0.0, 0.08, 0.10)
    row = df.iloc[0]
    return (
        float(row.get("current_hc") or 0),
        float(row.get("attrition") or 0.08),
        float(row.get("hiring_rate") or 0.10),
    )


def _load_role_population_synthetic(cfg, target_role):
    # Scale from skill synthetic defaults when names overlap; else mid demo size.
    pops = {
        "Data Analyst": (420.0, 0.09, 0.12),
        "Software Engineer": (1800.0, 0.11, 0.14),
        "Systems Engineer": (950.0, 0.08, 0.10),
        "Process Engineer": (610.0, 0.07, 0.09),
        "Quality Engineer": (480.0, 0.06, 0.08),
        "Project Manager": (720.0, 0.08, 0.09),
        "Financial Analyst": (310.0, 0.10, 0.11),
    }
    if target_role in pops:
        return pops[target_role]
    h = abs(hash(str(target_role))) % 800 + 200
    return (float(h), 0.08, 0.10)


def load_role_activity_pathways(cfg, target_role):
    """Role→role feeders; overlap = activity Jaccard (mirrors skill pathways)."""
    if not target_role:
        raise ValueError("No target role selected.")

    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        return _load_role_activity_pathways_synthetic(cfg, target_role)

    peers = se._resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    peer_rcids = [int(r) for r in (peers.get("peer_rcids") or [])
                  if int(r) != int(company_rcid)]
    peer_sql = se._sql_quote_list(peer_rcids)
    country = cfg.get("country", "United States")
    years = int(cfg.get("pathway_years", 2))
    # Flat min_pool=50 zeros out mid-size firms: roles fragment into many
    # small titles, so observed feeders sit at 10–35 HC even when transitions
    # are real (Wayfair → Data Scientist). Scale the floor with workforce size
    # inside the SQL (LEAST of configured floor and HC/400, floored at 10).
    base_pool = int(cfg.get("pathway_min_pool", 50))
    max_gap = int(cfg.get("max_gap_days", 180))
    role_sql = str(target_role).replace("'", "''")

    if peer_sql:
        peer_cte = f"""
    peer_pos AS (
        SELECT
            p.user_id, r.role, COALESCE(p.weight_v2_1, 1) AS wt,
            p.startdate, p.enddate, p.enddate_primary, p.position_id
        FROM service_pipelines.output_current.individual_position p
        JOIN roles r ON p.role_v3_id = r.role_v3_id
        WHERE p.ultimate_parent_rcid IN ({peer_sql})
          AND p.country = '{country}'
          AND {se._pos_filter(cfg)}
          AND p.startdate IS NOT NULL
    ),
    peer_current AS (
        SELECT user_id, role, wt FROM (
            SELECT p.*, ROW_NUMBER() OVER (
                PARTITION BY p.user_id
                ORDER BY p.startdate DESC NULLS LAST, p.position_id DESC
            ) AS rn
            FROM peer_pos p WHERE p.enddate_primary IS NULL
        ) x WHERE rn = 1
    ),
    peer_seq AS (
        SELECT
            user_id, role AS source_role, wt,
            LEAD(role) OVER (
                PARTITION BY user_id ORDER BY startdate,
                COALESCE(enddate_primary, '9999-12-31')
            ) AS dest_role,
            LEAD(startdate) OVER (
                PARTITION BY user_id ORDER BY startdate,
                COALESCE(enddate_primary, '9999-12-31')
            ) AS to_start,
            enddate_primary
        FROM peer_pos
    ),
    peer_transitions AS (
        SELECT source_role, SUM(wt) AS peer_transition_wt
        FROM peer_seq
        WHERE dest_role = '{role_sql}'
          AND source_role <> '{role_sql}'
          AND to_start >= DATEADD('year', -{years}, CURRENT_DATE())
          AND (
                (enddate_primary IS NOT NULL
                 AND ABS(DATEDIFF('day', enddate_primary, to_start)) <= {max_gap})
             OR (enddate_primary IS NULL
                 AND DATEDIFF('day', to_start, CURRENT_DATE()) <= {max_gap})
          )
        GROUP BY 1
    ),
    peer_feeder AS (
        SELECT role AS source_role, SUM(wt) AS peer_feeder_pool
        FROM peer_current
        WHERE role <> '{role_sql}'
        GROUP BY 1
    ),
    peer_rates AS (
        SELECT
            f.source_role,
            COALESCE(t.peer_transition_wt / NULLIF(f.peer_feeder_pool, 0), 0)
                AS peer_conversion_rate
        FROM peer_feeder f
        LEFT JOIN peer_transitions t ON f.source_role = t.source_role
    ),
"""
        peer_select = "COALESCE(pr.peer_conversion_rate, 0) AS peer_conversion_rate,"
        peer_join = " LEFT JOIN peer_rates pr ON f.source_role = pr.source_role"
    else:
        peer_cte = ""
        peer_select = "0::FLOAT AS peer_conversion_rate,"
        peer_join = ""

    q = f"""
    WITH roles AS ({_role_taxonomy_with_k10()}),
    act_names AS ({_activity_taxonomy(cfg)}),
    pos AS (
        SELECT
            p.user_id, p.position_id, r.role, p.startdate, p.enddate,
            p.enddate_primary,
            COALESCE(p.total_compensation_v2_1, p.total_compensation) AS comp,
            COALESCE(p.weight_v2_1, 1) AS wt
        FROM service_pipelines.output_current.individual_position p
        JOIN roles r ON p.role_v3_id = r.role_v3_id
        WHERE p.ultimate_parent_rcid = {company_rcid}
          AND p.country = '{country}'
          AND {se._pos_filter(cfg)}
          AND p.startdate IS NOT NULL
    ),
    current_pos AS (
        SELECT user_id, role, comp, wt, position_id, startdate
        FROM (
            SELECT p.*, ROW_NUMBER() OVER (
                PARTITION BY p.user_id
                ORDER BY p.startdate DESC NULLS LAST, p.position_id DESC
            ) AS rn
            FROM pos p WHERE p.enddate_primary IS NULL
        ) x WHERE rn = 1
    ),
    target_users AS (
        SELECT DISTINCT user_id FROM current_pos WHERE role = '{role_sql}'
    ),
    seq AS (
        SELECT
            user_id, role AS source_role, wt,
            LEAD(role) OVER (
                PARTITION BY user_id ORDER BY startdate,
                COALESCE(enddate_primary, '9999-12-31')
            ) AS dest_role,
            LEAD(startdate) OVER (
                PARTITION BY user_id ORDER BY startdate,
                COALESCE(enddate_primary, '9999-12-31')
            ) AS to_start,
            enddate, enddate_primary
        FROM pos
    ),
    transitions AS (
        SELECT source_role, SUM(wt) AS transition_wt
        FROM seq
        WHERE dest_role = '{role_sql}'
          AND source_role <> '{role_sql}'
          AND to_start >= DATEADD('year', -{years}, CURRENT_DATE())
          AND (
                (enddate_primary IS NOT NULL
                 AND ABS(DATEDIFF('day', enddate_primary, to_start)) <= {max_gap})
             OR (enddate_primary IS NULL
                 AND DATEDIFF('day', to_start, CURRENT_DATE()) <= {max_gap})
          )
        GROUP BY 1
    ),
    feeder AS (
        SELECT role AS source_role, SUM(wt) AS feeder_pool
        FROM current_pos
        WHERE role <> '{role_sql}'
        GROUP BY 1
    ),
    {peer_cte}
    source_activities AS (
        SELECT cp.role AS source_role, an.activity
        FROM current_pos cp
        JOIN service_pipelines.output_current.individual_activities a
          ON cp.user_id = a.user_id
        JOIN act_names an ON a.activity_v3_id = an.activity_v3_id
        GROUP BY 1, 2
    ),
    target_activities AS (
        SELECT DISTINCT an.activity
        FROM target_users tu
        JOIN service_pipelines.output_current.individual_activities a
          ON tu.user_id = a.user_id
        JOIN act_names an ON a.activity_v3_id = an.activity_v3_id
    ),
    overlap AS (
        SELECT
            sa.source_role,
            COUNT(DISTINCT sa.activity) AS n_source,
            (SELECT COUNT(*) FROM target_activities) AS n_target,
            COUNT(DISTINCT CASE WHEN ta.activity IS NOT NULL THEN sa.activity END)
                AS n_overlap
        FROM source_activities sa
        LEFT JOIN target_activities ta ON sa.activity = ta.activity
        GROUP BY sa.source_role
    ),
    target_comp AS (
        SELECT MEDIAN(comp) AS target_comp
        FROM current_pos WHERE role = '{role_sql}'
    ),
    source_comp AS (
        SELECT role AS source_role, MEDIAN(comp) AS source_comp
        FROM current_pos GROUP BY 1
    )
    SELECT
        f.source_role,
        f.feeder_pool,
        COALESCE(t.transition_wt, 0) AS transition_wt,
        COALESCE(t.transition_wt / NULLIF(f.feeder_pool, 0), 0) AS conversion_rate,
        {peer_select}
        COALESCE(sc.source_comp / NULLIF(tc.target_comp, 0) - 1, 0) AS wage_gap,
        COALESCE(
            o.n_overlap / NULLIF(o.n_source + o.n_target - o.n_overlap, 0), 0
        ) AS activity_jaccard,
        COALESCE(sc.source_comp, 0) AS source_median_comp,
        COALESCE(tc.target_comp, 0) AS target_median_comp
    FROM feeder f
    LEFT JOIN transitions t ON f.source_role = t.source_role
    LEFT JOIN overlap o ON f.source_role = o.source_role
    LEFT JOIN source_comp sc ON f.source_role = sc.source_role
    CROSS JOIN target_comp tc{peer_join}
    WHERE COALESCE(t.transition_wt, 0) > 0
      AND f.feeder_pool >= GREATEST(
            10,
            LEAST(
              {base_pool},
              (SELECT COUNT(DISTINCT user_id) FROM current_pos) / 400
            )
          )
    {se._pathway_candidate_qualify(cfg)}
    """
    df = se._lower_cols(se._sf(cfg).load_df(q))
    empty_cols = [
        "source_role", "feeder_pool", "transition_freq", "activity_overlap",
        "wage_gap", "conversion_rate", "peer_conversion_rate", "mobility_gap",
        "source_median_comp", "target_median_comp", "activity_jaccard",
        "transition_wt"]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)
    df["transition_freq"] = df["transition_wt"].map(se._freq_label)
    df["activity_overlap"] = df["activity_jaccard"].map(se._overlap_label)
    if "peer_conversion_rate" not in df.columns:
        df["peer_conversion_rate"] = 0.0
    df["mobility_gap"] = (
        df["peer_conversion_rate"].astype(float) - df["conversion_rate"].astype(float)
    )
    # Alias for shared feasibility scorer (expects skill_overlap column name).
    df["skill_overlap"] = df["activity_overlap"]
    cols = [
        "source_role", "feeder_pool", "transition_freq", "activity_overlap",
        "skill_overlap", "wage_gap", "conversion_rate", "peer_conversion_rate",
        "mobility_gap", "source_median_comp", "target_median_comp",
        "activity_jaccard", "transition_wt"]
    return se._apply_signed_wage_gap(df[cols])


def _load_role_activity_pathways_synthetic(cfg, target_role):
    """Frozen activity-overlap pathways for offline demos.

    Feeder pools / rates are deliberately below the skill-track synthetic
    supply (~135) so the two angles don't show an identical internal figure.
    """
    data = [
        ("Process Engineer", 280, "med", "high", -0.05, 0.055, 0.028, 98000, 115000, 0.41),
        ("Quality Engineer", 210, "low-med", "med-high", -0.08, 0.042, 0.022, 95000, 115000, 0.33),
        ("Systems Engineer", 360, "med", "high", 0.04, 0.036, 0.031, 120000, 115000, 0.38),
        ("Data Analyst", 160, "low-med", "med", -0.12, 0.048, 0.019, 88000, 115000, 0.22),
        ("Project Manager", 250, "low-med", "med", 0.10, 0.028, 0.025, 125000, 115000, 0.18),
        ("Manufacturing Engineer", 300, "med", "med-high", -0.02, 0.031, 0.020, 105000, 115000, 0.29),
        ("Operations Manager", 190, "low-med", "med", 0.15, 0.024, 0.018, 132000, 115000, 0.15),
        ("Software Engineer", 520, "med", "med-high", 0.08, 0.018, 0.035, 135000, 115000, 0.27),
    ]
    # Drop self if target appears as source
    data = [r for r in data if r[0] != target_role]
    df = pd.DataFrame(data, columns=[
        "source_role", "feeder_pool", "transition_freq", "activity_overlap",
        "wage_gap", "conversion_rate", "peer_conversion_rate",
        "source_median_comp", "target_median_comp", "activity_jaccard"])
    df["skill_overlap"] = df["activity_overlap"]
    df["mobility_gap"] = df["peer_conversion_rate"] - df["conversion_rate"]
    df["transition_wt"] = df["feeder_pool"] * df["conversion_rate"]
    return df


def build_role_pathways(cfg, target_role, role_categories=None):
    """Activity-overlap feeders + feasibility (reuse skill pathway scorer).

    Unlike skill pathways, we keep ``transition_freq == "low"`` rows. The
    absolute count thresholds behind "low" (< 3 weighted moves) were
    calibrated on large employers; at mid-size firms like Wayfair every
    real feeder into Data Scientist sat at 1–2 moves and was wiped, leaving
    a false "0 redeployment" story. Feasibility already haircuts thin
    transitions via the freq score.
    """
    df = load_role_activity_pathways(cfg, target_role).copy()
    if df is None or getattr(df, "empty", True):
        return df
    df = se.drop_excluded_roles(df, cfg, col="source_role")
    if df is None or getattr(df, "empty", True):
        return df
    # Feasibility expects skill_overlap label column — already aliased.
    if "skill_overlap" not in df.columns and "activity_overlap" in df.columns:
        df["skill_overlap"] = df["activity_overlap"]

    sources = df["source_role"].dropna().astype(str).unique().tolist() if len(df) else []
    if role_categories is None or getattr(role_categories, "empty", True):
        role_categories = se.classify_roles(cfg, include_roles=sources)
    else:
        have = set(role_categories["role"].astype(str))
        missing = [r for r in sources if r not in have]
        if missing:
            role_categories = se.classify_roles(cfg, include_roles=list(have) + missing)

    if role_categories is not None and not role_categories.empty:
        cats = role_categories[["role", "category"]].drop_duplicates("role")
        df = df.merge(cats, left_on="source_role", right_on="role", how="left")
        df["category"] = df["category"].fillna("stable")
        df = df.drop(columns=["role"], errors="ignore")
    else:
        df["category"] = "stable"

    boosts = cfg.get("pathway_category_boost", se._DEFAULT_CATEGORY_BOOST)
    cat_boost = df["category"].map(boosts).fillna(0)
    f = df["transition_freq"].map(_FREQ)
    o = df["skill_overlap"].map(_OVL)
    wage_pen = 1 - (df["wage_gap"].clip(0, 0.4) / 0.4) * 0.4
    score = (0.45 * f + 0.4 * o + 0.15 * wage_pen + cat_boost).clip(0, 1.2)
    df["feasibility_score"] = score
    df["feasibility"] = np.where(
        score >= 0.75, "high", np.where(score >= 0.55, "med", "low"))
    return se.sort_pathways(df, cfg)


def _role_radar_as_skill_frame(target_role, role_radar):
    """Map a role radar row into the skill-shaped frame ``run_scenario`` reads."""
    if role_radar is None or getattr(role_radar, "empty", True):
        return None
    row = role_radar[role_radar["role"].astype(str) == str(target_role)]
    if not len(row):
        return None
    r = row.iloc[0]
    peer = r.get("peer_share")
    company = r.get("company_share")
    if peer is None or company is None or pd.isna(peer) or pd.isna(company):
        return None
    peer = float(peer)
    company = float(company)
    under = float(r["under_index"]) if pd.notna(r.get("under_index")) else peer - company
    ratio = (float(r["index_ratio"]) if pd.notna(r.get("index_ratio"))
             else se._index_ratio(company, peer))
    return pd.DataFrame([{
        "skill": target_role,
        "peer_share": peer,
        "company_share": company,
        "under_index": under,
        "index_ratio": ratio,
        "bucket": "emerging",
        "momentum": float(r.get("momentum", 0) or 0),
        "blended_growth": float(r.get("hiring_growth", 0) or 0),
    }])


def run_role_scenario(cfg, target_role, pathways, role_radar=None, population=None):
    """Same growth/replacement/retention math; target is a role name.

    Gap basis uses real role workforce shares (company vs peers) when the
    role radar carries ``peer_share`` / ``company_share``. No fabricated shares.
    """
    if population is None:
        population = load_role_population(cfg, target_role)
    radar = _role_radar_as_skill_frame(target_role, role_radar)
    if radar is None and cfg.get("growth_primary") == "gap":
        # Honest fallback: can't earn a gap without shares — size on fixed %.
        cfg = dict(cfg)
        cfg["growth_primary"] = "fixed"
    return se.run_scenario(cfg, target_role, pathways, radar=radar,
                           population=population)


def load_role_competitor_outflows(cfg, target_role):
    """People leaving the company from ``target_role`` → peer employers."""
    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        return _load_role_competitor_outflows_synthetic(cfg, target_role)

    peers = se._resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    peer_sql = se._sql_quote_list(peers["peer_rcids"])
    if not peer_sql:
        return pd.DataFrame(columns=["dest_rcid", "dest_company", "outflow_wt"])

    years = int(cfg.get("outflow_years", 2))
    max_gap = int(cfg.get("max_gap_days", 180))
    top_n = int(cfg.get("outflow_top_n", 10))
    country = cfg.get("country", "United States")
    role_sql = str(target_role).replace("'", "''")
    batchtime = cfg.get("batchtime", "202602")

    q = f"""
    WITH roles AS ({_role_taxonomy_with_k10()}),
    peer_rcids AS (
        SELECT value::INT AS rcid
        FROM TABLE(SPLIT_TO_TABLE('{peer_sql}', ','))
    ),
    pos AS (
        SELECT
            p.user_id,
            p.ultimate_parent_rcid,
            p.startdate,
            p.enddate,
            p.enddate_primary,
            r.role,
            COALESCE(p.weight_v2_1, 1) AS wt
        FROM service_pipelines.output_current.individual_position p
        JOIN roles r ON p.role_v3_id = r.role_v3_id
        WHERE p.country = '{country}'
          AND {se._pos_filter(cfg)}
          AND p.startdate IS NOT NULL
    ),
    seq AS (
        SELECT
            user_id,
            ultimate_parent_rcid AS from_rcid,
            role AS from_role,
            enddate_primary,
            LEAD(ultimate_parent_rcid) OVER (
                PARTITION BY user_id ORDER BY startdate, COALESCE(enddate_primary, '9999-12-31')
            ) AS to_rcid,
            LEAD(startdate) OVER (
                PARTITION BY user_id ORDER BY startdate, COALESCE(enddate_primary, '9999-12-31')
            ) AS to_start,
            wt
        FROM pos
    ),
    departures AS (
        SELECT s.to_rcid, s.wt
        FROM seq s
        WHERE s.from_rcid = {company_rcid}
          AND s.from_role = '{role_sql}'
          AND s.to_rcid IS NOT NULL
          AND s.to_rcid <> s.from_rcid
          AND s.enddate_primary >= DATEADD('year', -{years}, CURRENT_DATE())
          AND s.to_rcid IN (SELECT rcid FROM peer_rcids)
          AND (
                (s.enddate_primary IS NOT NULL AND ABS(DATEDIFF('day', s.enddate_primary, s.to_start)) <= {max_gap})
             OR (s.enddate_primary IS NULL AND DATEDIFF('day', s.to_start, CURRENT_DATE()) <= {max_gap})
          )
    )
    SELECT
        d.to_rcid AS dest_rcid,
        COALESCE(c.company_name, CAST(d.to_rcid AS VARCHAR)) AS dest_company,
        SUM(d.wt) AS outflow_wt
    FROM departures d
    LEFT JOIN model_compuniv.v1_internal.rcid_full_company_ref_dashboard_{batchtime} c
      ON c.rcid = d.to_rcid
    GROUP BY 1, 2
    ORDER BY outflow_wt DESC
    LIMIT {top_n}
    """
    return se._lower_cols(se._sf(cfg).load_df(q))


def _load_role_competitor_outflows_synthetic(cfg, target_role):
    """Offline outflows — same rivals as skill track, nudged weights by role."""
    base = se._load_competitor_outflows_synthetic(cfg, target_role)
    if base is None or getattr(base, "empty", True):
        return base
    out = base.copy()
    # Scale so role-track Sankey isn't a carbon copy of the skill figure.
    h = abs(hash(str(target_role))) % 40
    out["outflow_wt"] = out["outflow_wt"].astype(float) * (0.55 + h / 100.0)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- narratives

# ---------------------------------------------------------------- adjacency (skills angle)
# "Which skills sit next to the target, and are they already common outside
# the holder group?" Skills explain *why* people are close; roles (in the
# pathways table) do the counting, because roles partition the workforce and
# skills do not — one person holds many skills, so skill-level headcounts
# must never be summed into supply.
#
# Role-angle adjacency is intentionally omitted: activity Jaccard on the
# pathways table already answers "what sits next to this role."

def _adjacency_frame(df, min_lift=1.2, top_n=12):
    """Shared post-processing: lift and ranking.

    This used to split rows into ``bridge`` (already widespread outside the
    holder group, ``share_outside >= 20%``) and ``teach`` (the curriculum), but
    the two rules fought each other: surviving rows have a median lift near 15x,
    so a 20% outside share would imply an impossible share among holders. Across
    348 rows of real adjacency the largest outside share was 6%, so every row
    was ``teach`` and the bridge column was permanently empty. One ranked list
    is what the data supports.
    """
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=[
            "name", "share_among_target", "share_outside", "lift",
            "holders_company", "gap_pp"])
    out = df.copy()
    for c in ("share_among_target", "share_outside", "holders_company"):
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0.0)
    # Lift vs the rest of the workforce: how much more this skill
    # concentrates among target holders. Same logic that keeps the radar from
    # surfacing generic skills.
    out["lift"] = np.where(
        out["share_outside"] > 0,
        out["share_among_target"] / out["share_outside"],
        np.where(out["share_among_target"] > 0, 99.0, 0.0))
    out["gap_pp"] = out["share_among_target"] - out["share_outside"]
    out = out[out["lift"] >= float(min_lift)]
    return (out.sort_values(["lift", "share_among_target"],
                            ascending=[False, False])
            .head(int(top_n)).reset_index(drop=True))


def _adjacency_sql(cfg, taxonomy_sql, source_table, id_col, label_col,
                   target, min_holders):
    """One query: share among target holders vs share outside, + holder count."""
    rcid = _company_rcid(cfg)
    country = cfg.get("country", "United States")
    tgt = str(target).replace("'", "''")
    return f"""
    WITH labels AS ({taxonomy_sql}),
    company_pos AS (
        SELECT p.user_id, COALESCE(p.weight_v2_1, 1) AS wt
        FROM service_pipelines.output_current.individual_position p
        WHERE p.ultimate_parent_rcid = {int(rcid)}
          AND p.country = '{country}'
          AND {se._pos_filter(cfg)}
          AND p.enddate_primary IS NULL
    ),
    company_users AS (
        SELECT user_id, MAX(wt) AS wt FROM company_pos GROUP BY user_id
    ),
    user_items AS (
        SELECT DISTINCT s.user_id, l.{label_col} AS name
        FROM {source_table} s
        JOIN labels l ON s.{id_col} = l.{id_col}
        JOIN company_users cu ON cu.user_id = s.user_id
    ),
    holders AS (
        SELECT DISTINCT user_id FROM user_items WHERE name = '{tgt}'
    ),
    totals AS (
        SELECT
            SUM(CASE WHEN h.user_id IS NOT NULL THEN cu.wt ELSE 0 END) AS tgt_wt,
            SUM(CASE WHEN h.user_id IS NULL THEN cu.wt ELSE 0 END) AS out_wt
        FROM company_users cu
        LEFT JOIN holders h ON h.user_id = cu.user_id
    )
    SELECT
        ui.name,
        SUM(CASE WHEN h.user_id IS NOT NULL THEN cu.wt ELSE 0 END)
            / NULLIF(MAX(t.tgt_wt), 0) AS share_among_target,
        SUM(CASE WHEN h.user_id IS NULL THEN cu.wt ELSE 0 END)
            / NULLIF(MAX(t.out_wt), 0) AS share_outside,
        SUM(cu.wt) AS holders_company
    FROM user_items ui
    JOIN company_users cu ON cu.user_id = ui.user_id
    LEFT JOIN holders h ON h.user_id = ui.user_id
    CROSS JOIN totals t
    WHERE ui.name <> '{tgt}'
    GROUP BY ui.name
    HAVING SUM(cu.wt) >= {int(min_holders)}
"""


def build_adjacent_skills(cfg, target_skill):
    """Skills nearest the target, by lift among holders vs the rest."""
    if not target_skill:
        return _adjacency_frame(None)
    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        return _adjacency_frame(_adjacent_skills_synthetic(target_skill))
    sql = _adjacency_sql(
        cfg, se._skill_taxonomy(cfg),
        "service_pipelines.output_current.individual_skills",
        "skill_v3_id", "skill", target_skill,
        cfg.get("adjacency_min_holders", 25))
    df = se._lower_cols(se._sf(cfg).load_df(sql))
    return _adjacency_frame(
        df, cfg.get("adjacency_min_lift", 1.2), cfg.get("adjacency_top_n", 12))


def load_skill_time_to_report(cfg, target_skill):
    """Median months from entering a skill-heavy role to first reporting it.

    Feasibility clock for a reskill programme: among people who moved into a
    role where ``target_skill`` is concentrated, how long until the skill
    shows on their profile? Uses destination roles from the same
    concentration logic as pathways (skill holders' current roles).
    """
    if not target_skill:
        return None
    if not cfg.get("use_snowflake") or _company_rcid(cfg) is None:
        h = abs(hash(str(target_skill))) % 18
        return {
            "median_months": float(6 + h),
            "n": 40,
            "synthetic": True,
            "basis": "role_entry",
        }
    peers = se._resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    country = cfg.get("country", "United States")
    skill_sql = str(target_skill).replace("'", "''")
    min_holders = int(cfg.get("pathway_min_pool", 50)) // 2
    years = int(cfg.get("pathway_years", 2))
    q = f"""
    WITH skill_names AS ({se._skill_taxonomy(cfg)}),
    roles AS ({_role_taxonomy_with_k10()}),
    -- Roles where the target skill is concentrated (same idea as pathways).
    target_roles AS (
        SELECT r.role, SUM(COALESCE(p.weight_v2_1, 1)) AS wt
        FROM service_pipelines.output_current.individual_position p
        JOIN roles r ON p.role_v3_id = r.role_v3_id
        JOIN service_pipelines.output_current.individual_skills sk
          ON p.user_id = sk.user_id
        JOIN skill_names sn ON sk.skill_v3_id = sn.skill_v3_id
        WHERE p.ultimate_parent_rcid = {company_rcid}
          AND p.country = '{country}'
          AND {se._pos_filter(cfg)}
          AND p.enddate_primary IS NULL
          AND sn.skill = '{skill_sql}'
        GROUP BY 1
        HAVING SUM(COALESCE(p.weight_v2_1, 1)) >= {max(5, min_holders)}
    ),
    -- First time each person entered one of those destination roles.
    role_entries AS (
        SELECT
            p.user_id,
            MIN(p.startdate) AS role_start
        FROM service_pipelines.output_current.individual_position p
        JOIN roles r ON p.role_v3_id = r.role_v3_id
        WHERE p.ultimate_parent_rcid = {company_rcid}
          AND p.country = '{country}'
          AND {se._pos_filter(cfg)}
          AND p.startdate IS NOT NULL
          AND r.role IN (SELECT role FROM target_roles)
          AND p.startdate >= DATEADD('year', -{years + 3}, CURRENT_DATE())
        GROUP BY 1
    )
    SELECT
        MEDIAN(DATEDIFF(
            'month', e.role_start, TRY_TO_DATE(s.first_reported::STRING)
        )) AS median_months,
        COUNT(*) AS n
    FROM service_pipelines.output_current.individual_skills s
    JOIN skill_names sn ON s.skill_v3_id = sn.skill_v3_id
    JOIN role_entries e ON s.user_id = e.user_id
    WHERE sn.skill = '{skill_sql}'
      AND s.first_reported IS NOT NULL
      AND TRY_TO_DATE(s.first_reported::STRING) >= e.role_start
      AND DATEDIFF(
            'month', e.role_start, TRY_TO_DATE(s.first_reported::STRING)
          ) BETWEEN 0 AND 36
    """
    try:
        df = se._lower_cols(se._sf(cfg).load_df(q))
    except Exception:
        return None
    if df.empty or pd.isna(df.iloc[0].get("median_months")):
        return None
    row = df.iloc[0]
    n = int(row.get("n") or 0)
    if n < 10:
        return None
    return {
        "median_months": float(row["median_months"]),
        "n": n,
        "synthetic": False,
        "basis": "role_entry",
    }


def _adjacent_skills_synthetic(target_skill):
    """Frozen adjacency for offline demos (shaped like the live query)."""
    rows = [
        # name, share_among_target, share_outside, holders_company
        ("Data Modeling", 0.71, 0.14, 980),
        ("Statistical Analysis", 0.63, 0.09, 640),
        ("SQL And Databases", 0.68, 0.21, 1480),
        ("Requirements Analysis", 0.58, 0.31, 2170),
        ("Business Intelligence", 0.49, 0.11, 760),
        ("Data Visualization", 0.55, 0.18, 1240),
        ("Process Improvement", 0.44, 0.34, 2390),
        ("Python Programming", 0.37, 0.08, 520),
        ("Technical Reporting", 0.52, 0.29, 1910),
        ("Quality Assurance", 0.31, 0.26, 1750),
    ]
    return pd.DataFrame(rows, columns=[
        "name", "share_among_target", "share_outside", "holders_company"])


def skill_pathway_blurb(target_skill, paths):
    if paths is None or getattr(paths, "empty", True):
        return (f"No observed feeder roles into roles where **{target_skill}** "
                "is concentrated.")
    top = paths.head(3)["source_role"].tolist()
    names = ", ".join(f"**{r}**" for r in top)
    return (
        f"People in roles like {names} already hold overlapping skills and "
        f"move into roles where **{target_skill}** lives. Treat them as "
        f"reskill candidates for **{target_skill}** — not as a role redesign."
    )


def role_pathway_blurb(target_role, paths):
    if paths is None or getattr(paths, "empty", True):
        return f"No observed feeders into **{target_role}** with shared activities."
    top = paths.head(3)["source_role"].tolist()
    names = ", ".join(f"**{r}**" for r in top)
    return (
        f"Roles like {names} share day-to-day **activities** with "
        f"**{target_role}** and already produce internal moves. Those are the "
        f"natural role-based hiring / mobility paths."
    )


# ---------------------------------------------------------------- snapshot io

_SKILL_FRAMES = ["radar", "funnel", "paths", "metros", "outflows", "roles",
                 "adjacency"]
_ROLE_FRAMES = ["role_radar", "funnel", "paths", "roles", "outflows"]


def _meta_safe(v):
    """Numpy / Decimal scalars → Python, so meta.json holds numbers not strings."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_meta_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _meta_safe(x) for k, x in v.items()}
    return v


def _angle_root(angle: str) -> Path:
    return SNAP_ROOT / ("skills" if angle == "skills" else "roles")


def snapshot_dir(angle: str, company, target) -> Path:
    """``<root>/<angle>/<company>/<target>`` — company-scoped so the same skill
    can be frozen for several companies without clobbering."""
    return _angle_root(angle) / _slug(company or "company") / _slug(target)


def _iter_snapshot_dirs(angle: str):
    """Company-scoped dirs plus legacy flat ``<angle>/<target>`` ones."""
    root = _angle_root(angle)
    if not root.exists():
        return
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "meta.json").exists():
            yield d  # legacy flat layout
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and (sub / "meta.json").exists():
                yield sub


def save_skill_snapshot(data: dict):
    d = snapshot_dir("skills", data.get("company"), data["target"])
    d.mkdir(parents=True, exist_ok=True)
    for k in _SKILL_FRAMES:
        if k in data and isinstance(data[k], pd.DataFrame):
            data[k].to_json(d / f"{k}.json", orient="split")
    meta = {k: _meta_safe(data[k]) for k in (
        "target", "angle", "population", "attrition", "hiring_rate",
        "company", "company_rcid", "role_k10_filter", "tight", "avail",
        "skill_time")
        if k in data and data[k] is not None}
    meta["angle"] = "skills"
    meta["saved_at"] = pd.Timestamp.utcnow().isoformat()
    (d / "meta.json").write_text(json.dumps(meta, default=str))
    return d


def save_role_snapshot(data: dict):
    d = snapshot_dir("roles", data.get("company"), data["target"])
    d.mkdir(parents=True, exist_ok=True)
    for k in _ROLE_FRAMES:
        if k in data and isinstance(data[k], pd.DataFrame):
            data[k].to_json(d / f"{k}.json", orient="split")
    meta = {k: _meta_safe(data[k]) for k in (
        "target", "angle", "population", "attrition", "hiring_rate",
        "company", "company_rcid", "role_k10_filter")
        if k in data and data[k] is not None}
    meta["angle"] = "roles"
    meta["saved_at"] = pd.Timestamp.utcnow().isoformat()
    (d / "meta.json").write_text(json.dumps(meta, default=str))
    return d


def list_snapshots(angle: str, company=None):
    """target → dir for this angle, optionally limited to one company.

    A company-scoped dir beats a legacy flat one for the same target: the flat
    layout predates company scoping, so it is the older copy. Without this the
    winner fell out of alphabetical order — ``roles/lockheed-martin/...`` sorts
    before ``roles/manufacturing-engineer/``, so a stale hand-made snapshot
    quietly shadowed the fresh live pull that replaced it.
    """
    root = _angle_root(angle)
    scoped, legacy = {}, {}
    for d in _iter_snapshot_dirs(angle):
        meta = json.loads((d / "meta.json").read_text())
        name = meta.get("company")
        if company is not None and str(name or "") != str(company):
            continue
        key = meta["target"]
        out = legacy if d.parent == root else scoped
        if company is None and key in out:
            key = f"{key} — {name}"  # same target frozen for several companies
        out[key] = d
    for key, d in legacy.items():
        scoped.setdefault(key, d)
    return scoped


def list_snapshot_companies(angle: str):
    """Companies that have at least one saved snapshot for this angle."""
    names = []
    for d in _iter_snapshot_dirs(angle):
        meta = json.loads((d / "meta.json").read_text())
        name = meta.get("company") or "Company"
        if name not in names:
            names.append(name)
    return sorted(names)


def load_snapshot(path):
    d = Path(path)
    meta = json.loads((d / "meta.json").read_text())
    angle = meta.get("angle", "skills")
    frames = _SKILL_FRAMES if angle == "skills" else _ROLE_FRAMES
    out = dict(meta)
    for k in frames:
        p = d / f"{k}.json"
        if p.exists():
            out[k] = pd.read_json(p, orient="split")
    return out
