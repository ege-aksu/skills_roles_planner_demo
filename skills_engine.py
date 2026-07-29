# %% [markdown]
# # Skills scenario planner (v1)
#
# Dual-audience workforce intelligence notebook. One engine, two decks:
#
# **SWP** (HR / workforce strategy): capability gap, reskill pathways, retention,
# build vs buy.
#
# **TI** (talent acquisition / location strategy): competitive skill momentum,
# talent maps, competitor outflows, external pool sizing.
#
# Set `deck_lead` to `swp`, `ti`, or `dual`. Same data spine; slide order and
# headlines follow the buyer.
#
# **How to reuse:** edit `CONFIG`, run all cells.

# %%
import numpy as np
import pandas as pd

# %%
sfClient = None
s3_client = None


def connect(cfg=None):
    """Lazy Snowflake/S3 connection via revelio.base (no Secrets Manager fetch).

    Safe for offline / Streamlit import: does nothing when use_snowflake is False.
    Callers should not import boto3 or hit AWS at module load.
    """
    global sfClient, s3_client
    if cfg is not None and not cfg.get("use_snowflake", True):
        return None
    if sfClient is not None:
        return sfClient
    import revelio.base
    user = "ege"
    sfClient = revelio.base.client("snowflake", f"{user}@reveliolabs.com")
    s3_client = revelio.base.client("s3", f"{user}@reveliolabs.com")
    sfClient.warehouse = "cst_transformer_3"
    return sfClient


def _sf(cfg=None):
    """Return a connected Snowflake client, connecting lazily if needed."""
    client = connect(cfg)
    if client is None:
        raise RuntimeError(
            "Snowflake client unavailable. Set use_snowflake=True and ensure "
            "revelio.base credentials work, or use synthetic mode."
        )
    return client


def _lower_cols(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    return df

# %% [markdown]
# ## Config
# The only block an analyst edits to run a new company / cluster / scenario.
#
# **Snowflake setup**
# 1. Connection is **lazy** — Snowflake only connects when a loader needs it.
# 2. Set `company_rcid` (ultimate-parent rcid) and `company` name.
# 3. Keep `use_snowflake=True` (default). Set `use_snowflake=False` (or omit
#    `company_rcid`) to run fully offline on synthetic demo data — no AWS/
#    revelio access required.
#
# **Peer benchmark (recommended)**
# - Skill and role signals are benchmarked against **competitors + your company**.
# - Growth/momentum uses **competitors only**; shares compare **company vs peer average**.
# - `peer_set='competitors'` uses `model_industry.v1_inference.competitors_revelioshared_latest`.
# - If that is empty, peers fall back to the largest companies in the same `rics_k200`.
# - Or set `peer_set='rcid_list'` and pass explicit `peer_rcids`.
# - Set `benchmark_mode='peers'` (default) to skip noisy whole-industry slides.
# - Position queries use **primary** roles (`is_primary`), `is_bad_user_v3_1`,
#   and `enddate_primary IS NULL` for current HC; contingent excluded via
#   `jobtype_v1 <> 'Contingent'` when configured.
#
# **Deck lead (`deck_lead`)**
# - `swp`: internal-first — gap, pathways, retention, hire plan.
# - `ti`: external-first — competitive radar, talent map, outflows, pool sizing.
# - `dual` (default): TI block then SWP block with an integration bridge.
#
# **Radar pipeline (three stages)**
# 1. **Universe (entry):** base size (`min_skill_headcount`) → **lift floor**
#    (peer_share / economy_share ≥ percentile of size-gated lifts) → postings
#    noise (`min_peer_postings`) → **specialized** (`skill_tags.is_specialized`).
#    No by-name exclude lists. No global top-N — cache once.
# 2. **Radar (display):** strategic + watch buckets, then **cap within bucket**
#    (`present_rows_per_bucket`): emerging/nascent→momentum, declining→steepest
#    decline, core→peer share. Growth signals are winsorized before blend;
#    Optional dual-positive (both postings+hires ≥ 0) is off by default.
# 3. **Funnel (selection):** **emerging** only + under-index + feeders.
#    Nascent is watch-only (show ≠ pick). KL / TF-IDF specificity is deferred.
#
# **Radar thresholds (split by job)**
# - **Entry:** `min_skill_headcount` (500) → `lift_floor_percentile` (50) →
#   `min_peer_postings` (50) → `require_specialized` (True).
#   Lift = peer_share / country-wide economy_share.
# - **Share floor (classifier):** `share_floor` (0.3%) on peer share splits
#   emerging (material + hot) vs **nascent** (hot but below materiality).
# - **Buckets:** emerging / nascent / core / declining (+ `other` for sub-floor
#   non-hot skills — held in universe, omitted from present caps).
# - Hot bar: blended growth ≥ `emerging_growth_percentile` (default 95th) of
#   the post-entry universe's blended-growth distribution (adapts to cycle).
#   Optional absolute `emerging_growth_threshold` overrides if percentile is None.
# - Growth signals are **composition** change: (skill ÷ all peer postings/hires)
#   recent vs prior — not raw volume (so market-wide hiring declines don't
#   make every skill look negative).
# - `max_signal_growth` (2.0): cap each growth signal before blend/momentum.
# - `require_dual_positive_growth` (False): if True, hot only if both raw signals ≥ 0.
# - Selection under-index: **ratio** gate — company_share / peer_share
#   < `max_index_ratio` (0.90), plus absolute floor `min_under_index` (0.2pp)
#   so noise on tiny skills doesn't count. Funnel also requires sized pathway
#   supply in the **same** pass (`min_internal_supply`).
# - Gap blow-ups from tiny company shares: `max_gap_multiple`, not the share floor.
#
# **Industry benchmark (optional)**
# - Only runs when `benchmark_mode` is `industry` or `both`.
# - Whole-sector views can be noisy; prefer peers for client-facing output.
#
# **Costs**
# - Deck default is **comp-based** (`use_comp_based_costs=True`): external hire ≈ target median comp × 1.25.
# - Flat `$45k/$110k/$15k` rates are fallbacks when comp is missing or the toggle is off.
# - The integrated plan slide always prints the cost basis — pick one and leave it for client decks.
#
# **Growth need**
# - `growth_mode='fixed'`: client ask — grow target skill HC by `growth_target` (e.g. 30%).
# - `growth_mode='gap'`: our suggestion — heads to close peer `under_index`.
# - `growth_mode='both'` (default): report both; size the plan on `growth_primary` (`gap` or `fixed`).
# - Gap formula: `current × (peer_share / company_share − 1)` when under-indexed.
#
# **Internal supply / conversion**
# - Role-pathway `conversion_rate` = (all source→target-role moves) / (full source-role HC)
#   over `pathway_years`, then **annualized** × `horizon_years` × persistence.
# - `skill_mover_share` is a separate quality column (share of those movers who hold
#   the skill) — not baked into the rate.
# - `max_feeder_conversion` caps how much of a feeder pool can move in the plan window.
# - Selection requires sized internal supply (`min_internal_supply`), not just feeder count.
# - Rates are within-company; low organic mobility at large primes is a real finding.

# %%
CONFIG = {
    "use_snowflake": True,
    "company": "Lockheed Martin",
    "company_rcid": "20921805",          # e.g. 20921805 = Lockheed Martin
    "peer_set": "competitors",     # competitors | rics_k200 | rcid_list
    "peer_rcids": [],              # used when peer_set='rcid_list'
    "peer_limit": 50,
    "rics_k200": None,             # fallback peer universe when competitors empty
    "batchtime": "202602",
    "country": "United States",
    "exclude_contingent": True,  # jobtype_v1 <> Contingent
    "skill_level": "skill_k1500",   # skill cluster grain in taxonomy
    # --- industry benchmark ---
    "industry_level": "rics_k200",  # rics_k200 | rics_k50
    "industry_label": None,         # auto-resolve from company rcid if None
    "deck_lead": "dual",              # swp | ti | dual
    "benchmark_mode": "peers",         # peers | industry | both
    # --- radar ---
    "radar_weights": {"postings": 0.60, "hires": 0.40},
    # Selection ranks on a hires-heavier blend (harder to inflate via profile wording).
    "selection_radar_weights": {"postings": 0.40, "hires": 0.60},
    # Classifier: peer-share materiality for emerging vs nascent (not an entry gate).
    "share_floor": 0.003,
    # Hot bar: ≥ this percentile of blended growth in the post-entry universe.
    "emerging_growth_percentile": 95,
    # Absolute override only when emerging_growth_percentile is None.
    "emerging_growth_threshold": None,
    "min_skill_headcount": 500,  # entry: weighted peer heads (size gate)
    "lift_floor_percentile": 50,  # entry: keep lifts ≥ this pct of size-gated dist
    "min_peer_postings": 50,     # entry: peer job posts in growth window
    # Drop soft/generic clusters via skill_tags_all_latest (no by-name list).
    "require_specialized": True,
    # Winsorize each postings/hires growth signal before blend (tagging cliffs).
    "max_signal_growth": 2.0,
    # Optional: emerging/nascent only if both raw growth signals ≥ 0 (off by default).
    "require_dual_positive_growth": False,
    # Selection under-index: relative shortfall vs peers (not absolute pp alone).
    # Eligible if company/peer < max_index_ratio AND (peer − company) ≥ floor.
    "max_index_ratio": 0.90,   # below ~90% of peer rate
    "min_under_index": 0.002,  # 0.2pp absolute floor (noise on tiny skills)
    "min_feeder_roles": 1,  # at least one non-thin pathway row
    # Horizon-scaled internal heads from role pathways (not just feeder count).
    # Applied in the same select_target pass as the ratio gate.
    "min_internal_supply": 50,
    "selection_max_candidates": 5,  # cap emerging skills tested for pathways
    "recent_months": 12,
    "prior_months": 12,
    # Display cap *within each bucket* (not a global entry top-N).
    "present_rows_per_bucket": 8,
    # Optional safety valve on universe size (None = no LIMIT). Prefer caching.
    "radar_universe_max_skills": None,
    "outflow_years": 2,
    "outflow_top_n": 10,  # rows per bucket/category on slides
    # --- pathways ---
    "pathway_years": 2,
    "pathway_min_pool": 50,
    # Rank feeders by conversion rate (moves ÷ current pool) or absolute
    # move volume. Load keeps a union of top-N by each so a UI toggle can
    # switch without re-querying; display then shows pathway_display_n.
    "pathway_rank_by": "conversion",  # conversion | volume
    "pathway_candidate_n": 25,
    "pathway_display_n": 15,
    # Past conversion is annualized to horizon_years; persistence haircuts the rate
    # (1.0 = assume historical mobility continues; 0.7 = 30% haircut).
    "conversion_persistence": 1.0,
    "max_feeder_conversion": 1.0,  # cap: cannot convert more than this share of feeder pool
    "role_category_mode": "relative",  # relative | absolute
    "role_exclude_names": ["Retired", "Unknown", "On Leave", "empty"],
    "role_min_per_bucket": 3,  # backfill empty buckets (relative mode)
    "role_category_top_n": 50,
    # absolute-mode thresholds (real-data scale)
    "role_expanding_hiring": 0.05,
    "role_expanding_skill_mix": 0.0005,
    "role_transforming_ai": 0.30,
    "role_transforming_skill_mix": 0.001,
    "role_atrisk_ai": 0.25,
    "role_atrisk_hiring": -0.15,
    "max_gap_days": 180,
    # --- scenario ---
    # growth_mode: fixed = client % target; gap = close peer under-index;
    # both = size on growth_primary, always report fixed + gap side-by-side
    "growth_mode": "both",       # fixed | gap | both
    "growth_primary": "gap",     # which drives net_need/cost when mode=both
    "growth_target": 0.30,       # used for fixed (and shown under both)
    "horizon_years": 2,
    # Retention what-if: assumed share of skill attrition a program cuts.
    # improved_rate = attrition × (1 − improvement);
    # heads_saved = C × (attrition − improved_rate) × H.
    # Replacement / net need / external hire size on improved_rate (hole shrinks).
    # Demo-facing (not "should you invest?"); default 15% is modest.
    "retention_improvement": 0.15,
    # --- pathway category boosts (feasibility score, not hard filter) ---
    "pathway_category_boost": {
        "transforming": 0.08,
        "at-risk": 0.06,
        "expanding": -0.05,
        "stable": 0.0,
    },
    # --- costs ---
    "use_comp_based_costs": True,
    "cost_hire_multiplier": 1.25,
    "cost_reskill_training_pct": 0.15,
    # Share of monthly comp treated as opportunity cost during the observed
    # ramp-to-report window (skill time-to-report). 0.25 ≈ quarter-productive
    # while learning — conservative and easy to defend in a room.
    "cost_reskill_ramp_pct": 0.25,
    "cost_retain_pct": 0.08,
    # flat fallbacks when comp missing or use_comp_based_costs=False
    "cost_reskill": 45_000,
    "cost_hire": 110_000,
    "cost_retain": 15_000,
    "force_target_skill": None,
    # --- staging (user_ege only) ---
    "staging_schema": "user_ege.tmp_daily",
    "staging_prefix": "skills_scenario",
    "refresh_staging": True,
}

rng = np.random.default_rng(7)

# %% [markdown]
# ## Data loaders
# Each returns a tidy dataframe with fixed column names. When `use_snowflake=True`,
# queries hit `service_pipelines.output_current` (positions, skills, postings) plus
# the competitor model for the peer set. Set `use_snowflake=False` to use the
# synthetic walkthrough data instead.

# %%
# --- Snowflake helpers -------------------------------------------------------

def _pos_filter(cfg=None):
    """Standard individual_position filters (primary roles, bad/platform, optional contingent)."""
    parts = [
        "COALESCE(p.is_bad_user_v3_1, FALSE) = FALSE",
        "COALESCE(p.is_platform_user, FALSE) = FALSE",
        "COALESCE(p.is_primary, FALSE) = TRUE",
    ]
    if cfg is None or cfg.get("exclude_contingent", True):
        parts.append("COALESCE(p.jobtype_v1, 'Full-time') <> 'Contingent'")
    return "\n    AND ".join(parts)


def _pos_is_current(alias="p"):
    """Primary position is currently open (prefer enddate_primary over enddate coalesce)."""
    return f"{alias}.enddate_primary IS NULL"

_SKILL_LEVELS = ("skill_k150", "skill_k500", "skill_k1500")
_SKILL_TAG_LEVEL = {
    "skill_k150": "k1500",    # no k150 grain in skill_tags_all_latest
    "skill_k500": "k5000",
    "skill_k1500": "k1500",
}


def _skill_tag_level(cfg):
    """skill_tags_all_latest.skill_k_level matching CONFIG skill_level grain."""
    level = cfg.get("skill_level", "skill_k1500")
    return _SKILL_TAG_LEVEL.get(level, "k1500")


def _ai_skill_tags_join(cfg, taxonomy_alias="sn"):
    """Join skill taxonomy row -> skill_tags_all_latest.is_ai at CONFIG grain."""
    tag_level = _skill_tag_level(cfg)
    col = f"{cfg.get('skill_level', 'skill_k1500')}_name"
    return f"""
        LEFT JOIN model_skills.v3_reference.skill_tags_all_latest t
          ON t.skill_k_level = '{tag_level}'
         AND {taxonomy_alias}.{col} = t.skill_k_label"""


def _skill_tags_on_label_join(cfg, skill_expr="s.skill", alias="t"):
    """Join aggregated skill label -> skill_tags_all_latest at CONFIG grain."""
    tag_level = _skill_tag_level(cfg)
    return f"""
        LEFT JOIN model_skills.v3_reference.skill_tags_all_latest {alias}
          ON {alias}.skill_k_level = '{tag_level}'
         AND {alias}.skill_k_label = {skill_expr}"""




def _skill_taxonomy(cfg):
    """Map skill_v3_id -> skill label at CONFIG skill_level grain."""
    level = cfg.get("skill_level", "skill_k1500")
    if level not in _SKILL_LEVELS:
        raise ValueError(f"skill_level must be one of {_SKILL_LEVELS}")
    col = f"{level}_name"
    return f"""
    SELECT skill_v3_id, {col} AS skill
    FROM service_pipelines.global_ref.custom_skills_taxonomy_v3_overall_latest
    WHERE taxonomy_name = 'default'
      AND {col} IS NOT NULL
"""

_ROLE_TAXONOMY = """
    SELECT role_v3_id, role_k1500 AS role
    FROM model_jobembedding.v3_reference.role_taxonomy_current
    WHERE taxonomy_id = 0
      AND role_k1500 IS NOT NULL
      AND LOWER(role_k1500) NOT IN ('unknown', 'retired', 'on leave', 'empty')
"""


def excluded_role_names(cfg=None):
    """Lowercased role labels to drop (Unknown / Retired / …)."""
    raw = (cfg or {}).get("role_exclude_names",
                          ["Retired", "Unknown", "On Leave", "empty"])
    return {str(n).strip().lower() for n in (raw or []) if str(n).strip()}


def drop_excluded_roles(df, cfg=None, col="role"):
    """Drop excluded role labels (case-insensitive) from a frame."""
    if df is None or getattr(df, "empty", True) or col not in getattr(df, "columns", []):
        return df
    ban = excluded_role_names(cfg)
    if not ban:
        return df
    keep = ~df[col].astype(str).str.strip().str.lower().isin(ban)
    return df.loc[keep].copy()

_PEER_CACHE = {}
_SKILL_USERS_CTE = """
    skill_users AS (
        SELECT DISTINCT sk.user_id, sn.skill
        FROM service_pipelines.output_current.individual_skills sk
        JOIN skill_names sn ON sk.skill_v3_id = sn.skill_v3_id
    ),"""



def _sql_quote_list(values):
    return ", ".join(str(int(v)) for v in values)


def _sql_quote_list_str(values):
    """Comma-separated SQL string literals; skips empties / Retired / Unknown."""
    parts = []
    skip = {"", "retired", "unknown"}
    for v in values or []:
        s = str(v).strip()
        if not s or s.lower() in skip:
            continue
        parts.append("'" + s.replace("'", "''") + "'")
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return ", ".join(out)



def _rn_latest_position(partition_sql):
    """ROW_NUMBER to keep one position weight per partition (latest start)."""
    return (
        f"ROW_NUMBER() OVER (PARTITION BY {partition_sql} "
        f"ORDER BY p.startdate DESC NULLS LAST, p.position_id DESC)"
    )


def _resolve_peer_rcids(cfg):
    cache_key = (
        cfg.get("company_rcid"), cfg.get("peer_set"),
        tuple(cfg.get("peer_rcids") or ()), cfg.get("peer_limit"),
        cfg.get("rics_k200"), cfg.get("batchtime"),
    )
    if cache_key in _PEER_CACHE and not cfg.get("refresh_staging"):
        return _PEER_CACHE[cache_key]

    company_rcid = cfg["company_rcid"]
    if company_rcid is None:
        raise ValueError("Set CONFIG['company_rcid'] before running Snowflake loaders.")

    peer_set = cfg.get("peer_set", "competitors")
    if peer_set == "rcid_list":
        peer_rcids = [int(r) for r in (cfg.get("peer_rcids") or []) if r != company_rcid]
    elif peer_set == "competitors":
        q = f"""
        SELECT competitor_rcid AS rcid
        FROM model_industry.v1_inference.competitors_revelioshared_latest
        WHERE rcid = {int(company_rcid)}
        ORDER BY similarity_score DESC
        LIMIT {int(cfg.get('peer_limit', 50))}
        """
        peer_rcids = _lower_cols(_sf(cfg).load_df(q))["rcid"].astype(int).tolist()
        if not peer_rcids:
            q = f"""
            WITH anchor AS (
                SELECT y.rics_k200
                FROM model_compuniv.v1_internal.rcid_full_company_ref_dashboard_{cfg['batchtime']} x
                JOIN model_industry.v1_reference.rics_cluster_lookup_latest y
                  ON x.rics_k400 = y.rics_k400
                WHERE x.rcid = {int(company_rcid)}
            )
            SELECT p.ultimate_parent_rcid AS rcid
            FROM service_pipelines.output_current.individual_position p
            JOIN model_compuniv.v1_internal.rcid_full_company_ref_dashboard_{cfg['batchtime']} x
              ON p.ultimate_parent_rcid = x.rcid
            JOIN model_industry.v1_reference.rics_cluster_lookup_latest y
              ON x.rics_k400 = y.rics_k400
            WHERE p.enddate_primary IS NULL
              AND y.rics_k200 = (SELECT rics_k200 FROM anchor)
              AND p.ultimate_parent_rcid <> {int(company_rcid)}
            GROUP BY 1
            ORDER BY COUNT(DISTINCT p.position_id) DESC
            LIMIT {int(cfg.get('peer_limit', 50))}
            """
            peer_rcids = _lower_cols(_sf(cfg).load_df(q))["rcid"].astype(int).tolist()
    else:
        raise ValueError("peer_set must be 'competitors' or 'rcid_list'")

    out = {"company_rcid": int(company_rcid), "peer_rcids": peer_rcids}
    _PEER_CACHE[cache_key] = out
    return out


def _freq_label(n):
    # thresholds on weighted transition counts (real employers, not demo scale)
    if n < 3:
        return "low"
    if n < 15:
        return "low-med"
    if n < 50:
        return "med"
    if n < 150:
        return "med-high"
    return "high"


def sort_pathways(pathways, cfg, display_n=None):
    """Pick top feeders by conversion or volume, then order high→low feasibility.

    ``pathway_rank_by``: ``conversion`` (default) = moves ÷ current feeder HC;
    ``volume`` = absolute weighted moves. Selection uses that metric; the
    table is then sorted by feasibility (high → med → low), with the metric
    as a tie-break so both toggle modes read cleanly.
    """
    if pathways is None or getattr(pathways, "empty", True):
        return pathways
    out = pathways.copy()
    rank = str(cfg.get("pathway_rank_by", "conversion") or "conversion").lower()
    by_volume = (rank in ("volume", "moves", "transition_wt")
                 and "transition_wt" in out.columns)

    # 1) Select by the chosen metric (optionally trim to display_n).
    if by_volume:
        sel = ["transition_wt"]
        if "conversion_rate" in out.columns:
            sel.append("conversion_rate")
    else:
        sel = [c for c in ("conversion_rate", "feasibility_score", "transition_wt")
               if c in out.columns]
    if sel:
        out = out.sort_values(sel, ascending=[False] * len(sel))
    if display_n is not None:
        out = out.head(int(display_n))

    # 2) Display order: feasibility high → low, then the same metric.
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


def _pathway_candidate_qualify(cfg):
    """SQL QUALIFY: keep top-N by conversion **or** top-N by move volume."""
    n = int(cfg.get("pathway_candidate_n", 25))
    return (
        f"QUALIFY ROW_NUMBER() OVER (ORDER BY conversion_rate DESC) <= {n}\n"
        f"     OR ROW_NUMBER() OVER (ORDER BY transition_wt DESC) <= {n}"
    )


def _overlap_label(jaccard):
    if jaccard >= 0.45:
        return "high"
    if jaccard >= 0.25:
        return "med-high"
    if jaccard >= 0.12:
        return "med"
    return "low"


# --- Synthetic fallbacks (frozen from live Lockheed Martin pull) -------------
# Size + lift + postings + specialized entry; composition growth; P95 emerging;
# ratio under-index; retention = max(company attrition, peer skill P10).
# Default target="Data Analysis". Snapshot: demo_snapshot/_synthetic_from_live.json
# Refreshed: 2026-07-27T21:47:25.954762+00:00

_SYNTHETIC_COMPANY_ATTRITION = 0.055603
_SYNTHETIC_DEFAULT_TARGET = "Data Analysis"
_SYNTHETIC_PEER_FLOORS = {
        "Data Analysis": 0.106562
}


def _load_skill_radar_synthetic(cfg):
    data = [
        (".NET Development", -0.168286, -0.007496, 0.013441, 0.012752, 12097.112301, 21210.0, 1.840357, 0.007303, True),
        ("3D Design", -0.095077, 0.087973, 0.010645, 0.011871, 9581.213644, 56613.0, 2.087837, 0.005099, True),
        ("3D Graphics", 0.093327, 0.02811, 0.069032, 0.073748, 62131.242295, 14832.0, 9.757872, 0.007075, True),
        ("3D Plant Engineering", -0.024897, 0.142182, 0.022446, 0.022165, 20201.981976, 15900.0, 5.785695, 0.00388, True),
        ("AI Business Integration", 0.135063, 0.104439, 0.006534, 0.007652, 5881.147099, 12811.0, 1.746753, 0.003741, True),
        ("AI Modeling", 0.197476, 0.019263, 0.012544, 0.014372, 11290.207213, 39092.0, 1.682536, 0.007456, True),
        ("Access Control", -0.215743, -0.023308, 0.002577, 0.002071, 2319.68168, 14845.0, 1.265894, 0.002036, True),
        ("Access Security", -0.112972, 0.068769, 0.001786, 0.001301, 1607.386105, 16373.0, 1.615608, 0.001105, True),
        ("Aerospace Fabrication", -0.153964, 0.100631, 0.041371, 0.043177, 37235.518412, 95617.0, 8.914242, 0.004641, True),
        ("Aerospace Marine Design", -0.327558, 0.007252, 0.113585, 0.180412, 102230.833769, 60949.0, 28.739883, 0.003952, True),
        ("Aerospace Technologies", -0.145528, -0.003007, 0.110855, 0.175694, 99773.528219, 114287.0, 16.741929, 0.006621, True),
        ("Agile Collaboration", -0.041038, 0.046263, 0.064597, 0.06098, 58139.32277, 89084.0, 6.907085, 0.009352, True),
        ("Agile Practices", -0.113248, -0.012465, 0.035507, 0.035402, 31957.706239, 98616.0, 2.366785, 0.015002, True),
        ("Agile Project Delivery", -0.06881, -0.05094, 0.095557, 0.129293, 86004.757776, 62790.0, 2.860708, 0.033403, True),
        ("Air Quality Regulation", -0.304717, -0.041968, 0.004178, 0.003083, 3760.779067, 8760.0, 1.430249, 0.002922, True),
        ("Airway Care", 0.030213, 0.059749, 0.000663, 0.000488, 597.066637, 102.0, 1.23997, 0.000535, True),
        ("Analytical Techniques", -0.161806, 0.112899, 0.010313, 0.012872, 9281.694674, 12256.0, 2.614008, 0.003945, True),
        ("Apple Technology", -0.425838, -0.030444, 0.007695, 0.009766, 6925.985645, 6511.0, 2.31795, 0.00332, True),
        ("Application Development", -0.356117, -0.122457, 0.015129, 0.020947, 13616.374435, 27207.0, 3.360371, 0.004502, True),
        ("Application Integration", -0.374143, -0.189805, 0.00552, 0.004963, 4968.596693, 13846.0, 2.143708, 0.002575, True),
        ("Artificial Intelligence", 0.508898, 0.112674, 0.010411, 0.01031, 9370.681577, 24841.0, 1.621192, 0.006422, True),
        ("Asset Development", -0.031759, -0.056885, 0.102532, 0.082129, 92282.460439, 16803.0, 2.529432, 0.040536, True),
        ("Audit Compliance", -0.162785, 0.009415, 0.050853, 0.03707, 45769.104885, 53722.0, 2.388643, 0.021289, True),
        ("Augmented Reality", -0.167928, 0.156215, 0.011247, 0.015565, 10122.426381, 4501.0, 5.317536, 0.002115, True),
        ("Automotive", -0.262109, -0.012664, 0.02052, 0.010442, 18468.84726, 25551.0, 2.349509, 0.008734, True),
        ("Automotive Electronics", -0.006594, -0.065483, 0.014972, 0.018554, 13475.715573, 24263.0, 13.427281, 0.001115, True),
        ("Automotive Engineering", -0.185734, 0.047499, 0.001261, 0.001212, 1135.110474, 15033.0, 5.644099, 0.000223, True),
        ("Automotive Maintenance", -0.256421, -0.004257, 0.001323, 0.001336, 1191.09793, 8440.0, 1.975304, 0.00067, True),
        ("Automotive Software", 0.077333, -0.10993, 0.013374, 0.02383, 12036.679558, 21291.0, 10.635953, 0.001257, True),
        ("Automotive Systems", -0.008696, 0.045385, 0.024771, 0.017411, 22294.479506, 19281.0, 7.53225, 0.003289, True),
        ("Aviation Compliance", -0.150384, -0.056129, 0.02267, 0.019912, 20403.814317, 16776.0, 5.778684, 0.003923, True),
        ("Aviation Coordination", -0.315306, -0.043129, 0.050095, 0.059697, 45087.145446, 25533.0, 3.628513, 0.013806, True),
        ("Aviation Operations", -0.188768, 0.033849, 0.039828, 0.045853, 35846.815654, 34403.0, 5.687058, 0.007003, True),
        ("Aviation Systems", -0.127629, -0.006186, 0.086083, 0.119437, 77477.300998, 89591.0, 11.978279, 0.007187, True),
        ("Aviation Technology", -0.090764, -0.018648, 0.047341, 0.057179, 42608.113154, 52181.0, 9.321865, 0.005078, True),
        ("Aviation and Aerospace Technology", -0.245145, 0.013829, 0.024982, 0.025518, 22484.653965, 26035.0, 15.094076, 0.001655, True),
        ("Azure Cloud", -0.166495, -0.02777, 0.003096, 0.002065, 2786.891638, 31784.0, 1.829084, 0.001693, True),
        ("Backend Development", 0.030494, 0.010144, 0.009592, 0.010419, 8632.950988, 38428.0, 1.438422, 0.006668, True),
        ("Biochemical Research", 0.003344, -0.000377, 0.00495, 0.002343, 4455.475908, 12546.0, 3.458062, 0.001432, True),
        ("Biological Production Technologies", -0.489506, -0.26519, 0.008192, 0.003125, 7373.041026, 667.0, 4.105854, 0.001995, True),
        ("Biomedical Technology", -0.43674, 0.001051, 0.061291, 0.060207, 55163.829604, 3180.0, 5.348903, 0.011459, True),
        ("Biopharmaceutical Technology", -0.231571, -0.025174, 0.034217, 0.024217, 30796.780012, 10164.0, 5.074768, 0.006743, True),
        ("Biotechnology", -0.18795, -0.120671, 0.008054, 0.006464, 7249.177935, 1095.0, 1.349275, 0.005969, True),
        ("Building Automation", -0.035082, -0.202561, 0.002459, 0.001248, 2213.441023, 16116.0, 2.117002, 0.001162, True),
        ("Building Construction", -0.500963, -0.085012, 0.002167, 0.000574, 1949.932133, 8078.0, 1.580216, 0.001371, True),
        ("Building Systems Maintenance", -0.304595, -0.027824, 0.006038, 0.00377, 5434.154349, 15489.0, 1.534607, 0.003934, True),
        ("Business Analytics", -0.219337, 0.023369, 0.019074, 0.015064, 17167.497592, 12062.0, 1.67781, 0.011369, True),
        ("Business Continuity", -0.277154, -0.018957, 0.030625, 0.020674, 27563.67951, 9630.0, 2.400054, 0.01276, True),
        ("Business Intelligence", -0.051073, 0.048626, 0.017558, 0.021696, 15803.094426, 18065.0, 1.357452, 0.012935, True),
        ("Business Intelligence Reporting", -0.048737, -0.006165, 0.008531, 0.004906, 7678.132886, 15156.0, 1.710693, 0.004987, True),
        ("Business Process", -0.373026, -0.004701, 0.064225, 0.050965, 57804.979288, 7514.0, 2.454243, 0.026169, True),
        ("Business Process Automation", -0.296241, -0.031113, 0.030649, 0.024913, 27585.256393, 21476.0, 2.535608, 0.012087, True),
        ("Business Process Strategy", -0.275698, 0.033223, 0.014689, 0.011923, 13220.931753, 8727.0, 1.274075, 0.011529, True),
        ("Business Resilience", -0.18107, -0.011227, 0.004917, 0.003302, 4425.175778, 4022.0, 2.179436, 0.002256, True),
        ("Business Technology", -0.511097, 0.010213, 0.008138, 0.006468, 7324.525258, 5709.0, 1.254538, 0.006487, True),
        ("C# User Interface Development", 0.327699, -0.053646, 0.064149, 0.086642, 57736.712962, 24066.0, 2.696917, 0.023786, True),
        ("CAD Drafting", -0.028715, 0.045266, 0.147019, 0.144653, 132321.780018, 128969.0, 4.707867, 0.031228, True),
        ("Cellular Imaging", 0.084943, -0.068224, 0.007851, 0.003683, 7066.248306, 6958.0, 2.511849, 0.003126, True),
        ("Chemical Engineering", -0.447524, -0.083844, 0.008759, 0.003907, 7883.531354, 9656.0, 4.303893, 0.002035, True),
        ("Chemical Processing", -0.003886, 0.001058, 0.00101, 0.000431, 909.096118, 7091.0, 2.097672, 0.000482, True),
        ("Chemical Production", -0.397745, -0.214229, 0.022234, 0.002871, 20011.398392, 8757.0, 4.040782, 0.005502, True),
        ("Chromatography Techniques", -0.027589, 0.079947, 0.0142, 0.00825, 12780.135042, 3783.0, 2.156705, 0.006584, True),
        ("Circuit and Hardware Design", 0.085699, -0.066021, 0.055712, 0.068423, 50143.080301, 86985.0, 15.271155, 0.003648, True),
        ("Client Service Support", -0.511474, -0.054137, 0.003024, 0.001954, 2721.951133, 2190.0, 1.425634, 0.002121, True),
        ("Cloud Architecture", 0.135742, -0.037987, 0.015023, 0.014529, 13521.245822, 36856.0, 4.960554, 0.003028, True),
        ("Cloud Communication", -0.091117, 0.070162, 0.009752, 0.005568, 8777.275733, 8462.0, 1.621922, 0.006013, True),
        ("Cloud Computing", -0.343119, -0.125635, 0.003013, 0.002057, 2711.780287, 20468.0, 1.844309, 0.001634, True),
        ("Cloud Containers", 0.089851, -0.129201, 0.012834, 0.016054, 11550.895496, 57860.0, 2.299606, 0.005581, True),
        ("Cloud Data", 0.083814, -0.002133, 0.005667, 0.005655, 5100.690551, 13155.0, 1.633062, 0.00347, True),
        ("Cloud Infrastructure", -0.452915, -0.035326, 0.005368, 0.005268, 4831.255151, 5846.0, 2.428782, 0.00221, True),
        ("Cloud Infrastructure Reliability", -0.137481, -0.127422, 0.009034, 0.006278, 8130.565081, 16726.0, 2.695397, 0.003351, True),
        ("Cloud Integration", 0.176478, 0.047634, 0.003021, 0.002931, 2719.384715, 40432.0, 1.3376, 0.002259, True),
        ("Cloud Solutions", -0.002334, 0.002722, 0.011726, 0.013017, 10553.60301, 14734.0, 1.527394, 0.007677, True),
        ("Cloud Technologies", -0.087004, -0.007462, 0.043412, 0.029766, 39071.977263, 107726.0, 1.587328, 0.027349, True),
        ("Cloud Virtualization", -0.046257, -0.090837, 0.006913, 0.006685, 6221.914354, 28601.0, 2.686989, 0.002573, True),
        ("Code Quality", 0.139476, -0.089442, 0.033511, 0.052399, 30161.13675, 42840.0, 7.017468, 0.004775, True),
        ("Collaboration", -0.237069, -0.068543, 0.000748, 0.000813, 672.992005, 1248.0, 1.46488, 0.00051, True),
        ("Collaborative Programming", -0.292712, 0.052461, 0.023859, 0.035497, 21474.099885, 12276.0, 4.205148, 0.005674, True),
        ("Collaborative Technologies", 0.160412, 0.071538, 0.011114, 0.012852, 10003.222744, 21195.0, 1.466047, 0.007581, True),
        ("Collaborative Web Technologies", 0.039374, 0.063315, 0.002106, 0.002169, 1895.334853, 21381.0, 2.238884, 0.000941, True),
        ("Commercial Contracting", -0.678314, 0.054663, 0.023997, 0.015355, 21598.357514, 5317.0, 1.606928, 0.014934, True),
        ("Commodity and Energy Trading", 0.311095, -0.046881, 0.005577, 0.008224, 5019.410171, 36910.0, 3.269298, 0.001706, True),
        ("Communication Technologies", 0.153927, 0.046343, 0.022225, 0.022576, 20002.89112, 46571.0, 4.379178, 0.005075, True),
        ("Communication Technology Integration", -0.07197, 0.113149, 0.009148, 0.011405, 8233.654079, 2754.0, 2.049036, 0.004465, True),
        ("Compensation Strategy", -0.344095, -0.043523, 0.004517, 0.003898, 4065.688494, 6088.0, 1.50242, 0.003007, True),
        ("Compliance Assessment", -0.422027, 0.087657, 0.001573, 0.001539, 1415.975914, 7720.0, 2.563578, 0.000614, True),
        ("Compliance Inspection", -0.720414, 0.006872, 0.002841, 0.001024, 2556.868087, 7322.0, 2.490087, 0.001141, True),
        ("Computational Technologies", -0.267524, 0.029199, 0.010249, 0.010894, 9224.096152, 33065.0, 3.515366, 0.002915, True),
        ("Computer Hardware Design", -0.076385, -0.029727, 0.052851, 0.054176, 47567.493506, 78281.0, 12.688922, 0.004165, True),
        ("Computer Hardware Support", -0.367415, -0.031885, 0.076031, 0.094506, 68430.525916, 52950.0, 4.497291, 0.016906, True),
        ("Configuration Management", 0.013079, -0.151351, 0.009388, 0.018156, 8449.529098, 19800.0, 18.673263, 0.000503, True),
        ("Construction", -0.313046, -0.069862, 0.005104, 0.002776, 4593.521705, 35891.0, 1.618778, 0.003153, True),
        ("Construction Coordination", -0.407739, 0.004125, 0.061246, 0.042923, 55123.271683, 24744.0, 1.57473, 0.038893, True),
        ("Construction Documentation", -0.197266, 0.046391, 0.100573, 0.092588, 90519.539166, 63210.0, 5.16622, 0.019468, True),
        ("Construction Materials", -0.136681, -0.009317, 0.009993, 0.009456, 8993.840314, 6880.0, 2.975418, 0.003358, True),
        ("Construction Project Coordination", -0.372746, 0.000691, 0.060569, 0.046892, 54514.512996, 30416.0, 1.647279, 0.036769, True),
        ("Construction Quality Assurance", -0.629624, 0.01319, 0.028674, 0.022364, 25807.225337, 13099.0, 2.622402, 0.010934, True),
        ("Consumer Marketing", -0.321869, -0.035139, 0.042354, 0.032403, 38119.695538, 30434.0, 1.734398, 0.02442, True),
        ("Content Design", 1.042663, -0.208485, 0.001119, 0.000771, 1007.166022, 1728.0, 5.149773, 0.000217, True),
        ("Contract Administration", -0.349597, -0.066904, 0.065045, 0.075211, 58542.706767, 32804.0, 2.44383, 0.026616, True),
        ("Contract and License Management", -0.26296, -0.052158, 0.003224, 0.0025, 2901.74672, 14483.0, 1.757483, 0.001834, True),
        ("Contracting Processes", -0.232887, -0.072177, 0.006445, 0.00552, 5801.079546, 29129.0, 2.28961, 0.002815, True),
        ("Control Systems", -0.027434, -0.016204, 0.04949, 0.081491, 44542.537191, 32052.0, 14.963953, 0.003307, True),
        ("Control Systems Automation", -0.167288, 0.082999, 0.022759, 0.016108, 20483.861026, 43014.0, 4.962625, 0.004586, True),
        ("Cost Analysis", -0.370102, 0.026322, 0.031996, 0.024944, 28797.113909, 27101.0, 4.830943, 0.006623, True),
        ("Cost Assessment", -0.109302, 0.069781, 0.008183, 0.007961, 7365.237582, 21110.0, 2.060645, 0.003971, True),
        ("Cost Estimation", -0.335471, -0.071365, 0.022306, 0.025712, 20075.848212, 32968.0, 1.385234, 0.016102, True),
        ("Cross-Platform Development", -0.178754, -0.033712, 0.019469, 0.023377, 17522.505548, 15265.0, 3.344866, 0.00582, True),
        ("Cyber Threat Defense", -0.296748, -0.041681, 0.007525, 0.008156, 6772.38596, 20573.0, 5.00649, 0.001503, True),
        ("Cybersecurity", -0.044578, -0.032474, 0.083472, 0.094065, 75127.901001, 196679.0, 2.640102, 0.031617, True),
        ("Cybersecurity Exploitation", -0.077131, -0.011058, 0.012258, 0.009771, 11032.835406, 9913.0, 1.599012, 0.007666, True),
        ("Cybersecurity Response", -0.125538, 0.033846, 0.02041, 0.023443, 18370.102562, 31797.0, 1.807638, 0.011291, True),
        ("Cybersecurity Risk Management", -0.079936, 0.055211, 0.017609, 0.018567, 15848.588094, 35883.0, 2.535759, 0.006944, True),
        ("Data", -0.028095, 0.065204, 0.013818, 0.00971, 12436.924616, 39096.0, 1.529523, 0.009034, True),
        ("Data Analysis", 0.363219, 0.062147, 0.029456, 0.022838, 26510.974904, 20052.0, 1.58571, 0.018576, True),
        ("Data Analytics", -0.309825, 0.059381, 0.029738, 0.033491, 26765.242089, 26258.0, 2.234646, 0.013308, True),
        ("Data Architecture", -0.142846, -0.043516, 0.007977, 0.006718, 7179.414962, 35405.0, 1.296053, 0.006155, True),
        ("Data Automation", 0.331922, 0.016754, 0.011438, 0.011125, 10294.467427, 18384.0, 2.079719, 0.0055, True),
        ("Data Backup Solutions", -0.147009, -0.044846, 0.004257, 0.003368, 3831.491036, 15779.0, 2.115033, 0.002013, True),
        ("Data Center Infrastructure", -0.2639, 0.03699, 0.013684, 0.009097, 12316.297445, 29056.0, 2.679985, 0.005106, True),
        ("Data Communication", -0.15662, 0.094133, 0.000657, 0.00046, 591.732778, 4908.0, 1.477833, 0.000445, True),
        ("Data Engineering", 0.100023, -0.029567, 0.007494, 0.007159, 6744.892771, 30794.0, 2.079925, 0.003603, True),
        ("Data Exploration", -0.027725, -0.017258, 0.00869, 0.007813, 7821.392043, 14094.0, 1.561307, 0.005566, True),
        ("Data Governance", -0.151177, -0.034385, 0.005461, 0.004653, 4915.009889, 23558.0, 1.256224, 0.004347, True),
        ("Data Integration", 0.032189, -0.07075, 0.014965, 0.014305, 13469.232035, 32889.0, 1.905516, 0.007854, True),
        ("Data Intelligence", -0.223683, -0.01551, 0.023227, 0.020582, 20905.50569, 30722.0, 3.308355, 0.007021, True),
        ("Data Management", -0.245036, -0.004977, 0.01481, 0.0145, 13329.279044, 18697.0, 1.674922, 0.008842, True),
        ("Data Modeling", 0.318158, 0.054417, 0.005909, 0.007478, 5318.750162, 9983.0, 2.208768, 0.002675, True),
        ("Data Processing", 0.226329, -0.017289, 0.033722, 0.037904, 30350.847664, 77368.0, 3.230146, 0.01044, True),
        ("Data Processing Infrastructure", 0.191684, -0.110458, 0.003161, 0.003629, 2845.248638, 24079.0, 1.51468, 0.002087, True),
        ("Data Programming", 0.059982, -0.001867, 0.06797, 0.105907, 61175.451161, 59789.0, 3.519696, 0.019311, True),
        ("Data Quality Assurance", -0.256223, -0.087632, 0.005008, 0.005823, 4507.325734, 8412.0, 3.981672, 0.001258, True),
        ("Data Science Algorithms", 0.196412, 0.013388, 0.015267, 0.016258, 13740.505011, 38154.0, 3.17021, 0.004816, True),
        ("Data Security Analytics", 0.007231, 0.046327, 0.002189, 0.002038, 1970.400766, 10400.0, 2.352929, 0.00093, True),
        ("Data Stewardship", -0.348903, -0.23841, 0.008461, 0.013844, 7615.209101, 16731.0, 5.246315, 0.001613, True),
        ("Data Storage Solutions", -0.265843, -0.058737, 0.00311, 0.002461, 2799.251182, 9815.0, 1.745467, 0.001782, True),
        ("Data Systems", -0.187924, 0.013126, 0.006184, 0.005414, 5565.569225, 39900.0, 1.571955, 0.003934, True),
        ("Data Visualization", -0.158083, 0.025732, 0.026472, 0.022114, 23825.479165, 43342.0, 1.947869, 0.01359, True),
        ("Database", -0.296708, 0.071942, 0.002715, 0.002866, 2443.853707, 9757.0, 1.831808, 0.001482, True),
        ("Database Development", -0.120632, -0.001203, 0.028209, 0.027686, 25388.656254, 35970.0, 1.7892, 0.015766, True),
        ("Database Systems", -0.264978, -0.091203, 0.02317, 0.020512, 20854.093192, 17770.0, 2.427134, 0.009546, True),
        ("Databases", -0.125256, -0.071372, 0.113284, 0.130352, 101959.381955, 37958.0, 2.543725, 0.044535, True),
        ("Defect Tracking", 0.235223, -0.059508, 0.001429, 0.00098, 1285.70481, 5763.0, 2.011511, 0.00071, True),
        ("Defense Analysis", -0.198783, -0.050829, 0.003132, 0.002819, 2818.572129, 12284.0, 1.927099, 0.001625, True),
        ("Demand Planning", -0.276185, 0.131384, 0.01717, 0.012397, 15453.999617, 15802.0, 2.331889, 0.007363, True),
        ("Design Engineering", -0.169513, 0.009034, 0.008406, 0.010169, 7565.838994, 24584.0, 5.32446, 0.001579, True),
        ("Design Management", -0.155994, 0.008577, 0.021396, 0.016004, 19256.916187, 18678.0, 2.864307, 0.00747, True),
        ("Design Prototyping", -0.128854, 0.055554, 0.009026, 0.008303, 8123.96534, 18798.0, 3.666411, 0.002462, True),
        ("Design Tools", -0.272878, 0.075716, 0.017564, 0.021497, 15808.483894, 52240.0, 4.565585, 0.003847, True),
        ("Desktop Support", -0.412121, -0.048937, 0.029124, 0.026987, 26212.902914, 53442.0, 2.0635, 0.014114, True),
        ("DevOps Automation", -0.051298, -0.114108, 0.008508, 0.011008, 7657.677619, 82627.0, 2.697818, 0.003154, True),
        ("Development Collaboration", 0.148419, -0.166397, 0.00623, 0.005939, 5607.154754, 36196.0, 2.419345, 0.002575, True),
        ("Development Processes", -0.015631, -0.147056, 0.003944, 0.003389, 3550.083194, 15798.0, 2.787144, 0.001415, True),
        ("Device Integration", -0.355191, -0.004449, 0.029245, 0.032193, 26321.840177, 53854.0, 2.93219, 0.009974, True),
        ("Digital Fabrication", -0.099062, 0.188239, 0.011843, 0.015369, 10658.898049, 8522.0, 6.535902, 0.001812, True),
        ("Digital Forensics", -0.194632, -0.065601, 0.002268, 0.002591, 2040.889296, 3371.0, 2.649244, 0.000856, True),
        ("Digital Governance", -0.176323, -0.174746, 0.027154, 0.022403, 24439.771206, 21177.0, 7.01164, 0.003873, True),
        ("Digital Hardware Design", -0.153405, -0.06952, 0.05531, 0.067132, 49781.150994, 53788.0, 14.954592, 0.003699, True),
        ("Digital Security", -0.167841, 0.097398, 0.001081, 0.001014, 973.286634, 11333.0, 3.545728, 0.000305, True),
        ("Distributed Systems", -0.0683, -0.098445, 0.015783, 0.017405, 14204.989835, 28756.0, 3.811389, 0.004141, True),
        ("Document Management", -0.173886, 0.024517, 0.001743, 0.001512, 1568.558328, 966.0, 1.779698, 0.000979, True),
        ("Document Preparation", -0.348906, 0.01147, 0.006702, 0.006085, 6032.181449, 10351.0, 1.488511, 0.004503, True),
        ("Documentation", -0.34334, 0.002234, 0.04491, 0.046495, 40420.606786, 49800.0, 3.614775, 0.012424, True),
        ("Dynamic Web Development", -0.239789, -0.00039, 0.016496, 0.017192, 14846.898493, 5276.0, 1.432481, 0.011516, True),
        ("E-commerce Development", -0.493273, -0.019976, 0.039138, 0.048547, 35225.648842, 2561.0, 2.07348, 0.018876, True),
        ("ERP Systems Integration", -0.444498, -0.025773, 0.005815, 0.005342, 5233.589503, 4055.0, 2.000483, 0.002907, True),
        ("Earth Sciences", -0.339201, 0.060653, 0.00318, 0.001223, 2862.462316, 1929.0, 1.491455, 0.002132, True),
        ("Economic Analysis", -0.254671, 0.083219, 0.004422, 0.004172, 3980.247256, 2721.0, 1.263212, 0.003501, True),
        ("Electrical Assembly", 0.000915, -0.129078, 0.004766, 0.004912, 4289.180683, 36739.0, 1.63606, 0.002913, True),
        ("Electrical Compliance", -0.473172, -0.157173, 0.005411, 0.002317, 4869.931568, 10045.0, 2.103945, 0.002572, True),
        ("Electrical Design", -0.035537, 0.002787, 0.159515, 0.189151, 143568.625949, 198072.0, 8.719685, 0.018294, True),
        ("Electrical Engineering", -0.173643, 0.003698, 0.155434, 0.217999, 139895.969172, 54361.0, 17.125565, 0.009076, True),
        ("Electrical Power Systems", -0.18396, 0.05193, 0.003238, 0.002729, 2913.903054, 16601.0, 2.332267, 0.001388, True),
        ("Electrical Systems", -0.140878, -0.011414, 0.029051, 0.021364, 26147.032952, 102659.0, 2.940864, 0.009878, True),
        ("Electromagnetic Engineering", 0.041634, -0.006593, 0.040953, 0.046875, 36859.17387, 64605.0, 4.270671, 0.009589, True),
        ("Electromagnetic Technologies", 0.285197, 0.120194, 0.001043, 0.001007, 938.520214, 6658.0, 2.581486, 0.000404, True),
        ("Electronic Assembly", -0.04872, 0.007156, 0.025925, 0.030787, 23333.665239, 63907.0, 11.282316, 0.002298, True),
        ("Electronic Communication", 0.029092, -0.080042, 0.005403, 0.004666, 4862.581729, 24789.0, 7.708995, 0.000701, True),
        ("Electronic Design", 0.359241, 0.005154, 0.009663, 0.009461, 8696.902797, 22382.0, 10.918522, 0.000885, True),
        ("Electronic Systems Production", 0.042731, 0.087657, 0.006819, 0.007319, 6137.016413, 56872.0, 11.053016, 0.000617, True),
        ("Electronic Testing", 0.114144, -0.020289, 0.024894, 0.031048, 22405.662243, 68873.0, 13.428528, 0.001854, True),
        ("Electronics Repair", -0.364418, -0.049641, 0.026691, 0.038859, 24023.094865, 29666.0, 8.239321, 0.00324, True),
        ("Embedded Software Development", 0.24827, -0.135646, 0.054823, 0.066485, 49342.175407, 27034.0, 8.689721, 0.006309, True),
        ("Embedded Systems Engineering", -0.0658, -0.066479, 0.17761, 0.284897, 159854.892824, 144912.0, 12.915509, 0.013752, True),
        ("Energy Economics", 0.552261, -0.13636, 0.006729, 0.001695, 6055.893903, 1489.0, 2.356477, 0.002855, True),
        ("Energy Engineering", -0.106379, -0.020142, 0.0263, 0.025853, 23671.208234, 24515.0, 7.682526, 0.003423, True),
        ("Energy Production Safety", -0.473158, -0.080595, 0.01037, 0.004457, 9333.713904, 23529.0, 3.319838, 0.003124, True),
        ("Energy Systems", -0.225083, -0.081735, 0.028101, 0.00808, 25291.975358, 14002.0, 2.154037, 0.013046, True),
        ("Energy Systems Engineering", -0.1661, -0.020036, 0.007273, 0.005732, 6546.036471, 49607.0, 4.476918, 0.001625, True),
        ("Engineering Design", -0.11747, 0.014652, 0.040427, 0.028452, 36385.884707, 42341.0, 9.461418, 0.004273, True),
        ("Engineering Project Delivery", -0.449106, 0.023654, 0.255883, 0.299963, 230303.07542, 36787.0, 7.408978, 0.034537, True),
        ("Engineering Quality", -0.33967, 0.033289, 0.077465, 0.072424, 69721.023204, 43060.0, 7.411966, 0.010451, True),
        ("Engineering Reliability", -0.143017, 0.002622, 0.132728, 0.104777, 119459.679729, 34138.0, 11.971549, 0.011087, True),
        ("Engineering Simulation", -0.101035, 0.026637, 0.202462, 0.273919, 182223.009039, 125033.0, 9.774236, 0.020714, True),
        ("Enterprise Architecture", -0.073543, -0.100261, 0.007655, 0.006094, 6889.371969, 40042.0, 2.062444, 0.003711, True),
        ("Enterprise Mobility", -0.193433, -0.130357, 0.005362, 0.004745, 4825.884506, 15377.0, 2.228091, 0.002406, True),
        ("Enterprise Productivity Tools", -0.535998, 0.016689, 0.003092, 0.002605, 2782.468536, 5307.0, 1.625184, 0.001902, True),
        ("Enterprise Resource Planning", -0.432927, 0.034573, 0.004716, 0.002852, 4244.906156, 3561.0, 1.334258, 0.003535, True),
        ("Environmental Analysis", -0.46444, -0.070868, 0.055615, 0.088241, 50055.047904, 103250.0, 7.950021, 0.006996, True),
        ("Environmental Construction", -0.25637, -0.120938, 0.002031, 0.000995, 1827.977522, 6439.0, 1.68067, 0.001208, True),
        ("Environmental Policy", -0.296912, 0.016582, 0.054141, 0.070554, 48729.12439, 4465.0, 3.78935, 0.014288, True),
        ("Environmental Quality Assessment", -0.231483, -0.099018, 0.008794, 0.005288, 7915.1803, 25874.0, 1.615227, 0.005445, True),
        ("Environmental Remediation", -0.256062, -0.207193, 0.000766, 0.000465, 689.798609, 2674.0, 1.337442, 0.000573, True),
        ("Environmental Safety", -0.35672, -0.099096, 0.005282, 0.00259, 4753.63386, 22576.0, 1.398447, 0.003777, True),
        ("Environmental Stewardship", -0.466023, -0.160099, 0.002247, 0.001324, 2022.134923, 4503.0, 2.540411, 0.000884, True),
        ("Equipment Reliability", -0.264187, -0.021679, 0.013379, 0.010068, 12041.281413, 13444.0, 6.928332, 0.001931, True),
        ("Facility Development", -0.37855, 0.017619, 0.012066, 0.009599, 10860.22736, 5953.0, 1.319303, 0.009146, True),
        ("Fiber Optic Technology", -0.13217, 0.088267, 0.006141, 0.005046, 5526.839959, 11080.0, 2.698819, 0.002275, True),
        ("Field Service", -0.292423, -0.19404, 0.005199, 0.002552, 4679.042133, 16739.0, 2.834608, 0.001834, True),
        ("Financial Analysis and ERP", -0.4085, -0.044283, 0.018226, 0.018647, 16403.668689, 15601.0, 2.877684, 0.006333, True),
        ("Financial Integration", -0.16953, -0.004897, 0.007415, 0.006381, 6673.595938, 6750.0, 2.034583, 0.003644, True),
        ("Financial Management", -0.02792, -0.009293, 0.01169, 0.009579, 10521.746159, 4963.0, 1.460697, 0.008003, True),
        ("Fintech", -0.296544, 0.048619, 0.005811, 0.005555, 5229.764229, 4270.0, 1.463872, 0.003969, True),
        ("Fire Safety Engineering", -0.373828, -0.058889, 0.004524, 0.001981, 4072.107936, 11293.0, 1.86343, 0.002428, True),
        ("Fluid Dynamics", 0.089151, 0.011819, 0.004136, 0.003119, 3722.219963, 5335.0, 9.592629, 0.000431, True),
        ("Fluid and Material Analysis", 0.101165, 0.099004, 0.013698, 0.014734, 12328.760562, 17791.0, 9.603176, 0.001426, True),
        ("Fluid and Thermal Technologies", -0.208098, -0.178338, 0.004433, 0.00096, 3989.901656, 21735.0, 4.084915, 0.001085, True),
        ("Front-End Development", 0.07532, -0.161182, 0.002367, 0.003308, 2130.645592, 7898.0, 1.652181, 0.001433, True),
        ("Frontend Development", -0.182962, -0.021492, 0.013569, 0.013717, 12212.479284, 9936.0, 1.270216, 0.010682, True),
        ("Fuel Production", -0.312108, -0.225564, 0.022285, 0.003963, 20057.559918, 9622.0, 4.093415, 0.005444, True),
        ("Furniture and Room Design", -0.326522, 0.123605, 0.004632, 0.004514, 4169.015323, 2972.0, 1.255971, 0.003688, True),
        ("Game Development", -0.159623, 0.126401, 0.005198, 0.0081, 4678.601716, 9568.0, 1.371076, 0.003791, True),
        ("Gas and Liquid Processing", -0.287108, -0.179841, 0.011637, 0.001617, 10473.717046, 5681.0, 3.768417, 0.003088, True),
        ("Genomic Data Technologies", -0.72394, -0.257746, 0.002137, 0.004412, 1923.563845, 1026.0, 2.384716, 0.000896, True),
        ("Geological Sciences", -0.11669, -0.222393, 0.014975, 0.002632, 13478.315279, 2218.0, 3.952542, 0.003789, True),
        ("Geospatial Data Integration", 0.298747, 0.060814, 0.003282, 0.003132, 2954.318091, 5451.0, 1.870797, 0.001755, True),
        ("Geospatial Navigation", -0.123738, -0.02591, 0.006262, 0.010279, 5636.404526, 13638.0, 8.395262, 0.000746, True),
        ("Geospatial Visualization", 0.132514, -0.009866, 0.001407, 0.00097, 1266.54786, 2287.0, 1.289108, 0.001092, True),
        ("Geotechnical Exploration", -0.201853, -0.143762, 0.005425, 0.001717, 4882.320211, 5874.0, 1.531932, 0.003541, True),
        ("Global Business Coordination", -0.681846, 0.172105, 0.002636, 0.002121, 2372.210378, 550.0, 1.854944, 0.001421, True),
        ("Global Logistics Coordination", -0.250269, 0.071707, 0.057955, 0.042575, 52161.334402, 34340.0, 1.67148, 0.034673, True),
        ("Global Project Delivery", -0.567542, -0.067634, 0.056467, 0.081225, 50822.56828, 3107.0, 4.030483, 0.01401, True),
        ("Graphics Programming", 0.109374, -0.037335, 0.179588, 0.264548, 161635.811964, 95711.0, 5.537801, 0.03243, True),
        ("Grinding Operations", -0.438346, 0.036489, 0.001187, 0.000915, 1068.094732, 6292.0, 5.450325, 0.000218, True),
        ("HR Process Analytics", -0.507843, -0.029588, 0.009847, 0.007427, 8862.216696, 7198.0, 1.624137, 0.006063, True),
        ("HVAC Appliances", -0.15469, -0.147614, 0.005609, 0.002697, 5048.195472, 18688.0, 1.667811, 0.003363, True),
        ("Hazardous Materials Safety", -0.332267, -0.054591, 0.008959, 0.00749, 8062.965454, 20731.0, 1.808952, 0.004952, True),
        ("Heavy Load Operations", -0.447826, -0.05318, 0.006263, 0.00355, 5637.22782, 48948.0, 2.234644, 0.002803, True),
        ("Home Repair", -0.268554, -0.110073, 0.003912, 0.003792, 3521.095632, 47260.0, 4.024154, 0.000972, True),
        ("Human Factors Engineering", -0.101162, 0.003733, 0.00564, 0.005193, 5076.039439, 6802.0, 4.797101, 0.001176, True),
        ("Human Resources Technology", -0.38349, -0.024293, 0.007797, 0.004764, 7017.651723, 8392.0, 1.523297, 0.005119, True),
        ("Hydraulic Mechanics", -0.237815, -0.078253, 0.003694, 0.002978, 3324.767168, 18657.0, 5.278517, 0.0007, True),
        ("IBM Mainframe Technology", -0.226899, -0.067495, 0.007136, 0.006141, 6422.917495, 17526.0, 3.163352, 0.002256, True),
        ("IT Delivery", -0.238794, -0.043024, 0.04592, 0.038326, 41329.816442, 33487.0, 2.321295, 0.019782, True),
        ("IT Infrastructure", -0.311546, -0.014403, 0.034873, 0.033907, 31387.067328, 36826.0, 2.573853, 0.013549, True),
        ("IT Integration", -0.357473, -0.057454, 0.027863, 0.015498, 25077.874483, 27101.0, 2.338441, 0.011915, True),
        ("IT Solutions", -0.146827, 0.020759, 0.012953, 0.011379, 11658.535763, 52980.0, 2.279861, 0.005682, True),
        ("IT Systems Development", -0.198147, -0.21581, 0.025657, 0.023843, 23091.911379, 20819.0, 6.593447, 0.003891, True),
        ("IT Transformation", -0.703291, -0.007633, 0.037699, 0.031686, 33930.18973, 12077.0, 1.954359, 0.01929, True),
        ("Identity Security", -0.303196, -0.091012, 0.002609, 0.00163, 2348.16553, 6790.0, 1.964358, 0.001328, True),
        ("Identity Software Security", -0.146336, -0.13656, 0.001756, 0.001352, 1580.634696, 3954.0, 1.465095, 0.001199, True),
        ("Image Data Processing", -0.022807, 0.026548, 0.002533, 0.00223, 2279.978125, 3856.0, 2.199799, 0.001152, True),
        ("Imaging Technologies", 0.012202, 0.223652, 0.001228, 0.001416, 1105.268787, 2120.0, 1.652609, 0.000743, True),
        ("Industrial Automation", -0.259929, 0.008991, 0.07075, 0.069461, 63677.323362, 51766.0, 5.797529, 0.012203, True),
        ("Industrial Equipment Safety", -0.26741, -0.079067, 0.004135, 0.001863, 3721.786976, 51191.0, 2.020443, 0.002047, True),
        ("Industrial Systems", -0.406314, -0.056375, 0.040461, 0.011174, 36416.232279, 21730.0, 3.939473, 0.010271, True),
        ("Industrial Technology", -0.37237, 0.035467, 0.003415, 0.001543, 3073.728629, 7576.0, 3.895733, 0.000877, True),
        ("Information Governance", -0.546706, -0.081322, 0.000846, 0.000616, 761.282976, 846.0, 1.636699, 0.000517, True),
        ("Information Retrieval", -0.575525, 0.047768, 0.001844, 0.002465, 1659.892013, 819.0, 1.994083, 0.000925, True),
        ("Information Security", -0.107769, -0.031577, 0.079673, 0.111379, 71708.571141, 118823.0, 4.290064, 0.018572, True),
        ("Information Systems Integration", -0.294366, 0.055129, 0.058609, 0.070358, 52750.121954, 14525.0, 1.88067, 0.031164, True),
        ("Infrastructure Engineering", -0.167662, 0.044299, 0.017175, 0.01028, 15458.415149, 27354.0, 1.930012, 0.008899, True),
        ("Infrastructure Storage", -0.318817, 0.079629, 0.001316, 0.000696, 1184.187084, 15170.0, 1.815963, 0.000725, True),
        ("Innovation Commercialization", -0.507838, 0.080834, 0.003665, 0.002324, 3298.835853, 878.0, 1.57525, 0.002327, True),
        ("Innovation and Technology", -0.709189, 0.091458, 0.006509, 0.006531, 5858.115489, 4689.0, 2.255368, 0.002886, True),
        ("Instrumentation and Power Systems", 0.198029, 0.007602, 0.043114, 0.06021, 38803.948861, 26154.0, 13.823883, 0.003119, True),
        ("Intelligence Collection", -0.216473, -0.020014, 0.000918, 0.000668, 826.531606, 2390.0, 1.254265, 0.000732, True),
        ("Intelligent Software", -0.107734, -0.014976, 0.078855, 0.114108, 70972.506885, 150820.0, 3.62966, 0.021725, True),
        ("Internal Controls", -0.074418, 0.052381, 0.006589, 0.004959, 5930.213454, 9308.0, 1.334137, 0.004939, True),
        ("International Trade", -0.546738, 0.031196, 0.011779, 0.009179, 10601.661316, 27840.0, 2.186332, 0.005388, True),
        ("International Trade Practices", 0.003312, 0.124572, 0.002882, 0.001952, 2594.225445, 8749.0, 2.018964, 0.001428, True),
        ("Internet Security", -0.208332, -0.075617, 0.01179, 0.009409, 10611.035907, 17256.0, 3.058026, 0.003855, True),
        ("JD Edwards ERP", -0.29816, 0.017215, 0.012625, 0.009857, 11362.515732, 9580.0, 2.628433, 0.004803, True),
        ("Java Architecture", -0.099706, -0.113241, 0.006624, 0.007556, 5962.235722, 5578.0, 2.516306, 0.002633, True),
        ("Java Development", -0.027199, -0.075289, 0.055292, 0.058804, 49764.976424, 39911.0, 2.54824, 0.021698, True),
        ("Java Microservices", -0.20546, -0.064842, 0.011564, 0.010782, 10408.195807, 21428.0, 1.568099, 0.007375, True),
        ("Java Security", -0.184269, -0.036426, 0.022308, 0.027623, 20078.177746, 24317.0, 3.182511, 0.00701, True),
        ("JavaScript Frameworks", 0.047601, -0.026131, 0.049614, 0.064682, 44653.907215, 15278.0, 2.653229, 0.018699, True),
        ("Laboratory Analysis", -0.111505, -0.115345, 0.002516, 0.001995, 2264.472376, 4447.0, 1.913273, 0.001315, True),
        ("Laboratory Quality Assurance", -0.135645, 0.055868, 0.009847, 0.00679, 8862.898241, 11960.0, 1.743497, 0.005648, True),
        ("Language Analysis", -0.297476, -0.076445, 0.000812, 0.000769, 730.937922, 2092.0, 1.521317, 0.000534, True),
        ("Language Technology", 0.445605, 0.165147, 0.003344, 0.003519, 3009.79341, 47074.0, 1.581765, 0.002114, True),
        ("Lean Quality", -0.261685, 0.026067, 0.259105, 0.245804, 233203.144507, 43430.0, 6.617384, 0.039155, True),
        ("Lifting Equipment", -0.321388, -0.088188, 0.007155, 0.003922, 6440.001393, 55099.0, 1.621903, 0.004412, True),
        ("Linux", -0.028445, -0.057383, 0.04104, 0.065819, 36937.188602, 25167.0, 4.513026, 0.009094, True),
        ("Linux and Unix Systems", -0.150953, -0.024973, 0.008092, 0.00962, 7282.652772, 11483.0, 3.366175, 0.002404, True),
        ("Log Monitoring and Analysis", -0.325967, 0.039953, 0.004137, 0.002581, 3722.998098, 9091.0, 3.725045, 0.00111, True),
        ("Machining", -0.124156, 0.125492, 0.02104, 0.01514, 18936.892726, 42773.0, 6.367461, 0.003304, True),
        ("Maintenance Reliability", -0.445295, -0.061564, 0.027175, 0.025654, 24458.677584, 26152.0, 5.978318, 0.004546, True),
        ("Manufacturing", -0.315417, 0.015689, 0.242531, 0.205707, 218286.260842, 90757.0, 7.903014, 0.030688, True),
        ("Manufacturing Integration", -0.546029, -0.109443, 0.040326, 0.026077, 36294.353086, 13220.0, 5.359557, 0.007524, True),
        ("Marine Operations", -0.54109, 0.007046, 0.002908, 0.001733, 2617.705269, 5321.0, 2.186106, 0.00133, True),
        ("Marine Transportation Engineering", 0.142539, -0.002307, 0.025178, 0.027541, 22660.980442, 10404.0, 11.027551, 0.002283, True),
        ("Maritime Logistics", -0.374034, 0.190982, 0.001237, 0.000692, 1113.377172, 2038.0, 1.398858, 0.000884, True),
        ("Maritime Safety", -0.306241, 0.043375, 0.009105, 0.00762, 8194.480297, 10319.0, 1.987589, 0.004581, True),
        ("Material Analysis", -0.096572, -0.016202, 0.009176, 0.009625, 8258.534118, 33655.0, 9.619218, 0.000954, True),
        ("Material Characterization", -0.029858, -0.121733, 0.003551, 0.001795, 3195.860802, 3956.0, 4.473606, 0.000794, True),
        ("Material Compliance", -0.126103, -0.256303, 0.000617, 0.000379, 555.612037, 7944.0, 1.301356, 0.000474, True),
        ("Material Evaluation", -0.128832, -0.02154, 0.010984, 0.010295, 9885.944467, 16282.0, 8.003116, 0.001372, True),
        ("Material Processing", -0.070959, -0.057412, 0.016267, 0.008639, 14640.736362, 62921.0, 3.054115, 0.005326, True),
        ("Materials", -0.454248, -0.019084, 0.010629, 0.005517, 9566.581482, 8862.0, 3.541514, 0.003001, True),
        ("Materials Science", -0.249748, -0.009195, 0.030889, 0.020906, 27801.149941, 45740.0, 6.094439, 0.005068, True),
        ("Mathematical Analysis", -0.36985, 0.22528, 0.002046, 0.00198, 1841.398603, 6630.0, 3.69129, 0.000554, True),
        ("Mathematical Modeling", -0.137874, -0.021283, 0.138277, 0.173805, 124454.305312, 57019.0, 6.257849, 0.022097, True),
        ("Mathematics", -0.120515, 0.103131, 0.011545, 0.01431, 10391.071642, 61436.0, 1.542317, 0.007486, True),
        ("Measurement", -0.397431, 0.000723, 0.011781, 0.009854, 10603.583351, 17456.0, 4.536349, 0.002597, True),
        ("Mechanical Design Engineering", -0.082371, 0.016218, 0.033731, 0.026288, 30359.028804, 35391.0, 8.0037, 0.004214, True),
        ("Mechanical Maintenance", -0.28338, -0.048596, 0.052748, 0.045587, 47474.689591, 109106.0, 3.815356, 0.013825, True),
        ("Mechanical Repair", -0.324255, -0.134495, 0.002908, 0.001621, 2617.729375, 23285.0, 2.646477, 0.001099, True),
        ("Mechanical Systems Maintenance", -0.227202, -0.21808, 0.015124, 0.006254, 13611.761736, 32491.0, 4.159416, 0.003636, True),
        ("Mechanical and Hydraulic Systems", -0.101683, -0.003402, 0.005279, 0.003343, 4751.700685, 26051.0, 5.147765, 0.001026, True),
        ("Messaging Integration", -0.249085, -0.014324, 0.012957, 0.012029, 11661.848988, 9723.0, 1.732278, 0.00748, True),
        ("Metal Processing", -0.129735, 0.052837, 0.00295, 0.00178, 2655.236569, 13633.0, 4.90288, 0.000602, True),
        ("Metal Shaping", -0.182439, 0.045234, 0.02061, 0.012027, 18549.450687, 47428.0, 6.056085, 0.003403, True),
        ("Military Aviation", -0.186726, 0.021702, 0.063045, 0.08907, 56742.832259, 38038.0, 6.817642, 0.009247, True),
        ("Military Communication", -0.08334, -0.045702, 0.022914, 0.02921, 20623.189819, 22948.0, 4.53827, 0.005049, True),
        ("Military Readiness", -0.561217, -0.039346, 0.13635, 0.198664, 122719.656713, 30170.0, 3.979388, 0.034264, True),
        ("Military Tactics", -0.255028, -0.076868, 0.010195, 0.008907, 9176.260953, 10240.0, 2.687181, 0.003794, True),
        ("Military and Naval Operations", -0.216421, -0.002116, 0.037466, 0.047242, 33720.440489, 19030.0, 2.212801, 0.016931, True),
        ("Mining Operations", -0.80426, -0.204269, 0.006292, 0.003332, 5663.157965, 207.0, 3.788675, 0.001661, True),
        ("Mobile Development", -0.194028, -0.005181, 0.023136, 0.024712, 20822.885307, 26511.0, 1.976047, 0.011708, True),
        ("Mobile Quality Assurance", -0.00396, -0.119415, 0.009628, 0.009827, 8665.578681, 13797.0, 4.460691, 0.002158, True),
        ("Mobile Technology", -0.612975, 0.049295, 0.013746, 0.010048, 12372.286725, 5839.0, 2.844438, 0.004833, True),
        ("Motion Dynamics Analysis", -0.054159, 0.072242, 0.023937, 0.025908, 21543.958389, 18770.0, 14.1582, 0.001691, True),
        ("Music Data Technology", -0.073904, 0.044522, 0.014064, 0.012935, 12657.90789, 263.0, 2.428265, 0.005792, True),
        ("Nanoscale Technology", -0.11244, -0.141514, 0.025956, 0.034016, 23361.088597, 7935.0, 16.833561, 0.001542, True),
        ("Network Administration", -0.401636, -0.083365, 0.018239, 0.017657, 16415.304267, 22682.0, 3.428421, 0.00532, True),
        ("Network Architecture", -0.239535, 0.055109, 0.008481, 0.006968, 7633.016489, 55804.0, 2.889192, 0.002935, True),
        ("Network Communication", -0.036952, 0.043444, 0.005191, 0.003429, 4672.173575, 19829.0, 2.736146, 0.001897, True),
        ("Network Connectivity", -0.26818, -0.010828, 0.021387, 0.01623, 19249.297351, 14747.0, 3.007284, 0.007112, True),
        ("Network Design and Analysis", 0.103858, 0.056707, 0.003845, 0.004258, 3460.91662, 7892.0, 3.117696, 0.001233, True),
        ("Network Engineering", -0.251932, -0.014374, 0.008417, 0.007849, 7575.420205, 10530.0, 2.998245, 0.002807, True),
        ("Network Infrastructure", -0.242487, -0.000542, 0.059388, 0.059378, 53451.137535, 146825.0, 3.002219, 0.019781, True),
        ("Network Monitoring", -0.289051, 0.065947, 0.003492, 0.003712, 3142.721181, 16997.0, 3.432017, 0.001017, True),
        ("Network Performance", 0.191041, -0.05063, 0.017644, 0.012537, 15880.515964, 37342.0, 3.918005, 0.004503, True),
        ("Network Performance Assurance", -0.468258, 0.066586, 0.003167, 0.002851, 2850.70285, 15606.0, 2.168838, 0.00146, True),
        ("Network Security", -0.09133, -0.0315, 0.041837, 0.039792, 37654.597911, 52503.0, 3.711913, 0.011271, True),
        ("Network Systems Integration", -0.233794, -0.07461, 0.00732, 0.006782, 6588.396392, 37822.0, 3.719469, 0.001968, True),
        ("Network Technologies", -0.150233, -0.003103, 0.048757, 0.042406, 43882.592265, 97514.0, 3.336909, 0.014611, True),
        ("Network and System Diagnostics", -0.121426, -0.075613, 0.004912, 0.003692, 4421.235645, 30633.0, 2.782401, 0.001765, True),
        ("Networking Infrastructure", -0.043848, -0.14954, 0.012004, 0.00758, 10803.705485, 26620.0, 6.527246, 0.001839, True),
        ("Networking Technologies", -0.144397, -0.067958, 0.063887, 0.067968, 57500.856891, 65525.0, 5.561006, 0.011488, True),
        ("Nuclear Tech", -0.481336, -0.002484, 0.004145, 0.001465, 3730.445531, 4068.0, 3.364493, 0.001232, True),
        ("Oil Production Techniques", -0.282205, -0.254403, 0.025872, 0.002498, 23285.486252, 12719.0, 5.574532, 0.004641, True),
        ("Optical Networking", -0.288181, -0.057483, 0.003241, 0.003263, 2916.86283, 6917.0, 2.752242, 0.001178, True),
        ("Optical Technologies", -0.038731, -0.138851, 0.009561, 0.013036, 8605.444243, 21307.0, 13.873124, 0.000689, True),
        ("Oracle Data Analytics", -0.253984, 0.004303, 0.003746, 0.00208, 3371.732659, 1136.0, 1.453271, 0.002578, True),
        ("Oracle Technologies", -0.250994, 0.004809, 0.012685, 0.010766, 11416.57365, 13895.0, 1.999915, 0.006343, True),
        ("PHP Development", -0.242637, 0.009456, 0.015283, 0.016, 13755.517049, 6114.0, 1.36338, 0.01121, True),
        ("Packaging Technology", -0.454169, -0.077837, 0.00698, 0.003629, 6282.544247, 28660.0, 2.599289, 0.002685, True),
        ("Parallel Programming", 0.107349, -0.09794, 0.012402, 0.016224, 11162.502772, 38008.0, 2.888283, 0.004294, True),
        ("Passenger Transport", -0.805467, -0.080339, 0.038814, 0.058739, 34934.319796, 4994.0, 5.669239, 0.006846, True),
        ("Performance", -0.111499, 0.031621, 0.008371, 0.008282, 7534.149727, 19017.0, 2.698282, 0.003102, True),
        ("Performance Data Analysis", -0.586671, -0.103021, 0.004581, 0.00399, 4123.434815, 10901.0, 1.634453, 0.002803, True),
        ("Performance Engineering", -0.087704, -0.07558, 0.014993, 0.019875, 13494.021221, 27636.0, 3.978468, 0.003768, True),
        ("Petroleum Engineering", -0.517005, -0.205936, 0.054113, 0.00879, 48703.367056, 2269.0, 4.760008, 0.011368, True),
        ("Physical Science", -0.223164, 0.095067, 0.011995, 0.013011, 10796.045463, 43004.0, 2.135824, 0.005616, True),
        ("Physical System Analysis", 0.078664, 0.164847, 0.00581, 0.006479, 5229.527082, 9638.0, 10.718001, 0.000542, True),
        ("Piping Systems", -0.388164, -0.154929, 0.012916, 0.003337, 11624.827945, 12967.0, 4.056275, 0.003184, True),
        ("Piping Techniques", -0.256839, -0.250517, 0.007342, 0.001055, 6608.209567, 15377.0, 2.975273, 0.002468, True),
        ("Plastic Manufacturing", -0.223941, 0.08516, 0.008199, 0.005219, 7379.774078, 24497.0, 2.176725, 0.003767, True),
        ("Platform Development", 0.027978, 0.041202, 0.002989, 0.003999, 2690.388442, 4143.0, 2.703463, 0.001106, True),
        ("Power Generation Technologies", -0.304358, -0.04003, 0.003719, 0.002322, 3347.276934, 21246.0, 1.800756, 0.002065, True),
        ("Power System Automation", -0.474598, 0.010584, 0.007061, 0.007975, 6355.537816, 10308.0, 6.891117, 0.001025, True),
        ("Power Systems Engineering", -0.181799, -0.047765, 0.004976, 0.003301, 4478.413378, 10364.0, 2.208574, 0.002253, True),
        ("Precision Manufacturing", -0.297042, 0.086527, 0.093906, 0.08787, 84518.656049, 90636.0, 8.676128, 0.010823, True),
        ("Precision Measurement Techniques", 0.045302, 0.024782, 0.003386, 0.002597, 3047.228443, 29369.0, 7.653258, 0.000442, True),
        ("Predictive Analytics", 0.082784, 0.06119, 0.015635, 0.018258, 14072.3055, 26808.0, 2.140532, 0.007304, True),
        ("Prenatal Diagnostics", -0.720212, -0.345624, 0.0014, 0.003284, 1260.11673, 429.0, 1.500174, 0.000933, True),
        ("Printing Technology", -0.312085, 0.007202, 0.00524, 0.003963, 4716.232932, 4713.0, 1.513358, 0.003463, True),
        ("Probabilistic and Statistical Modeling", 0.000971, 0.125115, 0.001294, 0.001918, 1165.041447, 674.0, 2.453791, 0.000528, True),
        ("Problem Solving", -0.610124, 0.084413, 0.016988, 0.022616, 15289.980959, 30243.0, 2.89679, 0.005864, True),
        ("Process", -0.26647, 0.03164, 0.260926, 0.238634, 234842.196325, 25496.0, 6.234869, 0.041849, True),
        ("Process Analysis", -0.522056, 0.045544, 0.003013, 0.002756, 2711.573097, 2886.0, 3.85512, 0.000781, True),
        ("Process Quality", -0.270706, 0.011276, 0.068595, 0.062723, 61738.120558, 40303.0, 10.955779, 0.006261, True),
        ("Process Visualization", -0.331708, 0.037206, 0.005522, 0.003849, 4969.778574, 8351.0, 2.565281, 0.002152, True),
        ("Procurement", -0.295528, 0.078289, 0.048314, 0.023991, 43483.964542, 48264.0, 2.13543, 0.022625, True),
        ("Product Design", -0.099516, 0.120809, 0.016632, 0.014883, 14968.970023, 46092.0, 3.685333, 0.004513, True),
        ("Product Development", -0.349217, 0.001843, 0.065019, 0.048264, 58518.949793, 42708.0, 4.901422, 0.013265, True),
        ("Product Implementation", -0.330341, -0.039588, 0.007171, 0.005266, 6454.21029, 5711.0, 2.816968, 0.002546, True),
        ("Product Lifecycle Strategy", -0.140245, -0.052217, 0.004583, 0.003615, 4124.700416, 7735.0, 2.831742, 0.001618, True),
        ("Product Strategy", -0.241493, -0.05106, 0.048294, 0.036068, 43466.174943, 42342.0, 1.516842, 0.031838, True),
        ("Production", -0.4102, 0.005751, 0.223418, 0.175414, 201084.041853, 54022.0, 3.722305, 0.060021, True),
        ("Production Efficiency", -0.414725, -0.023291, 0.141421, 0.102905, 127283.934232, 29200.0, 5.71031, 0.024766, True),
        ("Production Engineering", -0.202017, -0.015745, 0.251434, 0.279447, 226298.939742, 158981.0, 10.569986, 0.023788, True),
        ("Production Planning", -0.137082, -0.023505, 0.041934, 0.031814, 37741.9871, 49299.0, 5.263012, 0.007968, True),
        ("Production Programming", -0.251643, -0.001041, 0.085467, 0.096766, 76922.812622, 27788.0, 10.712284, 0.007978, True),
        ("Programming Language", 0.047539, -0.031782, 0.016037, 0.017293, 14434.09896, 17407.0, 5.539379, 0.002895, True),
        ("Project Oversight", -0.372426, -0.044657, 0.0173, 0.014461, 15570.282248, 18847.0, 1.677184, 0.010315, True),
        ("Project Performance", 0.013824, -0.097883, 0.070376, 0.138801, 63340.419859, 54576.0, 15.943206, 0.004414, True),
        ("Project Scheduling", -0.233258, 0.02171, 0.013759, 0.009793, 12383.328054, 20254.0, 1.930487, 0.007127, True),
        ("Property Compliance", -0.281957, 0.014088, 0.003431, 0.003195, 3088.170825, 9212.0, 1.237712, 0.002772, True),
        ("Quality", -0.316498, -0.09467, 0.026923, 0.012825, 24232.032043, 55348.0, 4.928185, 0.005463, True),
        ("Quality Assurance", -0.09395, -0.004562, 0.080266, 0.054203, 72242.46278, 147778.0, 2.411438, 0.033286, True),
        ("Quality Assurance Standards", 0.370551, -0.001227, 0.00143, 0.000827, 1286.740215, 205.0, 1.379426, 0.001036, True),
        ("Quality Standards", -0.082117, 0.03682, 0.056273, 0.050041, 50647.31967, 55110.0, 6.354929, 0.008855, True),
        ("Quantum Space Technologies", -0.186804, 0.064587, 0.007326, 0.008627, 6593.67999, 11305.0, 6.26033, 0.00117, True),
        ("Radar and Satellite Technologies", -0.236535, 0.000525, 0.021477, 0.035034, 19329.755897, 29272.0, 8.624938, 0.00249, True),
        ("Radio Frequency Technology", 0.091609, -0.012966, 0.003342, 0.003379, 3008.25164, 16042.0, 12.857231, 0.00026, True),
        ("Recreational Equipment Maintenance", -0.420009, -0.016275, 0.004572, 0.006127, 4114.870475, 10751.0, 2.004993, 0.00228, True),
        ("Refrigeration and Air Quality", -0.175843, -0.025335, 0.00333, 0.001317, 2996.747045, 20519.0, 2.212946, 0.001505, True),
        ("Reliability Analysis", -0.326758, 0.002921, 0.128426, 0.102598, 115587.825941, 33343.0, 8.709952, 0.014745, True),
        ("Renewable Energy", -0.374541, -0.11213, 0.012078, 0.003586, 10870.416022, 9023.0, 3.766125, 0.003207, True),
        ("Requirements Analysis", -0.248058, -0.057486, 0.106073, 0.156338, 95469.724692, 75225.0, 4.450143, 0.023836, True),
        ("Resource Acquisition", -0.138557, 0.052445, 0.051271, 0.044436, 46145.800193, 42087.0, 5.00634, 0.010241, True),
        ("Resource Allocation", -0.219504, -0.145653, 0.003723, 0.003255, 3351.102063, 6283.0, 2.656591, 0.001402, True),
        ("Revenue Generation", -0.42047, 0.062753, 0.00419, 0.006648, 3770.944199, 3908.0, 1.276763, 0.003282, True),
        ("Risk Management", -0.016384, -0.062969, 0.006984, 0.006974, 6285.627553, 39529.0, 1.445468, 0.004831, True),
        ("Robotics Automation", -0.328589, 0.088651, 0.012541, 0.012836, 11287.507745, 12715.0, 5.430847, 0.002309, True),
        ("SAP", -0.325049, -0.020958, 0.039422, 0.038364, 35481.488964, 30644.0, 3.542462, 0.011129, True),
        ("SAP Logistics", -0.226016, 0.007717, 0.0127, 0.012389, 11430.444867, 21412.0, 2.187395, 0.005806, True),
        ("SQL Development", -0.086785, -0.017062, 0.055882, 0.055961, 50295.888794, 40018.0, 1.863775, 0.029983, True),
        ("Safety", -0.099023, -0.015371, 0.015376, 0.009816, 13838.793246, 34680.0, 1.869425, 0.008225, True),
        ("Safety Environment", -0.343862, -0.087208, 0.011191, 0.006894, 10072.528523, 28195.0, 2.461458, 0.004547, True),
        ("Safety and Risk Engineering", -0.206266, -0.018814, 0.005981, 0.005407, 5383.272525, 10682.0, 4.7939, 0.001248, True),
        ("Safety and Security", 0.021863, 0.087194, 0.000864, 0.000651, 777.430063, 5338.0, 1.90679, 0.000453, True),
        ("Scientific Analysis", -0.107843, 0.13474, 0.012999, 0.012861, 11699.520707, 9327.0, 3.556884, 0.003655, True),
        ("Security Administration", -0.120726, -0.063656, 0.043121, 0.03461, 38810.77406, 29257.0, 3.169016, 0.013607, True),
        ("Security Enforcement", -0.580701, -0.025919, 0.075713, 0.113266, 68143.956345, 33599.0, 3.447649, 0.021961, True),
        ("Security Operations", -0.26017, -0.018859, 0.01473, 0.014869, 13257.445664, 34461.0, 1.533347, 0.009606, True),
        ("Security Practices", -0.100816, -0.076396, 0.011134, 0.01114, 10021.080858, 74261.0, 3.006295, 0.003704, True),
        ("Security Systems", -0.051669, -0.106124, 0.015295, 0.020599, 13765.63982, 38033.0, 7.775512, 0.001967, True),
        ("Security Technology", -0.125897, -0.057057, 0.00957, 0.009452, 8613.257976, 17344.0, 1.405654, 0.006808, True),
        ("Security Testing", -0.007603, -0.07592, 0.01994, 0.033589, 17946.816336, 26111.0, 7.523229, 0.00265, True),
        ("Security and Defense", -0.575176, -0.020578, 0.0526, 0.064499, 47341.451794, 10359.0, 3.13876, 0.016758, True),
        ("Semiconductor Electronics", -0.208836, -0.06546, 0.123965, 0.162504, 111572.316503, 89513.0, 15.06249, 0.00823, True),
        ("Semiconductor Fabrication Techniques", -0.308358, -0.06703, 0.052768, 0.062302, 47492.933693, 14813.0, 11.342046, 0.004652, True),
        ("Sensor Technology Design", -0.400375, 0.057682, 0.034198, 0.036266, 30779.395946, 66658.0, 6.640548, 0.00515, True),
        ("Sensor and Mobile Technologies", -0.162973, -0.319788, 0.000698, 0.000619, 628.087612, 8203.0, 3.218589, 0.000217, True),
        ("Server Technologies", -0.279834, -0.062569, 0.030559, 0.022921, 27504.065335, 39439.0, 2.835397, 0.010778, True),
        ("SharePoint Solutions", -0.449712, -0.040843, 0.020226, 0.015637, 18203.986676, 5572.0, 2.856959, 0.00708, True),
        ("Signal Processing", 0.211449, -0.049371, 0.021433, 0.028857, 19290.493317, 65438.0, 10.906866, 0.001965, True),
        ("Site Development", -0.307672, -0.088336, 0.006927, 0.001592, 6234.748915, 6636.0, 1.785542, 0.00388, True),
        ("Smart Energy Systems", -0.28878, 0.116197, 0.005678, 0.002719, 5110.758612, 7485.0, 4.162365, 0.001364, True),
        ("Software Architecture", -0.001226, -0.052378, 0.019312, 0.016864, 17381.790923, 74879.0, 2.019051, 0.009565, True),
        ("Software Development", -0.108132, -0.051352, 0.082905, 0.13262, 74616.949208, 138740.0, 3.608763, 0.022973, True),
        ("Software Development Practices", 0.026314, -0.01783, 0.04497, 0.059179, 40474.920129, 112683.0, 2.726972, 0.016491, True),
        ("Software Development Technologies", -0.070819, -0.061589, 0.070789, 0.070923, 63712.976, 133119.0, 2.702595, 0.026193, True),
        ("Software Engineering", -0.476082, -0.003427, 0.010833, 0.011096, 9750.183134, 25825.0, 4.032846, 0.002686, True),
        ("Software Integration", 0.013791, 0.019629, 0.01756, 0.021374, 15804.964439, 67311.0, 3.193698, 0.005498, True),
        ("Software Quality", 0.029441, -0.063743, 0.074919, 0.099185, 67429.55925, 122093.0, 6.344767, 0.011808, True),
        ("Software Testing", -0.0794, -0.041446, 0.143815, 0.245919, 129438.140836, 88909.0, 11.845835, 0.012141, True),
        ("Solar Energy", -0.646688, -0.150368, 0.004065, 0.000729, 3658.893828, 4149.0, 2.316791, 0.001755, True),
        ("Spectroscopy", 0.167873, -0.042966, 0.008067, 0.005492, 7260.344259, 12409.0, 3.509848, 0.002298, True),
        ("Stakeholder Engagement", -0.564222, -0.005203, 0.003659, 0.002473, 3293.604592, 1798.0, 1.634139, 0.002239, True),
        ("Statistical Analysis Software", -0.05747, 0.07487, 0.009396, 0.008575, 8456.367279, 6367.0, 2.287222, 0.004108, True),
        ("Stochastic Modeling", -0.21041, 0.026109, 0.004068, 0.006131, 3660.967595, 11598.0, 1.985005, 0.002049, True),
        ("Storage", -0.352088, 0.093934, 0.003426, 0.003719, 3083.512897, 10535.0, 1.576245, 0.002174, True),
        ("Strategic Collaboration", -0.389402, 0.015595, 0.026526, 0.021196, 23874.3218, 9030.0, 1.69551, 0.015645, True),
        ("Strategic Sourcing", -0.13054, 0.064466, 0.058075, 0.036159, 52269.160604, 67905.0, 1.747625, 0.033231, True),
        ("Structural Analysis", -0.240459, 0.080841, 0.020516, 0.015088, 18465.283647, 56647.0, 5.778489, 0.00355, True),
        ("Structural Engineering", -0.346809, -0.080835, 0.001774, 0.00071, 1596.47032, 8410.0, 1.349028, 0.001315, True),
        ("Structural Metal Applications", -0.050205, -0.063698, 0.014835, 0.015961, 13351.699862, 10875.0, 4.538233, 0.003269, True),
        ("Supplier Relations", -0.154088, 0.03519, 0.112596, 0.1091, 101339.921344, 59920.0, 5.396555, 0.020864, True),
        ("Supply Chain Coordination", -0.247059, 0.006471, 0.084351, 0.044415, 75918.489072, 61703.0, 4.270343, 0.019753, True),
        ("Supply Chain Event", -0.501596, 0.067558, 0.024065, 0.012805, 21659.033881, 11711.0, 3.664166, 0.006568, True),
        ("Supply Chain Logistics", -0.485803, -0.000484, 0.034847, 0.033069, 31363.703087, 14709.0, 3.942743, 0.008838, True),
        ("Supply Chain Strategy", -0.259838, 0.052046, 0.082479, 0.064248, 74234.174824, 56144.0, 3.588559, 0.022984, True),
        ("Surface Finishing", 0.032512, 0.020103, 0.00258, 0.002978, 2321.898712, 18174.0, 1.244327, 0.002073, True),
        ("Surface Treatment Techniques", -0.078743, -0.039696, 0.002557, 0.002084, 2301.06769, 12972.0, 2.774272, 0.000922, True),
        ("Sustainable Energy", 0.083222, -0.040076, 0.019519, 0.010221, 17567.40952, 2774.0, 3.028337, 0.006445, True),
        ("Sustainable Environment", -0.113704, -0.067981, 0.004823, 0.005136, 4340.739281, 10020.0, 1.856969, 0.002597, True),
        ("Sustainable Waste Practices", -0.225075, -0.017166, 0.003055, 0.003218, 2749.295737, 21067.0, 2.016657, 0.001515, True),
        ("System Analysis", -0.381723, -0.164126, 0.008757, 0.014838, 7881.879359, 9129.0, 11.523315, 0.00076, True),
        ("System Integration", -0.51708, 0.199577, 0.019533, 0.02136, 17579.965361, 16868.0, 3.805616, 0.005133, True),
        ("System Security", -0.081712, -0.122114, 0.010644, 0.006491, 9579.534303, 29814.0, 4.305483, 0.002472, True),
        ("System Testing", 0.004173, -0.010137, 0.027328, 0.033687, 24595.743783, 65053.0, 10.60458, 0.002577, True),
        ("Systems Integration", -0.341093, -0.093573, 0.090849, 0.155188, 81766.829405, 61362.0, 11.280997, 0.008053, True),
        ("Systems Visualization", 0.048199, -0.053107, 0.09393, 0.182538, 84540.21588, 72055.0, 27.973799, 0.003358, True),
        ("Tactical Communication", -0.184517, -0.038428, 0.003829, 0.004173, 3446.267844, 8669.0, 3.801414, 0.001007, True),
        ("Tactical Operations", -0.300712, -0.009074, 0.067565, 0.087928, 60811.016275, 29387.0, 4.344573, 0.015552, True),
        ("Technical", -0.299983, 0.018148, 0.104667, 0.12882, 94204.173211, 51027.0, 2.705233, 0.038691, True),
        ("Technical Coordination", -0.339197, -0.007029, 0.038552, 0.038809, 34697.859595, 38224.0, 3.126807, 0.012329, True),
        ("Technical Evaluation", -0.392661, -0.009902, 0.041633, 0.046647, 37470.889472, 58597.0, 2.429512, 0.017136, True),
        ("Technical Leadership", -0.466465, 0.067556, 0.098525, 0.139155, 88675.692802, 31353.0, 4.659982, 0.021143, True),
        ("Technology Infrastructure", -0.541877, 0.072237, 0.002539, 0.00245, 2285.550012, 15542.0, 1.851094, 0.001372, True),
        ("Technology Integration", -0.093882, 0.099486, 0.068929, 0.075145, 62038.743335, 46441.0, 2.301575, 0.029949, True),
        ("Technology Strategy", -0.407715, 0.020526, 0.090155, 0.07653, 81143.018388, 10790.0, 1.30758, 0.068948, True),
        ("Telecommunications", -0.107149, -0.048894, 0.005887, 0.004042, 5298.600488, 11562.0, 1.658867, 0.003549, True),
        ("Telecommunications Infrastructure", -0.477403, -0.08629, 0.025351, 0.021298, 22817.102645, 7758.0, 3.31049, 0.007658, True),
        ("Telecommunications Technologies", -0.022422, 0.023397, 0.011558, 0.007843, 10402.286625, 18717.0, 4.201994, 0.002751, True),
        ("Testing", 0.156834, -0.011136, 0.016929, 0.018703, 15236.591342, 97725.0, 4.637193, 0.003651, True),
        ("Textile Manufacturing", -0.106596, -0.252737, 0.001228, 0.00117, 1105.51663, 1359.0, 1.252081, 0.000981, True),
        ("Thermal Fluid", -0.211845, -0.045405, 0.003677, 0.002363, 3309.875405, 7107.0, 7.842467, 0.000469, True),
        ("Thermal Systems", -0.197521, -0.069762, 0.007908, 0.004139, 7117.458945, 18514.0, 2.984793, 0.002649, True),
        ("Training", -0.034824, 0.109556, 0.014558, 0.011069, 13102.31557, 4279.0, 2.086719, 0.006976, True),
        ("Transmission Systems", -0.197694, -0.05205, 0.003127, 0.00232, 2814.670041, 4266.0, 3.363661, 0.00093, True),
        ("Transportation Demand Analysis", -0.47466, -0.042542, 0.00094, 0.001112, 846.253449, 305.0, 1.534532, 0.000613, True),
        ("Transportation Infrastructure", -0.250941, 0.105026, 0.003019, 0.00086, 2717.321216, 4928.0, 1.274203, 0.002369, True),
        ("Underground Construction", -0.305588, -0.061028, 0.012724, 0.007862, 11452.299477, 20891.0, 1.918859, 0.006631, True),
        ("Unix and Linux Systems", -0.008727, -0.098721, 0.103601, 0.151029, 93244.439172, 84720.0, 4.891163, 0.021181, True),
        ("Unmanned Technology", -0.012244, 0.06596, 0.006898, 0.009486, 6208.555277, 12283.0, 7.483304, 0.000922, True),
        ("User Administration", -0.559553, -0.086278, 0.003999, 0.002904, 3599.289118, 15525.0, 1.459549, 0.00274, True),
        ("User Experience Design", -0.185492, 0.073252, 0.024239, 0.021382, 21816.237239, 62115.0, 1.35906, 0.017835, True),
        ("User Experience Research", -0.194419, 0.087212, 0.014304, 0.019278, 12873.700767, 8284.0, 2.235946, 0.006397, True),
        ("User Interface Design", -0.545776, -0.06529, 0.002729, 0.002281, 2456.425138, 9505.0, 2.219464, 0.00123, True),
        ("Utility Maintenance Coordination", -0.246213, -0.056444, 0.029624, 0.020024, 26662.768646, 33987.0, 2.201015, 0.013459, True),
        ("Validation", -0.583162, 0.093331, 0.006371, 0.004319, 5733.823307, 20940.0, 1.678328, 0.003796, True),
        ("Vehicle Engineering", -0.126145, 0.03657, 0.154201, 0.158476, 138786.465453, 95859.0, 10.727287, 0.014375, True),
        ("Vehicle Maintenance", -0.257084, -0.197001, 0.001314, 0.000504, 1182.436202, 3722.0, 1.671139, 0.000786, True),
        ("Vehicle Operations", -0.057955, -0.246919, 0.000738, 0.000165, 664.35322, 7652.0, 1.41884, 0.00052, True),
        ("Vehicle Restoration", -0.220799, 0.34123, 0.001058, 0.000757, 952.440172, 3166.0, 1.500276, 0.000705, True),
        ("Vehicle Technology", 0.112594, 0.011898, 0.00528, 0.007203, 4752.193501, 5394.0, 8.496657, 0.000621, True),
        ("Vehicle Tracking", -0.076944, -0.075485, 0.002289, 0.002448, 2059.798181, 6469.0, 1.962495, 0.001166, True),
        ("Vendor Relations", -0.32631, 0.045111, 0.012556, 0.007747, 11300.424426, 1027.0, 1.237371, 0.010147, True),
        ("Virtual Infrastructure", -0.374259, -0.080257, 0.012505, 0.007988, 11254.813136, 15014.0, 2.870453, 0.004356, True),
        ("Virtualization", -0.177517, -0.05543, 0.023673, 0.022914, 21306.849125, 54713.0, 3.081371, 0.007683, True),
        ("Virtualization Infrastructure", -0.259842, -0.096701, 0.031196, 0.021598, 28077.241736, 34721.0, 3.077013, 0.010138, True),
        ("Visual Data Processing", -0.12064, 0.076499, 0.001095, 0.001042, 985.820533, 3411.0, 1.471804, 0.000744, True),
        ("Visual Drawing", 0.178424, 0.059425, 0.063269, 0.057289, 56944.168968, 11655.0, 6.498191, 0.009736, True),
        ("Voice Communication Systems", -0.292195, -0.004997, 0.008484, 0.005109, 7635.823425, 5902.0, 1.747116, 0.004856, True),
        ("Voice Networking", -0.143014, -0.132455, 0.003287, 0.001578, 2958.49651, 8998.0, 2.151876, 0.001528, True),
        ("Water Resource Assessment", -0.636523, -0.343461, 0.003846, 0.000274, 3461.221824, 1211.0, 4.776776, 0.000805, True),
        ("Water Treatment", -0.259904, -0.097914, 0.004696, 0.001349, 4226.755765, 15939.0, 1.796781, 0.002614, True),
        ("Wearable Technology Integration", -0.115898, -0.158662, 0.005102, 0.004668, 4592.069474, 32349.0, 2.834326, 0.0018, True),
        ("Weather Science", -0.281714, -0.019937, 0.00146, 0.001785, 1314.051908, 4049.0, 2.955338, 0.000494, True),
        ("Web Development", -0.071486, 0.013353, 0.116863, 0.142766, 105180.452415, 133419.0, 1.941883, 0.06018, True),
        ("Web Scripting", 0.242053, -0.224747, 0.001336, 0.001099, 1202.205861, 1576.0, 2.204145, 0.000606, True),
        ("Web Server", -0.267491, 0.203528, 0.000969, 0.001363, 872.137229, 1274.0, 1.239447, 0.000782, True),
        ("Web and Server Technologies", -0.007656, -0.087714, 0.007883, 0.005118, 7095.264326, 10111.0, 3.847842, 0.002049, True),
        ("Welding Techniques", -0.339908, -0.063833, 0.018707, 0.00818, 16837.020138, 35997.0, 3.859902, 0.004847, True),
        ("Windows Systems", -0.21318, 0.001474, 0.19543, 0.223763, 175893.958394, 99664.0, 2.836286, 0.068904, True),
        ("Wireless Communication", -0.197785, -0.030935, 0.016446, 0.018078, 14801.948351, 10296.0, 6.630844, 0.00248, True),
        ("Wireless Networking", 0.093611, 0.227199, 0.000688, 0.000545, 618.983529, 1015.0, 4.183978, 0.000164, True),
        ("Wireless and Virtualization Technologies", -0.176492, -0.082391, 0.051926, 0.102384, 46735.381394, 35961.0, 24.332722, 0.002134, True),
        ("Workflow Integration", -0.338977, -0.055956, 0.015146, 0.015966, 13631.532936, 3877.0, 5.898158, 0.002568, True),
        ("Workforce Technology", -0.372874, -0.039869, 0.004449, 0.003797, 4004.636728, 3732.0, 1.549448, 0.002872, True),
        ("XML", -0.232287, 0.033594, 0.013142, 0.01465, 11827.835769, 9984.0, 2.185759, 0.006012, True),
    ]
    df = pd.DataFrame(data, columns=[
        "skill", "peer_postings_share_growth", "peer_hires_share_growth",
        "peer_share", "company_share", "peer_headcount", "peer_postings",
        "lift", "economy_share", "is_specialized"])
    # Already live-gated; re-apply gates so CONFIG knobs still work offline.
    # Already live-gated at freeze time. Do NOT recompute lift percentile —
    # that raises the floor on an already-truncated set and drops live targets.
    out = df.copy()
    min_posts = int(cfg.get("min_peer_postings", 50))
    if "peer_postings" in out.columns:
        out = out[out["peer_postings"].fillna(0) >= min_posts]
    if cfg.get("require_specialized", True) and "is_specialized" in out.columns:
        out = out[out["is_specialized"].fillna(False).astype(bool)]
    return out.reset_index(drop=True)


def _load_role_categories_synthetic(_cfg, include_roles=None):
    data = [
        ("IT Engineer", 0.181062, 7.6e-05, -0.17043, 0.103572, 12000),
        ("Software Engineering", 0.215543, -0.000239, -0.486781, 0.095449, 11000),
        ("Engineering Managers", 0.243932, -0.001034, 0.05354, 0.112305, 10500),
        ("Aerospace Engineer", 0.199386, 0.001257, -0.152368, 0.129601, 10000),
        ("Senior Engineer", 0.121123, 0.001643, -0.155555, 0.020713, 9500),
        ("Domain Software Engineer", 0.3073, -0.000449, 0.194491, 0.11753, 9000),
        ("Mechanical Engineering", 0.194409, -0.000835, -0.062648, 0.099606, 8500),
        ("Industrial Quality Engineer", 0.19313, 0.000621, -0.199814, 0.149805, 8000),
        ("Manufacturing Engineer", 0.196886, 0.000227, 0.019817, 0.167167, 7500),
        ("Senior Program Manager", 0.106293, 0.000415, -0.440533, 0.06873, 7000),
        ("Project Engineer", 0.144138, 0.000408, 0.00918, 0.139424, 6500),
        ("Aircraft Mechanic", 0.101615, 0.000722, -0.462624, 0.045117, 6000),
        ("Electrical Power Engineer", 0.199097, 0.001645, -0.09871, 0.107545, 5500),
        ("Finance Manager", 0.241932, -0.000296, 0.161754, 0.108557, 5000),
        ("Systems Analyst", 0.260069, 0.001645, -0.389914, 0.057567, 4500),
        ("Optical Engineer", 0.203404, -0.000151, -0.067827, 0.133168, 4000),
        ("Test Engineer", 0.249641, -0.000931, 0.049068, 0.101745, 3800),
        ("Production Planner", 0.204194, -0.000192, -0.122652, 0.118001, 3600),
        ("Security Architect", 0.20984, -0.000382, -0.139545, 0.129023, 3400),
        ("Assembly Technician", 0.141204, -0.000397, -0.486563, 0.054176, 3200),
        ("Hardware Engineer", 0.210921, 0.004741, -0.004431, 0.115828, 3000),
        ("Administrative Assistant", 0.185947, 0.0, -0.169003, 0.015885, 2800),
        ("System Administrator", 0.281206, 0.001112, -0.572708, 0.060446, 2600),
        ("Project Planner", 0.161426, 0.001947, -0.162764, 0.091201, 2400),
        ("Aerospace Manager", 0.143682, 0.000815, -0.455493, 0.088299, 2200),
        # Pathway feeders for default target (Data Analysis) — included so
        # source_role→category merges are not all fillna("stable").
        ("Technical Reporting Analyst", 0.22, 0.0012, 0.08, 0.09, 900),
        ("Process Engineer", 0.18, 0.0004, -0.05, 0.11, 880),
        ("Site Engineer", 0.15, 0.0002, 0.12, 0.10, 860),
        ("Systems Design Engineers", 0.28, 0.0021, -0.25, 0.08, 840),
        ("Category Manager", 0.12, 0.0001, 0.05, 0.07, 820),
        ("Operations Engineering Specialist", 0.20, 0.0008, -0.18, 0.12, 800),
        ("Thermal Engineer", 0.17, 0.0005, -0.02, 0.09, 780),
        ("Product Design Engineer", 0.24, 0.0018, 0.03, 0.10, 760),
        ("Quality Analyst", 0.16, 0.0003, -0.22, 0.13, 740),
        ("Engineering Technician", 0.14, 0.0006, -0.35, 0.11, 720),
        ("Solutions Engineer", 0.26, 0.0015, 0.15, 0.09, 700),
        ("Manufacturing Excellence Engineer", 0.19, 0.0009, -0.08, 0.14, 680),
    ]
    df = pd.DataFrame(data, columns=[
        "role", "ai_exposure", "skill_mix_change", "hiring_growth", "attrition",
        "headcount"])
    # Ensure any requested include_roles exist (generic mid metrics if unknown).
    have = set(df["role"])
    extra = []
    for r in include_roles or []:
        if r and r not in have:
            h = abs(hash(r)) % 1000 + 200
            extra.append((r, 0.18, 0.0005, -0.05, 0.10, h))
            have.add(r)
    if extra:
        df = pd.concat([df, pd.DataFrame(extra, columns=df.columns)], ignore_index=True)
    top_n = int((_cfg or {}).get("role_category_top_n", 25))
    # Keep top_n by HC plus every include_role.
    include = set(include_roles or [])
    df = df.sort_values("headcount", ascending=False)
    keep = set(df.head(top_n)["role"]) | include
    return df[df["role"].isin(keep)].reset_index(drop=True)



def _load_pathways_synthetic(_cfg, _target_skill):
    # Frozen pathways for default target "Data Analysis" (offline ignores skill).
    data = [
        ("Technical Reporting Analyst", 79.0648, "low-med", "med-high", -0.11475847103702119, 0.141166, 0.033222, 0.729047, 110044.233951, 124309.84127),
        ("Process Engineer", 97.559046, "low-med", "med-high", -0.06321080561057979, 0.084267, 0.025143, 0.121738, 116452.116058, 124309.84127),
        ("Site Engineer", 136.328771, "low-med", "high", -0.34961363203420326, 0.06739, 0.027732, 0.115384, 80849.426166, 124309.84127),
        ("Systems Design Engineers", 383.814425, "med", "high", 0.16475679209915217, 0.058222, 0.012818, 0.04475, 144790.731944, 124309.84127),
        ("Category Manager", 81.237663, "low-med", "med-high", 0.3595971691968358, 0.049291, 0.088, 0.0, 169011.308294, 124309.84127),
        ("Operations Engineering Specialist", 149.583634, "low-med", "med-high", -0.10635564935107578, 0.047509, 0.030342, 0.428175, 111088.787381, 124309.84127),
        ("Thermal Engineer", 537.848674, "med", "high", -0.04612189197914818, 0.046178, 0.047824, 0.086598, 118576.436199, 124309.84127),
        ("Product Design Engineer", 395.773133, "med", "high", 0.08526055587167591, 0.044288, 0.019913, 0.057691, 134908.567437, 124309.84127),
        ("Quality Analyst", 119.119142, "low-med", "med-high", 0.18047481723729164, 0.042728, 0.00171, 0.803526, 146744.637154, 124309.84127),
        ("Engineering Technician", 204.932874, "low-med", "high", -0.5884938706430061, 0.041412, 0.057191, 0.117832, 51154.261622, 124309.84127),
        ("Solutions Engineer", 82.973126, "low-med", "med-high", 0.06159413137990888, 0.03755, 0.003069, 0.321865, 131966.597965, 124309.84127),
        ("Manufacturing Excellence Engineer", 393.451599, "low-med", "high", 0.30518881961725786, 0.03606, 0.0227, 0.07054, 162247.814994, 124309.84127),
    ]
    df = pd.DataFrame(data, columns=["source_role", "feeder_pool", "transition_freq", "skill_overlap", "wage_gap", "conversion_rate", "peer_conversion_rate", "skill_mover_share", "source_median_comp", "target_median_comp"])
    if "peer_conversion_rate" in df.columns and "conversion_rate" in df.columns:
        df["mobility_gap"] = df["peer_conversion_rate"] - df["conversion_rate"]
    df["transition_wt"] = df["feeder_pool"] * df["conversion_rate"]
    return df
def _load_target_population_synthetic(_cfg, target_skill):
    _POPS = {
        "Data Analysis": (2535.1412, 0.091832, 0.129366),
        "Energy Economics": (188.1794, 0.038661, 0.089871),
        "Process": (26489.611, 0.062417, 0.080572),
        "Lean Quality": (27285.5373, 0.05383, 0.064282),
        "Engineering Project Delivery": (33297.5056, 0.053718, 0.06668),
        "Production Engineering": (31020.1393, 0.055438, 0.055444),
        "Manufacturing": (22834.6227, 0.055651, 0.058959),
        "Production": (19471.8695, 0.051468, 0.060636),
        "Engineering Simulation": (30406.4839, 0.065615, 0.081116),
        "Windows Systems": (24838.9087, 0.063887, 0.082393),
        "Graphics Programming": (29366.1875, 0.065405, 0.071927),
        "Embedded Systems Engineering": (31625.0355, 0.055537, 0.047495),
        "Electrical Design": (20996.7284, 0.065529, 0.080633),
        "Electrical Engineering": (24199.0946, 0.052702, 0.051709),
        "Vehicle Engineering": (17591.6894, 0.059303, 0.069342),
        "CAD Drafting": (16057.302, 0.072759, 0.097594),
        "Software Testing": (27298.2831, 0.049403, 0.04958)
    }
    if target_skill in _POPS:
        return _POPS[target_skill]
    # Fallback: median-ish demo scale
    return (1500.0, 0.08, 0.10)


def _load_metro_supply_synthetic(_cfg, _target_skill):
    data = [
        ("new york city metropolitan area", 195575.489641, 12528, True),
        ("washington metropolitan area", 104934.397957, 76408, True),
        ("chicago metropolitan area", 80519.289437, 11529, True),
        ("san francisco metropolitan area", 75073.783505, 4814, True),
        ("boston metropolitan area", 74801.936643, 33386, True),
        ("atlanta metropolitan area", 57571.657301, 3987, True),
        ("dallas metropolitan area", 55492.783851, 13079, True),
        ("los angeles metropolitan area", 53686.999129, 18875, True),
        ("philadelphia metropolitan area", 53640.268401, 9667, True),
        ("seattle metropolitan area", 51388.191413, 32928, True),
        ("houston metropolitan area", 44242.869534, 10375, True),
        ("san jose metropolitan area (california)", 40551.193374, 6492, True),
        ("denver metropolitan area", 34645.565895, 22299, True),
        ("austin metropolitan area", 32032.802274, 9644, True),
        ("minneapolis metropolitan area", 31943.588772, 6294, True),
        ("detroit metropolitan area", 28497.319573, 2832, True),
        ("raleigh metropolitan area", 27768.413079, 2217, True),
        ("phoenix metropolitan area", 27641.79348, 14528, True),
        ("anaheim metropolitan area", 26526.683609, 5244, True),
    ]
    return pd.DataFrame(data, columns=["metro", "external_supply", "competitor_demand", "company_presence"])


def _load_competitor_outflows_synthetic(_cfg, _target_skill):
    data = [
        (851738, "Northrop Grumman Corp.", 11.361188),
        (396968, "RTX Corp.", 10.665959),
        (2623, "BAE Systems Plc", 8.216626),
        (1418721, "L3Harris Technologies, Inc.", 7.063995),
        (961524, "The Boeing Co.", 3.172272),
        (694089, "Textron, Inc.", 2.037461),
        (949998, "CACI International, Inc.", 2.019265),
        (1012186, "Space Exploration Technologies Corp.", 2.016367),
        (828625, "General Dynamics Corp.", 2.012339),
        (371876, "Parker-Hannifin Corp.", 1.235501),
    ]
    return pd.DataFrame(data, columns=["dest_rcid", "dest_company", "outflow_wt"])


def _load_peer_role_categories_synthetic(_cfg):
    data = [
        ("Software Engineering", 0.240112, -0.000648, -0.516949, 0.091914),
        ("IT Engineer", 0.206662, 0.000146, -0.238605, 0.103365),
        ("Senior Engineer", 0.149725, -0.000371, -0.170619, 0.043979),
        ("Engineering Managers", 0.240251, 0.000519, -0.201092, 0.100813),
        ("Domain Software Engineer", 0.318737, -0.000364, 0.029737, 0.136115),
        ("Aerospace Engineer", 0.225415, 0.00028, -0.172933, 0.127131),
        ("Mechanical Engineering", 0.207583, -0.000283, -0.052374, 0.119528),
        ("Manufacturing Engineer", 0.233266, 0.000216, -0.110222, 0.155205),
        ("Electrical Power Engineer", 0.219303, 0.000305, -0.198032, 0.119892),
        ("Senior Program Manager", 0.119233, 0.000137, -0.393961, 0.076104),
        ("Industrial Quality Engineer", 0.228246, 0.000156, -0.16911, 0.141603),
        ("Product Design Engineer", 0.244321, -0.000473, -0.25207, 0.103748),
        ("Aircraft Mechanic", 0.105299, -0.00011, -0.389082, 0.064858),
        ("Procurement Specialist", 0.168548, 6.5e-05, -0.00468, 0.128349),
        ("Project Manager", 0.129104, -0.000112, -0.487507, 0.062907),
        ("Assembly Technician", 0.16324, 3.5e-05, -0.356449, 0.050676),
        ("Production Planner", 0.210872, 0.000116, -0.184387, 0.114131),
        ("Hardware Engineer", 0.267733, 0.000832, -0.271337, 0.132824),
        ("Project Engineer", 0.185112, 0.000665, -0.19448, 0.101007),
        ("Test Engineer", 0.263418, -0.000114, -0.271218, 0.116419),
        ("Thermal Engineer", 0.309943, 0.00035, -0.249981, 0.208087),
        ("System Administrator", 0.27712, 0.000456, -0.36769, 0.095742),
        ("Welder", 0.084799, -0.000305, -0.558958, 0.038293),
        ("Electronics Technician", 0.22113, 0.000407, -0.245533, 0.076664),
        ("Optical Engineer", 0.249101, -0.000106, -0.230844, 0.115177),
    ]
    return pd.DataFrame(data, columns=[
        "role", "ai_exposure", "skill_mix_change", "hiring_growth", "attrition"])



# --- Snowflake loaders -------------------------------------------------------

_RADAR_UNIVERSE_CACHE = {}


def load_skill_radar(cfg):
    """Skill universe: size gate in SQL; lift percentile + postings in Python.

    Cached by company/peer/entry knobs — recompute with refresh_staging=True.
    Share floor is applied later in bucketing (emerging vs nascent).
    """
    if not cfg.get("use_snowflake") or cfg.get("company_rcid") is None:
        return _load_skill_radar_synthetic(cfg)

    peers = _resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    peer_sql = _sql_quote_list(peers["peer_rcids"]) or str(company_rcid)
    recent_m = int(cfg.get("recent_months", 12))
    prior_m = int(cfg.get("prior_months", 12))
    min_hc = int(cfg.get("min_skill_headcount", 500))
    min_posts = int(cfg.get("min_peer_postings", 50))
    lift_pct = float(cfg.get("lift_floor_percentile", 50))
    country = cfg.get("country", "United States")
    max_n = cfg.get("radar_universe_max_skills")
    # backward compat: old top_skills acted as a hard cap
    if max_n is None and cfg.get("top_skills") is not None:
        # Only honor top_skills if explicitly kept as safety valve via radar_universe_max_skills;
        # ignore legacy top_skills for entry (display caps are within-bucket).
        max_n = None

    cache_key = (
        int(company_rcid), tuple(peers.get("peer_rcids") or ()),
        cfg.get("skill_level"), country, recent_m, prior_m, min_hc, min_posts,
        lift_pct, max_n, cfg.get("exclude_contingent", True),
        bool(cfg.get("require_specialized", True)),
        "share_growth_v2",  # composition vs market, not within-skill volume
    )
    if cache_key in _RADAR_UNIVERSE_CACHE and not cfg.get("refresh_staging"):
        return _RADAR_UNIVERSE_CACHE[cache_key].copy()

    limit_sql = f"\n    LIMIT {int(max_n)}" if max_n else ""

    q = f"""
    WITH skill_names AS ({_skill_taxonomy(cfg)}),
    peer_rcids AS (
        SELECT {company_rcid} AS rcid, 'company' AS peer_type
        UNION ALL
        SELECT rcid, 'peer' AS peer_type
        FROM (SELECT value::INT AS rcid FROM TABLE(SPLIT_TO_TABLE('{peer_sql}', ',')))
    ),
    postings AS (
        SELECT
            pr.peer_type,
            sn.skill,
            CASE
                WHEN p.post_date >= DATEADD('month', -{recent_m}, CURRENT_DATE()) THEN 'recent'
                WHEN p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE()) THEN 'prior'
            END AS period,
            COUNT(DISTINCT p.job_id) AS n_jobs
        FROM service_pipelines.output_current.postings_unique_unified p
        JOIN service_pipelines.output_current.postings_unique_unified_skills_v3 ps
          USING (job_id)
        JOIN skill_names sn ON ps.skill_v3_id = sn.skill_v3_id
        JOIN peer_rcids pr ON p.rcid = pr.rcid
        WHERE p.country_v3 = '{country}'
          AND p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
        GROUP BY 1, 2, 3
    ),
    peer_postings_n AS (
        SELECT skill, SUM(n_jobs) AS peer_postings
        FROM postings
        WHERE peer_type = 'peer' AND period IS NOT NULL
        GROUP BY 1
    ),
    -- Period totals = distinct peer jobs (not sum over skills; jobs are multi-tagged).
    peer_postings_period_tot AS (
        SELECT
            CASE
                WHEN p.post_date >= DATEADD('month', -{recent_m}, CURRENT_DATE()) THEN 'recent'
                WHEN p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE()) THEN 'prior'
            END AS period,
            COUNT(DISTINCT p.job_id) AS tot_jobs
        FROM service_pipelines.output_current.postings_unique_unified p
        JOIN peer_rcids pr ON p.rcid = pr.rcid
        WHERE pr.peer_type = 'peer'
          AND p.country_v3 = '{country}'
          AND p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
        GROUP BY 1
    ),
    -- True composition growth: (skill jobs / all peer jobs)_recent
    --   / (skill jobs / all peer jobs)_prior − 1
    postings_growth AS (
        SELECT
            skill,
            (recent_share / NULLIF(prior_share, 0)) - 1 AS peer_postings_share_growth
        FROM (
            SELECT
                skill,
                SUM(CASE WHEN period = 'recent' THEN n_jobs ELSE 0 END)::FLOAT
                    / NULLIF((SELECT tot_jobs FROM peer_postings_period_tot WHERE period = 'recent'), 0)
                    AS recent_share,
                SUM(CASE WHEN period = 'prior' THEN n_jobs ELSE 0 END)::FLOAT
                    / NULLIF((SELECT tot_jobs FROM peer_postings_period_tot WHERE period = 'prior'), 0)
                    AS prior_share
            FROM postings
            WHERE peer_type = 'peer' AND period IS NOT NULL
            GROUP BY skill
        ) x
    ),
    {_SKILL_USERS_CTE}
    hires AS (
        SELECT
            pr.peer_type,
            su.skill,
            CASE
                WHEN p.startdate >= DATEADD('month', -{recent_m}, CURRENT_DATE()) THEN 'recent'
                WHEN p.startdate >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE()) THEN 'prior'
            END AS period,
            SUM(COALESCE(p.weight_v2_1, 1)) AS n_hires
        FROM service_pipelines.output_current.individual_position p
        JOIN peer_rcids pr ON p.ultimate_parent_rcid = pr.rcid
        JOIN skill_users su ON p.user_id = su.user_id
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND p.startdate >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
        GROUP BY 1, 2, 3
    ),
    -- Period totals = all peer hires (no skill join; skill shares may sum > 1).
    peer_hires_period_tot AS (
        SELECT
            CASE
                WHEN p.startdate >= DATEADD('month', -{recent_m}, CURRENT_DATE()) THEN 'recent'
                WHEN p.startdate >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE()) THEN 'prior'
            END AS period,
            SUM(COALESCE(p.weight_v2_1, 1)) AS tot_hires
        FROM service_pipelines.output_current.individual_position p
        JOIN peer_rcids pr ON p.ultimate_parent_rcid = pr.rcid
        WHERE pr.peer_type = 'peer'
          AND p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND p.startdate >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
        GROUP BY 1
    ),
    hires_growth AS (
        SELECT
            skill,
            (recent_share / NULLIF(prior_share, 0)) - 1 AS peer_hires_share_growth
        FROM (
            SELECT
                skill,
                SUM(CASE WHEN period = 'recent' THEN n_hires ELSE 0 END)::FLOAT
                    / NULLIF((SELECT tot_hires FROM peer_hires_period_tot WHERE period = 'recent'), 0)
                    AS recent_share,
                SUM(CASE WHEN period = 'prior' THEN n_hires ELSE 0 END)::FLOAT
                    / NULLIF((SELECT tot_hires FROM peer_hires_period_tot WHERE period = 'prior'), 0)
                    AS prior_share
            FROM hires
            WHERE peer_type = 'peer' AND period IS NOT NULL
            GROUP BY skill
        ) x
    ),
    headcount AS (
        SELECT peer_type, skill, SUM(wt) AS headcount
        FROM (
            SELECT
                pr.peer_type,
                su.skill,
                p.user_id,
                COALESCE(p.weight_v2_1, 1) AS wt,
                {_rn_latest_position("pr.peer_type, su.skill, p.user_id")} AS rn
            FROM service_pipelines.output_current.individual_position p
            JOIN peer_rcids pr ON p.ultimate_parent_rcid = pr.rcid
            JOIN skill_users su ON p.user_id = su.user_id
            WHERE p.country = '{country}'
              AND {_pos_filter(cfg)}
              AND p.enddate_primary IS NULL
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
                {_rn_latest_position("pr.peer_type, p.user_id")} AS rn
            FROM service_pipelines.output_current.individual_position p
            JOIN peer_rcids pr ON p.ultimate_parent_rcid = pr.rcid
            WHERE p.country = '{country}'
              AND {_pos_filter(cfg)}
              AND p.enddate_primary IS NULL
        ) x
        WHERE rn = 1
        GROUP BY 1
    ),
    shares AS (
        SELECT
            h.skill,
            SUM(CASE WHEN h.peer_type = 'peer' THEN h.headcount ELSE 0 END) AS peer_headcount,
            SUM(CASE WHEN h.peer_type = 'peer' THEN h.headcount ELSE 0 END)::FLOAT
                / NULLIF((SELECT total_hc FROM workforce WHERE peer_type = 'peer'), 0) AS peer_share,
            SUM(CASE WHEN h.peer_type = 'company' THEN h.headcount ELSE 0 END)::FLOAT
                / NULLIF((SELECT total_hc FROM workforce WHERE peer_type = 'company'), 0) AS company_share
        FROM headcount h
        GROUP BY h.skill
    ),
    economy_headcount AS (
        SELECT skill, SUM(wt) AS headcount
        FROM (
            SELECT
                su.skill,
                p.user_id,
                COALESCE(p.weight_v2_1, 1) AS wt,
                {_rn_latest_position("su.skill, p.user_id")} AS rn
            FROM service_pipelines.output_current.individual_position p
            JOIN skill_users su ON p.user_id = su.user_id
            WHERE p.country = '{country}'
              AND {_pos_filter(cfg)}
              AND p.enddate_primary IS NULL
        ) x
        WHERE rn = 1
        GROUP BY 1
    ),
    economy_workforce AS (
        SELECT SUM(wt) AS total_hc
        FROM (
            SELECT
                p.user_id,
                COALESCE(p.weight_v2_1, 1) AS wt,
                {_rn_latest_position("p.user_id")} AS rn
            FROM service_pipelines.output_current.individual_position p
            WHERE p.country = '{country}'
              AND {_pos_filter(cfg)}
              AND p.enddate_primary IS NULL
        ) x
        WHERE rn = 1
    ),
    economy_shares AS (
        SELECT
            h.skill,
            h.headcount::FLOAT
                / NULLIF((SELECT total_hc FROM economy_workforce), 0) AS economy_share
        FROM economy_headcount h
    )
    SELECT
        s.skill,
        COALESCE(pg.peer_postings_share_growth, 0) AS peer_postings_share_growth,
        COALESCE(hg.peer_hires_share_growth, 0) AS peer_hires_share_growth,
        COALESCE(s.peer_headcount, 0) AS peer_headcount,
        COALESCE(s.peer_share, 0) AS peer_share,
        COALESCE(s.company_share, 0) AS company_share,
        COALESCE(es.economy_share, 0) AS economy_share,
        COALESCE(s.peer_share, 0)
            / NULLIF(COALESCE(es.economy_share, 0), 0) AS lift,
        COALESCE(pn.peer_postings, 0) AS peer_postings,
        COALESCE(t.is_specialized, FALSE) AS is_specialized
    FROM shares s
    LEFT JOIN postings_growth pg USING (skill)
    LEFT JOIN hires_growth hg USING (skill)
    LEFT JOIN peer_postings_n pn USING (skill)
    LEFT JOIN economy_shares es USING (skill)
    {_skill_tags_on_label_join(cfg, "s.skill", "t")}
    WHERE s.peer_headcount >= {min_hc}
    ORDER BY s.skill
    {limit_sql}
    """
    df = _lower_cols(_sf(cfg).load_df(q))
    df = _apply_lift_and_postings_gates(df, cfg)
    _RADAR_UNIVERSE_CACHE[cache_key] = df
    return df.copy()


def load_role_categories(cfg, include_roles=None):
    """Company role disruption metrics.

    Returns top ``role_category_top_n`` roles by HC, plus any ``include_roles``
    (e.g. pathway feeders) so relative categories cover feeder names.
    """
    if not cfg.get("use_snowflake") or cfg.get("company_rcid") is None:
        return _load_role_categories_synthetic(cfg, include_roles=include_roles)

    peers = _resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    country = cfg.get("country", "United States")
    recent_m = int(cfg.get("recent_months", 12))
    prior_m = int(cfg.get("prior_months", 12))
    min_pool = int(cfg.get("pathway_min_pool", 50))
    top_n = int(cfg.get("role_category_top_n", 25))
    include_sql = _sql_quote_list_str(include_roles or [])

    q = f"""
    WITH roles AS ({_ROLE_TAXONOMY}),
    pos AS (
        SELECT
            p.position_id,
            p.user_id,
            p.startdate,
            p.enddate,
            p.enddate_primary,
            rt.role,
            COALESCE(p.weight_v2_1, 1) AS wt,
            COALESCE(p.ai_exposure_v1_upsell, 0) AS ai_exposure
        FROM service_pipelines.output_current.individual_position p
        JOIN roles rt ON p.role_v3_id = rt.role_v3_id
        WHERE p.ultimate_parent_rcid = {company_rcid}
          AND p.country = '{country}'
          AND {_pos_filter(cfg)}
    ),
    ai_skill_mix AS (
        SELECT
            p.role,
            CASE
                WHEN p.startdate >= DATEADD('month', -{recent_m}, CURRENT_DATE()) THEN 'recent'
                WHEN p.startdate >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE()) THEN 'prior'
            END AS period,
            AVG(CASE WHEN COALESCE(t.is_ai, FALSE) THEN 1 ELSE 0 END) AS ai_skill_share
        FROM pos p
        LEFT JOIN service_pipelines.output_current.individual_skills sk ON p.user_id = sk.user_id
        LEFT JOIN service_pipelines.global_ref.custom_skills_taxonomy_v3_overall_latest sn
          ON sk.skill_v3_id = sn.skill_v3_id AND sn.taxonomy_name = 'default'
{_ai_skill_tags_join(cfg)}
        WHERE period IS NOT NULL
        GROUP BY 1, 2
    ),
    mix_change AS (
        SELECT
            role,
            COALESCE(
                MAX(CASE WHEN period = 'recent' THEN ai_skill_share END)
                - MAX(CASE WHEN period = 'prior' THEN ai_skill_share END),
                0
            ) AS skill_mix_change
        FROM ai_skill_mix
        GROUP BY 1
    ),
    flows AS (
        SELECT
            rt.role,
            SUM(CASE WHEN DATE_TRUNC('month', p.startdate) >= DATEADD('month', -{recent_m}, CURRENT_DATE())
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS inflow_recent,
            SUM(CASE WHEN DATE_TRUNC('month', p.startdate) >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
                      AND DATE_TRUNC('month', p.startdate) < DATEADD('month', -{recent_m}, CURRENT_DATE())
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS inflow_prior,
            SUM(CASE WHEN DATE_TRUNC('month', p.enddate_primary) >= DATEADD('month', -{recent_m}, CURRENT_DATE())
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS outflow_recent,
            SUM(CASE WHEN p.enddate_primary IS NULL
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS headcount
        FROM service_pipelines.output_current.individual_position p
        JOIN roles rt ON p.role_v3_id = rt.role_v3_id
        WHERE p.ultimate_parent_rcid = {company_rcid}
          AND p.country = '{country}'
          AND {_pos_filter(cfg)}
        GROUP BY 1
    ),
    ai_by_role AS (
        SELECT role, AVG(ai_exposure) AS ai_exposure
        FROM pos
        GROUP BY 1
    ),
    scored AS (
        SELECT
            f.role,
            COALESCE(a.ai_exposure, 0) AS ai_exposure,
            COALESCE(m.skill_mix_change, 0) AS skill_mix_change,
            (f.inflow_recent / NULLIF(f.inflow_prior, 0)) - 1 AS hiring_growth,
            f.outflow_recent / NULLIF(f.headcount, 0) AS attrition,
            f.headcount,
            ROW_NUMBER() OVER (ORDER BY f.headcount DESC) AS rn
        FROM flows f
        LEFT JOIN ai_by_role a ON f.role = a.role
        LEFT JOIN mix_change m ON f.role = m.role
        WHERE f.headcount >= {min_pool}
          AND LOWER(f.role) NOT IN ('retired', 'unknown', 'on leave', 'empty')
    )
    SELECT role, ai_exposure, skill_mix_change, hiring_growth, attrition, headcount
    FROM scored
    WHERE rn <= {top_n}
       {f"OR role IN ({include_sql})" if include_sql else ""}
    ORDER BY headcount DESC
    """
    return _lower_cols(_sf(cfg).load_df(q))


def _apply_signed_wage_gap(df):
    """wage_gap = source/target − 1 (positive = pay cut, negative = raise)."""
    if df is None or getattr(df, "empty", True):
        return df
    if "source_median_comp" not in df.columns or "target_median_comp" not in df.columns:
        return df
    out = df.copy()
    src = pd.to_numeric(out["source_median_comp"], errors="coerce")
    tgt = pd.to_numeric(out["target_median_comp"], errors="coerce")
    out["wage_gap"] = np.where(tgt > 0, src / tgt - 1.0, 0.0)
    return out

def load_pathways(cfg, target_skill):
    """Role-pathway feeders + peer mobility benchmark.

    Company:
    - target roles = current company roles with enough skill holders
    - conversion_rate = (all source→target-role moves) / (full source-role HC)

    Peers (same target-role set):
    - peer_conversion_rate = peer moves into those roles / peer source-role HC
    - mobility_gap = peer_conversion_rate − conversion_rate
    """
    if not target_skill:
        raise ValueError(
            "No target skill selected. select_target found no emerging + "
            "under-indexed skill with sized pathway supply, and "
            "CONFIG['force_target_skill'] is unset. Set force_target_skill "
            "or relax min_under_index / min_internal_supply."
        )

    if not cfg.get("use_snowflake") or cfg.get("company_rcid") is None:
        return _apply_signed_wage_gap(_load_pathways_synthetic(cfg, target_skill))

    peers = _resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    peer_rcids = [int(r) for r in (peers.get("peer_rcids") or []) if int(r) != int(company_rcid)]
    peer_sql = _sql_quote_list(peer_rcids)
    country = cfg.get("country", "United States")
    years = int(cfg.get("pathway_years", 2))
    # Flat min_pool=50 zeros mid-size firms (New Balance ~4k US HC): skill
    # holders fragment across titles, so concentration roles sit at 5–30 and
    # feeders at 10–35. Scale like role pathways — LEAST(configured, HC/N),
    # with a lower floor for holder concentration (subset of role HC).
    base_pool = int(cfg.get("pathway_min_pool", 50))
    max_gap = int(cfg.get("max_gap_days", 180))
    skill_sql = str(target_skill).replace("'", "''")

    if peer_sql:
        peer_cte = f"""
    peer_pos AS (
        SELECT
            p.user_id,
            p.position_id,
            rt.role,
            p.startdate,
            p.enddate,
            p.enddate_primary,
            COALESCE(p.weight_v2_1, 1) AS wt
        FROM service_pipelines.output_current.individual_position p
        JOIN roles rt ON p.role_v3_id = rt.role_v3_id
        WHERE p.ultimate_parent_rcid IN ({peer_sql})
          AND p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND p.startdate IS NOT NULL
    ),
    peer_current AS (
        SELECT user_id, role, wt
        FROM (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY p.user_id
                    ORDER BY p.startdate DESC NULLS LAST, p.position_id DESC
                ) AS rn
            FROM peer_pos p
            WHERE p.enddate_primary IS NULL
        ) x
        WHERE rn = 1
    ),
    peer_seq AS (
        SELECT
            user_id,
            role AS source_role,
            wt,
            LEAD(role) OVER (
                PARTITION BY user_id ORDER BY startdate, COALESCE(enddate_primary, '9999-12-31')
            ) AS dest_role,
            LEAD(startdate) OVER (
                PARTITION BY user_id ORDER BY startdate, COALESCE(enddate_primary, '9999-12-31')
            ) AS to_start,
            enddate,
            enddate_primary
        FROM peer_pos
    ),
    peer_transitions AS (
        SELECT
            s.source_role,
            SUM(s.wt) AS peer_transition_wt
        FROM peer_seq s
        WHERE s.dest_role IS NOT NULL
          AND s.source_role <> s.dest_role
          AND s.dest_role IN (SELECT role FROM target_roles)
          AND s.source_role NOT IN (SELECT role FROM target_roles)
          AND s.to_start >= DATEADD('year', -{years}, CURRENT_DATE())
          AND (
                (s.enddate_primary IS NOT NULL AND ABS(DATEDIFF('day', s.enddate_primary, s.to_start)) <= {max_gap})
             OR (s.enddate_primary IS NULL AND DATEDIFF('day', s.to_start, CURRENT_DATE()) <= {max_gap})
          )
        GROUP BY 1
    ),
    peer_feeder AS (
        SELECT
            cp.role AS source_role,
            SUM(cp.wt) AS peer_feeder_pool
        FROM peer_current cp
        WHERE cp.role NOT IN (SELECT role FROM target_roles)
        GROUP BY 1
    ),
    peer_rates AS (
        SELECT
            f.source_role,
            f.peer_feeder_pool,
            COALESCE(t.peer_transition_wt, 0) AS peer_transition_wt,
            COALESCE(t.peer_transition_wt / NULLIF(f.peer_feeder_pool, 0), 0)
                AS peer_conversion_rate
        FROM peer_feeder f
        LEFT JOIN peer_transitions t USING (source_role)
    ),"""
        peer_select = """
        COALESCE(pr.peer_feeder_pool, 0) AS peer_feeder_pool,
        COALESCE(pr.peer_conversion_rate, 0) AS peer_conversion_rate,"""
        peer_join = "\n    LEFT JOIN peer_rates pr ON f.source_role = pr.source_role"
    else:
        peer_cte = ""
        peer_select = """
        0::FLOAT AS peer_feeder_pool,
        0::FLOAT AS peer_conversion_rate,"""
        peer_join = ""

    q = f"""
    WITH roles AS ({_ROLE_TAXONOMY}),
    skill_names AS ({_skill_taxonomy(cfg)}),
    skill_holders AS (
        SELECT DISTINCT sk.user_id
        FROM service_pipelines.output_current.individual_skills sk
        JOIN skill_names sn ON sk.skill_v3_id = sn.skill_v3_id
        WHERE sn.skill = '{skill_sql}'
    ),
    target_users AS (
        SELECT sh.user_id
        FROM skill_holders sh
        WHERE EXISTS (
            SELECT 1
            FROM service_pipelines.output_current.individual_position p
            WHERE p.user_id = sh.user_id
              AND p.ultimate_parent_rcid = {company_rcid}
              AND p.country = '{country}'
              AND {_pos_filter(cfg)}
        )
    ),
    pos AS (
        SELECT
            p.user_id,
            p.position_id,
            rt.role,
            p.startdate,
            p.enddate,
            p.enddate_primary,
            COALESCE(p.total_compensation_v2_1, p.total_compensation) AS comp,
            COALESCE(p.weight_v2_1, 1) AS wt
        FROM service_pipelines.output_current.individual_position p
        JOIN roles rt ON p.role_v3_id = rt.role_v3_id
        WHERE p.ultimate_parent_rcid = {company_rcid}
          AND p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND p.startdate IS NOT NULL
    ),
    current_pos AS (
        SELECT user_id, role, comp, wt, position_id, startdate
        FROM (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY p.user_id
                    ORDER BY p.startdate DESC NULLS LAST, p.position_id DESC
                ) AS rn
            FROM pos p
            WHERE p.enddate_primary IS NULL
        ) x
        WHERE rn = 1
    ),
    target_roles AS (
        SELECT cp.role
        FROM current_pos cp
        JOIN target_users tu ON cp.user_id = tu.user_id
        GROUP BY 1
        HAVING COUNT(*) >= GREATEST(
            5,
            LEAST(
              {base_pool},
              (SELECT COUNT(DISTINCT user_id) FROM current_pos) / 800
            )
          )
    ),
    seq AS (
        SELECT
            user_id,
            role AS source_role,
            wt,
            LEAD(role) OVER (
                PARTITION BY user_id ORDER BY startdate, COALESCE(enddate_primary, '9999-12-31')
            ) AS dest_role,
            LEAD(startdate) OVER (
                PARTITION BY user_id ORDER BY startdate, COALESCE(enddate_primary, '9999-12-31')
            ) AS to_start,
            enddate,
            enddate_primary
        FROM pos
    ),
    transitions AS (
        SELECT
            s.source_role,
            SUM(s.wt) AS transition_wt,
            SUM(CASE WHEN tu.user_id IS NOT NULL THEN s.wt ELSE 0 END)
                AS skill_holder_transition_wt
        FROM seq s
        LEFT JOIN target_users tu ON s.user_id = tu.user_id
        WHERE s.dest_role IS NOT NULL
          AND s.source_role <> s.dest_role
          AND s.dest_role IN (SELECT role FROM target_roles)
          AND s.source_role NOT IN (SELECT role FROM target_roles)
          AND s.to_start >= DATEADD('year', -{years}, CURRENT_DATE())
          AND (
                (s.enddate_primary IS NOT NULL AND ABS(DATEDIFF('day', s.enddate_primary, s.to_start)) <= {max_gap})
             OR (s.enddate_primary IS NULL AND DATEDIFF('day', s.to_start, CURRENT_DATE()) <= {max_gap})
          )
        GROUP BY 1
    ),
    feeder AS (
        SELECT
            cp.role AS source_role,
            SUM(cp.wt) AS feeder_pool
        FROM current_pos cp
        WHERE cp.role NOT IN (SELECT role FROM target_roles)
        GROUP BY 1
    ),
    {peer_cte}
    source_skills AS (
        SELECT cp.role AS source_role, sn.skill
        FROM current_pos cp
        JOIN service_pipelines.output_current.individual_skills sk ON cp.user_id = sk.user_id
        JOIN skill_names sn ON sk.skill_v3_id = sn.skill_v3_id
        GROUP BY 1, 2
    ),
    target_holder_skills AS (
        SELECT DISTINCT sn.skill
        FROM target_users tu
        JOIN service_pipelines.output_current.individual_skills sk ON tu.user_id = sk.user_id
        JOIN skill_names sn ON sk.skill_v3_id = sn.skill_v3_id
    ),
    overlap AS (
        SELECT
            ss.source_role,
            COUNT(DISTINCT ss.skill) AS n_source,
            (SELECT COUNT(*) FROM target_holder_skills) AS n_target,
            COUNT(DISTINCT CASE WHEN ths.skill IS NOT NULL THEN ss.skill END) AS n_overlap
        FROM source_skills ss
        LEFT JOIN target_holder_skills ths ON ss.skill = ths.skill
        GROUP BY ss.source_role
    ),
    target_comp AS (
        SELECT MEDIAN(cp.comp) AS target_comp
        FROM current_pos cp
        JOIN target_users tu ON cp.user_id = tu.user_id
    ),
    source_comp AS (
        SELECT role AS source_role, MEDIAN(comp) AS source_comp
        FROM current_pos
        GROUP BY 1
    )
    SELECT
        f.source_role,
        f.feeder_pool,
        COALESCE(t.transition_wt, 0) AS transition_wt,
        COALESCE(t.transition_wt / NULLIF(f.feeder_pool, 0), 0) AS conversion_rate,
        {peer_select}
        COALESCE(
            t.skill_holder_transition_wt / NULLIF(t.transition_wt, 0), 0
        ) AS skill_mover_share,
        -- Signed: source/target − 1; positive = pay cut to move, negative = raise.
        COALESCE(sc.source_comp / NULLIF(tc.target_comp, 0) - 1, 0) AS wage_gap,
        COALESCE(o.n_overlap / NULLIF(o.n_source + o.n_target - o.n_overlap, 0), 0) AS skill_jaccard,
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
    {_pathway_candidate_qualify(cfg)}
    """
    df = _lower_cols(_sf(cfg).load_df(q))
    empty_cols = [
        "source_role", "feeder_pool", "transition_freq", "skill_overlap",
        "wage_gap", "conversion_rate", "peer_conversion_rate", "mobility_gap",
        "skill_mover_share", "source_median_comp", "target_median_comp",
        "transition_wt"]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)
    df["transition_freq"] = df["transition_wt"].map(_freq_label)
    df["skill_overlap"] = df["skill_jaccard"].map(_overlap_label)
    if "peer_conversion_rate" not in df.columns:
        df["peer_conversion_rate"] = 0.0
    df["mobility_gap"] = (
        df["peer_conversion_rate"].astype(float) - df["conversion_rate"].astype(float)
    )
    cols = [
        "source_role", "feeder_pool", "transition_freq", "skill_overlap",
        "wage_gap", "conversion_rate", "peer_conversion_rate", "mobility_gap",
        "skill_mover_share", "source_median_comp", "target_median_comp",
        "transition_wt"]
    return _apply_signed_wage_gap(df[cols])
def annotate_pathway_supply(cfg, pathways):
    """Annualized conversion rates + ``supply_heads`` on a copy of the frame.

    Shared with the dashboard so its pathway table shows the same heads the
    scenario sized on, rather than re-deriving the formula and drifting.
    """
    if pathways is None:
        return pd.DataFrame()
    if getattr(pathways, "empty", True):
        return pathways.copy()
    H = float(cfg.get("horizon_years", 3))
    P = float(cfg.get("pathway_years", H) or H)
    if P <= 0:
        raise ValueError("pathway_years must be > 0")
    persistence = float(cfg.get("conversion_persistence", 1.0))
    max_conv = float(cfg.get("max_feeder_conversion", 1.0))
    out = pathways.copy()
    pool = out["feeder_pool"].astype(float)
    conv = out["conversion_rate"].astype(float)
    annual = conv / P
    out["annual_conversion_rate"] = annual
    out["effective_conversion_rate"] = (annual * H * persistence).clip(
        lower=0, upper=max_conv)
    out["supply_heads_raw"] = pool * conv          # unscaled observed window
    out["supply_heads"] = pool * out["effective_conversion_rate"]
    return out


def _horizon_pathway_supply(cfg, pathways):
    """Annualize role-pathway rates into horizon-scaled internal heads."""
    if pathways is None or getattr(pathways, "empty", True):
        return 0.0
    H = float(cfg.get("horizon_years", 3))
    P = float(cfg.get("pathway_years", H) or H)
    if P <= 0:
        raise ValueError("pathway_years must be > 0")
    persistence = float(cfg.get("conversion_persistence", 1.0))
    max_conv = float(cfg.get("max_feeder_conversion", 1.0))
    annual = pathways["conversion_rate"].astype(float) / P
    effective = (annual * H * persistence).clip(lower=0, upper=max_conv)
    return float((pathways["feeder_pool"].astype(float) * effective).sum())


def count_feeder_roles(cfg, target_skill):
    """Count observed role-pathway feeders (uses load_pathways; no duplicate SQL).

    Includes ``transition_freq == "low"`` — thin counts are real at mid-size
    firms; feasibility haircuts them later rather than zeroing supply.
    """
    paths = load_pathways(cfg, target_skill)
    if paths is None or getattr(paths, "empty", True):
        return 0
    return int(len(paths))


def pathway_supply_summary(cfg, target_skill):
    """Feeder count + horizon-scaled supply for selection funnel."""
    paths = load_pathways(cfg, target_skill)
    if paths is None or getattr(paths, "empty", True):
        empty = pd.DataFrame()
        return {"feeder_roles": 0, "supply_heads": 0.0, "pathways": empty}
    # Keep low-freq: absolute <3-move thresholds were calibrated on large
    # employers; mid-size firms would otherwise look falsely unbuildable.
    usable = paths.copy()
    supply = _horizon_pathway_supply(cfg, usable)
    return {
        "feeder_roles": int(len(usable)),
        "supply_heads": supply,
        "pathways": usable,
    }



def load_target_population(cfg, target_skill):
    """Current skill headcount + 12-month flow rates for the target skill.

    Returns ``(current_hc, attrition_rate, hiring_rate)``.

    - **Current HC:** one ``weight_v2_1`` per skill holder on their latest
      current company position (US + position filters).
    - **Attrition rate:** 12-month sum of **employer-exit outflows** /
      12-month average monthly skill HC (``recent_months``).
    - **Hiring rate:** 12-month sum of **company-entry inflows** /
      the same 12-month average HC. Internal role changes are neither.
    """
    if not cfg.get("use_snowflake") or cfg.get("company_rcid") is None:
        return _load_target_population_synthetic(cfg, target_skill)

    peers = _resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    country = cfg.get("country", "United States")
    recent_m = int(cfg.get("recent_months", 12))
    skill_sql = target_skill.replace("'", "''")

    q = f"""
    WITH skill_names AS ({_skill_taxonomy(cfg)}),
    skill_holders AS (
        SELECT DISTINCT sk.user_id
        FROM service_pipelines.output_current.individual_skills sk
        JOIN skill_names sn ON sk.skill_v3_id = sn.skill_v3_id
        WHERE sn.skill = '{skill_sql}'
    ),
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
            p.user_id,
            p.position_id,
            p.startdate,
            p.enddate,
            p.enddate_primary,
            p.ultimate_parent_rcid,
            COALESCE(p.weight_v2_1, 1) AS wt,
            LAG(p.ultimate_parent_rcid) OVER (
                PARTITION BY p.user_id
                ORDER BY p.startdate, COALESCE(p.enddate_primary, '9999-12-31'), p.position_id
            ) AS prev_rcid,
            LEAD(p.ultimate_parent_rcid) OVER (
                PARTITION BY p.user_id
                ORDER BY p.startdate, COALESCE(p.enddate_primary, '9999-12-31'), p.position_id
            ) AS next_rcid
        FROM service_pipelines.output_current.individual_position p
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND p.user_id IN (SELECT user_id FROM skill_holders)
    ),
    company_pos AS (
        SELECT * FROM pos WHERE ultimate_parent_rcid = {company_rcid}
    ),
    current_hc AS (
        SELECT SUM(wt) AS headcount
        FROM (
            SELECT
                cp.user_id,
                cp.wt,
                ROW_NUMBER() OVER (
                    PARTITION BY cp.user_id
                    ORDER BY cp.startdate DESC NULLS LAST, cp.position_id DESC
                ) AS rn
            FROM company_pos cp
            WHERE cp.enddate_primary IS NULL
        ) x
        WHERE rn = 1
    ),
    monthly_hc AS (
        SELECT m.month_start, SUM(u.wt) AS hc
        FROM months m
        JOIN (
            SELECT
                m2.month_start,
                cp.user_id,
                MAX(cp.wt) AS wt
            FROM months m2
            JOIN company_pos cp
              ON cp.startdate <= m2.month_end
             AND COALESCE(cp.enddate_primary, '9999-12-31'::DATE) >= m2.month_start
            GROUP BY 1, 2
        ) u ON m.month_start = u.month_start
        GROUP BY 1
    ),
    monthly_outflow AS (
        SELECT m.month_start, SUM(e.wt) AS outflow
        FROM months m
        JOIN (
            SELECT
                user_id,
                wt,
                DATE_TRUNC('month', enddate_primary) AS end_month,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, DATE_TRUNC('month', enddate_primary)
                    ORDER BY enddate_primary DESC, position_id DESC
                ) AS rn
            FROM company_pos
            WHERE enddate_primary IS NOT NULL
              AND (next_rcid IS NULL OR next_rcid <> {company_rcid})
        ) e ON e.end_month = m.month_start AND e.rn = 1
        GROUP BY 1
    ),
    monthly_inflow AS (
        SELECT m.month_start, SUM(i.wt) AS inflow
        FROM months m
        JOIN (
            SELECT
                user_id,
                wt,
                DATE_TRUNC('month', startdate) AS start_month,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, DATE_TRUNC('month', startdate)
                    ORDER BY startdate, position_id
                ) AS rn
            FROM company_pos
            WHERE prev_rcid IS NULL OR prev_rcid <> {company_rcid}
        ) i ON i.start_month = m.month_start AND i.rn = 1
        GROUP BY 1
    ),
    rates AS (
        SELECT
            SUM(COALESCE(o.outflow, 0)) AS out_sum,
            SUM(COALESCE(i.inflow, 0)) AS in_sum,
            AVG(h.hc) AS hc_avg
        FROM monthly_hc h
        LEFT JOIN monthly_outflow o USING (month_start)
        LEFT JOIN monthly_inflow i USING (month_start)
    )
    SELECT
        COALESCE((SELECT headcount FROM current_hc), 0) AS headcount,
        COALESCE(
            (SELECT out_sum FROM rates) / NULLIF((SELECT hc_avg FROM rates), 0),
            0
        ) AS attrition,
        COALESCE(
            (SELECT in_sum FROM rates) / NULLIF((SELECT hc_avg FROM rates), 0),
            0
        ) AS hiring_rate,
        COALESCE((SELECT hc_avg FROM rates), 0) AS hc_avg_12m,
        COALESCE((SELECT out_sum FROM rates), 0) AS outflow_sum_12m,
        COALESCE((SELECT in_sum FROM rates), 0) AS inflow_sum_12m
    """
    row = _lower_cols(_sf(cfg).load_df(q)).iloc[0]
    return (
        float(row["headcount"] or 0),
        float(row["attrition"] or 0),
        float(row["hiring_rate"] or 0),
    )



_COMPANY_ATTRITION_CACHE = {}


def load_company_attrition(cfg):
    """Company-wide US attrition (employer exits / avg monthly HC), same window.

    Benchmark for the retention lever: is the target skill leaking faster than
    the rest of this company's workforce?
    """
    if not cfg.get("use_snowflake") or cfg.get("company_rcid") is None:
        # Synthetic: slightly below typical skill rates so excess can fire.
        return float(cfg.get(
            "synthetic_company_attrition",
            globals().get("_SYNTHETIC_COMPANY_ATTRITION", 0.06)))

    peers = _resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    country = cfg.get("country", "United States")
    recent_m = int(cfg.get("recent_months", 12))
    cache_key = (
        int(company_rcid), country, recent_m, cfg.get("exclude_contingent", True),
    )
    if cache_key in _COMPANY_ATTRITION_CACHE and not cfg.get("refresh_staging"):
        return _COMPANY_ATTRITION_CACHE[cache_key]

    q = f"""
    WITH months AS (
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
            p.user_id,
            p.position_id,
            p.startdate,
            p.enddate_primary,
            COALESCE(p.weight_v2_1, 1) AS wt,
            LEAD(p.ultimate_parent_rcid) OVER (
                PARTITION BY p.user_id
                ORDER BY p.startdate, COALESCE(p.enddate_primary, '9999-12-31'), p.position_id
            ) AS next_rcid
        FROM service_pipelines.output_current.individual_position p
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND p.ultimate_parent_rcid = {company_rcid}
    ),
    monthly_hc AS (
        SELECT m.month_start, SUM(u.wt) AS hc
        FROM months m
        JOIN (
            SELECT
                m2.month_start,
                p.user_id,
                MAX(p.wt) AS wt
            FROM months m2
            JOIN pos p
              ON p.startdate <= m2.month_end
             AND COALESCE(p.enddate_primary, '9999-12-31'::DATE) >= m2.month_start
            GROUP BY 1, 2
        ) u ON m.month_start = u.month_start
        GROUP BY 1
    ),
    monthly_outflow AS (
        SELECT m.month_start, SUM(e.wt) AS outflow
        FROM months m
        JOIN (
            SELECT
                user_id,
                wt,
                DATE_TRUNC('month', enddate_primary) AS end_month,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, DATE_TRUNC('month', enddate_primary)
                    ORDER BY enddate_primary DESC, position_id DESC
                ) AS rn
            FROM pos
            WHERE enddate_primary IS NOT NULL
              AND (next_rcid IS NULL OR next_rcid <> {company_rcid})
        ) e ON e.end_month = m.month_start AND e.rn = 1
        GROUP BY 1
    ),
    rates AS (
        SELECT
            SUM(COALESCE(o.outflow, 0)) AS out_sum,
            AVG(h.hc) AS hc_avg
        FROM monthly_hc h
        LEFT JOIN monthly_outflow o USING (month_start)
    )
    SELECT
        COALESCE(
            (SELECT out_sum FROM rates) / NULLIF((SELECT hc_avg FROM rates), 0),
            0
        ) AS attrition
    """
    row = _lower_cols(_sf(cfg).load_df(q)).iloc[0]
    rate = float(row["attrition"] or 0)
    _COMPANY_ATTRITION_CACHE[cache_key] = rate
    return rate


_PEER_SKILL_ATTRITION_FLOOR_CACHE = {}


def load_peer_skill_attrition_floor(cfg, target_skill):
    """P10 of peer employers' attrition among holders of ``target_skill``.

    Structural retention floor: the best-retaining comparable employers still
    lose ~X% a year on this skill; treat X as unavoidable. Adapts by skill
    (e.g. AI vs facilities engineering).

    Returns ``(floor_rate, meta)`` where meta includes ``source``, ``n_peers``,
    and the configured percentile.
    """
    pct = float(cfg.get("retention_peer_attrition_percentile", 0.10))
    min_hc = float(cfg.get("retention_min_peer_skill_hc", 50))
    fallback = float(cfg.get(
        "retention_attrition_floor_fallback",
        cfg.get("retention_attrition_floor",
                cfg.get("target_attrition_for_retention", 0.08))))
    override = cfg.get("retention_attrition_floor_override")
    if override is not None:
        return float(override), {
            "source": "override",
            "n_peers": None,
            "percentile": pct,
            "floor": float(override),
        }

    if not cfg.get("use_snowflake") or cfg.get("company_rcid") is None:
        frozen = globals().get("_SYNTHETIC_PEER_FLOORS") or {}
        if target_skill in frozen and frozen[target_skill] is not None:
            syn = float(frozen[target_skill])
            return syn, {
                "source": "synthetic_frozen",
                "n_peers": None,
                "percentile": pct,
                "floor": syn,
            }
        try:
            _c, skill_attr, _h = _load_target_population_synthetic(cfg, target_skill)
            syn = max(0.02, float(skill_attr) * 0.85)
        except Exception:
            syn = float(cfg.get("synthetic_peer_attrition_p10", fallback))
        return syn, {
            "source": "synthetic",
            "n_peers": None,
            "percentile": pct,
            "floor": syn,
        }

    peers = _resolve_peer_rcids(cfg)
    peer_rcids = [int(r) for r in (peers.get("peer_rcids") or [])
                  if int(r) != int(peers["company_rcid"])]
    if not peer_rcids:
        return fallback, {
            "source": "fallback_no_peers",
            "n_peers": 0,
            "percentile": pct,
            "floor": fallback,
        }

    country = cfg.get("country", "United States")
    recent_m = int(cfg.get("recent_months", 12))
    skill_sql = target_skill.replace("'", "''")
    peer_sql = _sql_quote_list(peer_rcids)
    cache_key = (
        int(peers["company_rcid"]), peer_sql, skill_sql, country, recent_m,
        pct, min_hc, cfg.get("exclude_contingent", True),
    )
    if cache_key in _PEER_SKILL_ATTRITION_FLOOR_CACHE and not cfg.get("refresh_staging"):
        return _PEER_SKILL_ATTRITION_FLOOR_CACHE[cache_key]

    q = f"""
    WITH skill_names AS ({_skill_taxonomy(cfg)}),
    skill_holders AS (
        SELECT DISTINCT sk.user_id
        FROM service_pipelines.output_current.individual_skills sk
        JOIN skill_names sn ON sk.skill_v3_id = sn.skill_v3_id
        WHERE sn.skill = '{skill_sql}'
    ),
    months AS (
        SELECT
            DATEADD('month', -seq, DATE_TRUNC('month', CURRENT_DATE())) AS month_start,
            LAST_DAY(DATEADD('month', -seq, DATE_TRUNC('month', CURRENT_DATE()))) AS month_end
        FROM (
            SELECT ROW_NUMBER() OVER (ORDER BY SEQ4()) - 1 AS seq
            FROM TABLE(GENERATOR(ROWCOUNT => {recent_m}))
        )
    ),
    peer_list AS (
        SELECT value::INT AS rcid
        FROM TABLE(SPLIT_TO_TABLE('{peer_sql}', ','))
    ),
    pos AS (
        SELECT
            p.user_id,
            p.position_id,
            p.startdate,
            p.enddate_primary,
            p.ultimate_parent_rcid,
            COALESCE(p.weight_v2_1, 1) AS wt,
            LEAD(p.ultimate_parent_rcid) OVER (
                PARTITION BY p.user_id
                ORDER BY p.startdate, COALESCE(p.enddate_primary, '9999-12-31'), p.position_id
            ) AS next_rcid
        FROM service_pipelines.output_current.individual_position p
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND p.user_id IN (SELECT user_id FROM skill_holders)
    ),
    peer_pos AS (
        SELECT p.*
        FROM pos p
        JOIN peer_list pl ON p.ultimate_parent_rcid = pl.rcid
    ),
    monthly_hc AS (
        SELECT u.rcid, u.month_start, SUM(u.wt) AS hc
        FROM (
            SELECT
                m2.month_start,
                pp.ultimate_parent_rcid AS rcid,
                pp.user_id,
                MAX(pp.wt) AS wt
            FROM months m2
            JOIN peer_pos pp
              ON pp.startdate <= m2.month_end
             AND COALESCE(pp.enddate_primary, '9999-12-31'::DATE) >= m2.month_start
            GROUP BY 1, 2, 3
        ) u
        GROUP BY 1, 2
    ),
    monthly_outflow AS (
        SELECT e.rcid, e.end_month AS month_start, SUM(e.wt) AS outflow
        FROM (
            SELECT
                ultimate_parent_rcid AS rcid,
                user_id,
                wt,
                DATE_TRUNC('month', enddate_primary) AS end_month,
                ROW_NUMBER() OVER (
                    PARTITION BY ultimate_parent_rcid, user_id,
                                 DATE_TRUNC('month', enddate_primary)
                    ORDER BY enddate_primary DESC, position_id DESC
                ) AS rn
            FROM peer_pos
            WHERE enddate_primary IS NOT NULL
              AND (next_rcid IS NULL OR next_rcid <> ultimate_parent_rcid)
        ) e
        WHERE e.rn = 1
        GROUP BY 1, 2
    ),
    peer_rates AS (
        SELECT
            h.rcid,
            SUM(COALESCE(o.outflow, 0)) AS out_sum,
            AVG(h.hc) AS hc_avg,
            SUM(COALESCE(o.outflow, 0)) / NULLIF(AVG(h.hc), 0) AS attrition
        FROM monthly_hc h
        LEFT JOIN monthly_outflow o
          ON h.rcid = o.rcid AND h.month_start = o.month_start
        GROUP BY h.rcid
        HAVING AVG(h.hc) >= {min_hc}
    )
    SELECT
        PERCENTILE_CONT({pct}) WITHIN GROUP (ORDER BY attrition) AS floor_rate,
        COUNT(*) AS n_peers,
        MIN(attrition) AS min_attr,
        AVG(attrition) AS mean_attr,
        MAX(attrition) AS max_attr
    FROM peer_rates
    WHERE attrition IS NOT NULL
    """
    row = _lower_cols(_sf(cfg).load_df(q)).iloc[0]
    n_peers = int(row["n_peers"] or 0)
    if n_peers <= 0 or row["floor_rate"] is None or (isinstance(row["floor_rate"], float) and pd.isna(row["floor_rate"])):
        result = (fallback, {
            "source": "fallback_empty",
            "n_peers": n_peers,
            "percentile": pct,
            "floor": fallback,
        })
    else:
        floor = float(row["floor_rate"])
        result = (floor, {
            "source": "peer_p10" if abs(pct - 0.10) < 1e-9 else f"peer_p{int(round(pct * 100))}",
            "n_peers": n_peers,
            "percentile": pct,
            "floor": floor,
            "min_attr": float(row["min_attr"]) if row["min_attr"] is not None else None,
            "mean_attr": float(row["mean_attr"]) if row["mean_attr"] is not None else None,
            "max_attr": float(row["max_attr"]) if row["max_attr"] is not None else None,
        })
    _PEER_SKILL_ATTRITION_FLOOR_CACHE[cache_key] = result
    return result



def load_metro_supply(cfg, target_skill):
    if not cfg.get("use_snowflake") or cfg.get("company_rcid") is None:
        return _load_metro_supply_synthetic(cfg, target_skill)

    peers = _resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    peer_sql = _sql_quote_list(peers["peer_rcids"]) or str(company_rcid)
    country = cfg.get("country", "United States")
    recent_m = int(cfg.get("recent_months", 12))
    skill_sql = target_skill.replace("'", "''")

    q = f"""
    WITH skill_names AS ({_skill_taxonomy(cfg)}),
    peer_rcids AS (
        SELECT {company_rcid} AS rcid, TRUE AS is_company
        UNION ALL
        SELECT value::INT AS rcid, FALSE AS is_company
        FROM TABLE(SPLIT_TO_TABLE('{peer_sql}', ','))
    ),
    skill_holders AS (
        SELECT DISTINCT sk.user_id
        FROM service_pipelines.output_current.individual_skills sk
        JOIN skill_names sn ON sk.skill_v3_id = sn.skill_v3_id
        WHERE sn.skill = '{skill_sql}'
    ),
    supply AS (
        SELECT
            p.metro_area AS metro,
            SUM(wt) AS external_supply
        FROM (
            SELECT
                p.user_id,
                p.metro_area,
                COALESCE(p.weight_v2_1, 1) AS wt,
                {_rn_latest_position("p.user_id")} AS rn
            FROM service_pipelines.output_current.individual_position p
            JOIN skill_holders sh ON p.user_id = sh.user_id
            WHERE p.country = '{country}'
              AND {_pos_filter(cfg)}
              AND p.enddate_primary IS NULL
              AND p.metro_area IS NOT NULL
              AND p.ultimate_parent_rcid <> {company_rcid}
        ) p
        WHERE rn = 1
        GROUP BY 1
    ),
    demand AS (
        SELECT
            COALESCE(p.msa_v3, 'unknown') AS metro,
            COUNT(DISTINCT p.job_id) AS competitor_demand
        FROM service_pipelines.output_current.postings_unique_unified p
        JOIN service_pipelines.output_current.postings_unique_unified_skills_v3 ps USING (job_id)
        JOIN skill_names sn ON ps.skill_v3_id = sn.skill_v3_id
        JOIN peer_rcids pr ON p.rcid = pr.rcid
        WHERE p.country_v3 = '{country}'
          AND NOT pr.is_company
          AND p.post_date >= DATEADD('month', -{recent_m}, CURRENT_DATE())
          AND p.msa_v3 IS NOT NULL
        GROUP BY 1
    ),
    company_presence AS (
        SELECT DISTINCT p.metro_area AS metro, TRUE AS company_presence
        FROM service_pipelines.output_current.individual_position p
        WHERE p.ultimate_parent_rcid = {company_rcid}
          AND p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND p.enddate_primary IS NULL
          AND p.metro_area IS NOT NULL
    )
    SELECT
        s.metro,
        s.external_supply,
        COALESCE(d.competitor_demand, 0) AS competitor_demand,
        COALESCE(cp.company_presence, FALSE) AS company_presence
    FROM supply s
    LEFT JOIN demand d ON s.metro = d.metro
    LEFT JOIN company_presence cp ON s.metro = cp.metro
    WHERE s.external_supply >= 25
    ORDER BY s.external_supply DESC
    LIMIT 20
    """
    df = _lower_cols(_sf(cfg).load_df(q))
    junk = {"empty", "unknown", "", "none"}
    df = df[~df["metro"].str.lower().isin(junk)]
    return df.reset_index(drop=True)


def load_competitor_outflows(cfg, target_skill):
    """Skill holders leaving company -> next employer among peer set."""
    if not cfg.get("use_snowflake") or cfg.get("company_rcid") is None:
        return _load_competitor_outflows_synthetic(cfg, target_skill)

    peers = _resolve_peer_rcids(cfg)
    company_rcid = peers["company_rcid"]
    peer_sql = _sql_quote_list(peers["peer_rcids"])
    if not peer_sql:
        return pd.DataFrame(columns=["dest_rcid", "dest_company", "outflow_wt"])

    years = int(cfg.get("outflow_years", 2))
    max_gap = int(cfg.get("max_gap_days", 180))
    top_n = int(cfg.get("outflow_top_n", 10))
    country = cfg.get("country", "United States")
    skill_sql = target_skill.replace("'", "''")
    batchtime = cfg.get("batchtime", "202602")

    q = f"""
    WITH skill_names AS ({_skill_taxonomy(cfg)}),
    skill_holders AS (
        SELECT DISTINCT sk.user_id
        FROM service_pipelines.output_current.individual_skills sk
        JOIN skill_names sn ON sk.skill_v3_id = sn.skill_v3_id
        WHERE sn.skill = '{skill_sql}'
    ),
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
            COALESCE(p.weight_v2_1, 1) AS wt
        FROM service_pipelines.output_current.individual_position p
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND p.startdate IS NOT NULL
    ),
    seq AS (
        SELECT
            user_id,
            ultimate_parent_rcid AS from_rcid,
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
        JOIN skill_holders sh ON s.user_id = sh.user_id
        WHERE s.from_rcid = {company_rcid}
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
    return _lower_cols(_sf(cfg).load_df(q))

_INDUSTRY_CACHE = {}


def _industry_col(cfg):
    level = cfg.get("industry_level", "rics_k200")
    if level not in ("rics_k200", "rics_k50"):
        raise ValueError("industry_level must be 'rics_k200' or 'rics_k50'")
    return level


def _resolve_industry(cfg):
    col = _industry_col(cfg)
    company_rcid = cfg.get("company_rcid")
    cache_key = (company_rcid, col, cfg.get("industry_label"), cfg.get("batchtime"))
    if cache_key in _INDUSTRY_CACHE and not cfg.get("refresh_staging"):
        return _INDUSTRY_CACHE[cache_key]

    label = cfg.get("industry_label")
    if label is None:
        if company_rcid is None:
            return {"industry_col": col, "industry_label": "Synthetic Industry"}
        q = f"""
        SELECT y.{col} AS industry_label
        FROM model_compuniv.v1_internal.rcid_full_company_ref_dashboard_{cfg['batchtime']} x
        JOIN model_industry.v1_reference.rics_cluster_lookup_latest y
          ON x.rics_k400 = y.rics_k400
        WHERE x.rcid = {int(company_rcid)}
        LIMIT 1
        """
        row = _lower_cols(_sf(cfg).load_df(q))
        if row.empty:
            raise ValueError(f"Could not resolve {col} for rcid {company_rcid}")
        label = row.iloc[0]["industry_label"]

    out = {"industry_col": col, "industry_label": label}
    _INDUSTRY_CACHE[cache_key] = out
    return out


def _industry_filter(alias, industry):
    col = industry["industry_col"]
    label = str(industry["industry_label"]).replace("'", "''")
    return f"""
        EXISTS (
            SELECT 1
            FROM model_industry.v1_reference.rics_cluster_lookup_latest ind
            WHERE {alias}.rics_k400 = ind.rics_k400
              AND ind.{col} = '{label}'
        )"""


def _load_industry_skill_radar_synthetic(cfg):
    df = _load_skill_radar_synthetic(cfg).rename(columns={
        "peer_postings_share_growth": "industry_postings_share_growth",
        "peer_hires_share_growth": "industry_hires_share_growth",
        "peer_share": "industry_share",
    })
    return df[["skill", "industry_postings_share_growth",
               "industry_hires_share_growth", "industry_share"]]


def load_industry_skill_radar(cfg):
    if not cfg.get("use_snowflake") or cfg.get("company_rcid") is None:
        return _load_industry_skill_radar_synthetic(cfg)

    industry = _resolve_industry(cfg)
    ind_filter = _industry_filter("p", industry)
    recent_m = int(cfg.get("recent_months", 12))
    prior_m = int(cfg.get("prior_months", 12))
    min_hc = int(cfg.get("min_skill_headcount", 500))
    max_n = cfg.get("radar_universe_max_skills")
    country = cfg.get("country", "United States")
    limit_sql = f"\n    LIMIT {int(max_n)}" if max_n else ""

    q = f"""
    WITH skill_names AS ({_skill_taxonomy(cfg)}),
    postings AS (
        SELECT
            sn.skill,
            CASE
                WHEN p.post_date >= DATEADD('month', -{recent_m}, CURRENT_DATE()) THEN 'recent'
                WHEN p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE()) THEN 'prior'
            END AS period,
            COUNT(DISTINCT p.job_id) AS n_jobs
        FROM service_pipelines.output_current.postings_unique_unified p
        JOIN service_pipelines.output_current.postings_unique_unified_skills_v3 ps
          USING (job_id)
        JOIN skill_names sn ON ps.skill_v3_id = sn.skill_v3_id
        WHERE p.country_v3 = '{country}'
          AND p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
          AND {ind_filter}
        GROUP BY 1, 2
    ),
    industry_postings_period_tot AS (
        SELECT
            CASE
                WHEN p.post_date >= DATEADD('month', -{recent_m}, CURRENT_DATE()) THEN 'recent'
                WHEN p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE()) THEN 'prior'
            END AS period,
            COUNT(DISTINCT p.job_id) AS tot_jobs
        FROM service_pipelines.output_current.postings_unique_unified p
        WHERE p.country_v3 = '{country}'
          AND p.post_date >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
          AND {ind_filter}
        GROUP BY 1
    ),
    postings_growth AS (
        SELECT
            skill,
            (recent_share / NULLIF(prior_share, 0)) - 1 AS industry_postings_share_growth
        FROM (
            SELECT
                skill,
                SUM(CASE WHEN period = 'recent' THEN n_jobs ELSE 0 END)::FLOAT
                    / NULLIF((SELECT tot_jobs FROM industry_postings_period_tot WHERE period = 'recent'), 0)
                    AS recent_share,
                SUM(CASE WHEN period = 'prior' THEN n_jobs ELSE 0 END)::FLOAT
                    / NULLIF((SELECT tot_jobs FROM industry_postings_period_tot WHERE period = 'prior'), 0)
                    AS prior_share
            FROM postings
            WHERE period IS NOT NULL
            GROUP BY skill
        ) x
    ),
    {_SKILL_USERS_CTE}
    hires AS (
        SELECT
            su.skill,
            CASE
                WHEN p.startdate >= DATEADD('month', -{recent_m}, CURRENT_DATE()) THEN 'recent'
                WHEN p.startdate >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE()) THEN 'prior'
            END AS period,
            SUM(COALESCE(p.weight_v2_1, 1)) AS n_hires
        FROM service_pipelines.output_current.individual_position p
        JOIN skill_users su ON p.user_id = su.user_id
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND {ind_filter}
          AND p.startdate >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
        GROUP BY 1, 2
    ),
    industry_hires_period_tot AS (
        SELECT
            CASE
                WHEN p.startdate >= DATEADD('month', -{recent_m}, CURRENT_DATE()) THEN 'recent'
                WHEN p.startdate >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE()) THEN 'prior'
            END AS period,
            SUM(COALESCE(p.weight_v2_1, 1)) AS tot_hires
        FROM service_pipelines.output_current.individual_position p
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND {ind_filter}
          AND p.startdate >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
        GROUP BY 1
    ),
    hires_growth AS (
        SELECT
            skill,
            (recent_share / NULLIF(prior_share, 0)) - 1 AS industry_hires_share_growth
        FROM (
            SELECT
                skill,
                SUM(CASE WHEN period = 'recent' THEN n_hires ELSE 0 END)::FLOAT
                    / NULLIF((SELECT tot_hires FROM industry_hires_period_tot WHERE period = 'recent'), 0)
                    AS recent_share,
                SUM(CASE WHEN period = 'prior' THEN n_hires ELSE 0 END)::FLOAT
                    / NULLIF((SELECT tot_hires FROM industry_hires_period_tot WHERE period = 'prior'), 0)
                    AS prior_share
            FROM hires
            WHERE period IS NOT NULL
            GROUP BY skill
        ) x
    ),
    headcount AS (
        SELECT su.skill, SUM(COALESCE(p.weight_v2_1, 1)) AS headcount
        FROM service_pipelines.output_current.individual_position p
        JOIN skill_users su ON p.user_id = su.user_id
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND {ind_filter}
          AND p.enddate_primary IS NULL
        GROUP BY 1
    ),
    workforce AS (
        SELECT SUM(COALESCE(p.weight_v2_1, 1)) AS total_hc
        FROM service_pipelines.output_current.individual_position p
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND {ind_filter}
          AND p.enddate_primary IS NULL
    ),
    shares AS (
        SELECT h.skill,
               h.headcount AS industry_headcount,
               h.headcount / NULLIF((SELECT total_hc FROM workforce), 0) AS industry_share
        FROM headcount h
    )
    SELECT
        s.skill,
        COALESCE(pg.industry_postings_share_growth, 0) AS industry_postings_share_growth,
        COALESCE(hg.industry_hires_share_growth, 0) AS industry_hires_share_growth,
        COALESCE(s.industry_headcount, 0) AS industry_headcount,
        COALESCE(s.industry_share, 0) AS industry_share
    FROM shares s
    LEFT JOIN postings_growth pg USING (skill)
    LEFT JOIN hires_growth hg USING (skill)
    WHERE s.industry_headcount >= {min_hc}
    ORDER BY s.skill
    {limit_sql}
    """
    return _lower_cols(_sf(cfg).load_df(q))


def _load_industry_role_categories_synthetic(_cfg):
    return _load_role_categories_synthetic(_cfg)


def load_industry_role_categories(cfg):
    if not cfg.get("use_snowflake") or cfg.get("company_rcid") is None:
        return _load_industry_role_categories_synthetic(cfg)

    industry = _resolve_industry(cfg)
    ind_filter = _industry_filter("p", industry)
    country = cfg.get("country", "United States")
    recent_m = int(cfg.get("recent_months", 12))
    prior_m = int(cfg.get("prior_months", 12))
    min_pool = int(cfg.get("pathway_min_pool", 50))

    q = f"""
    WITH roles AS ({_ROLE_TAXONOMY}),
    pos AS (
        SELECT
            p.position_id,
            p.user_id,
            p.startdate,
            p.enddate,
            p.enddate_primary,
            rt.role,
            COALESCE(p.weight_v2_1, 1) AS wt,
            COALESCE(p.ai_exposure_v1_upsell, 0) AS ai_exposure
        FROM service_pipelines.output_current.individual_position p
        JOIN roles rt ON p.role_v3_id = rt.role_v3_id
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND {ind_filter}
    ),
    ai_skill_mix AS (
        SELECT
            p.role,
            CASE
                WHEN p.startdate >= DATEADD('month', -{recent_m}, CURRENT_DATE()) THEN 'recent'
                WHEN p.startdate >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE()) THEN 'prior'
            END AS period,
            AVG(CASE WHEN COALESCE(t.is_ai, FALSE) THEN 1 ELSE 0 END) AS ai_skill_share
        FROM pos p
        LEFT JOIN service_pipelines.output_current.individual_skills sk ON p.user_id = sk.user_id
        LEFT JOIN service_pipelines.global_ref.custom_skills_taxonomy_v3_overall_latest sn
          ON sk.skill_v3_id = sn.skill_v3_id AND sn.taxonomy_name = 'default'
{_ai_skill_tags_join(cfg)}
        WHERE period IS NOT NULL
        GROUP BY 1, 2
    ),
    mix_change AS (
        SELECT
            role,
            COALESCE(
                MAX(CASE WHEN period = 'recent' THEN ai_skill_share END)
                - MAX(CASE WHEN period = 'prior' THEN ai_skill_share END),
                0
            ) AS skill_mix_change
        FROM ai_skill_mix
        GROUP BY 1
    ),
    flows AS (
        SELECT
            rt.role,
            SUM(CASE WHEN DATE_TRUNC('month', p.startdate) >= DATEADD('month', -{recent_m}, CURRENT_DATE())
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS inflow_recent,
            SUM(CASE WHEN DATE_TRUNC('month', p.startdate) >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
                      AND DATE_TRUNC('month', p.startdate) < DATEADD('month', -{recent_m}, CURRENT_DATE())
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS inflow_prior,
            SUM(CASE WHEN DATE_TRUNC('month', p.enddate_primary) >= DATEADD('month', -{recent_m}, CURRENT_DATE())
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS outflow_recent,
            SUM(CASE WHEN p.enddate_primary IS NULL
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS headcount
        FROM service_pipelines.output_current.individual_position p
        JOIN roles rt ON p.role_v3_id = rt.role_v3_id
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND {ind_filter}
        GROUP BY 1
    ),
    ai_by_role AS (
        SELECT role, AVG(ai_exposure) AS ai_exposure
        FROM pos
        GROUP BY 1
    )
    SELECT
        f.role,
        COALESCE(a.ai_exposure, 0) AS ai_exposure,
        COALESCE(m.skill_mix_change, 0) AS skill_mix_change,
        (f.inflow_recent / NULLIF(f.inflow_prior, 0)) - 1 AS hiring_growth,
        f.outflow_recent / NULLIF(f.headcount, 0) AS attrition
    FROM flows f
    LEFT JOIN ai_by_role a ON f.role = a.role
    LEFT JOIN mix_change m ON f.role = m.role
    WHERE f.headcount >= {min_pool}
      AND LOWER(f.role) NOT IN ('retired', 'unknown', 'on leave', 'empty')
    ORDER BY f.headcount DESC
    LIMIT 25
    """
    return _lower_cols(_sf(cfg).load_df(q))



def load_peer_role_categories(cfg):
    """Role disruption signals aggregated across competitors (peer benchmark)."""
    if not cfg.get("use_snowflake") or cfg.get("company_rcid") is None:
        return _load_peer_role_categories_synthetic(cfg)

    peers = _resolve_peer_rcids(cfg)
    peer_sql = _sql_quote_list(peers["peer_rcids"])
    if not peer_sql:
        return pd.DataFrame(columns=[
            "role", "ai_exposure", "skill_mix_change", "hiring_growth", "attrition"])

    country = cfg.get("country", "United States")
    recent_m = int(cfg.get("recent_months", 12))
    prior_m = int(cfg.get("prior_months", 12))
    min_pool = int(cfg.get("pathway_min_pool", 50))

    q = f"""
    WITH roles AS ({_ROLE_TAXONOMY}),
    pos AS (
        SELECT
            p.position_id,
            p.user_id,
            p.startdate,
            p.enddate,
            p.enddate_primary,
            rt.role,
            COALESCE(p.weight_v2_1, 1) AS wt,
            COALESCE(p.ai_exposure_v1_upsell, 0) AS ai_exposure
        FROM service_pipelines.output_current.individual_position p
        JOIN roles rt ON p.role_v3_id = rt.role_v3_id
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND p.ultimate_parent_rcid IN (
              SELECT value::INT AS rcid
              FROM TABLE(SPLIT_TO_TABLE('{peer_sql}', ','))
          )
    ),
    ai_skill_mix AS (
        SELECT
            p.role,
            CASE
                WHEN p.startdate >= DATEADD('month', -{recent_m}, CURRENT_DATE()) THEN 'recent'
                WHEN p.startdate >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE()) THEN 'prior'
            END AS period,
            AVG(CASE WHEN COALESCE(t.is_ai, FALSE) THEN 1 ELSE 0 END) AS ai_skill_share
        FROM pos p
        LEFT JOIN service_pipelines.output_current.individual_skills sk ON p.user_id = sk.user_id
        LEFT JOIN service_pipelines.global_ref.custom_skills_taxonomy_v3_overall_latest sn
          ON sk.skill_v3_id = sn.skill_v3_id AND sn.taxonomy_name = 'default'
{_ai_skill_tags_join(cfg)}
        WHERE period IS NOT NULL
        GROUP BY 1, 2
    ),
    mix_change AS (
        SELECT
            role,
            COALESCE(
                MAX(CASE WHEN period = 'recent' THEN ai_skill_share END)
                - MAX(CASE WHEN period = 'prior' THEN ai_skill_share END),
                0
            ) AS skill_mix_change
        FROM ai_skill_mix
        GROUP BY 1
    ),
    flows AS (
        SELECT
            rt.role,
            SUM(CASE WHEN DATE_TRUNC('month', p.startdate) >= DATEADD('month', -{recent_m}, CURRENT_DATE())
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS inflow_recent,
            SUM(CASE WHEN DATE_TRUNC('month', p.startdate) >= DATEADD('month', -{recent_m + prior_m}, CURRENT_DATE())
                      AND DATE_TRUNC('month', p.startdate) < DATEADD('month', -{recent_m}, CURRENT_DATE())
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS inflow_prior,
            SUM(CASE WHEN DATE_TRUNC('month', p.enddate_primary) >= DATEADD('month', -{recent_m}, CURRENT_DATE())
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS outflow_recent,
            SUM(CASE WHEN p.enddate_primary IS NULL
                     THEN COALESCE(p.weight_v2_1, 1) ELSE 0 END) AS headcount
        FROM service_pipelines.output_current.individual_position p
        JOIN roles rt ON p.role_v3_id = rt.role_v3_id
        WHERE p.country = '{country}'
          AND {_pos_filter(cfg)}
          AND p.ultimate_parent_rcid IN (
              SELECT value::INT AS rcid
              FROM TABLE(SPLIT_TO_TABLE('{peer_sql}', ','))
          )
        GROUP BY 1
    ),
    ai_by_role AS (
        SELECT role, AVG(ai_exposure) AS ai_exposure
        FROM pos
        GROUP BY 1
    )
    SELECT
        f.role,
        COALESCE(a.ai_exposure, 0) AS ai_exposure,
        COALESCE(m.skill_mix_change, 0) AS skill_mix_change,
        (f.inflow_recent / NULLIF(f.inflow_prior, 0)) - 1 AS hiring_growth,
        f.outflow_recent / NULLIF(f.headcount, 0) AS attrition
    FROM flows f
    LEFT JOIN ai_by_role a ON f.role = a.role
    LEFT JOIN mix_change m ON f.role = m.role
    WHERE f.headcount >= {min_pool}
      AND LOWER(f.role) NOT IN ('retired', 'unknown', 'on leave', 'empty')
    ORDER BY f.headcount DESC
    LIMIT 25
    """
    return _lower_cols(_sf(cfg).load_df(q))

# %% [markdown]
# ## Peer benchmark radar (skills)
# Skills at `CONFIG["skill_level"]` (default k1500). Growth/momentum from
# competitors; shares = skill holders / total workforce (deduped).
# **Entry:** size (≥500 peer HC) → lift vs economy (percentile floor) →
# peer postings noise → specialized tag; no by-name excludes; no global top-N.
# **Buckets:** emerging (hot + material), nascent/watch (hot below share floor),
# core (material, slower growth), declining (material, shrinking). Sub-floor
# non-hot skills are labeled `other` and omitted from display caps.
# Growth signals are capped before blend; dual-positive is optional (off by default).

# %%
def _norm(s):
    rng_ = s.max() - s.min()
    return (s - s.min()) / rng_ if rng_ else s * 0.0

def _apply_lift_and_postings_gates(df, cfg):
    """After size gate: lift percentile floor, min peer postings, specialized.

    Lift = peer_share / economy_share. Floor is the configured percentile of the
    lift distribution among size-gated rows (protects relevance without name lists).
    When require_specialized, keep only skill_tags.is_specialized (untagged drop).
    """
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    if "lift" not in out.columns:
        if "economy_share" in out.columns and "peer_share" in out.columns:
            out["lift"] = out["peer_share"] / out["economy_share"].replace(0, np.nan)
        else:
            out["lift"] = 1.0
    out["lift"] = out["lift"].replace([np.inf, -np.inf], np.nan)
    pct = float(cfg.get("lift_floor_percentile", 50))
    valid = out["lift"].dropna()
    floor = float(np.nanpercentile(valid, pct)) if len(valid) else 0.0
    out["lift_floor"] = floor
    out = out[out["lift"].fillna(-np.inf) >= floor]
    min_posts = int(cfg.get("min_peer_postings", 50))
    if "peer_postings" in out.columns:
        out = out[out["peer_postings"].fillna(0) >= min_posts]
    if cfg.get("require_specialized", True) and "is_specialized" in out.columns:
        out = out[out["is_specialized"].fillna(False).astype(bool)]
    return out.reset_index(drop=True)

def _skill_bucket(growth, share, hot_bar, decline_bar, floor, dual_ok=True):
    """Assign bucket from share (materiality) + growth vs hot/decline bars.

    Order: nascent → emerging → declining → growing → core → other (bin).
    ``dual_ok`` encodes "trend consistent" for hot (nascent/emerging).
    """
    material = share >= floor
    hot = (growth >= hot_bar) and dual_ok
    if hot and not material:
        return "nascent"
    if hot and material:
        return "emerging"
    if material and growth <= decline_bar:
        return "declining"
    if material and growth > 0:
        return "growing"
    if material:
        return "core"
    return "other"


def _growth_signals(df, postings_col, hires_col, cfg):
    """Raw + capped growth columns for blend/momentum and dual-positive hot."""
    cap = float(cfg.get("max_signal_growth", 2.0))
    pg = df[postings_col].astype(float)
    hg = df[hires_col].astype(float)
    pg_c = pg.clip(upper=cap)
    hg_c = hg.clip(upper=cap)
    require_dual = bool(cfg.get("require_dual_positive_growth", False))
    dual_ok = ((pg >= 0) & (hg >= 0)) if require_dual else pd.Series(True, index=df.index)
    return pg_c, hg_c, dual_ok

def _growth_bars(blended, cfg):
    """Hot / decline bars from the universe growth distribution.

    hot_bar = max(P95, 0); decline_bar = min(P10, 0).
    If emerging_growth_percentile is None, hot_bar uses absolute
    emerging_growth_threshold (still floored at 0).
    """
    s = pd.to_numeric(blended, errors="coerce").dropna()
    hot_pct = cfg.get("emerging_growth_percentile", 95)
    dec_pct = float(cfg.get("declining_growth_percentile", 10))
    if hot_pct is not None and len(s):
        hot_raw = float(np.nanpercentile(s, float(hot_pct)))
    else:
        hot_raw = float(cfg.get("emerging_growth_threshold") or 0.20)
    dec_raw = float(np.nanpercentile(s, dec_pct)) if len(s) else 0.0
    return max(hot_raw, 0.0), min(dec_raw, 0.0)


def build_radar(cfg):
    """Size + lift + postings + specialized universe with strategic/watch buckets.

    Use `present_radar` for slide/dashboard tables (cap within each bucket).
    """
    df = load_skill_radar(cfg).copy()
    w = cfg["radar_weights"]
    pg_c, hg_c, dual_ok = _growth_signals(
        df, "peer_postings_share_growth", "peer_hires_share_growth", cfg)
    # blended growth for interpretability (capped share-weighted average)
    df["blended_growth"] = w["postings"] * pg_c + w["hires"] * hg_c
    # momentum score for ranking (normalized so postings/hires sit on one scale)
    df["momentum"] = (w["postings"] * _norm(pg_c) + w["hires"] * _norm(hg_c))
    df["under_index"] = df["peer_share"] - df["company_share"]
    df["index_ratio"] = (
        df["company_share"] / df["peer_share"].replace(0, np.nan))

    floor = float(cfg.get("share_floor", 0.003))
    hot_bar, decline_bar = _growth_bars(df["blended_growth"], cfg)
    df["growth_floor"] = hot_bar          # hot bar (compat name)
    df["hot_bar"] = hot_bar
    df["decline_bar"] = decline_bar
    df["bucket"] = [
        _skill_bucket(g, s, hot_bar, decline_bar, floor, d)
        for g, s, d in zip(df["blended_growth"], df["peer_share"], dual_ok)
    ]
    return df.reset_index(drop=True)

def present_radar(radar, cfg):
    """Display trim: top-N *within each bucket* by the bucket-appropriate metric.

    - emerging / nascent / growing: highest momentum
    - declining: steepest decline (most negative blended_growth)
    - core: largest peer share
    `other` (bin — sub-floor, not hot) is omitted from the slide view.
    """
    if radar is None or getattr(radar, "empty", True):
        return radar
    n = int(cfg.get("present_rows_per_bucket", 8))
    # peer_share for core; industry_share if this is an industry radar frame
    share_col = "peer_share" if "peer_share" in radar.columns else "industry_share"
    specs = [
        ("emerging", "momentum", False),
        ("nascent", "momentum", False),
        ("growing", "momentum", False),
        ("core", share_col, False),
        ("declining", "blended_growth", True),
    ]
    parts = []
    for bucket, sort_col, asc in specs:
        sub = radar[radar["bucket"] == bucket].copy()
        if sub.empty:
            continue
        if sort_col in sub.columns:
            sub = sub.sort_values(sort_col, ascending=asc)
        parts.append(sub.head(n))
    if not parts:
        return radar.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)

# %% [markdown]
# ## Optional industry layer + peer role benchmark
# Industry slides run only when `benchmark_mode` is `industry` or `both`.
# Peer role categories always compare **competitors** to **your company**.

# %%
def build_industry_radar(cfg):
    df = load_industry_skill_radar(cfg).copy()
    w = cfg["radar_weights"]
    pg_c, hg_c, dual_ok = _growth_signals(
        df, "industry_postings_share_growth", "industry_hires_share_growth", cfg)
    df["blended_growth"] = w["postings"] * pg_c + w["hires"] * hg_c
    df["momentum"] = w["postings"] * _norm(pg_c) + w["hires"] * _norm(hg_c)
    floor = float(cfg.get("share_floor", 0.003))
    hot_bar, decline_bar = _growth_bars(df["blended_growth"], cfg)
    df["growth_floor"] = hot_bar
    df["hot_bar"] = hot_bar
    df["decline_bar"] = decline_bar
    df["bucket"] = [
        _skill_bucket(g, s, hot_bar, decline_bar, floor, d)
        for g, s, d in zip(df["blended_growth"], df["industry_share"], dual_ok)
    ]
    return df.sort_values("momentum", ascending=False).reset_index(drop=True)


def build_company_vs_industry(cfg, industry_radar, peer_radar):
    """Merge industry + peer + company shares on skill."""
    df = industry_radar.merge(
        peer_radar[["skill", "peer_share", "company_share", "peer_postings_share_growth",
                    "peer_hires_share_growth"]],
        on="skill", how="outer",
    )
    df["index_vs_industry"] = df["company_share"] - df["industry_share"]
    df["index_vs_peers"] = df["company_share"] - df["peer_share"]
    df["industry_under_indexed"] = df["index_vs_industry"] < 0
    df["peer_under_indexed"] = df["index_vs_peers"] < 0
    df = df[df["skill"].notna() & (df["skill"].str.lower() != "unknown")]
    return df.sort_values("industry_share", ascending=False, na_position="last")


def _assign_role_categories(df, cfg):
    """Role disruption buckets — relative to cohort on real data."""
    df = df.copy()
    ban = excluded_role_names(cfg)
    work = df[~df["role"].astype(str).str.strip().str.lower().isin(ban)].copy()
    if work.empty:
        df["category"] = "stable"
        return drop_excluded_roles(
            df[df["category"] != "excluded"], cfg).reset_index(drop=True)

    for col in ("hiring_growth", "skill_mix_change", "ai_exposure", "attrition"):
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    mode = cfg.get("role_category_mode", "relative")
    min_bucket = int(cfg.get("role_min_per_bucket", 3))
    categories = {}

    if mode == "relative":
        work = work.copy()
        work["_exp"] = _norm(work["hiring_growth"]) + 0.5 * _norm(work["skill_mix_change"])
        work["_trans"] = _norm(work["ai_exposure"]) + _norm(work["skill_mix_change"])
        work["_risk"] = (
            _norm(work["ai_exposure"])
            - _norm(work["hiring_growth"])
            + 0.3 * _norm(work["attrition"])
        )

        hg_hi = work["hiring_growth"].quantile(0.70)
        hg_lo = work["hiring_growth"].quantile(0.30)
        sm_med = work["skill_mix_change"].median()
        sm_hi = work["skill_mix_change"].quantile(0.70)
        ai_med = work["ai_exposure"].median()
        ai_hi = work["ai_exposure"].quantile(0.70)
        any_growth = (work["hiring_growth"] > 0).any()

        def rule_expanding(r):
            if any_growth and r["hiring_growth"] <= 0:
                return False
            return r["hiring_growth"] >= hg_hi and r["skill_mix_change"] >= sm_med

        def rule_transforming(r):
            return r["ai_exposure"] >= ai_hi and r["skill_mix_change"] >= sm_hi

        def rule_atrisk(r):
            return r["ai_exposure"] >= ai_med and r["hiring_growth"] <= hg_lo

        for _, r in work.sort_values("_trans", ascending=False).iterrows():
            if rule_transforming(r) and r["role"] not in categories:
                categories[r["role"]] = "transforming"
        for _, r in work.sort_values("_risk", ascending=False).iterrows():
            if rule_atrisk(r) and r["role"] not in categories:
                categories[r["role"]] = "at-risk"
        for _, r in work.sort_values("_exp", ascending=False).iterrows():
            if rule_expanding(r) and r["role"] not in categories:
                categories[r["role"]] = "expanding"

        backfill = [
            ("expanding", "_exp", lambda w: w[w["hiring_growth"] > 0] if any_growth else w),
            ("transforming", "_trans", lambda w: w),
            ("at-risk", "_risk", lambda w: w[w["ai_exposure"] >= ai_med]),
        ]
        for cat, score_col, pool_fn in backfill:
            need = max(0, min_bucket - sum(v == cat for v in categories.values()))
            if need:
                pool = pool_fn(work[~work["role"].isin(categories)])
                for role in pool.sort_values(score_col, ascending=False).head(need)["role"]:
                    categories[role] = cat
    else:
        eh = float(cfg.get("role_expanding_hiring", 0.05))
        es = float(cfg.get("role_expanding_skill_mix", 0.0005))
        ta = float(cfg.get("role_transforming_ai", 0.30))
        ts = float(cfg.get("role_transforming_skill_mix", 0.001))
        aa = float(cfg.get("role_atrisk_ai", 0.25))
        ah = float(cfg.get("role_atrisk_hiring", -0.15))

        for _, r in work.iterrows():
            if r["hiring_growth"] >= eh and r["skill_mix_change"] >= es:
                categories[r["role"]] = "expanding"
            elif r["ai_exposure"] >= ta and r["skill_mix_change"] >= ts:
                categories[r["role"]] = "transforming"
            elif r["ai_exposure"] >= aa and r["hiring_growth"] <= ah:
                categories[r["role"]] = "at-risk"

    def bucket(r):
        if str(r["role"]).strip().lower() in ban:
            return "excluded"
        return categories.get(r["role"], "stable")

    df["category"] = df.apply(bucket, axis=1)
    return drop_excluded_roles(
        df[df["category"] != "excluded"], cfg).reset_index(drop=True)


def classify_industry_roles(cfg):
    return _assign_role_categories(load_industry_role_categories(cfg), cfg)


def classify_peer_roles(cfg):
    return _assign_role_categories(load_peer_role_categories(cfg), cfg)


def peer_benchmark_label(cfg):
    peers = _resolve_peer_rcids(cfg) if cfg.get("use_snowflake") and cfg.get("company_rcid") else None
    n = len(peers["peer_rcids"]) if peers else 0
    return f"{cfg.get('company', 'Company')} vs {n} competitors"

def print_by_category(df, cat_col, cat_order, cols, top_n=8, sort_map=None):
    """Stacked top/bottom slide sections: one block per category."""
    cols = [c for c in cols if c in df.columns]
    sort_map = sort_map or {}
    for cat in cat_order:
        sub = df[df[cat_col] == cat].copy()
        total = len(sub)
        if sub.empty:
            continue
        sort_col, asc = sort_map.get(cat, (None, False))
        if sort_col and sort_col in sub.columns:
            sub = sub.sort_values(sort_col, ascending=asc)
        print(f"\n--- {cat.upper()} ({total} total, showing top {min(top_n, total)}) ---")
        print(sub[cols].head(top_n).to_string(index=False))


def print_skill_radar(radar, cfg):
    """Print within-bucket capped radar (display view)."""
    view = present_radar(radar, cfg)
    cols = ["skill", "momentum", "blended_growth", "peer_share", "industry_share",
            "company_share", "under_index"]
    for cat in ["emerging", "nascent", "growing", "core", "declining"]:
        total = int((radar["bucket"] == cat).sum()) if "bucket" in radar.columns else 0
        sub = view[view["bucket"] == cat] if "bucket" in view.columns else view.iloc[0:0]
        if sub.empty and total == 0:
            continue
        label = {
            "nascent": "NASCENT (WATCH)",
            "growing": "GROWING",
            "other": "BIN",
        }.get(cat, cat.upper())
        print(f"\n--- {label} ({total} in universe, showing {len(sub)}) ---")
        if not sub.empty:
            show_cols = [c for c in cols if c in sub.columns]
            print(sub[show_cols].to_string(index=False))


def print_role_categories(roles, cfg):
    n = int(cfg.get("present_rows_per_bucket", 8))
    cols = ["role", "ai_exposure", "skill_mix_change", "hiring_growth", "attrition"]
    sort_map = {
        "expanding": ("hiring_growth", False),
        "transforming": ("skill_mix_change", False),
        "at-risk": ("ai_exposure", False),
        "stable": ("hiring_growth", False),
    }
    print_by_category(roles, "category",
                      ["expanding", "transforming", "at-risk", "stable"],
                      cols, n, sort_map)

# %% [markdown]
# ## Selection funnel
# Not "rank one, take the top." A skill must be **emerging among peers**,
# the company must be **under-indexed** on it, and pathways must supply
# meaningful **internal heads** (`min_internal_supply`), not merely N feeder roles.
# Low within-company mobility is a real finding — the gate sizes it, not hides it.

# %%
def _index_ratio(company_share, peer_share):
    """company_share / peer_share; NaN when peer share is 0."""
    peer = float(peer_share) if peer_share is not None and pd.notna(peer_share) else 0.0
    company = float(company_share) if company_share is not None and pd.notna(company_share) else 0.0
    if peer <= 0:
        return np.nan
    return company / peer

def _under_indexed_mask(df, cfg):
    """Relative shortfall vs peers, with a small absolute pp floor.

    Eligible when company/peer < max_index_ratio AND (peer − company) ≥
    min_under_index. Same meaning at 1% or 20% peer share.
    """
    max_r = float(cfg.get("max_index_ratio", 0.90))
    min_pp = float(cfg.get("min_under_index", 0.002))
    peer = df["peer_share"].astype(float)
    company = df["company_share"].astype(float)
    under = peer - company
    ratio = company / peer.replace(0, np.nan)
    return (peer > 0) & (under >= min_pp) & (ratio < max_r)

def select_target(cfg, radar):
    """Pick target among emerging, ratio-under-indexed skills with pathway supply.

    Nascent/watch skills are visible on the radar but ineligible here.
    **One pass:** emerging → index-ratio gate (+ absolute pp floor) → rank →
    sized pathway supply (`min_internal_supply`) + feeder count. Ratio widens
    the pool; supply narrows it — apply together so the demo target doesn't churn.
    """
    min_feeders = int(cfg.get("min_feeder_roles", 1))
    min_supply = float(cfg.get("min_internal_supply", 50))
    under = _under_indexed_mask(radar, cfg) if radar is not None and len(radar) else False
    cands = radar[(radar["bucket"] == "emerging") & under].copy()
    if cands.empty:
        return (cfg.get("force_target_skill"), pd.DataFrame(columns=[
            "skill", "momentum", "pct_of_peer", "feeder_roles",
            "supply_heads", "passes"]))

    if "index_ratio" not in cands.columns:
        cands["index_ratio"] = [
            _index_ratio(c, p) for c, p in zip(cands["company_share"], cands["peer_share"])]

    w = cfg.get("selection_radar_weights") or cfg.get("radar_weights") or {
        "postings": 0.6, "hires": 0.4}
    if {"peer_postings_share_growth", "peer_hires_share_growth"} <= set(cands.columns):
        cands = cands.copy()
        pg_c, hg_c, _ = _growth_signals(
            cands, "peer_postings_share_growth", "peer_hires_share_growth", cfg)
        cands["selection_momentum"] = (
            float(w.get("postings", 0.6)) * _norm(pg_c)
            + float(w.get("hires", 0.4)) * _norm(hg_c))
        rank_col = "selection_momentum"
    else:
        rank_col = "momentum"
    cands = cands.sort_values(rank_col, ascending=False)
    max_cands = int(cfg.get("selection_max_candidates", 5))
    cands = cands.head(max_cands)

    chosen, trace = None, []
    for _, r in cands.iterrows():
        summary = pathway_supply_summary(cfg, r["skill"])
        n_feeders = int(summary["feeder_roles"])
        supply = float(summary["supply_heads"])
        ok = (n_feeders >= min_feeders) and (supply >= min_supply)
        mom = float(r[rank_col]) if rank_col in r.index else float(r["momentum"])
        ratio = float(r["index_ratio"]) if pd.notna(r.get("index_ratio")) else np.nan
        pct_peer = None if np.isnan(ratio) else round(ratio * 100, 1)
        trace.append((r["skill"], round(mom, 2), pct_peer,
                      n_feeders, round(supply, 1), ok))
        if ok and chosen is None:
            chosen = r["skill"]
    funnel = pd.DataFrame(trace, columns=[
        "skill", "momentum", "pct_of_peer", "feeder_roles",
        "supply_heads", "passes"])
    return (cfg.get("force_target_skill") or chosen), funnel


# %% [markdown]
# ## Role categories (workforce context slides)
#
# Role buckets are **context for the SWP deck** — they do not currently feed
# target selection, pathways, scenario sizing, or cost. Those steps use **skills**
# and observed role-to-role transitions, not category labels.
#
# **Where they appear:** `render_deck()` prints company role disruption via
# `print_role_categories()` for SWP context. Peer mobility vs company pathways
# is the TI benchmark (not peer role-category tables).
#
# **What drives the buckets:** `load_role_categories()` SQL computes per role:
# - `ai_exposure` — mean `ai_exposure_v1_upsell` on positions
# - `skill_mix_change` — change in share of **AI-tagged skills** (`skill_tags_all_latest.is_ai`)
# - `hiring_growth` — YoY weighted inflow ratio
# - `attrition` — recent outflow / headcount
#
# `_assign_role_categories()` labels expanding / transforming / at-risk / stable
# relative to the cohort (`role_category_mode: relative`).
#
# Run output stacks non-empty categories top-to-bottom for slides.
# Re-run definition cells after changing CONFIG.

# %%
def classify_roles(cfg, include_roles=None):
    """Classify company roles; union ``include_roles`` (pathway feeders) into cohort."""
    return _assign_role_categories(
        load_role_categories(cfg, include_roles=include_roles), cfg)

# %% [markdown]
# ## Pathways
# Observed **role-pathway** feeders into roles where the target skill lives:
# - destination / target roles = current roles with enough skill holders
# - transitions = **all** within-company moves from a non-target role → target role
# - feeder_pool = **full** current HC in the source role (same universe as numerator)
# - conversion_rate = transition_wt / feeder_pool (then annualized in `run_scenario`)
# - skill_mover_share = share of movers who hold the skill (quality column only)
# - **peer_conversion_rate** = same role-pathway rate at competitors (same
#   destination roles); **mobility_gap** = peer − company
#
# **Role categories** soft-boost feasibility only — they do not size supply.

# %%
_FREQ = {"low": 0.2, "low-med": 0.45, "med": 0.65, "med-high": 0.8, "high": 1.0}
_OVL = {"low": 0.2, "med": 0.5, "med-high": 0.75, "high": 1.0}
_DEFAULT_CATEGORY_BOOST = {
    "transforming": 0.08,
    "at-risk": 0.06,
    "expanding": -0.05,
    "stable": 0.0,
}


def build_pathways(cfg, target_skill, role_categories=None):
    """Role-pathway feeders; role categories annotate + soft-rank feasibility.

    Categories are scored on top HC roles ∪ these pathway source roles so
    feeder names are not left-join orphans (previously all ``stable``).

    Keeps ``transition_freq == "low"`` rows. Absolute count thresholds behind
    "low" (< 3 weighted moves) were calibrated on large employers; at mid-size
    firms (New Balance, Wayfair) every real feeder often sat at 1–2 moves and
    was wiped, leaving a false "0 reskilling" story. Feasibility already
    haircuts thin transitions via the freq score.
    """
    df = load_pathways(cfg, target_skill).copy()
    if df is None or getattr(df, "empty", True):
        return df
    df = drop_excluded_roles(df, cfg, col="source_role")
    if df is None or getattr(df, "empty", True):
        return df

    sources = (
        df["source_role"].dropna().astype(str).unique().tolist()
        if "source_role" in df.columns and len(df) else []
    )
    if role_categories is not None and not role_categories.empty:
        have = set(role_categories["role"].astype(str))
        missing = [r for r in sources if r not in have]
        if missing:
            role_categories = classify_roles(
                cfg, include_roles=list(have) + missing)
    else:
        role_categories = classify_roles(cfg, include_roles=sources)

    if role_categories is not None and not role_categories.empty:
        cats = role_categories[["role", "category"]].drop_duplicates("role")
        df = df.merge(cats, left_on="source_role", right_on="role", how="left")
        df["category"] = df["category"].fillna("stable")
        df = df.drop(columns=["role"], errors="ignore")
    else:
        df["category"] = "stable"

    boosts = cfg.get("pathway_category_boost", _DEFAULT_CATEGORY_BOOST)
    cat_boost = df["category"].map(boosts).fillna(0)

    f = df["transition_freq"].map(_FREQ)
    o = df["skill_overlap"].map(_OVL)
    # Feasibility only penalizes pay cuts (positive wage_gap); raises (neg) → 0.
    wage_pen = 1 - (df["wage_gap"].clip(0, 0.4) / 0.4) * 0.4
    score = (0.45 * f + 0.4 * o + 0.15 * wage_pen + cat_boost).clip(0, 1.2)
    df["feasibility_score"] = score
    df["feasibility"] = np.where(
        score >= 0.75, "high", np.where(score >= 0.55, "med", "low"))
    return sort_pathways(df, cfg)

# %% [markdown]
# ## Scenario
# 1. **Net need = growth need + replacement need.**
# 2. **Growth need** — `fixed` client %, `gap` peer-parity, or `both` (size on `growth_primary`).
# 3. **Internal supply** — observed pathway conversion, **annualized** to the plan horizon:
#    `(conversion_rate / pathway_years) × horizon_years × conversion_persistence`,
#    capped at `max_feeder_conversion` of each feeder pool. Past rates are an assumption,
#    not a forecast guarantee.
# 4. Retention lowers replacement before build/buy.

# %%
def _radar_target_row(radar, target):
    if radar is None or getattr(radar, "empty", True):
        return None
    row = radar[radar["skill"] == target]
    return row.iloc[0] if len(row) else None


def _growth_need_gap(current, radar_row):
    """Heads to reach peer skill share at current company workforce."""
    meta = {
        "peer_share": None, "company_share": None, "under_index": None,
        "index_ratio": None, "gap_pct_of_current": None,
    }
    if radar_row is None:
        return 0.0, meta
    peer = float(radar_row["peer_share"])
    company = float(radar_row["company_share"])
    under = float(radar_row.get("under_index", peer - company))
    ratio = _index_ratio(company, peer)
    meta.update({
        "peer_share": peer, "company_share": company,
        "under_index": under, "index_ratio": ratio,
    })
    if company <= 0 or under <= 0 or current <= 0:
        return 0.0, meta
    gap = current * (peer / company - 1.0)
    meta["gap_pct_of_current"] = gap / current
    return max(gap, 0.0), meta


def _apply_gap_cap(gap, current, cfg, meta):
    """Cap gap growth at max_gap_multiple x current; flag when capped."""
    mult = float(cfg.get("max_gap_multiple", 3.0))
    cap = current * mult
    meta["gap_capped"] = bool(gap > cap)
    return (min(gap, cap), meta)


def run_scenario(cfg, target_skill, pathways, radar=None, population=None):
    """Size growth + replacement need; growth can be fixed % and/or peer-gap.

    population: optional ``(current_hc, attrition[, hiring_rate])`` — pass this
    from the dashboard so scenario sliders don't re-query Snowflake / get stuck
    on a monkeypatched loader.
    """
    hiring_rate = None
    if population is not None:
        if len(population) >= 3:
            C, attrition, hiring_rate = population[0], population[1], population[2]
        else:
            C, attrition = population[0], population[1]
    else:
        C, attrition, hiring_rate = load_target_population(cfg, target_skill)
    H = cfg["horizon_years"]
    g = float(cfg["growth_target"])

    row = _radar_target_row(radar, target_skill)
    growth_fixed = C * g
    growth_gap, gap_meta = _growth_need_gap(C, row)
    growth_gap, gap_meta = _apply_gap_cap(growth_gap, C, cfg, gap_meta)

    mode = cfg.get("growth_mode", "both")
    primary = cfg.get("growth_primary", "gap")
    if mode == "fixed":
        growth_need, growth_basis = growth_fixed, "fixed"
    elif mode == "gap":
        growth_need, growth_basis = growth_gap, "gap"
    else:  # both — report both; size on primary
        growth_basis = primary if primary in ("fixed", "gap") else "gap"
        growth_need = growth_gap if growth_basis == "gap" else growth_fixed

    # Retention what-if shrinks the hole: size replacement on improved attrition.
    # improved = attrition × (1 − improvement); heads_saved = baseline − improved.
    improv = float(cfg.get("retention_improvement", 0.15))
    improv = float(np.clip(improv, 0.0, 0.40))
    attr = float(attrition)
    improved_attr = attr * (1.0 - improv)
    replacement_baseline = C * attr * float(H)
    replacement_need = C * improved_attr * float(H)
    heads_saved = max(0.0, replacement_baseline - replacement_need)
    net_need_baseline = growth_need + replacement_baseline
    net_need = growth_need + replacement_need

    pathways = pathways if pathways is not None else pd.DataFrame()
    P = float(cfg.get("pathway_years", H) or H)
    persistence = float(cfg.get("conversion_persistence", 1.0))
    if P <= 0:
        raise ValueError("pathway_years must be > 0")
    if not pathways.empty:
        # Role-pathway rate over pathway_years → annualize into the plan horizon.
        pathways = annotate_pathway_supply(cfg, pathways)
        internal_supply = _horizon_pathway_supply(cfg, pathways)
        internal_supply_raw = float(pathways["supply_heads_raw"].sum())
    else:
        internal_supply = 0.0
        internal_supply_raw = 0.0

    # Cap applied reskill at net need — pathways can exceed the hole; don't
    # plan "reskill 135 into a 104-person need." Scale pathway heads for cost.
    internal_available = float(internal_supply)
    covers_full_need = bool(internal_available + 1e-9 >= max(net_need, 0.0)
                             and internal_available > 0)
    if internal_available > net_need > 0:
        if not pathways.empty and "supply_heads" in pathways.columns:
            pathways["supply_heads"] = (
                pathways["supply_heads"].astype(float)
                * (net_need / internal_available))
        internal_supply = float(net_need)
    elif net_need <= 0:
        internal_supply = 0.0
        if not pathways.empty and "supply_heads" in pathways.columns:
            pathways["supply_heads"] = 0.0

    external_need = max(net_need - internal_supply, 0.0)

    target_median_comp = None
    if not pathways.empty and "target_median_comp" in pathways.columns:
        target_median_comp = float(pathways["target_median_comp"].iloc[0] or 0) or None

    out = {
        "current": C, "attrition": attrition,
        "hiring_rate": hiring_rate,
        "growth_need": growth_need,
        "growth_need_fixed": growth_fixed,
        "growth_need_gap": growth_gap,
        "growth_basis": growth_basis,
        "growth_mode": mode,
        "growth_target_pct": g,
        "peer_share": gap_meta["peer_share"],
        "company_share": gap_meta["company_share"],
        "under_index": gap_meta["under_index"],
        "index_ratio": gap_meta.get("index_ratio"),
        "gap_pct_of_current": gap_meta["gap_pct_of_current"],
        "gap_capped": gap_meta.get("gap_capped", False),
        "replacement_need_baseline": replacement_baseline,
        "replacement_need": replacement_need,
        "net_need_baseline": net_need_baseline,
        "net_need": net_need, "internal_supply": internal_supply,
        "internal_supply_raw": internal_supply_raw,
        "internal_supply_available": internal_available,
        "pathways_cover_full_need": covers_full_need,
        "pathway_years": P, "horizon_years": float(H),
        "conversion_persistence": persistence,
        "external_need": external_need,
        "retention_improvement": improv,
        "attrition_after_retention": improved_attr,
        "heads_saved_by_retention": heads_saved,
        "target_median_comp": target_median_comp,
    }
    out.update(compute_plan_cost(cfg, out, pathways))
    return out
def compute_plan_cost(cfg, sc, paths):
    """Plan cost: comp-weighted by pathway when enabled, else flat CONFIG rates.

    When ``cfg['skill_time']`` carries a median months-to-report (from role
    entry to first showing the skill), reskill cost also includes a ramp
    opportunity cost: supply × (monthly target comp) × months × ramp_pct.
    That turns the feasibility clock into dollars instead of leaving it as
    a caption.
    """
    paths = paths if paths is not None else pd.DataFrame()
    use_comp = cfg.get("use_comp_based_costs", True)
    skill_time = cfg.get("skill_time") or sc.get("skill_time")
    ramp_months = None
    if isinstance(skill_time, dict) and skill_time.get("median_months") is not None:
        ramp_months = float(np.clip(skill_time["median_months"], 0.0, 36.0))
    ramp_pct = float(cfg.get("cost_reskill_ramp_pct", 0.25))

    if use_comp and not paths.empty and "target_median_comp" in paths.columns:
        target_comp = float(paths["target_median_comp"].iloc[0] or 0)
        if target_comp <= 0:
            hire_mult = float(cfg.get("cost_hire_multiplier", 1.25))
            target_comp = float(cfg.get("cost_hire", 110_000)) / hire_mult

        p = paths.copy()
        if "supply_heads" not in p.columns:
            p["supply_heads"] = p["feeder_pool"] * p["conversion_rate"]
        src = pd.to_numeric(p.get("source_median_comp", 0), errors="coerce").fillna(0)
        training_pct = float(cfg.get("cost_reskill_training_pct", 0.15))
        # Direct programme cost: wage catch-up + training as % of source comp.
        reskill_direct = float(
            (p["supply_heads"] * (
                np.maximum(target_comp - src, 0) + src * training_pct
            )).sum())
        # Ramp opportunity cost from observed time-to-report after role entry.
        supply = float(sc.get("internal_supply", 0) or 0)
        ramp_cost = 0.0
        if ramp_months and ramp_months > 0 and supply > 0:
            ramp_cost = supply * (target_comp / 12.0) * ramp_months * ramp_pct
        reskill_cost = reskill_direct + ramp_cost

        hire_mult = float(cfg.get("cost_hire_multiplier", 1.25))
        hire_cost = float(sc["external_need"]) * target_comp * hire_mult

        retain_pct = float(cfg.get("cost_retain_pct", 0.08))
        retain_cost = float(sc["heads_saved_by_retention"]) * target_comp * retain_pct

        plan_cost = reskill_cost + hire_cost + retain_cost
        naive_base = float(sc.get("net_need_baseline", sc["net_need"]))
        naive_cost = naive_base * target_comp * hire_mult
        out = {
            "plan_cost": plan_cost,
            "naive_cost": naive_cost,
            "reskill_cost": reskill_cost,
            "reskill_direct_cost": reskill_direct,
            "reskill_ramp_cost": ramp_cost,
            "hire_cost": hire_cost,
            "retain_cost": retain_cost,
            "cost_basis": "comp",
            "target_median_comp": target_comp,
        }
        if ramp_months is not None:
            out["skill_ramp_months"] = ramp_months
        return out

    reskill_direct = float(sc["internal_supply"]) * float(cfg["cost_reskill"])
    ramp_cost = 0.0
    if ramp_months and ramp_months > 0:
        # Flat path: approximate monthly comp from the flat hire rate.
        monthly = float(cfg["cost_hire"]) / 12.0
        ramp_cost = (float(sc["internal_supply"]) * monthly
                     * ramp_months * ramp_pct)
    reskill_cost = reskill_direct + ramp_cost
    hire_cost = float(sc["external_need"]) * float(cfg["cost_hire"])
    retain_cost = float(sc["heads_saved_by_retention"]) * float(cfg["cost_retain"])
    naive_base = float(sc.get("net_need_baseline", sc["net_need"]))
    out = {
        "plan_cost": reskill_cost + hire_cost + retain_cost,
        "naive_cost": naive_base * float(cfg["cost_hire"]),
        "reskill_cost": reskill_cost,
        "reskill_direct_cost": reskill_direct,
        "reskill_ramp_cost": ramp_cost,
        "hire_cost": hire_cost,
        "retain_cost": retain_cost,
        "cost_basis": "flat",
        "target_median_comp": sc.get("target_median_comp"),
    }
    if ramp_months is not None:
        out["skill_ramp_months"] = ramp_months
    return out

# %% [markdown]
# ## Metro read + dual narrative
# TI: talent map (supply vs competitor demand, winnable markets).
# SWP: hire strategy and location prioritization.
# `render_deck()` orders slides by `deck_lead`.

# %%

# %% [markdown]
# ## Metro read + dual narrative
# TI: talent map (supply vs competitor demand, winnable markets).
# SWP: hire strategy and location prioritization.
# `render_deck()` orders slides by `deck_lead`.

# %%
def read_metros(cfg, target_skill):
    df = load_metro_supply(cfg, target_skill).copy()
    df["ratio"] = df["external_supply"] / df["competitor_demand"].replace(0, np.nan)
    df["ratio"] = df["ratio"].fillna(np.inf)
    tight = df.sort_values("ratio").head(2)["metro"].tolist()
    avail = df[df["company_presence"].astype(bool)].sort_values(
        "ratio", ascending=False).head(2)["metro"].tolist()
    return df, tight, avail


def _target_row(radar, target):
    row = radar[radar["skill"] == target]
    return row.iloc[0] if len(row) else None


def recommend_swp(cfg, target, sc, tight, avail, paths):
    cost = sc.get("plan_cost", (
        sc["internal_supply"] * cfg["cost_reskill"]
        + sc["external_need"] * cfg["cost_hire"]
        + sc["heads_saved_by_retention"] * cfg["cost_retain"]))
    naive_cost = sc.get("naive_cost", sc["net_need"] * cfg["cost_hire"])
    t0, t1 = (tight + ["—", "—"])[:2]
    a0, a1 = (avail + ["—", "—"])[:2]
    basis = sc.get("growth_basis", "fixed")
    if basis == "gap":
        ratio = sc.get("index_ratio")
        if ratio is not None and pd.notna(ratio):
            rel = f"you're at {ratio*100:.0f}% of the peer rate"
        else:
            rel = "you're behind peers"
        growth_phrase = (
            f"close the gap on {target} ({rel}; "
            f"~{round(sc['growth_need'])} heads to parity)"
        )
    else:
        growth_phrase = (
            f"raise {target} capacity "
            f"{int(sc.get('growth_target_pct', cfg['growth_target'])*100)}%"
        )
    alt = ""
    if sc.get("growth_mode") == "both":
        alt = (
            f" Alternate view: fixed "
            f"{int(sc.get('growth_target_pct', cfg['growth_target'])*100)}% "
            f"→ {round(sc.get('growth_need_fixed', 0))} growth heads; "
            f"gap-to-peer → {round(sc.get('growth_need_gap', 0))}."
        )
    if basis == "gap":
        timing = "to reach peer parity"
    else:
        timing = f"over {cfg['horizon_years']*12} months"
    improv = float(sc.get("retention_improvement", 0) or 0)
    attr0 = float(sc.get("attrition", 0) or 0)
    attr1 = float(sc.get("attrition_after_retention", attr0 * (1 - improv)) or 0)
    base_net = float(sc.get("net_need_baseline", sc["net_need"]))
    if improv <= 0:
        retention_bit = (
            f"Skill attrition runs at {attr0*100:.1f}% with no assumed retention "
            f"improvement (slider at 0%)"
        )
    else:
        retention_bit = (
            f"This skill loses {attr0*100:.1f}% a year; a program cutting that by "
            f"{improv*100:.0f}% (to {attr1*100:.1f}%) retains "
            f"~{round(sc['heads_saved_by_retention'])} over the horizon and shrinks "
            f"net need from ~{round(base_net)} to ~{round(sc['net_need'])}"
        )
    close = (
        f"SWP view — To {growth_phrase} {timing}, "
        f"roughly {round(sc['net_need'])} positions need filling once expected "
        f"attrition is counted ({round(sc['growth_need'])} growth + "
        f"{round(sc['replacement_need'])} replacement after retention).{alt} "
        f"Observed internal pathways "
        f"could supply ~{round(sc['internal_supply'])} via "
        f"{cfg.get('build_term', 'reskilling')}; "
        f"~{round(sc['external_need'])} require external hire. {retention_bit}. "
        f"For external hiring, prioritize {a0} and {a1} (favorable supply/demand); "
        f"expect competition in {t0} and {t1}. "
        f"Plan cost uses {sc.get('cost_basis', 'flat')} basis"
        + (f" (target median ${sc['target_median_comp']:,.0f} × hire mult)."
           if sc.get("cost_basis") == "comp" and sc.get("target_median_comp")
           else ".")
    )
    return cost, naive_cost, close
def recommend_ti(cfg, target, sc, metros, outflows, radar):
    tr = _target_row(radar, target)
    peer_growth = round(float(tr["blended_growth"]) * 100, 1) if tr is not None else "—"
    if tr is not None:
        ratio = tr["index_ratio"] if "index_ratio" in tr.index and pd.notna(tr.get("index_ratio")) else _index_ratio(
            tr.get("company_share"), tr.get("peer_share"))
        peer_s = float(tr.get("peer_share") or 0)
        co_s = float(tr.get("company_share") or 0)
    else:
        ratio = sc.get("index_ratio")
        peer_s = float(sc.get("peer_share") or 0)
        co_s = float(sc.get("company_share") or 0)
    if ratio is not None and pd.notna(ratio):
        gap_txt = (
            f"at {float(ratio)*100:.0f}% of the peer rate "
            f"(peers {peer_s*100:.1f}% vs company {co_s*100:.1f}%)"
        )
    else:
        gap_txt = "near peer parity"
    top_dest = outflows.iloc[0]["dest_company"] if len(outflows) else "—"
    top_share = (
        round(100 * outflows.iloc[0]["outflow_wt"] / outflows["outflow_wt"].sum(), 1)
        if len(outflows) and outflows["outflow_wt"].sum() > 0 else "—"
    )
    bucket = str(tr["bucket"]) if tr is not None and "bucket" in tr.index else "unclassified"
    close = (
        f"TI view — Among peers, {target} is {bucket} ({peer_growth}% blended growth); "
        f"{cfg['company']} is {gap_txt}. "
        f"External talent pool to close the gap: ~{round(sc['external_need'])} weighted positions "
        f"(of {round(sc['net_need'])} total need). Top rival destination for departing "
        f"{target} talent: {top_dest} ({top_share}% of tracked outflows). "
        f"Internal {cfg.get('build_term', 'reskilling')} "
        f"(~{round(sc['internal_supply'])} heads over "
        f"{int(sc.get('horizon_years', cfg['horizon_years']))}y, "
        f"annualized from {int(sc.get('pathway_years', cfg.get('pathway_years', 2)))}y observed "
        f"mobility × persistence {sc.get('conversion_persistence', 1):.2f}) "
        f"reduces external dependence."
    )
    return close


def print_metro_table(metros, tight, avail, cfg):
    n = int(cfg.get("present_rows_per_bucket", 8))
    cols = ["metro", "external_supply", "competitor_demand", "ratio", "company_presence"]
    df = metros[cols].copy()
    df["ratio"] = df["ratio"].replace(np.inf, np.nan)
    print("Contested markets (low supply/demand ratio):")
    print(df.sort_values("ratio").head(n).to_string(index=False))
    print("\nFavorable markets (high ratio, company presence):")
    pref = df[df["company_presence"].astype(bool)].sort_values("ratio", ascending=False)
    print(pref.head(n).to_string(index=False))
    print(f"\nTightest: {', '.join(tight) or '—'}")
    print(f"Most winnable (with presence): {', '.join(avail) or '—'}")


def print_outflows(outflows, cfg):
    if outflows.empty:
        print("(no peer outflows in window)")
        return
    df = outflows.copy()
    total = df["outflow_wt"].sum()
    df["share"] = df["outflow_wt"] / total
    print(df[["dest_company", "outflow_wt", "share"]].to_string(index=False))


def render_deck(cfg, *, radar, funnel, target, peer_roles, roles,
                paths, sc, metros, tight, avail, outflows,
                industry_radar=None, vs_industry=None, industry_info=None,
                industry_roles=None):
    """Print slide deck tagged [TI]/[SWP]/[Both]; order follows deck_lead."""
    lead = cfg.get("deck_lead", "dual")
    peer_label = peer_benchmark_label(cfg) if cfg.get("use_snowflake") and cfg.get("company_rcid") else "Peer benchmark"
    mode = cfg.get("benchmark_mode", "peers")
    slide = [0]

    def hdr(title, tag):
        slide[0] += 1
        print("\n" + "=" * 70)
        print(f"SLIDE {slide[0]}  [{tag}]  {title}")
        print("=" * 70)

    cost, naive_cost, close_swp = recommend_swp(cfg, target, sc, tight, avail, paths)
    close_ti = recommend_ti(cfg, target, sc, metros, outflows, radar)

    # --- optional industry ---
    if mode in ("industry", "both") and industry_radar is not None and not industry_radar.empty:
        hdr(f"Industry skill context ({industry_info.get('industry_label', '')})", "TI")
        print_skill_radar(industry_radar, cfg)

    # --- TI block ---
    if lead in ("ti", "dual"):
        hdr("Executive summary — talent intelligence", "TI")
        print(close_ti)

        hdr(f"Competitive skill radar ({peer_label})", "TI")
        print("Which skills competitors are building (postings + hires growth).")
        print(f"Grain: {cfg.get('skill_level', 'skill_k1500')}")
        print_skill_radar(radar, cfg)

        hdr(f"Target skill — {target}", "TI+SWP")
        tr = _target_row(radar, target)
        if tr is not None:
            print(tr[["skill", "bucket", "momentum", "blended_growth",
                        "peer_share", "company_share", "under_index"]].to_string())
        print("\nSelection funnel:")
        print(funnel.to_string(index=False))

        hdr(f"Talent map — {target} supply vs competitor demand", "TI")
        print_metro_table(metros, tight, avail, cfg)

        hdr(f"Competitive outflows — where {target} talent goes", "TI")
        print(f"Departures to peer set (last {cfg.get('outflow_years', 2)}y).")
        print_outflows(outflows, cfg)

        hdr(f"External pool sizing — {target}", "TI")
        print(f"  current capability     : {round(sc['current']):,}")
        print(f"  total need (growth+repl): {round(sc['net_need']):,}")
        print(f"  external hire need     : {round(sc['external_need']):,}")
        print(f"  peer talent tracked    : {round(metros['external_supply'].sum()):,} (all metros)")

    # --- SWP block ---
    if lead in ("swp", "dual"):
        improv = float(sc.get("retention_improvement", 0) or 0)
        attr0 = float(sc.get("attrition", 0) or 0)
        attr1 = float(sc.get("attrition_after_retention", attr0) or 0)
        if lead == "dual":
            hdr("Bridge — internal levers cut external dependence", "Both")
            print(
                f"  External need is ~{round(sc['external_need']):,} after "
                f"internal pathways."
            )
        print(
            f"  retention lever  : {attr0*100:.1f}% → {attr1*100:.1f}% "
            f"({improv*100:.0f}% assumed improvement) → saves "
            f"{round(sc['heads_saved_by_retention']):,} heads"
        )

        if lead == "swp":
            hdr(f"Talent map (hire strategy) — {target}", "TI")
            print_metro_table(metros, tight, avail, cfg)
            hdr(f"Competitive outflows — {target}", "TI")
            print_outflows(outflows, cfg)

        hdr(f"Integrated plan — {target}", "Both")
        print("--- Talent intelligence ---")
        print(close_ti)
        print("\n--- Workforce planning ---")
        print(close_swp)
        basis = sc.get("cost_basis", "flat")
        print(f"\n  cost basis          : {basis}"
              + (f" (target median comp ${sc['target_median_comp']:,.0f})"
                 if sc.get("target_median_comp") else ""))
        print(f"  reskill             : ${sc.get('reskill_cost', 0)/1e6:,.1f}M")
        print(f"  external hire       : ${sc.get('hire_cost', 0)/1e6:,.1f}M")
        print(f"  retention           : ${sc.get('retain_cost', 0)/1e6:,.1f}M")
        print(f"  est. plan cost      : ${cost/1e6:,.1f}M")
        print(f"  vs buy-everything   : ${naive_cost/1e6:,.1f}M  "
              f"(saving ~${(naive_cost-cost)/1e6:,.1f}M)")

    return cost, naive_cost, close_swp, close_ti

# %%
def export_skills_engine(
    notebook_path=None,
    out_path=None,
    stop_heading="## Run",
):
    """Write notebook cells above the Run section to skills_engine.py.

    Mechanical sync: edit the notebook, run this (from the Run section), and
    the module regenerates. Cells at/after ``stop_heading`` stay notebook-only.
    """
    import inspect
    import json
    from pathlib import Path

    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    nb_path = Path(notebook_path) if notebook_path else here / "skills_scenario_planner.ipynb"
    dest = Path(out_path) if out_path else here / "skills_engine.py"
    nb = json.loads(nb_path.read_text())

    lines = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source") or [])
        first = next((ln.strip() for ln in src.splitlines() if ln.strip()), "")
        if cell.get("cell_type") == "markdown" and first.startswith(stop_heading):
            break
        if cell.get("cell_type") == "markdown":
            lines.append("# %% [markdown]\n")
            for ln in src.splitlines(keepends=True):
                if not ln.strip():
                    lines.append("#\n")
                elif ln.endswith("\n"):
                    lines.append("# " + ln)
                else:
                    lines.append("# " + ln + "\n")
            if src and not src.endswith("\n"):
                lines.append("\n")
        else:
            lines.append("# %%\n")
            lines.append(src if src.endswith("\n") else src + "\n")
        lines.append("\n")

    # Keep the exporter in the regenerated module.
    lines.append("# %%\n")
    lines.append(inspect.getsource(export_skills_engine))
    if not lines[-1].endswith("\n"):
        lines.append("\n")

    dest.write_text("".join(lines))
    return str(dest)

# %%
def export_skills_engine(
    notebook_path=None,
    out_path=None,
    stop_heading="## Run",
):
    """Write notebook cells above the Run section to skills_engine.py.

    Mechanical sync: edit the notebook, run this (from the Run section), and
    the module regenerates. Cells at/after ``stop_heading`` stay notebook-only.
    """
    import inspect
    import json
    from pathlib import Path

    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    nb_path = Path(notebook_path) if notebook_path else here / "skills_scenario_planner.ipynb"
    dest = Path(out_path) if out_path else here / "skills_engine.py"
    nb = json.loads(nb_path.read_text())

    lines = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source") or [])
        first = next((ln.strip() for ln in src.splitlines() if ln.strip()), "")
        if cell.get("cell_type") == "markdown" and first.startswith(stop_heading):
            break
        if cell.get("cell_type") == "markdown":
            lines.append("# %% [markdown]\n")
            for ln in src.splitlines(keepends=True):
                if not ln.strip():
                    lines.append("#\n")
                elif ln.endswith("\n"):
                    lines.append("# " + ln)
                else:
                    lines.append("# " + ln + "\n")
            if src and not src.endswith("\n"):
                lines.append("\n")
        else:
            lines.append("# %%\n")
            lines.append(src if src.endswith("\n") else src + "\n")
        lines.append("\n")

    # Keep the exporter in the regenerated module.
    lines.append("# %%\n")
    lines.append(inspect.getsource(export_skills_engine))
    if not lines[-1].endswith("\n"):
        lines.append("\n")

    dest.write_text("".join(lines))
    return str(dest)

# %%
def export_skills_engine(
    notebook_path=None,
    out_path=None,
    stop_heading="## Run",
):
    """Write notebook cells above the Run section to skills_engine.py.

    Mechanical sync: edit the notebook, run this (from the Run section), and
    the module regenerates. Cells at/after ``stop_heading`` stay notebook-only.
    """
    import inspect
    import json
    from pathlib import Path

    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    nb_path = Path(notebook_path) if notebook_path else here / "skills_scenario_planner.ipynb"
    dest = Path(out_path) if out_path else here / "skills_engine.py"
    nb = json.loads(nb_path.read_text())

    lines = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source") or [])
        first = next((ln.strip() for ln in src.splitlines() if ln.strip()), "")
        if cell.get("cell_type") == "markdown" and first.startswith(stop_heading):
            break
        if cell.get("cell_type") == "markdown":
            lines.append("# %% [markdown]\n")
            for ln in src.splitlines(keepends=True):
                if not ln.strip():
                    lines.append("#\n")
                elif ln.endswith("\n"):
                    lines.append("# " + ln)
                else:
                    lines.append("# " + ln + "\n")
            if src and not src.endswith("\n"):
                lines.append("\n")
        else:
            lines.append("# %%\n")
            lines.append(src if src.endswith("\n") else src + "\n")
        lines.append("\n")

    # Keep the exporter in the regenerated module.
    lines.append("# %%\n")
    lines.append(inspect.getsource(export_skills_engine))
    if not lines[-1].endswith("\n"):
        lines.append("\n")

    dest.write_text("".join(lines))
    return str(dest)

# %%
def export_skills_engine(
    notebook_path=None,
    out_path=None,
    stop_heading="## Run",
):
    """Write notebook cells above the Run section to skills_engine.py.

    Mechanical sync: edit the notebook, run this (from the Run section), and
    the module regenerates. Cells at/after ``stop_heading`` stay notebook-only.
    """
    import inspect
    import json
    from pathlib import Path

    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    nb_path = Path(notebook_path) if notebook_path else here / "skills_scenario_planner.ipynb"
    dest = Path(out_path) if out_path else here / "skills_engine.py"
    nb = json.loads(nb_path.read_text())

    lines = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source") or [])
        first = next((ln.strip() for ln in src.splitlines() if ln.strip()), "")
        if cell.get("cell_type") == "markdown" and first.startswith(stop_heading):
            break
        if cell.get("cell_type") == "markdown":
            lines.append("# %% [markdown]\n")
            for ln in src.splitlines(keepends=True):
                if not ln.strip():
                    lines.append("#\n")
                elif ln.endswith("\n"):
                    lines.append("# " + ln)
                else:
                    lines.append("# " + ln + "\n")
            if src and not src.endswith("\n"):
                lines.append("\n")
        else:
            lines.append("# %%\n")
            lines.append(src if src.endswith("\n") else src + "\n")
        lines.append("\n")

    # Keep the exporter in the regenerated module.
    lines.append("# %%\n")
    lines.append(inspect.getsource(export_skills_engine))
    if not lines[-1].endswith("\n"):
        lines.append("\n")

    dest.write_text("".join(lines))
    return str(dest)
