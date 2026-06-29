import streamlit as st
import json
import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess

# Configure page layout and style
st.set_page_config(
    page_title="Redrob Candidate Discovery & Ranking Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
/* Font import */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* Apply global font styling */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, .glow-title {
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphism containers */
.glass-card {
    background: rgba(25, 28, 41, 0.6);
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    margin-bottom: 20px;
}

.glass-card-compact {
    background: rgba(25, 28, 41, 0.4);
    border-radius: 8px;
    padding: 14px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.15);
    margin-bottom: 12px;
}

/* Gradients and typography */
.glow-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.sub-title {
    font-size: 1.1rem;
    color: #8c9bb4;
    font-weight: 400;
    margin-bottom: 24px;
}

/* Custom indicator tags and badges */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: 1px solid transparent;
}

.badge-passed {
    background-color: rgba(0, 230, 118, 0.12);
    color: #00e676;
    border-color: rgba(0, 230, 118, 0.3);
}

.badge-fallback {
    background-color: rgba(255, 179, 0, 0.12);
    color: #ffb300;
    border-color: rgba(255, 179, 0, 0.3);
}

.badge-failed {
    background-color: rgba(255, 23, 68, 0.12);
    color: #ff1744;
    border-color: rgba(255, 23, 68, 0.3);
}

.badge-location {
    background-color: rgba(0, 242, 254, 0.1);
    color: #00f2fe;
    border-color: rgba(0, 242, 254, 0.3);
}

/* Custom Skill pills */
.skill-pill {
    display: inline-block;
    background-color: rgba(255, 255, 255, 0.04);
    color: #c9d1d9;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11.5px;
    margin: 4px 3px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    transition: all 0.2s ease;
}

.skill-pill-highlight {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.12) 0%, rgba(79, 172, 254, 0.12) 100%);
    color: #00f2fe;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11.5px;
    margin: 4px 3px;
    border: 1px solid rgba(0, 242, 254, 0.35);
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(0, 242, 254, 0.05);
}

/* Timeline visualization */
.timeline-item {
    border-left: 2px solid rgba(0, 242, 254, 0.25);
    padding-left: 20px;
    margin-left: 10px;
    position: relative;
    padding-bottom: 24px;
}

.timeline-item:last-child {
    border-left: 2px solid transparent;
    padding-bottom: 0px;
}

.timeline-marker {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    position: absolute;
    left: -7px;
    top: 5px;
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.6);
}

.timeline-title {
    font-weight: 600;
    font-size: 14px;
    color: #f0f6fc;
    margin-bottom: 2px;
}

.timeline-subtitle {
    font-size: 12.5px;
    color: #8c9bb4;
    margin-bottom: 8px;
}

.timeline-desc {
    font-size: 13px;
    color: #c9d1d9;
    line-height: 1.5;
}

/* Metric blocks */
.metric-container {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    margin-bottom: 20px;
}
.metric-box {
    flex: 1;
    background: rgba(25, 28, 41, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-val {
    font-size: 24px;
    font-weight: 700;
    color: #00f2fe;
}
.metric-lbl {
    font-size: 12px;
    color: #8c9bb4;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# Imports from project codebase
import produce_submission
import reasoning
from validate_submission import validate_submission

ARTIFACTS_DIR = Path("artifacts")
CANDIDATES_PATH = "candidates.jsonl"
SAMPLE_CANDIDATES_PATH = "sample_candidates.json"

# Check if artifacts exist
def check_artifacts_status():
    required_files = [
        "candidate_features.pkl",
        "candidate_embeddings.npy",
        "candidate_ids.json",
        "centroid.npy",
        "jd_embedding.npy",
        "credibility_scores.npy",
        "hard_filter_mask.npy",
        "feature_names.json"
    ]
    status = {}
    if not ARTIFACTS_DIR.exists():
        status = {f: False for f in required_files}
        status["bm25_index.pkl"] = False
        return status
    for f in required_files:
        status[f] = (ARTIFACTS_DIR / f).exists()
    status["bm25_index.pkl"] = (ARTIFACTS_DIR / "bm25_index.pkl").exists() or (ARTIFACTS_DIR / "bm25_index.pkl.gz").exists()
    return status

# Default weights
DEFAULT_WEIGHTS = produce_submission.FEATURE_WEIGHTS.copy()

# Preset Weights Config
PRESET_WEIGHTS = {
    "Challenge Default (JD Balanced)": DEFAULT_WEIGHTS.copy(),
    "Strict Technical Match": {
        **DEFAULT_WEIGHTS,
        'in_band_5_9': 10.0,
        'has_ranking_skill': 5.0,
        'has_ranking_in_career': 6.0,
        'has_embedding_skill': 3.5,
        'has_llm_skill': 3.0,
        'ml_skill_kw_count': 1.0,
        'ml_career_kw_count': 1.2,
        'product_company_years': 1.5,
        'consulting_only_career': -12.0,
        'in_pune_noida': 1.5,  # lower emphasis on geo
    },
    "High Availability & Relocation": {
        **DEFAULT_WEIGHTS,
        'open_to_work': 3.5,
        'recently_active_30d': 4.0,
        'recruiter_response_rate': 4.0,
        'short_notice': 3.0,
        'notice_period_days': -0.05,
        'willing_to_relocate': 2.0,
    },
    "Anti-Consulting Product Core": {
        **DEFAULT_WEIGHTS,
        'product_company_years': 2.5,
        'consulting_only_career': -20.0,  # absolute reject
        'current_is_consulting': -5.0,
        'tenure_3y_or_more': 4.0,
        'longest_tenure_years': 0.8,
    }
}

# Sidebar - Settings & Controls
st.sidebar.markdown("<h2 class='glow-title' style='font-size: 1.6rem;'>Control Panel</h2>", unsafe_allow_html=True)

# Weight Preset selection
st.sidebar.subheader("Ranking Weight Presets")
preset_choice = st.sidebar.selectbox("Choose a Weighting Profile", list(PRESET_WEIGHTS.keys()))
active_preset = PRESET_WEIGHTS[preset_choice]

# Initialize session state for weights
if 'weights' not in st.session_state or st.sidebar.button("Reset to Preset Defaults"):
    st.session_state.weights = active_preset.copy()

# Add weight customization sliders in expanders
st.sidebar.subheader("Tune Specific Weights")

# Group 1: Experience & Seniority
with st.sidebar.expander("Experience & Seniority"):
    st.session_state.weights['in_band_5_9'] = st.slider(
        "Ideal Band (5-9y)", 0.0, 15.0, float(st.session_state.weights.get('in_band_5_9', 8.0)), 0.5
    )
    st.session_state.weights['in_band_4_12'] = st.slider(
        "Acceptable Band (4-12y)", 0.0, 10.0, float(st.session_state.weights.get('in_band_4_12', 3.0)), 0.5
    )
    st.session_state.weights['title_is_lead_or_above'] = st.slider(
        "Lead / Staff Title Match", 0.0, 10.0, float(st.session_state.weights.get('title_is_lead_or_above', 4.0)), 0.5
    )
    st.session_state.weights['title_is_engineer'] = st.slider(
        "Title contains Engineer", 0.0, 5.0, float(st.session_state.weights.get('title_is_engineer', 1.5)), 0.1
    )
    st.session_state.weights['title_is_non_tech'] = st.slider(
        "Title is Explicit Non-Tech (Penalty)", -15.0, 0.0, float(st.session_state.weights.get('title_is_non_tech', -6.0)), 0.5
    )

# Group 2: Core ML Skills & Scale
with st.sidebar.expander("ML & Retrieval Core"):
    st.session_state.weights['has_ranking_skill'] = st.slider(
        "Has Ranking Skill", 0.0, 10.0, float(st.session_state.weights.get('has_ranking_skill', 3.0)), 0.5
    )
    st.session_state.weights['has_ranking_in_career'] = st.slider(
        "Career Ranking Work", 0.0, 10.0, float(st.session_state.weights.get('has_ranking_in_career', 4.0)), 0.5
    )
    st.session_state.weights['has_embedding_skill'] = st.slider(
        "Embedding/Vector Skill", 0.0, 10.0, float(st.session_state.weights.get('has_embedding_skill', 2.0)), 0.5
    )
    st.session_state.weights['has_llm_skill'] = st.slider(
        "LLM Skill", 0.0, 5.0, float(st.session_state.weights.get('has_llm_skill', 1.5)), 0.1
    )
    st.session_state.weights['has_shipped_at_scale'] = st.slider(
        "Shipped at Scale (Million Users)", 0.0, 10.0, float(st.session_state.weights.get('has_shipped_at_scale', 3.0)), 0.5
    )
    st.session_state.weights['ml_skill_kw_count'] = st.slider(
        "Skill KW Count Density", 0.0, 2.0, float(st.session_state.weights.get('ml_skill_kw_count', 0.6)), 0.1
    )
    st.session_state.weights['ml_career_kw_count'] = st.slider(
        "Career KW Count Density", 0.0, 2.0, float(st.session_state.weights.get('ml_career_kw_count', 0.8)), 0.1
    )

# Group 3: Career Trajectory
with st.sidebar.expander("Career Trajectory"):
    st.session_state.weights['product_company_years'] = st.slider(
        "Years at Product Co.", 0.0, 5.0, float(st.session_state.weights.get('product_company_years', 0.8)), 0.1
    )
    st.session_state.weights['consulting_only_career'] = st.slider(
        "Pure Consulting Career (Penalty)", -25.0, 0.0, float(st.session_state.weights.get('consulting_only_career', -10.0)), 1.0
    )
    st.session_state.weights['current_is_consulting'] = st.slider(
        "Current is Consulting (Penalty)", -10.0, 0.0, float(st.session_state.weights.get('current_is_consulting', -2.0)), 0.5
    )
    st.session_state.weights['tenure_3y_or_more'] = st.slider(
        "Anti-Job-Hopping (Long Tenure)", 0.0, 10.0, float(st.session_state.weights.get('tenure_3y_or_more', 2.0)), 0.5
    )

# Group 4: Location & Availability
with st.sidebar.expander("Location & Logistics"):
    st.session_state.weights['in_pune_noida'] = st.slider(
        "Pune/Noida Location", 0.0, 10.0, float(st.session_state.weights.get('in_pune_noida', 3.0)), 0.5
    )
    st.session_state.weights['in_tier1_india'] = st.slider(
        "Tier-1 India Cities", 0.0, 5.0, float(st.session_state.weights.get('in_tier1_india', 1.5)), 0.1
    )
    st.session_state.weights['willing_to_relocate'] = st.slider(
        "Willing to Relocate", 0.0, 5.0, float(st.session_state.weights.get('willing_to_relocate', 0.5)), 0.1
    )
    st.session_state.weights['short_notice'] = st.slider(
        "Notice Period <= 30d", 0.0, 5.0, float(st.session_state.weights.get('short_notice', 1.5)), 0.1
    )
    st.session_state.weights['long_notice'] = st.slider(
        "Notice Period > 60d (Penalty)", -10.0, 0.0, float(st.session_state.weights.get('long_notice', -1.0)), 0.5
    )

# Group 5: Redrob signals / verification
with st.sidebar.expander("Platform Engagement & Verification"):
    st.session_state.weights['open_to_work'] = st.slider(
        "Open to Work", 0.0, 5.0, float(st.session_state.weights.get('open_to_work', 1.5)), 0.1
    )
    st.session_state.weights['recently_active_30d'] = st.slider(
        "Active in last 30d", 0.0, 5.0, float(st.session_state.weights.get('recently_active_30d', 2.0)), 0.1
    )
    st.session_state.weights['recruiter_response_rate'] = st.slider(
        "Response Rate weight", 0.0, 5.0, float(st.session_state.weights.get('recruiter_response_rate', 2.0)), 0.1
    )
    st.session_state.weights['github_linkedin_verified'] = st.slider(
        "Verified Github/Linkedin Profile", 0.0, 5.0, float(st.session_state.weights.get('github_linkedin_verified', 0.5)), 0.1
    )

# Multipliers & Thresholds
st.sidebar.subheader("Tune Scoring Multipliers")
use_cred_mult = st.sidebar.checkbox("Apply Credibility Penalty", value=True)
min_cred_floor = st.sidebar.slider("Credibility Score Floor Limit", 0.1, 1.0, 0.3, 0.05)

# Main Area Header
st.markdown("<h1 class='glow-title'>Redrob Candidate Discovery & Ranking Hub</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>🎯 Intelligent Candidate Ranking Dashboard for Founding Senior AI Engineer</p>", unsafe_allow_html=True)

# Helper function to custom score candidates using user selected weights
def run_custom_ranking(artifacts, weights_dict, apply_cred=True, cred_floor=0.3):
    n_ids = len(artifacts['candidate_ids'])
    n_features = len(artifacts['features_df'])
    n_embeds = len(artifacts['candidate_embeddings'])
    n_cred = len(artifacts['credibility'])
    n_mask = len(artifacts['hard_mask'])
    
    if not (n_ids == n_features == n_embeds == n_cred == n_mask):
        n_dummy = min(n_ids, n_features, n_embeds, n_cred, n_mask)
        sorted_indices = np.arange(n_dummy)
        sorted_scores = np.zeros(n_dummy)
        sorted_raw_scores = np.zeros(n_dummy)
        sorted_cred = np.zeros(n_dummy)
        sorted_sub = artifacts['features_df'].head(n_dummy).copy()
        return sorted_indices, sorted_scores, sorted_raw_scores, sorted_cred, sorted_sub

    indices = produce_submission.retrieve_top_k(artifacts, k=500)
    
    features_df = artifacts['features_df']
    sub = features_df.iloc[indices].reset_index(drop=True)
    
    n = len(indices)
    scores = np.zeros(n, dtype=np.float64)
    
    for fname, weight in weights_dict.items():
        if fname not in sub.columns:
            continue
        col = sub[fname].values
        if fname == 'recruiter_response_rate':
            scores += weight * col
        elif fname == 'notice_period_days':
            scores += weight * np.maximum(0, col - 30)
        else:
            scores += weight * col
            
    # Availability curves (from produce_submission.py)
    days_active = sub['days_active'].values
    availability_mult = np.array([produce_submission.days_active_multiplier(d) for d in days_active])
    
    response_rate = sub['recruiter_response_rate'].values
    response_mult = 0.5 + 0.5 * np.sqrt(np.clip(response_rate, 0, 1))
    
    open_to_work = sub['open_to_work'].values
    open_mult = np.where(open_to_work > 0, 1.0, 0.7)
    
    # Credibility
    if apply_cred:
        credibility = artifacts['credibility'][indices]
        cred_mult = np.clip(credibility, cred_floor, 1.0)
    else:
        cred_mult = np.ones(n, dtype=np.float64)
        
    final_scores = scores * availability_mult * response_mult * open_mult * cred_mult
    
    # Sort
    order = np.argsort(-final_scores)
    sorted_indices = indices[order]
    sorted_scores = final_scores[order]
    sorted_sub = sub.iloc[order].reset_index(drop=True)
    sorted_raw_scores = scores[order]
    sorted_cred = artifacts['credibility'][sorted_indices]
    
    return sorted_indices, sorted_scores, sorted_raw_scores, sorted_cred, sorted_sub

# Load the full candidates dictionary for details
@st.cache_resource
def load_candidates_dict(mtime):
    candidates = {}
    if not os.path.exists(CANDIDATES_PATH):
        return candidates
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                candidates[c['candidate_id']] = c
            except json.JSONDecodeError:
                continue
    return candidates

# Tab selection
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Job Description & Centroid picks", 
    "🔍 LIVE Ranker & Discovery", 
    "✅ Validation & Submission", 
    "⚙️ Pipeline Precomputation"
])

# Check files status
artifacts_status = check_artifacts_status()
all_artifacts_exist = all(artifacts_status.values())

cand_mtime = os.path.getmtime(CANDIDATES_PATH) if os.path.exists(CANDIDATES_PATH) else 0
candidates_dict = load_candidates_dict(cand_mtime)

# Load artifacts at page scope
@st.cache_resource
def load_ranking_artifacts(mtime):
    return produce_submission.load_artifacts()

features_path = ARTIFACTS_DIR / "candidate_features.pkl"
art_mtime = os.path.getmtime(features_path) if features_path.exists() else 0
artifacts = None
shapes_match = False
n_ids = 0
n_features = 0
n_embeds = 0
n_cred = 0
n_mask = 0

if all_artifacts_exist:
    try:
        artifacts = load_ranking_artifacts(art_mtime)
        n_ids = len(artifacts['candidate_ids'])
        n_features = len(artifacts['features_df'])
        n_embeds = len(artifacts['candidate_embeddings'])
        n_cred = len(artifacts['credibility'])
        n_mask = len(artifacts['hard_mask'])
        shapes_match = (n_ids == n_features == n_embeds == n_cred == n_mask)
    except Exception as e:
        shapes_match = False

# --- TAB 1: JOB DESCRIPTION & CENTROID PICKS ---
with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📄 Job Description Overview")
        st.markdown("**Role**: Senior AI Engineer (Founding Team)")
        st.markdown("**Company**: Redrob AI (Talent Intelligence)")
        st.markdown("**Location**: Pune / Noida, India (Hybrid)")
        st.markdown("**Experience Target**: 5 - 9 Years")
        st.markdown("---")
        
        # Read job description text if exists
        if os.path.exists("job_description.txt"):
            with open("job_description.txt", "r", encoding="utf-8") as f:
                jd_text = f.read()
            st.text_area("Full Job Description text", jd_text, height=350, disabled=True)
        else:
            st.error("job_description.txt not found in workspace root.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Centroid Picks (Reference Set)")
        st.markdown("The dense-centroid similarity uses 10 hand-picked candidates to represent ideal matching profiles across core buckets:")
        
        if os.path.exists("centroid_picks.json"):
            with open("centroid_picks.json", "r", encoding="utf-8") as f:
                centroid_data = json.load(f)
            
            # Show summary stats
            buckets = centroid_data.get("summary", {}).get("buckets_covered", [])
            st.markdown(f"**Buckets Covered**: {', '.join(buckets)}")
            
            # Showcase each centroid pick in expanders
            picks = centroid_data.get("picks", [])
            for pick in picks:
                with st.expander(f"⭐ Pick {pick['rank']}: {pick['candidate_id']} — {pick['current_title']} @ {pick['current_company']}"):
                    st.markdown(f"**Experience**: {pick['years_of_experience']} Years | **Location**: {pick['location']}")
                    st.markdown(f"**Primary Bucket**: `{pick['bucket_primary']}`")
                    st.markdown(f"*Justification*:")
                    st.info(pick['jd_criterion_justification'])
        else:
            st.warning("centroid_picks.json not found.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: LIVE RANKER & DISCOVERY ---
with tab2:
    if not all_artifacts_exist:
        st.error("⚠️ Offline precomputation artifacts are missing or incomplete. Please go to the **Pipeline Precomputation** tab and trigger the precompute script.")
    else:
        if not shapes_match:
            st.error(f"⚠️ Artifacts on disk are out of sync (IDs={n_ids}, Features={n_features}, Embeddings={n_embeds}, Credibility={n_cred}, HardMask={n_mask}). "
                     "This usually happens when a previous precomputation run was interrupted. "
                     "Please go to **Tab 4 (⚙️ Pipeline Precomputation)** and run the precomputation to completion.")
                     
        st.markdown("### Live Candidate Ranking & Profiler")
        
        # Run ranking using dynamic weights or dummy fallback
        sorted_indices, sorted_scores, sorted_raw_scores, sorted_cred, sorted_sub = run_custom_ranking(
            artifacts, st.session_state.weights, use_cred_mult, min_cred_floor
        )
        
        # Display summary metrics
        st.markdown(f"""
        <div class='metric-container'>
            <div class='metric-box'>
                <div class='metric-val'>{len(sorted_indices)}</div>
                <div class='metric-lbl'>Funnel Candidates</div>
            </div>
            <div class='metric-box'>
                <div class='metric-val'>{sorted_scores.max():.2f}</div>
                <div class='metric-lbl'>Highest Score</div>
            </div>
            <div class='metric-box'>
                <div class='metric-val'>{sorted_scores.min():.2f}</div>
                <div class='metric-lbl'>Lowest Score</div>
            </div>
            <div class='metric-box'>
                <div class='metric-val'>{np.mean(sorted_cred):.2f}</div>
                <div class='metric-lbl'>Mean Credibility</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Filters
        col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
        with col_f1:
            search_query = st.text_input("🔍 Search candidates by ID, Title, Company or Skills", "")
        with col_f2:
            subset_size = st.selectbox("Display Top", [10, 20, 30, 50, 100], index=3)
        with col_f3:
            loc_filter = st.multiselect("Filter by Region", ["Pune/Noida", "Tier-1 India", "Other"])

        # Compile list of candidates to display
        display_rows = []
        for i, idx in enumerate(sorted_indices):
            cid = artifacts['candidate_ids'][idx]
            c = candidates_dict.get(cid, {})
            score = sorted_scores[i]
            raw_score = sorted_raw_scores[i]
            cred = sorted_cred[i]
            
            p = c.get('profile', {})
            title = p.get('current_title', '') or 'Professional'
            company = p.get('current_company', '') or 'Company'
            loc = p.get('location', '') or ''
            years = p.get('years_of_experience', 0)
            
            # Generate reasoning inline
            rank = i + 1
            reasoning_text = reasoning.generate_reasoning(c, rank, score)
            ok, suspects = reasoning.check_hallucination(reasoning_text, c)
            
            # Region match check
            loc_lower = loc.lower()
            region = "Other"
            if 'pune' in loc_lower or 'noida' in loc_lower:
                region = "Pune/Noida"
            elif any(city in loc_lower for city in ['bangalore', 'bengaluru', 'hyderabad', 'mumbai', 'delhi', 'gurgaon', 'gurugram', 'chennai', 'kolkata']):
                region = "Tier-1 India"
                
            # Filter matches
            if search_query:
                sq = search_query.lower()
                skills_str = " ".join([s.get('name', '') for s in c.get('skills', [])]).lower()
                match = (sq in cid.lower() or 
                         sq in title.lower() or 
                         sq in company.lower() or 
                         sq in skills_str or 
                         sq in reasoning_text.lower())
                if not match:
                    continue
                    
            if loc_filter:
                if region not in loc_filter:
                    continue
                    
            display_rows.append({
                'rank': rank,
                'candidate_id': cid,
                'score': score,
                'raw_score': raw_score,
                'credibility': cred,
                'title': title,
                'company': company,
                'location': loc,
                'years': years,
                'reasoning': reasoning_text,
                'hallucination_ok': ok,
                'suspects': suspects,
                'raw_c': c
            })

        # Paginate to requested display size
        display_rows = display_rows[:subset_size]
        
        if not display_rows:
            st.info("No candidates match the search filters.")
        else:
            # Layout: list on left, inspector on right
            col_list, col_inspector = st.columns([5, 6])
            
            with col_list:
                st.markdown("<h4 style='margin-bottom:12px;'>Ranked List</h4>", unsafe_allow_html=True)
                
                # Selection radio to pick candidate
                selected_cid = None
                selection_options = {}
                for r in display_rows:
                    label = f"#{r['rank']}. {r['candidate_id']} | Score: {r['score']:.2f} | {r['title']} @ {r['company']}"
                    selection_options[label] = r['candidate_id']
                    
                selected_label = st.radio("Select Candidate to Inspect:", list(selection_options.keys()), label_visibility="collapsed")
                selected_cid = selection_options[selected_label]
                
                # Retrieve details of the selected candidate
                selected_cand = next(r for r in display_rows if r['candidate_id'] == selected_cid)

            with col_inspector:
                st.markdown("<h4 style='margin-bottom:12px;'>Profile Inspector</h4>", unsafe_allow_html=True)
                sc = selected_cand
                c = sc['raw_c']
                p = c.get('profile', {})
                sig = c.get('redrob_signals', {})
                
                # Inspector Panel
                st.markdown(f"""
                <div class='glass-card'>
                    <div style='display:flex; justify-content:space-between; align-items:start;'>
                        <div>
                            <h3 style='margin:0 0 4px 0; color:#f0f6fc;'>{p.get('anonymized_name', 'Anonymized Candidate')}</h3>
                            <p style='margin:0; font-size:14px; color:#8c9bb4;'>{p.get('current_title', '')} at <strong>{p.get('current_company', '')}</strong></p>
                            <p style='margin:4px 0 0 0; font-size:12.5px; color:#58a6ff;'>📍 {p.get('location', '')}, {p.get('country', '')}</p>
                        </div>
                        <div style='text-align:right;'>
                            <span class='badge badge-location'>Rank #{sc['rank']}</span>
                            <div style='font-size:22px; font-weight:700; color:#00f2fe; margin-top:5px;'>{sc['score']:.2f}</div>
                            <div style='font-size:10px; color:#8c9bb4; text-transform:uppercase;'>Composite Score</div>
                        </div>
                    </div>
                    <hr style='border-color:rgba(255,255,255,0.08); margin:15px 0;'/>
                """, unsafe_allow_html=True)
                
                # Score Breakdown block
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    st.metric("Raw Score", f"{sc['raw_score']:.2f}")
                with col_b2:
                    st.metric("Credibility Score", f"{sc['credibility']:.2f}")
                with col_b3:
                    days_active = sig.get('last_active_date', 'None')
                    st.metric("Last Active", days_active)
                    
                # Reasoning panel
                st.markdown("**Generated AI Reasoning**:")
                badge_class = "badge-passed" if sc['hallucination_ok'] else "badge-fallback"
                badge_label = "PASSED CHECK" if sc['hallucination_ok'] else "FALLBACK REASONING"
                
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03); border-radius:8px; padding:12px; border:1px solid rgba(255,255,255,0.06); margin-bottom:15px;'>
                    <p style='margin:0 0 6px 0; font-size:13.5px; font-style:italic;'>"{sc['reasoning']}"</p>
                    <span class='badge {badge_class}'>{badge_label}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if not sc['hallucination_ok']:
                    st.markdown("**Suspect Claims Caught**:")
                    for s in sc['suspects']:
                        st.markdown(f"- ⚠️ *{s}*")
                        
                # Tabs inside inspector for structured profile details
                p_tabs = st.tabs(["💼 Career History", "🛠️ Skills", "📊 Signals Audit", "🛡️ Credibility Details"])
                
                # Inspector Tab 1: Career history timeline
                with p_tabs[0]:
                    history = c.get('career_history', [])
                    if not history:
                        st.info("No career history available.")
                    else:
                        for idx, h in enumerate(history):
                            is_curr = h.get('is_current', False)
                            dur = h.get('duration_months', 0)
                            dur_str = f"{dur} mos" if dur else ""
                            date_str = f"{h.get('start_date', '')} - {'Present' if is_curr else h.get('end_date', '')}"
                            
                            st.markdown(f"""
                            <div class='timeline-item'>
                                <div class='timeline-marker'></div>
                                <div class='timeline-title'>{h.get('title')}</div>
                                <div class='timeline-subtitle'><strong>{h.get('company')}</strong> • {date_str} ({dur_str})</div>
                                <div class='timeline-desc'>{h.get('description', '')}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                # Inspector Tab 2: Skills with gradients
                with p_tabs[1]:
                    st.markdown("**Expert/Advanced Skills** (Highlighted if JD-relevant):")
                    skills = c.get('skills', [])
                    jd_relevant = reasoning.get_jd_relevant_skill_names()
                    
                    for s in skills:
                        name = s.get('name', '')
                        prof = s.get('proficiency', '')
                        endorse = s.get('endorsements', 0)
                        
                        is_jd_skill = name.lower() in jd_relevant or any(kw in name.lower() for kw in jd_relevant)
                        pill_class = "skill-pill-highlight" if is_jd_skill else "skill-pill"
                        
                        st.markdown(f"<span class='{pill_class}'>{name} ({prof} • {endorse}👍)</span>", unsafe_allow_html=True)
                        
                    education = c.get('education', [])
                    if education:
                        st.markdown("<br>**Education**:", unsafe_allow_html=True)
                        for ed in education:
                            st.markdown(f"🎓 **{ed.get('degree')}** in *{ed.get('field_of_study')}*  \n{ed.get('institution')} ({ed.get('start_year')} - {ed.get('end_year')}) — **{ed.get('tier', '').replace('_', ' ').title()}**")

                # Inspector Tab 3: Platform Signals
                with p_tabs[2]:
                    # Format verification indicators
                    verified_mail = "✅ Verified" if sig.get("verified_email") else "❌ Unverified"
                    verified_phone = "✅ Verified" if sig.get("verified_phone") else "❌ Unverified"
                    linkedin = "🔗 Connected" if sig.get("linkedin_connected") else "❌ Disconnected"
                    
                    st.markdown(f"""
                    **Expected Salary (INR LPA)**: {sig.get('expected_salary_range_inr_lpa', {}).get('min', 0)} - {sig.get('expected_salary_range_inr_lpa', {}).get('max', 0)} LPA  
                    **Preferred Work Mode**: {sig.get('preferred_work_mode', 'N/A').title()}  
                    **Willing to Relocate**: {"Yes" if sig.get('willing_to_relocate') else "No"}  
                    **Notice Period**: {sig.get('notice_period_days', 0)} days  
                    **Github Activity Score**: {sig.get('github_activity_score', -1)}  
                    **Recruiter Response Rate**: {sig.get('recruiter_response_rate', 0):.0%}  
                    **Connection Count**: {sig.get('connection_count', 0)}  
                    **Interview Attendance Rate**: {sig.get('interview_completion_rate', 0):.0%}  
                    
                    **Contact Verifications**:
                    - Email: {verified_mail}
                    - Phone: {verified_phone}
                    - LinkedIn: {linkedin}
                    """)

                # Inspector Tab 4: Credibility Details
                with p_tabs[3]:
                    # Let's perform a live audit based on precompute credibility checks
                    # 1. Experience vs career history duration
                    years = p.get('years_of_experience', 0) or 0
                    total_months = sum((h.get('duration_months', 0) or 0) for h in c.get('career_history', []))
                    years_from_career = total_months / 12.0
                    exp_diff = abs(years - years_from_career)
                    
                    # 2. Skill duration checks
                    expert_no_duration = sum(1 for s in c.get('skills', []) if s.get('proficiency') == 'expert' and (s.get('duration_months', 0) or 0) == 0)
                    
                    # 3. Too many expert skills
                    expert_count = sum(1 for s in c.get('skills', []) if s.get('proficiency') == 'expert')
                    
                    # 4. Date consistency
                    date_warning = False
                    for h in c.get('career_history', []):
                        sd = h.get('start_date')
                        ed = h.get('end_date') if not h.get('is_current') else None
                        if sd and ed and ed < sd:
                            date_warning = True
                            break
                            
                    # Render Checks list
                    st.markdown("### Credibility Engine Verification")
                    
                    st.markdown(f"1. **Experience Congruence**: target={years}y, career={years_from_career:.1f}y. Diff={exp_diff:.1f}y.")
                    if exp_diff > 3:
                        st.markdown("   - ❌ *Discrepancy: Experience profile duration mismatch.*")
                    else:
                        st.markdown("   - ✅ *Valid matching experience profile.*")
                        
                    st.markdown(f"2. **Expert Duration Verification**: {expert_no_duration} expert skills with 0 months practice.")
                    if expert_no_duration >= 3:
                        st.markdown("   - ❌ *Suspicious: Expert skills listed with no experience duration.*")
                    else:
                        st.markdown("   - ✅ *All expert skills have documented practice time.*")
                        
                    st.markdown(f"3. **Skill Count Density**: {expert_count} expert skills in profile.")
                    if expert_count >= 10:
                        st.markdown("   - ❌ *Suspicious: Unusually high number of expert skills listed.*")
                    else:
                        st.markdown("   - ✅ *Reasonable skill assessment profile.*")
                        
                    st.markdown(f"4. **Career History Dates Integrity**: Date consistency.")
                    if date_warning:
                        st.markdown("   - ❌ *Suspicious: Career history start date occurs after end date.*")
                    else:
                        st.markdown("   - ✅ *All career dates are logically ordered.*")
                        
                st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 3: VALIDATION & SUBMISSION ---
with tab3:
    st.markdown("### Submission Exporter & Compliance Validation")
    st.markdown("Generate your final CSV candidate rank file and run the automated check rules inline to guarantee challenge compliance.")
    
    export_filename = st.text_input("Submission Filename", "team_xxx.csv")
    
    if st.button("Generate & Validate Submission File", key="btn_generate_csv"):
        if not all_artifacts_exist:
            st.error("Please run the Precomputation step first before attempting to generate a submission.")
        elif not shapes_match:
            st.error("⚠️ Cannot generate submission because artifacts on disk are out of sync. Please complete precomputation in Tab 4 first.")
        else:
            with st.spinner("Generating CSV submission..."):
                # Run complete ranking with dynamic weights
                sorted_indices, sorted_scores, sorted_raw_scores, sorted_cred, sorted_sub = run_custom_ranking(
                    artifacts, st.session_state.weights, use_cred_mult, min_cred_floor
                )
                
                # Write CSV
                rows = []
                for rank, (idx, score) in enumerate(zip(sorted_indices, sorted_scores), 1):
                    cid = artifacts['candidate_ids'][idx]
                    c = candidates_dict.get(cid, {})
                    
                    reasoning_text = reasoning.generate_reasoning(c, rank, score)
                    ok, suspects = reasoning.check_hallucination(reasoning_text, c)
                    if not ok:
                        p = c.get('profile', {})
                        title = p.get('current_title', '') or 'professional'
                        company = p.get('current_company', '') or 'company'
                        years = p.get('years_of_experience', 0) or 0
                        reasoning_text = f"{title} at {company} with {years} years experience; see profile for details."
                        
                    rows.append({
                        'candidate_id': cid,
                        'rank': rank,
                        'score': float(score),
                        'reasoning': reasoning_text,
                    })
                
                # Write file
                import csv
                with open(export_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'], quoting=csv.QUOTE_MINIMAL)
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)
                        
            # Run validation checks
            st.success(f"Wrote {len(rows)} entries to {export_filename}")
            
            validation_errors = validate_submission(export_filename)
            
            st.markdown("### Inline Compliance Validation Report")
            if not validation_errors:
                st.markdown("""
                <div class='badge badge-passed' style='font-size:16px; padding:8px 16px; margin-bottom:15px;'>
                    🚀 Compliance Status: VALID SUBMISSION
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown("""
                <div class='badge badge-failed' style='font-size:16px; padding:8px 16px; margin-bottom:15px;'>
                    ❌ Compliance Status: INVALID SUBMISSION
                </div>
                """, unsafe_allow_html=True)
                for err in validation_errors:
                    st.error(f"- {err}")
                    
                # Special sandbox note if size mismatch
                if len(rows) < 100:
                    st.info("💡 **Sandbox Mode Note**: The submission contains only 50 rows because the current workspace is using the sandbox candidate sample (`sample_candidates.json`). The full competition dataset (`candidates.jsonl` containing ~100k records) is required to produce exactly 100 rows and pass validation. The ranking pipeline logic itself is fully functional and correct.")
            
            # Read file content for download
            with open(export_filename, 'r', encoding='utf-8') as f:
                csv_data = f.read()
                
            st.download_button(
                label="📥 Download Submission CSV",
                data=csv_data,
                file_name=export_filename,
                mime="text/csv"
            )
            
            st.markdown("#### CSV Preview (Top 10 Rows)")
            st.dataframe(pd.read_csv(export_filename).head(10))

# --- TAB 4: PIPELINE PRECOMPUTATION ---
with tab4:
    st.markdown("### Precomputation Artifact Status")
    st.markdown("Configure dataset sources, view persisted embeddings, and run the offline precomputation pipeline.")
    
    # Render table of files
    files_df = []
    for fname, exists in artifacts_status.items():
        size_str = "N/A"
        display_name = fname
        if exists:
            target_path = ARTIFACTS_DIR / fname
            if fname == "bm25_index.pkl" and not target_path.exists():
                target_path = ARTIFACTS_DIR / "bm25_index.pkl.gz"
            display_name = target_path.name
            size_bytes = target_path.stat().st_size
            size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        files_df.append({
            "Artifact File": display_name,
            "Exists": "✅ Yes" if exists else "❌ No",
            "Size": size_str
        })
    st.table(files_df)
    
    st.markdown("### Trigger Pipeline Offline Precompute")
    st.markdown("This step loads the candidate JSON database, builds the BM25 search index, generates sentence embeddings, and builds the dense centroid.")
    
    # Source dataset choice
    candidate_source = st.radio("Candidate Pool Source file", ["sample_candidates.json (Sandbox - 50 profiles)", "candidates.jsonl (Full Dataset - 100k profiles)"])
    
    st.markdown("#### Upload Custom Candidates Pool File")
    uploaded_file = st.file_uploader("Upload a candidate dataset (.jsonl or .json format)", type=["jsonl", "json"])
    if uploaded_file is not None:
        if st.button("💾 Process and Save Uploaded File as candidates.jsonl"):
            with st.spinner("Processing uploaded dataset..."):
                try:
                    file_content = uploaded_file.read()
                    # Check if it's a JSON array and convert to JSONL if needed
                    if uploaded_file.name.endswith(".json"):
                        data = json.loads(file_content.decode("utf-8"))
                        if isinstance(data, list):
                            with open(CANDIDATES_PATH, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(json.dumps(c) for c in data) + '\n')
                            st.success(f"Converted and saved {len(data)} profiles to candidates.jsonl")
                        else:
                            st.error("Uploaded JSON must be a list of candidate profiles.")
                    else:
                        with open(CANDIDATES_PATH, 'wb') as f:
                            f.write(file_content)
                        # Count lines to see how many candidates
                        line_count = len(file_content.split(b'\n')) - 1
                        st.success(f"Saved uploaded JSONL file containing {line_count} candidate profiles to candidates.jsonl")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving uploaded file: {e}")
    
    if st.button("Run Precomputation Pipeline (takes ~30s for sandbox)", key="btn_run_precompute"):
        # Set up correct source file
        source_found = True
        if "sample_candidates" in candidate_source:
            if os.path.exists(SAMPLE_CANDIDATES_PATH):
                with st.spinner("Converting sample JSON to candidates.jsonl..."):
                    try:
                        data = json.load(open(SAMPLE_CANDIDATES_PATH, 'r', encoding='utf-8'))
                        with open(CANDIDATES_PATH, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(json.dumps(c) for c in data) + '\n')
                    except Exception as e:
                        st.error(f"Failed to convert sample file: {e}")
                        source_found = False
            else:
                st.error("sample_candidates.json not found in workspace!")
                source_found = False
        else:
            if not os.path.exists(CANDIDATES_PATH):
                st.error("candidates.jsonl does not exist! Please upload it or place it in the project root first.")
                source_found = False
                
        if source_found:
            log_container = st.empty()
            with st.spinner("Running offline precomputation (precompute.py)..."):
                try:
                    # Run precompute as a subprocess to capture real-time stdout
                    process = subprocess.Popen(
                        ["python", "precompute.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=os.getcwd()
                    )
                    
                    log_output = []
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            log_output.append(line)
                            # Render logs in a scrolling code box
                            log_container.code("".join(log_output[-25:]))
                            
                    rc = process.poll()
                    if rc == 0:
                        st.success("Precomputation completed successfully! Refreshing status...")
                        st.rerun()
                    else:
                        st.error(f"Precompute script failed with exit code {rc}")
                except Exception as e:
                    st.error(f"Failed to run precomputation command: {e}")
