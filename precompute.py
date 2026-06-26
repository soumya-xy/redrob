"""
Offline Precomputation Pipeline
================================

Loads candidates.jsonl, computes everything expensive (embeddings, BM25
index, feature scores, credibility scores), and persists to disk.

The 5-minute wall is on the RANKING step. This script can take hours
because it produces the artifacts the ranker loads.

Outputs (in artifacts/):
  - candidate_features.parquet   (per-candidate engineered features)
  - candidate_embeddings.npy     (sentence-transformer vectors, 100k x 384)
  - candidate_ids.json           (parallel array of candidate_ids)
  - bm25_index.pkl               (BM25 over profile text)
  - centroid.npy                 (mean of 10 reference centroid picks)
  - jd_embedding.npy             (embedding of the job description)
  - credibility_scores.npy       (per-candidate credibility score)
  - hard_filter_mask.npy         (boolean: True = passed hard filter)
  - feature_names.json           (column names for the features parquet)
"""

import json
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------- Config ----------

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

CANDIDATES_PATH = "candidates.jsonl"
JD_PATH = "job_description.txt"
CENTROID_PICKS_PATH = "centroid_picks.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH = 128
EMBED_DIM = 384

CONSULTING_FIRMS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'mindtree', 'tech mahindra', 'hcl',
    'persistent', 'mphasis', 'ltimindtree', 'genpact', 'hexaware',
    'larsen infotech',
}

ML_KEYWORDS = [
    'embedding', 'vector', 'retrieval', 'rag', 'retrieval-augmented',
    'ranker', 'ranking', 'recommend', 'recommender', 'recsys',
    'search', 'semantic search', 'hybrid search', 'learning to rank',
    'ltor', 'letor', 'ndcg', 'mrr', 'map@', 're-ranking', 'reranking',
    'ann ', 'approximate nearest', 'similarity search', 'cosine',
    'information retrieval', 'tf-idf', 'bm25', 'matching engine',
    'sentence-transformer', 'sentence transformer', 'transformer',
    'bert', 'roberta', 'bge', 'e5', 'sentence-bert', 'sbert',
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss',
    'opensearch', 'elasticsearch', 'vespa',
    'llm', 'fine-tun', 'lora', 'qlora', 'peft', 'rlhf',
    'xgboost', 'learning-to-rank', 'evaluat', 'offline metric',
    'a/b test', 'ab test', 'ranking system', 'recommendation system',
    'candidate scoring', 'match scoring', 'match engine',
    'nlp', 'natural language', 'text classification',
]

NON_TECH_TITLE_PATTERNS = [
    r'\b(sales|marketing|recruiter|hr |human resources|account|finance|'
    r'operations|operations manager|project manager|product manager|'
    r'business analyst|data analyst|business development|customer support|'
    r'customer success|content|writer|designer|mechanical|civil|electrical|'
    r'supply chain|procurement|admin|office manager|teacher|professor)\b',
]

JUNIOR_TITLE_PATTERNS = [
    r'\b(junior|jr\.?|intern|internship|trainee|apprentice)\b',
    r'\b(associate|assistant)\b(?!\s+(engineer|technical|software))',
]

SENIOR_TITLE_PATTERNS = [
    r'\b(senior|sr\.?|staff|principal|lead|architect|fellow|distinguished)\b',
]

GOOD_TITLE_KEYWORDS = [
    'engineer', 'developer', 'scientist', 'architect',
    'ml ', 'ai ', 'data scient', 'machine learning', 'deep learning',
    'nlp', 'applied scientist', 'research engineer', 'software',
]

INDIA_TIER_1_CITIES = {
    'pune', 'noida', 'hyderabad', 'mumbai', 'bangalore', 'bengaluru',
    'delhi', 'gurgaon', 'gurugram', 'chennai', 'kolkata',
}

TODAY = datetime(2026, 6, 15)


# ---------- Helpers ----------

def is_consulting_company(name):
    if not name:
        return False
    n = name.lower()
    return any(f in n for f in CONSULTING_FIRMS)


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, '%Y-%m-%d')
    except (ValueError, TypeError):
        return None


def days_since(d):
    if d is None:
        return None
    return (TODAY - d).days


def text_for_bm25(c):
    """Concatenate searchable text fields for BM25."""
    p = c.get('profile', {})
    parts = [
        p.get('headline', '') or '',
        p.get('summary', '') or '',
        p.get('current_title', '') or '',
        p.get('current_company', '') or '',
        p.get('current_industry', '') or '',
    ]
    for h in c.get('career_history', []):
        parts.append(h.get('title', '') or '')
        parts.append(h.get('company', '') or '')
        parts.append(h.get('description', '') or '')
    for s in c.get('skills', []):
        parts.append(s.get('name', '') or '')
    for e in c.get('education', []):
        parts.append(e.get('degree', '') or '')
        parts.append(e.get('field_of_study', '') or '')
        parts.append(e.get('institution', '') or '')
    for cert in c.get('certifications', []):
        parts.append(cert.get('name', '') or '')
    return ' '.join(parts).lower()


def text_for_embedding(c):
    """Concise text for the embedding model.

    Kept short on purpose — the model is small (MiniLM-L6, 22M params, max
    256 tokens) and longer text dilutes the signal. The text here represents
    'what this person does' in a single sentence: current title + current
    role description + a curated skill phrase.
    """
    p = c.get('profile', {})
    parts = [
        (p.get('current_title', '') or '').strip(),
    ]
    # Add the current role's description (truncated to 800 chars)
    current = next(
        (h for h in c.get('career_history', []) if h.get('is_current')),
        None,
    )
    if current is None and c.get('career_history'):
        current = c['career_history'][-1]
    if current:
        desc = (current.get('description', '') or '').strip()
        if len(desc) > 800:
            desc = desc[:800]
        if desc:
            parts.append(desc)
    # Add advanced/expert skills (limit to 8 most relevant)
    jd_skill_words = {
        'embedding', 'vector', 'pinecone', 'weaviate', 'qdrant', 'milvus',
        'faiss', 'sentence-transformer', 'sentence transformer', 'bge', 'e5',
        'semantic search', 'similarity search', 'ranking', 'ranker',
        'learning to rank', 'ndcg', 'mrr', 'recommender', 'recommendation',
        'recsys', 'llm', 'fine-tun', 'lora', 'qlora', 'peft', 'rag',
        'retrieval', 'elasticsearch', 'opensearch', 'xgboost', 'lightgbm',
        'information retrieval', 'natural language', 'nlp', 'transformer',
        'bert', 'mlops', 'mlflow', 'kubeflow',
    }
    skill_strs = []
    seen = set()
    for s in c.get('skills', []):
        if s.get('proficiency') in ('advanced', 'expert'):
            name = (s.get('name', '') or '').strip()
            nl = name.lower()
            if name and nl not in seen and nl in jd_skill_words:
                skill_strs.append(name)
                seen.add(nl)
            if len(skill_strs) >= 8:
                break
    if skill_strs:
        parts.append('Skills: ' + ', '.join(skill_strs))
    return '. '.join(parts).strip()


def extract_features(c):
    """Return a flat dict of engineered features for one candidate."""
    p = c.get('profile', {})
    sig = c.get('redrob_signals', {})
    career = c.get('career_history', [])
    skills = c.get('skills', [])
    education = c.get('education', [])

    # ----- experience -----
    years = p.get('years_of_experience', 0) or 0
    in_band_5_9 = float(5 <= years <= 9)
    in_band_4_12 = float(4 <= years <= 12)

    # ----- title analysis -----
    title = (p.get('current_title', '') or '').lower()
    is_junior_title = any(re.search(p_, title) for p_ in JUNIOR_TITLE_PATTERNS)
    is_senior_title = any(re.search(p_, title) for p_ in SENIOR_TITLE_PATTERNS)
    is_non_tech_title = any(re.search(p_, title) for p_ in NON_TECH_TITLE_PATTERNS)
    is_good_title = any(kw in title for kw in GOOD_TITLE_KEYWORDS)
    title_is_engineer = float('engineer' in title or 'developer' in title)
    title_is_scientist = float('scientist' in title)
    title_is_lead_or_above = float(
        is_senior_title and not is_junior_title and not is_non_tech_title
    )

    # ----- skills -----
    skill_names = [(s.get('name', '') or '').lower() for s in skills]
    skill_text = ' '.join(skill_names)
    ml_skill_kw_count = sum(1 for kw in ML_KEYWORDS if kw in skill_text)
    has_embedding_skill = float(any(
        kw in skill_text for kw in
        ['embedding', 'vector', 'pinecone', 'weaviate', 'qdrant', 'milvus',
         'faiss', 'sentence-transformer', 'sentence transformer', 'bge', 'e5']
    ))
    has_ranking_skill = float(any(
        kw in skill_text for kw in
        ['ranking', 'ranker', 'learning to rank', 'ltor', 'ndcg', 'mrr']
    ))
    has_llm_skill = float(any(
        kw in skill_text for kw in
        ['llm', 'fine-tun', 'lora', 'qlora', 'peft', 'rlhf', 'prompt']
    ))
    skill_count = len(skills)
    advanced_skill_count = sum(
        1 for s in skills if s.get('proficiency') in ('advanced', 'expert')
    )

    # ----- career -----
    career_text = ' '.join(
        ((h.get('description', '') or '') + ' ' + (h.get('title', '') or ''))
        for h in career
    ).lower()
    ml_career_kw_count = sum(1 for kw in ML_KEYWORDS if kw in career_text)
    has_ranking_in_career = float(any(
        kw in career_text for kw in
        ['ranking system', 'recommendation system', 'retrieval system',
         'search system', 'match engine', 'match scoring',
         'learning to rank', 'embedding-based search', 'hybrid retrieval',
         're-ranking', 'ranker']
    ))
    has_shipped_at_scale = float(any(
        kw in career_text for kw in
        ['million queries', 'm+ queries', 'm+ users', 'million users',
         'production', 'shipped', 'launched', 'deployed',
         'served ', 'serving ']
    ))
    product_company_months = 0
    consulting_months = 0
    for h in career:
        months = h.get('duration_months', 0) or 0
        if is_consulting_company(h.get('company', '')):
            consulting_months += months
        else:
            product_company_months += months
    product_company_years = product_company_months / 12.0
    consulting_only = float(
        career and all(is_consulting_company(h.get('company', '')) for h in career)
    )
    current_is_consulting = float(
        is_consulting_company(p.get('current_company', ''))
    )
    longest_tenure_months = max(
        (h.get('duration_months', 0) or 0 for h in career), default=0
    )
    longest_tenure_years = longest_tenure_months / 12.0
    tenure_in_years = float(longest_tenure_years >= 3)  # anti title-chaser

    # ----- education -----
    highest_education_tier = 'unknown'
    for e in education:
        t = e.get('tier', 'unknown')
        if t == 'tier_1':
            highest_education_tier = 'tier_1'
            break
        elif t == 'tier_2' and highest_education_tier not in ('tier_1',):
            highest_education_tier = 'tier_2'
        elif t == 'tier_3' and highest_education_tier not in ('tier_1', 'tier_2'):
            highest_education_tier = 'tier_3'
    education_tier_numeric = {
        'tier_1': 4, 'tier_2': 3, 'tier_3': 2, 'tier_4': 1, 'unknown': 0,
    }.get(highest_education_tier, 0)
    education_count = len(education)

    # ----- location -----
    loc = (p.get('location', '') or '').lower()
    country = (p.get('country', '') or '').lower()
    in_india = float('india' in country)
    in_pune_noida = float('pune' in loc or 'noida' in loc)
    in_tier1_india = float(
        in_india and any(city in loc for city in INDIA_TIER_1_CITIES)
    )
    in_bangalore = float('bangalore' in loc or 'bengaluru' in loc)
    in_hyderabad = float('hyderabad' in loc)
    in_mumbai = float('mumbai' in loc)
    in_delhi_ncr = float(
        any(city in loc for city in ['delhi', 'gurgaon', 'gurugram', 'noida'])
    )
    willing_to_relocate = float(bool(sig.get('willing_to_relocate')))

    # ----- signals -----
    open_to_work = float(bool(sig.get('open_to_work_flag')))
    last_active = parse_date(sig.get('last_active_date'))
    days_active = days_since(last_active) if last_active else 9999
    recently_active_30d = float(days_active <= 30)
    recently_active_90d = float(days_active <= 90)
    recruiter_response_rate = sig.get('recruiter_response_rate', 0) or 0
    response_rate_high = float(recruiter_response_rate > 0.6)
    response_rate_low = float(recruiter_response_rate < 0.2)
    avg_response_hours = sig.get('avg_response_time_hours', 9999) or 9999
    fast_responder = float(avg_response_hours <= 24)
    github = sig.get('github_activity_score', -1) or -1
    has_github = float(github >= 0)
    github_high = float(github >= 30)
    github_linkedin_verified = float(
        bool(sig.get('linkedin_connected'))
        and bool(sig.get('verified_email'))
        and bool(sig.get('verified_phone'))
    )
    notice_period = sig.get('notice_period_days', 180) or 180
    short_notice = float(notice_period <= 30)
    long_notice = float(notice_period > 60)
    interview_completion = sig.get('interview_completion_rate', 0) or 0
    high_interview_completion = float(interview_completion > 0.7)
    offer_acceptance = sig.get('offer_acceptance_rate', -1) or -1
    has_offer_history = float(offer_acceptance >= 0)
    high_offer_acceptance = float(offer_acceptance >= 0.5)
    search_appearance_30d = sig.get('search_appearance_30d', 0) or 0
    saved_by_recruiters_30d = sig.get('saved_by_recruiters_30d', 0) or 0
    high_recruiter_interest = float(
        saved_by_recruiters_30d >= 5 or search_appearance_30d >= 20
    )

    # ----- expected salary (in INR lakhs per annum) -----
    salary = sig.get('expected_salary_range_inr_lpa', {}) or {}
    salary_min = salary.get('min', 0) or 0
    salary_max = salary.get('max', 0) or 0
    salary_mid = (salary_min + salary_max) / 2 if salary_min or salary_max else 0
    # Redrob's role is senior, so 30-60 LPA is the expected band
    salary_in_band = float(20 <= salary_mid <= 70) if salary_mid else 0
    salary_too_low = float(0 < salary_mid < 15)
    salary_too_high = float(salary_mid > 90)

    # ----- work mode -----
    work_mode = (sig.get('preferred_work_mode', '') or '').lower()
    open_to_hybrid = float(work_mode in ('hybrid', 'flexible'))

    features = {
        'years_of_experience': years,
        'in_band_5_9': in_band_5_9,
        'in_band_4_12': in_band_4_12,
        'title_is_engineer': title_is_engineer,
        'title_is_scientist': title_is_scientist,
        'title_is_lead_or_above': title_is_lead_or_above,
        'title_is_non_tech': float(is_non_tech_title and not is_good_title),
        'ml_skill_kw_count': ml_skill_kw_count,
        'ml_career_kw_count': ml_career_kw_count,
        'has_embedding_skill': has_embedding_skill,
        'has_ranking_skill': has_ranking_skill,
        'has_llm_skill': has_llm_skill,
        'has_ranking_in_career': has_ranking_in_career,
        'has_shipped_at_scale': has_shipped_at_scale,
        'skill_count': skill_count,
        'advanced_skill_count': advanced_skill_count,
        'product_company_years': product_company_years,
        'consulting_only_career': consulting_only,
        'current_is_consulting': current_is_consulting,
        'longest_tenure_years': longest_tenure_years,
        'tenure_3y_or_more': tenure_in_years,
        'education_tier_numeric': education_tier_numeric,
        'education_count': education_count,
        'in_india': in_india,
        'in_pune_noida': in_pune_noida,
        'in_tier1_india': in_tier1_india,
        'in_bangalore': in_bangalore,
        'in_hyderabad': in_hyderabad,
        'in_mumbai': in_mumbai,
        'in_delhi_ncr': in_delhi_ncr,
        'willing_to_relocate': willing_to_relocate,
        'open_to_work': open_to_work,
        'days_active': min(days_active, 9999),
        'recently_active_30d': recently_active_30d,
        'recently_active_90d': recently_active_90d,
        'recruiter_response_rate': recruiter_response_rate,
        'response_rate_high': response_rate_high,
        'response_rate_low': response_rate_low,
        'fast_responder': fast_responder,
        'has_github': has_github,
        'github_high': github_high,
        'github_linkedin_verified': github_linkedin_verified,
        'notice_period_days': notice_period,
        'short_notice': short_notice,
        'long_notice': long_notice,
        'high_interview_completion': high_interview_completion,
        'has_offer_history': has_offer_history,
        'high_offer_acceptance': high_offer_acceptance,
        'high_recruiter_interest': high_recruiter_interest,
        'salary_in_band': salary_in_band,
        'salary_too_low': salary_too_low,
        'salary_too_high': salary_too_high,
        'open_to_hybrid': open_to_hybrid,
    }
    return features


def credibility_score(c):
    """Per-candidate credibility score. 0 = clearly honeypot, 1 = clean.

    Heuristics that catch the README's 'subtly impossible profiles':
      - duration_months vs start/end date inconsistency
      - skill proficiency = expert but duration_months = 0
      - too many skills at expert level
      - years_of_experience vs total career months mismatch
      - company is in career_history but current_company is different
    """
    score = 1.0
    p = c.get('profile', {})
    career = c.get('career_history', [])
    skills = c.get('skills', [])

    years = p.get('years_of_experience', 0) or 0
    total_months = sum((h.get('duration_months', 0) or 0) for h in career)
    years_from_career = total_months / 12.0

    # 1. Years of experience vs career duration mismatch
    if years_from_career > 0:
        diff = abs(years - years_from_career)
        if diff > 5:
            score -= 0.5
        elif diff > 3:
            score -= 0.2
        elif diff > 2:
            score -= 0.1

    # 2. Skill proficiency inconsistency
    expert_no_duration = sum(
        1 for s in skills
        if s.get('proficiency') == 'expert'
        and (s.get('duration_months', 0) or 0) == 0
    )
    if expert_no_duration >= 5:
        score -= 0.5
    elif expert_no_duration >= 3:
        score -= 0.3
    elif expert_no_duration >= 1:
        score -= 0.1

    # 3. Too many expert skills
    expert_count = sum(1 for s in skills if s.get('proficiency') == 'expert')
    if expert_count >= 15:
        score -= 0.3
    elif expert_count >= 10:
        score -= 0.15

    # 4. Date consistency
    for h in career:
        sd = parse_date(h.get('start_date'))
        ed = parse_date(h.get('end_date')) if not h.get('is_current') else None
        if sd and ed and ed < sd:
            score -= 0.3
            break
        if sd and (h.get('duration_months', 0) or 0) > 0:
            # Rough check: duration in months should be close to (end - start) in months
            if ed:
                actual_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
                if actual_months > 0:
                    ratio = h.get('duration_months', 0) / actual_months
                    if ratio < 0.5 or ratio > 1.5:
                        score -= 0.2
                        break

    # 5. Career-history overlap (overlapping full-time roles is suspicious)
    ranges = []
    for h in career:
        sd = parse_date(h.get('start_date'))
        ed = parse_date(h.get('end_date')) if not h.get('is_current') else TODAY
        if sd and ed:
            ranges.append((sd, ed))
    ranges.sort()
    overlap_months = 0
    for i in range(1, len(ranges)):
        prev_start, prev_end = ranges[i-1]
        cur_start, cur_end = ranges[i]
        if cur_start < prev_end:
            overlap = (min(prev_end, cur_end) - cur_start).days / 30.0
            if overlap > 0:
                overlap_months += overlap
    if overlap_months > 24:
        score -= 0.3
    elif overlap_months > 12:
        score -= 0.15

    # 6. Current company mismatch
    if career:
        current_role = next((h for h in career if h.get('is_current')), None)
        if current_role and p.get('current_company'):
            if current_role.get('company', '').lower() != p.get('current_company', '').lower():
                # Not necessarily wrong, but flag
                score -= 0.05

    return max(0.0, min(1.0, score))


def hard_filter_pass(c):
    """Boolean: does this candidate pass the hard pre-filter?

    Conservative — we let the ranker handle most of the scoring.
    This filter is just to drop clear honeypots and corrupt profiles.
    """
    p = c.get('profile', {})
    cred = credibility_score(c)

    # Drop clear honeypots by credibility
    if cred < 0.3:
        return False

    # Drop profiles with no usable text
    text = text_for_bm25(c)
    if len(text) < 50:
        return False

    # Drop profiles with no skills, no career history
    if not c.get('skills') and not c.get('career_history'):
        return False

    return True


# ---------- Main ----------

def main():
    print("Loading candidates ...")
    candidates = []
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(candidates)} candidates")

    candidate_ids = [c['candidate_id'] for c in candidates]

    # ----- 1. Features -----
    print("Extracting features ...")
    feature_rows = [extract_features(c) for c in candidates]
    feature_names = list(feature_rows[0].keys())
    features_df = pd.DataFrame(feature_rows, columns=feature_names)
    features_df.insert(0, 'candidate_id', candidate_ids)
    # Use pickle instead of parquet to avoid pyarrow dependency
    features_df.to_pickle(ARTIFACTS_DIR / "candidate_features.pkl")
    print(f"  features shape: {features_df.shape}")
    with open(ARTIFACTS_DIR / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)

    # ----- 2. Credibility -----
    print("Computing credibility scores ...")
    cred_scores = np.array([credibility_score(c) for c in candidates],
                           dtype=np.float32)
    np.save(ARTIFACTS_DIR / "credibility_scores.npy", cred_scores)
    print(f"  credibility: min={cred_scores.min():.2f}, "
          f"mean={cred_scores.mean():.2f}, max={cred_scores.max():.2f}")

    # ----- 3. Hard filter mask -----
    print("Computing hard-filter mask ...")
    hard_mask = np.array([hard_filter_pass(c) for c in candidates],
                         dtype=bool)
    np.save(ARTIFACTS_DIR / "hard_filter_mask.npy", hard_mask)
    print(f"  passed: {hard_mask.sum()} / {len(hard_mask)} "
          f"({100*hard_mask.mean():.1f}%)")

    # ----- 4. BM25 index -----
    print("Building BM25 index ...")
    bm25_corpus = [text_for_bm25(c).split() for c in candidates]
    bm25 = BM25Okapi(bm25_corpus)
    with open(ARTIFACTS_DIR / "bm25_index.pkl", 'wb') as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  BM25 corpus: {len(bm25_corpus)} docs")

    # ----- 5. Sentence-transformer embeddings -----
    print(f"Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model dim: {model.get_sentence_embedding_dimension()}")

    print("Encoding candidates ...")
    candidate_texts = [text_for_embedding(c) for c in candidates]
    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  Embeddings shape: {candidate_embeddings.shape}")
    np.save(ARTIFACTS_DIR / "candidate_embeddings.npy", candidate_embeddings)

    # ----- 6. JD embedding -----
    print("Encoding JD ...")
    with open(JD_PATH, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    # Use the headline + summary + skills inventory + ideal-candidate paragraph
    jd_focus = jd_text  # full JD is fine for the embedding model
    jd_embedding = model.encode(
        [jd_focus],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    np.save(ARTIFACTS_DIR / "jd_embedding.npy", jd_embedding)

    # ----- 7. Centroid -----
    print("Building centroid from 10 reference picks ...")
    with open(CENTROID_PICKS_PATH, 'r', encoding='utf-8') as f:
        picks = json.load(f)
    pick_ids = [p['candidate_id'] for p in picks['picks']]
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}
    pick_indices = [id_to_idx[pid] for pid in pick_ids if pid in id_to_idx]
    print(f"  Found {len(pick_indices)}/{len(pick_ids)} picks in candidate pool")
    centroid = candidate_embeddings[pick_indices].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)  # re-normalize
    np.save(ARTIFACTS_DIR / "centroid.npy", centroid)
    print(f"  Centroid shape: {centroid.shape}")

    # ----- 8. Save candidate_ids array (for indexing) -----
    with open(ARTIFACTS_DIR / "candidate_ids.json", 'w') as f:
        json.dump(candidate_ids, f)

    # ----- 9. Sanity print -----
    print("\n=== Precomputation complete ===")
    print(f"Artifacts in: {ARTIFACTS_DIR.resolve()}")
    for f in sorted(ARTIFACTS_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:40s} {size_mb:8.2f} MB")


if __name__ == '__main__':
    main()
