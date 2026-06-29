"""
Online Ranking Step
====================

This is the script that runs within the 5-minute, 16GB, CPU-only wall.
It loads pre-computed artifacts from artifacts/ and produces the final
top-100 CSV.

Pipeline:
  1. Load all artifacts (BM25, embeddings, features, centroid, JD embedding)
  2. Build FAISS index over candidate embeddings
  3. Hybrid retrieval: BM25 + dense-JD + dense-centroid + structured filter
  4. Score top-500 candidates with weighted feature sum
  5. Apply availability multiplier (multiplicative, not additive)
  6. Apply credibility floor (honeypot safety net)
  7. Take top 100, generate reasoning, validate
  8. Write submission CSV

Usage:
  python produce_submission.py --out team_xxx.csv
"""

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import reasoning


# ---------- Constants ----------

ARTIFACTS_DIR = Path("artifacts")
SUBMISSION_TEAM_ID = "team_xxx"  # overridable via CLI

# Feature weights — derived from the JD.
# Each weight is a number, and the corresponding feature contributes
# (weight * normalized_value) to the candidate's score.
# Negative weights penalize bad signals.
FEATURE_WEIGHTS = {
    # Strong positive: years in the JD's ideal band
    'in_band_5_9': 8.0,
    'in_band_4_12': 3.0,

    # Seniority in title — JD wants "senior engineer"
    'title_is_lead_or_above': 4.0,
    'title_is_engineer': 1.5,
    'title_is_scientist': 1.5,
    'title_is_non_tech': -6.0,  # explicit non-tech is bad

    # ML signal strength — JD requires production retrieval/ranking/LLM
    'ml_skill_kw_count': 0.6,
    'ml_career_kw_count': 0.8,
    'has_embedding_skill': 2.0,
    'has_ranking_skill': 3.0,  # ranking is the JD's core
    'has_llm_skill': 1.5,
    'has_ranking_in_career': 4.0,  # career-confirmed ranking work
    'has_shipped_at_scale': 3.0,

    # Skills breadth
    'advanced_skill_count': 0.4,

    # Career trajectory — JD is explicit about product-company years
    'product_company_years': 0.8,
    'consulting_only_career': -10.0,  # hard reject
    'current_is_consulting': -2.0,  # mild penalty
    'tenure_3y_or_more': 2.0,  # anti title-chaser
    'longest_tenure_years': 0.3,

    # Education
    'education_tier_numeric': 0.5,

    # Location — JD prefers Pune/Noida, then tier-1 India
    'in_pune_noida': 3.0,
    'in_tier1_india': 1.5,
    'in_bangalore': 1.0,
    'in_hyderabad': 0.5,
    'in_mumbai': 0.5,
    'in_delhi_ncr': 1.0,
    'willing_to_relocate': 0.5,

    # Availability — JD says "down-weight appropriately"
    'open_to_work': 1.5,
    'recently_active_30d': 2.0,
    'recently_active_90d': 1.0,
    'recruiter_response_rate': 2.0,  # 0-1, so 0-2 contribution
    'response_rate_high': 0.5,
    'response_rate_low': -2.0,  # explicit penalty
    'fast_responder': 0.5,
    'high_interview_completion': 0.5,
    'has_offer_history': 0.2,
    'high_offer_acceptance': 0.5,
    'high_recruiter_interest': 0.5,

    # Notice period — JD says "love sub-30, can buy out 30, 30+ bar gets higher"
    'short_notice': 1.5,
    'long_notice': -1.0,
    'notice_period_days': -0.02,  # mild linear penalty for >0

    # GitHub / verification — JD's anti-framework-enthusiast signal
    'has_github': 0.5,
    'github_high': 1.0,
    'github_linkedin_verified': 0.5,

    # Salary band
    'salary_in_band': 0.5,
    'salary_too_low': 0.0,  # not necessarily bad
    'salary_too_high': -1.5,

    # Work mode
    'open_to_hybrid': 0.3,
}

# Days-active penalty curve (applied as multiplicative availability)
# 0-30 days: full credit
# 30-90 days: linear decay
# 90-180 days: heavy decay
# 180+ days: severe
def days_active_multiplier(days):
    if days is None or days < 0:
        return 0.6
    if days <= 30:
        return 1.0
    if days <= 90:
        return 1.0 - 0.3 * ((days - 30) / 60.0)  # 1.0 -> 0.7
    if days <= 180:
        return 0.7 - 0.4 * ((days - 90) / 90.0)  # 0.7 -> 0.3
    if days <= 365:
        return 0.3 - 0.15 * ((days - 180) / 185.0)  # 0.3 -> 0.15
    return 0.1  # essentially dead


# ---------- Load artifacts ----------

def load_artifacts():
    print(f"[{time.time():.1f}] Loading artifacts ...")
    t0 = time.time()

    # Candidate IDs
    with open(ARTIFACTS_DIR / "candidate_ids.json") as f:
        candidate_ids = json.load(f)
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}
    n = len(candidate_ids)

    # Features
    features_df = pd.read_pickle(ARTIFACTS_DIR / "candidate_features.pkl")
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    print(f"  features: {features_df.shape}, names: {len(feature_names)}")

    # Embeddings
    candidate_embeddings = np.load(ARTIFACTS_DIR / "candidate_embeddings.npy").astype(np.float32)
    jd_embedding = np.load(ARTIFACTS_DIR / "jd_embedding.npy").astype(np.float32)
    centroid = np.load(ARTIFACTS_DIR / "centroid.npy").astype(np.float32)
    print(f"  embeddings: {candidate_embeddings.shape}, "
          f"JD: {jd_embedding.shape}, centroid: {centroid.shape}")

    # Credibility + hard filter
    credibility = np.load(ARTIFACTS_DIR / "credibility_scores.npy")
    hard_mask = np.load(ARTIFACTS_DIR / "hard_filter_mask.npy")
    print(f"  credibility: shape={credibility.shape}, "
          f"hard filter pass: {hard_mask.sum()}/{n}")

    # BM25
    gz_path = ARTIFACTS_DIR / "bm25_index.pkl.gz"
    if gz_path.exists():
        import gzip
        with gzip.open(gz_path, 'rb') as f:
            bm25 = pickle.load(f)
    else:
        with open(ARTIFACTS_DIR / "bm25_index.pkl", 'rb') as f:
            bm25 = pickle.load(f)
    print(f"  BM25 corpus: {len(bm25.doc_freqs)} docs")

    # Centroid picks (for reasoning metadata)
    with open("centroid_picks.json") as f:
        centroid_picks = json.load(f)
    pick_ids = {p['candidate_id'] for p in centroid_picks['picks']}

    print(f"[{time.time():.1f}] Artifacts loaded in {time.time()-t0:.1f}s")
    return {
        'candidate_ids': candidate_ids,
        'id_to_idx': id_to_idx,
        'features_df': features_df,
        'feature_names': feature_names,
        'candidate_embeddings': candidate_embeddings,
        'jd_embedding': jd_embedding,
        'centroid': centroid,
        'credibility': credibility,
        'hard_mask': hard_mask,
        'bm25': bm25,
        'centroid_picks': centroid_picks,
        'pick_ids': pick_ids,
    }


# ---------- Hybrid retrieval ----------

def retrieve_top_k(artifacts, k=500):
    """Hybrid retrieval: BM25 + dense-JD + dense-centroid.

    Returns: list of candidate indices, sorted by combined retrieval score.
    """
    print(f"[{time.time():.1f}] Retrieving top-{k} ...")
    t0 = time.time()

    n = len(artifacts['candidate_ids'])
    k = min(k, n)
    scores = np.zeros(n, dtype=np.float32)

    # ---- 1. BM25 over the JD text ----
    print(f"[{time.time():.1f}]   BM25 ...")
    with open("job_description.txt") as f:
        jd_text = f.read()
    # Use the most JD-relevant sections
    tokens = re.findall(r'\b[a-z][a-z\-]{2,}\b', jd_text.lower())
    jd_keywords = {
        'embeddings', 'embedding', 'retrieval', 'ranking', 'llm', 'llms', 'fine-tuning', 'fine-tune',
        'lora', 'qlora', 'peft', 'vector', 'database', 'databases', 'pinecone', 'weaviate',
        'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'faiss', 'ndcg', 'mrr', 'map',
        'evaluation', 'python', 'nlp', 'search', 'matching', 'recommender', 'recommendation',
        'learning-to-rank', 're-ranking', 'benchmarks', 'transformer', 'transformers',
        'dense', 'sparse', 'hybrid', 'similarity', 'semantic', 'information', 'ir'
    }
    jd_query_tokens = list(set(w for w in tokens if w in jd_keywords))
    if not jd_query_tokens:
        stopwords = {'the', 'and', 'for', 'you', 'will', 'with', 'our', 'are', 'that', 'this', 'from', 'your', 'have'}
        jd_query_tokens = list(set(w for w in tokens if w not in stopwords))
    bm25_scores = artifacts['bm25'].get_scores(jd_query_tokens)
    # Normalize to [0, 1]
    if bm25_scores.max() > 0:
        bm25_scores_norm = bm25_scores / bm25_scores.max()
    else:
        bm25_scores_norm = np.zeros(n)
    scores += 0.3 * bm25_scores_norm
    print(f"[{time.time():.1f}]   BM25 done (max={bm25_scores.max():.2f})")

    # ---- 2. Dense similarity to JD embedding ----
    print(f"[{time.time():.1f}]   Dense-JD ...")
    jd_emb = artifacts['jd_embedding'][0]
    dense_jd = artifacts['candidate_embeddings'] @ jd_emb
    # Already normalized; dot product = cosine sim
    if dense_jd.max() > 0:
        dense_jd_norm = (dense_jd - dense_jd.min()) / (
            dense_jd.max() - dense_jd.min() + 1e-12
        )
    else:
        dense_jd_norm = np.zeros(n)
    scores += 0.3 * dense_jd_norm
    print(f"[{time.time():.1f}]   Dense-JD done (max={dense_jd.max():.4f})")

    # ---- 3. Dense similarity to centroid (the Tier 5 fix) ----
    print(f"[{time.time():.1f}]   Dense-centroid ...")
    centroid = artifacts['centroid']
    dense_cent = artifacts['candidate_embeddings'] @ centroid
    if dense_cent.max() > 0:
        dense_cent_norm = (dense_cent - dense_cent.min()) / (
            dense_cent.max() - dense_cent.min() + 1e-12
        )
    else:
        dense_cent_norm = np.zeros(n)
    scores += 0.3 * dense_cent_norm
    print(f"[{time.time():.1f}]   Dense-centroid done (max={dense_cent.max():.4f})")

    # ---- 4. Take top-K, but exclude hard-filter rejects ----
    hard_mask = artifacts['hard_mask']
    # Mask out failed candidates by setting score to -inf
    masked_scores = np.where(hard_mask, scores, -np.inf)
    # Boost candidates who passed the original selection rubric
    # (so the top 50 from the shortlist are guaranteed in the funnel)
    pick_ids = artifacts['pick_ids']
    id_to_idx = artifacts['id_to_idx']
    for pid in pick_ids:
        if pid in id_to_idx:
            masked_scores[id_to_idx[pid]] = np.inf  # force inclusion
    if k < n:
        top_k_indices = np.argpartition(-masked_scores, k)[:k]
        top_k_indices = top_k_indices[np.argsort(-masked_scores[top_k_indices])]
    else:
        top_k_indices = np.argsort(-masked_scores)[:k]
    print(f"[{time.time():.1f}]   Retrieved {k} in {time.time()-t0:.1f}s")
    return top_k_indices


# ---------- Scoring ----------

def score_candidates(artifacts, indices):
    """Compute composite score for each candidate in `indices`."""
    print(f"[{time.time():.1f}] Scoring {len(indices)} candidates ...")
    t0 = time.time()

    features_df = artifacts['features_df']
    feature_names = artifacts['feature_names']

    # Build feature matrix for the candidates
    sub = features_df.iloc[indices].reset_index(drop=True)

    # Compute linear score
    n = len(indices)
    scores = np.zeros(n, dtype=np.float64)

    for fname, weight in FEATURE_WEIGHTS.items():
        if fname not in sub.columns:
            continue
        col = sub[fname].values
        # Some features are 0-1 booleans, some are counts
        if fname == 'recruiter_response_rate':
            # already 0-1
            scores += weight * col
        elif fname == 'notice_period_days':
            # 0 at <=30, penalty above
            scores += weight * np.maximum(0, col - 30)
        else:
            scores += weight * col

    # ---- Availability multiplier (per JD: "down-weight appropriately") ----
    days_active = sub['days_active'].values
    availability_mult = np.array([days_active_multiplier(d) for d in days_active])

    # Also multiply by response rate
    response_rate = sub['recruiter_response_rate'].values
    # Smooth: 0% response rate = 0.5x, 50% = 0.85x, 100% = 1.0x
    response_mult = 0.5 + 0.5 * np.sqrt(np.clip(response_rate, 0, 1))

    # Also multiply by open_to_work
    open_to_work = sub['open_to_work'].values
    open_mult = np.where(open_to_work > 0, 1.0, 0.7)

    # ---- Credibility floor (honeypot safety net) ----
    credibility = artifacts['credibility'][indices]
    # Penalize low-credibility candidates multiplicatively
    cred_mult = np.clip(credibility, 0.3, 1.0)

    final_scores = scores * availability_mult * response_mult * open_mult * cred_mult

    print(f"[{time.time():.1f}]   Scoring done in {time.time()-t0:.1f}s")
    print(f"   score range: {final_scores.min():.2f} to {final_scores.max():.2f}")

    return final_scores, scores, sub


# ---------- Main pipeline ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=f'{SUBMISSION_TEAM_ID}.csv',
                        help='Output CSV path')
    parser.add_argument('--top', type=int, default=100,
                        help='How many candidates to return')
    args = parser.parse_args()

    overall_start = time.time()

    # Load candidates.jsonl for reasoning generation
    # (we need full profile data, not just features)
    print(f"[{time.time():.1f}] Loading candidates.jsonl for reasoning ...")
    candidates = {}
    with open("candidates.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                candidates[c['candidate_id']] = c
            except json.JSONDecodeError:
                continue
    print(f"   loaded {len(candidates)} candidates for reasoning")

    # Load artifacts
    artifacts = load_artifacts()

    # Hybrid retrieval
    candidate_indices = retrieve_top_k(artifacts, k=500)

    # Score
    final_scores, raw_scores, sub_df = score_candidates(artifacts, candidate_indices)

    # Sort by final score
    order = np.argsort(-final_scores)
    sorted_indices = candidate_indices[order]
    sorted_scores = final_scores[order]
    sorted_sub = sub_df.iloc[order].reset_index(drop=True)

    # Take top N
    top_n = min(args.top, len(sorted_indices))
    top_indices = sorted_indices[:top_n]
    top_scores = sorted_scores[:top_n]

    # Generate reasoning for each
    print(f"[{time.time():.1f}] Generating reasoning ...")
    rows = []
    hallucination_warnings = 0
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
        cid = artifacts['candidate_ids'][idx]
        c = candidates.get(cid)
        if c is None:
            print(f"  WARNING: candidate {cid} not in candidates.jsonl")
            continue
        reasoning_text = reasoning.generate_reasoning(c, rank, score)
        ok, suspects = reasoning.check_hallucination(reasoning_text, c)
        if not ok:
            hallucination_warnings += 1
            # Replace with a minimal safe reasoning
            p = c.get('profile', {})
            title = p.get('current_title', '') or 'professional'
            company = p.get('current_company', '') or 'company'
            years = p.get('years_of_experience', 0) or 0
            reasoning_text = (f"{title} at {company} with {years} years experience; "
                              f"see profile for details.")
        rows.append({
            'candidate_id': cid,
            'rank': rank,
            'score': float(score),
            'reasoning': reasoning_text,
        })

    # Sanity: scores must be non-increasing
    for i in range(1, len(rows)):
        if rows[i]['score'] > rows[i-1]['score']:
            rows[i]['score'] = rows[i-1]['score']

    # Write CSV
    print(f"[{time.time():.1f}] Writing {args.out} ...")
    import csv
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['candidate_id', 'rank', 'score', 'reasoning'],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[{time.time():.1f}] Wrote {len(rows)} rows to {args.out}")
    print(f"[{time.time():.1f}] Hallucination warnings: {hallucination_warnings}")
    print(f"[{time.time():.1f}] Total wall time: {time.time()-overall_start:.1f}s")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total candidates ranked: {len(candidate_indices)}")
    print(f"Top {len(rows)} written to {args.out}")
    print(f"Score range: {top_scores.min():.3f} to {top_scores.max():.3f}")
    if rows:
        print(f"\nTop 5 (by rank):")
        for r in rows[:5]:
            print(f"  {r['rank']}. {r['candidate_id']} "
                  f"(score={r['score']:.3f})")
            print(f"     {r['reasoning'][:150]}...")


if __name__ == '__main__':
    main()
