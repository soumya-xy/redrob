"""
Reference Candidate Selection Script
=====================================

Scans candidates.jsonl and produces a shortlist of ~30 strong candidates
grouped by JD criterion, to help the user hand-pick 10 for the centroid.

Run: python select_reference_candidates.py --in candidates.jsonl --out shortlist.json

The output is grouped by JD criterion so picking 10 can be done as
"one from each bucket" — this enforces the diversity the centroid needs.
"""

import json
import re
import argparse
from datetime import datetime
from collections import defaultdict

# ---------- Constants ----------

CONSULTING_FIRMS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'mindtree', 'tech mahindra', 'hcl',
    'ibm consult', 'persistent', 'mphasis', 'ltimindtree',
    'genpact', 'hexaware', 'larsen infotech',
}

# ML/NLP/retrieval keywords — counted in career_history descriptions
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

# Bucket-defining keywords — used to group candidates by JD criterion
BUCKETS = {
    'embedding_search': [
        'embedding', 'vector', 'pinecone', 'weaviate', 'qdrant', 'milvus',
        'faiss', 'sentence-transformer', 'sentence transformer', 'bge', 'e5',
        'semantic search', 'similarity search', 'ann ',
    ],
    'ranking_evaluation': [
        'ranker', 'ranking system', 'learning to rank', 'ltor', 'letor',
        'ndcg', 'mrr', 'map@', 'evaluat', 'a/b test', 'ab test',
        're-ranking', 'reranking', 'xgb', 'xgboost',
    ],
    'recsys_matching': [
        'recommend', 'recommender', 'recsys', 'matching engine',
        'match scoring', 'candidate scoring', 'collaborative filter',
    ],
    'llm_finetune': [
        'llm', 'fine-tun', 'lora', 'qlora', 'peft', 'rlhf',
        'prompt engineer', 'instruction tun',
    ],
    'ml_infra_distributed': [
        'distributed', 'spark', 'ray', 'kubeflow', 'mlflow',
        'feature store', 'data pipeline', 'data pipelines', 'airflow',
        'inference optimization', 'model serving',
    ],
}

JUNIOR_TITLE_PATTERNS = [
    r'\b(junior|jr\.?|intern|internship|trainee|apprentice)\b',
    r'\b(associate|assistant)\b(?!\s+(engineer|technical|software))',
]

SENIOR_TITLE_PATTERNS = [
    r'\b(senior|sr\.?|staff|principal|lead|architect|fellow|distinguished)\b',
]

NON_TECH_TITLE_PATTERNS = [
    r'\b(sales|marketing|recruiter|hr |human resources|account|finance|'
    r'operations|operations manager|project manager|product manager|'
    r'business analyst|data analyst|business development|customer support|'
    r'customer success|content|writer|designer|mechanical|civil|electrical|'
    r'supply chain|procurement|admin|office manager|teacher|professor|'
    r'research scientist|research engineer|postdoc)\b',
]

GOOD_TITLE_KEYWORDS = [
    'engineer', 'developer', 'scientist', 'architect', 'ml ', 'ai ',
    'data scient', 'machine learning', 'deep learning', 'nlp',
    'applied scientist', 'research engineer', 'software',
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


def product_company_years(c):
    """Years spent in non-consulting roles (any industry other than services)."""
    months = 0
    for h in c.get('career_history', []):
        if is_consulting_company(h.get('company', '')):
            continue
        months += h.get('duration_months', 0) or 0
    return months / 12.0


def is_consulting_only_career(c):
    """True if every role is at a consulting firm."""
    history = c.get('career_history', [])
    if not history:
        return False
    return all(is_consulting_company(h.get('company', '')) for h in history)


def ml_keyword_count_in_career(c):
    """Count of ML keywords across career_history descriptions (not skills list)."""
    text = ' '.join(
        (h.get('description', '') or '') for h in c.get('career_history', [])
    ).lower()
    return sum(1 for kw in ML_KEYWORDS if kw in text)


def ml_keyword_count_in_skills(c):
    """Count of ML keywords in skills list (proxy for claimed expertise)."""
    text = ' '.join((s.get('name', '') or '') for s in c.get('skills', [])).lower()
    return sum(1 for kw in ML_KEYWORDS if kw in text)


def title_seniority(title):
    if not title:
        return 0
    t = title.lower()
    is_junior = any(re.search(p, t) for p in JUNIOR_TITLE_PATTERNS)
    is_senior = any(re.search(p, t) for p in SENIOR_TITLE_PATTERNS)
    is_non_tech = any(re.search(p, t) for p in NON_TECH_TITLE_PATTERNS)
    is_good = any(kw in t for kw in GOOD_TITLE_KEYWORDS)

    if is_non_tech and not is_good:
        return -10  # explicit non-tech, exclude
    if is_junior and not is_senior:
        return -5
    if is_good and is_senior:
        return 3
    if is_good:
        return 1
    return 0


def has_strong_career_ml(c):
    """Used to rescue non-standard titles — strong ML work in role descriptions."""
    return ml_keyword_count_in_career(c) >= 3


def experience_score(years):
    if 5 <= years <= 9:
        return 3
    if 4 <= years < 5 or 9 < years <= 12:
        return 2
    if 3 <= years < 4 or 12 < years <= 15:
        return 1
    return 0


def availability_score(c):
    sig = c.get('redrob_signals', {})
    score = 0
    if sig.get('open_to_work_flag'):
        score += 1
    last_active = sig.get('last_active_date', '')
    if last_active:
        try:
            d = datetime.strptime(last_active, '%Y-%m-%d')
            days_ago = (TODAY - d).days
            if days_ago < 30:
                score += 2
            elif days_ago < 90:
                score += 1
        except (ValueError, TypeError):
            pass
    rr = sig.get('recruiter_response_rate', 0) or 0
    if rr > 0.6:
        score += 1
    return score


def location_score(c):
    loc = (c['profile'].get('location', '') or '').lower()
    country = (c['profile'].get('country', '') or '').lower()
    sig = c.get('redrob_signals', {})

    in_india_tier_1 = 'india' in country or any(city in loc for city in INDIA_TIER_1_CITIES)
    in_pune_noida = 'pune' in loc or 'noida' in loc

    if in_pune_noida:
        return 3
    if in_india_tier_1:
        return 2
    # Outside India: only count if willing_to_relocate or already shows India ties
    if sig.get('willing_to_relocate') and ('india' in country or 'indian' in loc):
        return 1
    if sig.get('willing_to_relocate'):
        return 0.5
    return 0


def github_score(c):
    g = c.get('redrob_signals', {}).get('github_activity_score', -1)
    if g == -1:
        return 0
    if g >= 30:
        return 2
    if g >= 10:
        return 1
    return 0.5


def determine_buckets(c):
    """Returns the JD-criterion bucket(s) this candidate exemplifies."""
    text = (
        ' '.join((h.get('description', '') or '') for h in c.get('career_history', []))
        + ' '
        + ' '.join((s.get('name', '') or '') for s in c.get('skills', []))
    ).lower()
    hits = []
    for bucket, kws in BUCKETS.items():
        if any(kw in text for kw in kws):
            hits.append(bucket)
    return hits


def score_candidate(c):
    title = c['profile'].get('current_title', '')
    years = c['profile'].get('years_of_experience', 0)

    # ----- Hard exclusions -----
    if is_consulting_only_career(c):
        return None, 'consulting_only'
    sen = title_seniority(title)
    if sen == -10:
        return None, 'non_tech_title'
    if sen == -5:
        return None, 'junior_title'
    if sen == 0 and not has_strong_career_ml(c):
        return None, 'title_not_eng'
    if years < 3:
        return None, 'too_junior'
    if ml_keyword_count_in_career(c) < 1 and ml_keyword_count_in_skills(c) < 2:
        return None, 'no_ml_signal'
    if years > 30:
        return None, 'implausible_years'

    # ----- Composite score -----
    score = 0.0
    score += experience_score(years) * 4
    score += max(sen, 0) * 2
    score += min(ml_keyword_count_in_career(c), 12) * 1.5
    score += min(ml_keyword_count_in_skills(c), 8) * 0.5
    score += availability_score(c) * 2
    score += location_score(c) * 1.5
    score += github_score(c) * 1
    score += min(product_company_years(c), 12) * 0.5
    return score, None


def explain(c, score):
    """Build a one-line JD-criterion justification for the shortlist output."""
    p = c['profile']
    buckets = determine_buckets(c)
    return {
        'candidate_id': c['candidate_id'],
        'score': round(score, 2),
        'current_title': p.get('current_title'),
        'current_company': p.get('current_company'),
        'current_company_size': p.get('current_company_size'),
        'years_of_experience': p.get('years_of_experience'),
        'location': p.get('location'),
        'country': p.get('country'),
        'career_history': [
            {
                'title': h.get('title'),
                'company': h.get('company'),
                'duration_months': h.get('duration_months'),
                'is_current': h.get('is_current'),
                'description_excerpt': (h.get('description', '') or '')[:200],
            }
            for h in c.get('career_history', [])
        ],
        'top_skills': [s.get('name') for s in c.get('skills', [])[:8]],
        'signals': {
            'open_to_work': c.get('redrob_signals', {}).get('open_to_work_flag'),
            'last_active': c.get('redrob_signals', {}).get('last_active_date'),
            'response_rate': c.get('redrob_signals', {}).get('recruiter_response_rate'),
            'github_score': c.get('redrob_signals', {}).get('github_activity_score'),
            'willing_to_relocate': c.get('redrob_signals', {}).get('willing_to_relocate'),
        },
        'product_company_years': round(product_company_years(c), 1),
        'ml_career_keywords': ml_keyword_count_in_career(c),
        'ml_skill_keywords': ml_keyword_count_in_skills(c),
        'jd_buckets': buckets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='inp', default='candidates.jsonl',
                        help='Input candidates.jsonl path')
    parser.add_argument('--out', dest='out', default='shortlist.json',
                        help='Output shortlist JSON path')
    parser.add_argument('--top', dest='top', type=int, default=30,
                        help='How many candidates to surface')
    parser.add_argument('--min-bucket', dest='min_bucket', type=int, default=2,
                        help='Min candidates per bucket before falling back')
    args = parser.parse_args()

    print(f"Loading {args.inp} ...")
    candidates = []
    with open(args.inp, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(candidates)} candidates")

    print("Scoring ...")
    scored = []
    excluded = defaultdict(int)
    for c in candidates:
        s, reason = score_candidate(c)
        if s is None:
            excluded[reason] += 1
        else:
            scored.append((s, c))
    scored.sort(key=lambda x: -x[0])

    print(f"Passed hard filter: {len(scored)}")
    print(f"Excluded: {dict(excluded)}")

    # Group by JD bucket
    by_bucket = defaultdict(list)
    bucketless = []
    seen_ids = set()
    for s, c in scored:
        buckets = determine_buckets(c)
        explained = explain(c, s)
        if not buckets:
            bucketless.append(explained)
            continue
        for b in buckets:
            if explained['candidate_id'] not in {x['candidate_id'] for x in by_bucket[b]}:
                by_bucket[b].append(explained)

    # Sort each bucket by score
    for b in by_bucket:
        by_bucket[b].sort(key=lambda x: -x['score'])
    bucketless.sort(key=lambda x: -x['score'])

    # Pick top candidates per bucket to fill top-N
    picked = []
    picked_ids = set()
    # Round-robin across buckets
    bucket_names = sorted(by_bucket.keys(),
                          key=lambda b: -len(by_bucket[b]))
    idx = 0
    while len(picked) < args.top and any(idx < len(by_bucket[b]) for b in bucket_names):
        for b in bucket_names:
            if idx < len(by_bucket[b]):
                cand = by_bucket[b][idx]
                if cand['candidate_id'] not in picked_ids:
                    picked.append(cand)
                    picked_ids.add(cand['candidate_id'])
                if len(picked) >= args.top:
                    break
        idx += 1

    # Fill remainder from bucketless if needed
    for cand in bucketless:
        if len(picked) >= args.top:
            break
        if cand['candidate_id'] not in picked_ids:
            picked.append(cand)
            picked_ids.add(cand['candidate_id'])

    output = {
        'metadata': {
            'total_loaded': len(candidates),
            'passed_filter': len(scored),
            'excluded_by_reason': dict(excluded),
            'shortlist_size': len(picked),
            'rubric': {
                'title_seniority': 'Senior+ Engineer/Scientist preferred, junior/non-tech rejected',
                'ml_threshold': '>=1 ML keyword in career_history OR >=2 in skills list',
                'consulting_filter': 'Reject if every role is at a consulting firm',
                'experience_band': 'JD ideal: 5-9 years; 4-12 years preferred; 3+ minimum',
                'location': 'Pune/Noida > tier-1 Indian cities > global+relocate',
                'availability': 'open_to_work + recent activity + high response rate',
            },
        },
        'by_bucket': {b: by_bucket[b][:10] for b in by_bucket},
        'shortlist': picked,
        'instructions': (
            'Pick 10 candidates for the centroid using diversity: '
            'aim for one from each jd_bucket where possible. '
            'You should be able to defend each pick with a one-sentence '
            'JD-criterion justification.'
        ),
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {args.out}")
    print(f"Shortlist: {len(picked)} candidates")
    print(f"Buckets represented: {sorted(by_bucket.keys())}")
    for b in sorted(by_bucket.keys()):
        print(f"  {b}: {len(by_bucket[b])} candidates")


if __name__ == '__main__':
    main()
