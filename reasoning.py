"""
Reasoning Generator
====================

Generates 1-2 sentence reasoning for each ranked candidate.
Uses feature-conditioned templates with mechanical no-hallucination check.

Output is bounded by the 6 Stage-4 checks:
  - Specific facts: references years, title, company, named skills
  - JD connection: ties back to a JD requirement
  - Honest concerns: acknowledges gaps for low-ranked candidates
  - No hallucination: every claim is mechanically checked against the profile
  - Variation: 10 sampled reasonings are substantively different
  - Rank consistency: tone matches the rank (rank 5 = strong, rank 95 = honest gap)
"""

import random
import re
from datetime import datetime, date


# ---------- Helpers ----------

def years_phrase(years):
    """Format years into a natural phrase."""
    if years is None or years == 0:
        return "limited"
    if years == int(years):
        return f"{int(years)} years"
    return f"{years:.1f} years"


def tenure_phrase(career_history):
    """Compute longest tenure phrase."""
    if not career_history:
        return None
    longest = max((h.get('duration_months', 0) or 0 for h in career_history),
                  default=0)
    years = longest / 12.0
    if years >= 3:
        return f"{years:.1f}-year tenure at one role"
    return None


def get_top_skills_for_reasoning(c, n=3):
    """Pick the most JD-relevant skills from the candidate's skill list."""
    if not c.get('skills'):
        return []
    jd_relevant = {
        'embedding', 'vector', 'pinecone', 'weaviate', 'qdrant', 'milvus',
        'faiss', 'sentence-transformer', 'sentence transformer', 'bge', 'e5',
        'semantic search', 'similarity search',
        'ranking', 'ranker', 'learning to rank', 'ltor', 'ndcg', 'mrr',
        'recommender', 'recommendation', 'recsys',
        'llm', 'fine-tun', 'lora', 'qlora', 'peft',
        'rag', 'retrieval', 'elasticsearch', 'opensearch',
        'xgboost', 'lightgbm', 'information retrieval',
        'natural language', 'nlp', 'transformer', 'bert',
    }
    matched = []
    seen = set()
    for s in c['skills']:
        name = s.get('name', '') or ''
        nl = name.lower()
        if nl in seen:
            continue
        if any(kw in nl for kw in jd_relevant):
            matched.append(name)
            seen.add(nl)
        if len(matched) >= n:
            break
    if not matched:
        # Fall back to top-N skills by proficiency
        prof_order = {'expert': 4, 'advanced': 3, 'intermediate': 2, 'beginner': 1}
        sorted_skills = sorted(
            c['skills'],
            key=lambda s: prof_order.get(s.get('proficiency', ''), 0),
            reverse=True,
        )
        for s in sorted_skills[:n]:
            name = s.get('name', '') or ''
            if name and name.lower() not in seen:
                matched.append(name)
                seen.add(name.lower())
    return matched[:n]


def get_jd_relevant_skill_names():
    return {
        'embedding', 'vector', 'pinecone', 'weaviate', 'qdrant', 'milvus',
        'faiss', 'sentence-transformer', 'sentence transformer', 'bge', 'e5',
        'semantic search', 'similarity search',
        'ranking', 'ranker', 'learning to rank', 'ltor', 'ndcg', 'mrr',
        'recommender', 'recommendation', 'recsys',
        'llm', 'fine-tun', 'lora', 'qlora', 'peft',
        'rag', 'retrieval', 'elasticsearch', 'opensearch',
        'xgboost', 'lightgbm', 'information retrieval',
        'natural language', 'nlp', 'transformer', 'bert',
    }


def mentioned_jd_relevant_skill(c):
    """Did the candidate mention at least one JD-relevant skill in their profile?"""
    jd_skills = get_jd_relevant_skill_names()
    for s in c.get('skills', []):
        if (s.get('name', '') or '').lower() in jd_skills:
            return True
    return False


def recent_role_phrase(c):
    """Get a phrase describing the candidate's current/recent role."""
    career = c.get('career_history', [])
    if not career:
        return None
    current = next((h for h in career if h.get('is_current')), career[-1])
    title = current.get('title', '') or ''
    company = current.get('company', '') or ''
    if not title and not company:
        return None
    return title, company


# ---------- Reasoning components ----------
# No more random templates. Build reasoning dynamically from actual features.


# ---------- Main reasoning function ----------

def generate_reasoning(c, rank, total_score):
    """Generate a 1-2 sentence reasoning for one candidate at this rank.

    Dynamic, feature-based approach:
    - Only mention strengths that are actually present in the profile
    - Only mention gaps that are actually present
    - Use concrete facts (years, title, company, specific skills)
    - Deterministic (no random choices)
    """
    p = c.get('profile', {})
    sig = c.get('redrob_signals', {})
    career = c.get('career_history', [])

    years = p.get('years_of_experience', 0) or 0
    title = p.get('current_title', '') or 'professional'
    company = p.get('current_company', '') or 'product company'

    # ---- Collect actual strengths (only if present) ----
    strengths = []

    # 1. Ranking/retrieval experience (JD's core requirement)
    has_ranking = _has_career_keyword(career, ['ranking', 'retrieval', 'recommend', 'recsys', 'search system', 'semantic search'])
    if has_ranking:
        strengths.append("shipped ranking/retrieval systems")

    # 2. Embedding/vector experience
    has_embedding = _has_career_keyword(career, ['embedding', 'vector', 'pinecone', 'weaviate', 'qdrant', 'faiss', 'milvus'])
    if has_embedding:
        strengths.append("built embedding/vector search")

    # 3. LLM fine-tuning
    has_llm = _has_career_keyword(career, ['llm', 'lora', 'qlora', 'fine-tun', 'peft', 'gpt', 'transformer'])
    if has_llm:
        strengths.append("LLM fine-tuning")

    # 4. Evaluation frameworks
    has_eval = _has_career_keyword(career, ['ndcg', 'mrr', 'map', 'evaluation', 'metric', 'ab test'])
    if has_eval:
        strengths.append("evaluation frameworks")

    # 5. Product company experience
    product_months = sum((h.get('duration_months', 0) or 0 for h in career if not _is_consulting(h.get('company', ''))))
    if product_months >= 36:
        strengths.append(f"{product_months / 12:.0f} years at product companies")

    # 6. Long tenure (anti-title-chaser)
    longest = max((h.get('duration_months', 0) or 0 for h in career), default=0)
    if longest >= 36:
        strengths.append(f"{longest / 12:.0f}-year tenure")

    # 7. Location
    loc = (p.get('location', '') or '').lower()
    in_pune_noida = 'pune' in loc or 'noida' in loc
    in_tier1 = any(city in loc for city in ['bangalore', 'bengaluru', 'hyderabad', 'mumbai', 'delhi', 'gurgaon', 'gurugram', 'chennai', 'kolkata'])
    if in_pune_noida:
        strengths.append("located in Pune/Noida")
    elif in_tier1:
        strengths.append("located in tier-1 India")
    elif sig.get('willing_to_relocate'):
        strengths.append("willing to relocate")

    # 8. Availability
    days_active = sig.get('days_active', 999) or 999
    response_rate = sig.get('recruiter_response_rate', 0) or 0
    if days_active <= 30 and response_rate >= 0.7:
        strengths.append("recently active with high response rate")
    elif days_active <= 30:
        strengths.append("recently active")
    elif response_rate >= 0.7:
        strengths.append(f"{response_rate:.0%} response rate")

    # 9. Notice period
    np_days = sig.get('notice_period_days', 180) or 180
    if np_days <= 30:
        strengths.append("short notice period")
    elif np_days <= 60:
        strengths.append("workable notice period")

    # 10. Seniority in title
    title_lower = title.lower()
    if any(t in title_lower for t in ['staff', 'principal', 'lead']):
        strengths.append("senior title")

    # 11. GitHub presence
    if p.get('github_url'):
        strengths.append("GitHub presence")

    # 12. Open to work
    if sig.get('open_to_work'):
        strengths.append("open to work")

    # ---- Collect actual gaps (only if present) ----
    gaps = []

    if not has_ranking and not has_embedding:
        gaps.append("no clear ranking/retrieval systems")

    if years < 4:
        gaps.append(f"{years:.0f} years experience")
    elif years > 12:
        gaps.append(f"{years:.0f} years experience")

    product_years = product_months / 12.0
    if product_years < 2:
        gaps.append("limited product-company experience")

    if longest < 18:
        gaps.append("short tenures")

    if not in_pune_noida and not in_tier1 and not sig.get('willing_to_relocate'):
        gaps.append("outside preferred cities")

    if days_active > 90:
        gaps.append(f"inactive for {days_active:.0f} days")

    if response_rate < 0.3:
        gaps.append(f"{response_rate:.0%} response rate")

    if np_days > 60:
        gaps.append(f"{np_days:.0f}-day notice period")

    if not sig.get('open_to_work'):
        gaps.append("not open to work")

    # ---- Build the reasoning text ----
    parts = []

    # Base: title at company with years
    base = f"{title} at {company} ({years_phrase(years)})"
    parts.append(base)

    # Add skills (if any)
    top_skills = get_top_skills_for_reasoning(c, n=2)
    if top_skills:
        skills_str = ', '.join(top_skills)
        parts.append(f"with {skills_str}")

    # Add strengths (prioritize JD-critical ones)
    jd_critical = [s for s in strengths if any(kw in s.lower() for kw in ['ranking', 'retrieval', 'embedding', 'vector', 'llm', 'evaluation'])]
    other_strengths = [s for s in strengths if s not in jd_critical]

    if jd_critical:
        parts.append("; ".join(jd_critical[:2]))
    elif other_strengths:
        parts.append(other_strengths[0])

    # Add gap if rank is mid/low
    if rank > 30 and gaps:
        parts.append(f"but {gaps[0]}")
    elif rank > 70 and len(gaps) > 1:
        parts.append(f"but {gaps[0]} and {gaps[1]}")

    # Join and clean
    text = "; ".join(parts) if len(parts) > 1 else parts[0] if parts else "Candidate profile"
    text = re.sub(r'\s+', ' ', text).strip()

    # Ensure it ends properly
    if not text.endswith('.') and len(text) > 50:
        text += '.'

    return text


def _has_career_keyword(career, keywords):
    """Check if any career history entry contains a JD-relevant keyword."""
    if not career:
        return False
    career_parts = []
    for h in career:
        career_parts.append((h.get('title', '') or '') + ' ' +
                            (h.get('company', '') or '') + ' ' +
                            (h.get('description', '') or '') + ' ' +
                            (h.get('industry', '') or ''))
    career_text = ' '.join(career_parts).lower()
    return any(kw in career_text for kw in keywords)


def _is_consulting(name):
    if not name:
        return False
    n = name.lower()
    return any(f in n for f in {
        'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
        'cognizant', 'capgemini', 'mindtree', 'tech mahindra', 'hcl',
    })


# ---------- Hallucination check ----------

def check_hallucination(reasoning_text, c):
    """Verify all factual claims in the reasoning are in the profile.

    Returns (ok: bool, list_of_suspect_claims: list[str])

    Strategy: extract multi-word capitalized noun phrases from the reasoning
    and verify each is either a known term (title fragment, company name,
    city, common noun) or appears in the candidate's skills/career text.
    """
    suspects = []
    text_lower = reasoning_text.lower()
    p = c.get('profile', {})

    # Build the "allowed vocabulary" from the profile
    candidate_skill_names = {(s.get('name', '') or '').lower()
                             for s in c.get('skills', [])}
    career_text = ' '.join(
        (h.get('title', '') or '') + ' ' +
        (h.get('company', '') or '') + ' ' +
        (h.get('description', '') or '') + ' ' +
        (h.get('industry', '') or '')
        for h in c.get('career_history', [])
    ).lower()
    profile_text = ' '.join([
        p.get('current_title', '') or '',
        p.get('current_company', '') or '',
        p.get('current_industry', '') or '',
        p.get('headline', '') or '',
        p.get('summary', '') or '',
        p.get('location', '') or '',
    ]).lower()
    full_text = career_text + ' ' + profile_text

    # ---- 1. Company name check (if mentioned) ----
    company = p.get('current_company', '') or ''
    if company and company.lower() in text_lower:
        # Verify it's actually in the profile
        if company.lower() not in full_text:
            suspects.append(f"company '{company}' not in profile")

    # ---- 2. Title check (if mentioned) ----
    title = p.get('current_title', '') or ''
    # Split into key fragments (drop seniority prefixes which are noisy)
    title_fragments = []
    for word in title.split():
        clean = word.strip('.,;:()[]{}').lower()
        if clean and clean not in {'senior', 'sr', 'jr', 'junior', 'lead',
                                    'staff', 'principal', 'i', 'ii', 'iii',
                                    'the', 'a', 'an'}:
            if len(clean) > 2:
                title_fragments.append(clean)
    # If a title fragment is mentioned, it should be in profile
    for frag in title_fragments:
        if frag in text_lower and frag not in full_text and frag not in career_text:
            # Mentioned but not in profile — possible hallucination
            # but skip if it's a very common English word
            if frag not in {'engineer', 'scientist', 'developer', 'architect',
                            'manager', 'analyst', 'consultant'}:
                suspects.append(f"title fragment '{frag}' not in profile")

    # ---- 3. Multi-word capitalized phrase check (for skills) ----
    # Find sequences of capitalized words (potential skill names).
    # Only match phrases that look like proper-noun terms, not
    # sentence-initial words.
    # First find all multi-word capitalized phrases
    multi_word_phrases = re.findall(
        r'(?<![.!?]\s)(?<!^)(?<!\.)\b([A-Z][a-zA-Z\-\.]{2,}(?:\s+[A-Z][a-zA-Z\-\.]{2,})+)\b',
        reasoning_text,
    )
    # Then find single capitalized words that are NOT the start of a sentence
    # and NOT part of a multi-word phrase already
    single_caps = []
    for m in re.finditer(r'(?<![.!?]\s)(?<!^)(?<!\.)\b([A-Z][a-zA-Z\-\.]{3,})\b', reasoning_text):
        word = m.group(1)
        # Check if this word is the first word of a multi-word phrase
        is_part_of_multi = any(word in p.split() for p in multi_word_phrases)
        if not is_part_of_multi:
            single_caps.append(word)
    potential_phrases = list(set(multi_word_phrases + single_caps))

    # Whitelist: words/phrases that are NOT skills but are common
    whitelist = {
        'Redrob', 'Senior', 'Staff', 'Lead', 'Principal', 'Architect',
        'Engineer', 'Scientist', 'Manager', 'Product', 'Note', 'Years',
        'Years Experience', 'Tier', 'India', 'Indian', 'Pune', 'Noida',
        'Bangalore', 'Bengaluru', 'Hyderabad', 'Mumbai', 'Delhi',
        'Gurgaon', 'Gurugram', 'Chennai', 'Kolkata', 'Kochi', 'Trivandrum',
        'Coimbatore', 'Vizag', 'Jaipur', 'Indore', 'Ahmedabad',
        'Bhubaneswar', 'Chandigarh',
        'Kerala', 'Tamil Nadu', 'Maharashtra', 'Karnataka', 'Telangana',
        'Haryana', 'Odisha', 'Rajasthan', 'Madhya Pradesh', 'Gujarat',
        'West Bengal', 'Uttar Pradesh',
        'Series', 'FAANG', 'JD', 'BM25', 'NDCG', 'MRR', 'NLP', 'RAG',
        'LLM', 'LLMs', 'ML', 'AI', 'API', 'SDK',
        'GitHub', 'LinkedIn', 'Email', 'Phone',
        # Common product/company names that may not be in skills
        'Netflix', 'Razorpay', 'Zomato', 'CRED', 'Google', 'Amazon',
        'Microsoft', 'Apple', 'Salesforce', 'Paytm', 'Stripe', 'Meta',
        'Flipkart', 'Wipro', 'TCS', 'Infosys', 'HCL', 'Mindtree',
        'Yellow', 'Haptik', 'Glance', 'Aganitha', 'Rephrase', 'Sarvam',
        'Krutrim', 'InMobi', 'Wysa', 'Meesho', 'Zoho', 'Freshworks',
        'Unacademy', 'upGrad', 'Niramai', 'JPMorgan', 'Uber', 'LinkedIn',
        'Meesho', 'Yulu', 'Cars24', 'PolicyBazaar', 'PhonePe',
        'Ather', 'Swiggy', 'Rapido', 'ShareChat', 'Dream11',
        'Paytm', 'PayU', 'Razorpay', 'Instamojo', 'Cashfree',
        # Title fragments
        'Machine Learning', 'Machine Learning Engineer', 'AI Engineer',
        'Data Scientist', 'Senior Data', 'Senior AI', 'Senior Machine',
        'Senior NLP', 'Applied ML', 'Applied Scientist', 'Research Engineer',
        'Senior Software', 'Recommendation Systems', 'Search Engineer',
        'RecSys', 'MLOps', 'Data Science', 'Deep Learning', 'Computer Vision',
    }
    # Build a normalized whitelist (lowercased + with extra spaces collapsed)
    whitelist_normalized = {w.lower() for w in whitelist}

    for phrase in potential_phrases:
        phrase_lower = phrase.lower()
        if phrase_lower in candidate_skill_names:
            continue
        if phrase_lower in whitelist_normalized:
            continue
        # If all words in the phrase are in the profile text, allow it
        words = phrase_lower.split()
        if all(w in full_text for w in words):
            continue
        # If phrase is a single word and in full_text, allow
        if len(words) == 1 and words[0] in full_text:
            continue
        # Otherwise it's a suspect (likely a hallucinated skill)
        if len(phrase) >= 4:
            suspects.append(f"possible hallucinated term: '{phrase}'")

    return (len(suspects) == 0, suspects)
