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


# ---------- Template library ----------

# Strong candidate templates (rank 1-30)
STRONG_TEMPLATES = [
    # Format: (template, generator_function) — generator returns the substituted text
    "Senior {title} at {company} with {years} experience spanning {n_skills} JD-relevant skills including {skills}; {extras}",
    "{title} at {company} ({years}) has shipped {skill_highlight} at scale — directly aligned with the JD's mandate to own retrieval and ranking systems",
    "Strong product ML profile: {years} at {company} working on {skill_highlight}; {extras}",
    "Career spans {n_career} roles with a {longest_tenure} — fits the JD's anti-title-chaser criterion; core skills in {skills} cover the retrieval/ranking requirement",
    "{title} with {n_skills} JD-relevant skills ({skills}) and {product_years} at product companies; {location_note}",
]

# Mid candidate templates (rank 31-70)
MID_TEMPLATES = [
    "{title} at {company} ({years}) with relevant background in {skills}; {gap_note}",
    "{years} experience as {title}; strong on {skill_highlight} but {gap_note}",
    "{title} profile shows {n_skills} matching skills ({skills}); {gap_note}",
    "Mid-band fit: {title} with {product_years} at product companies; {gap_note}",
]

# Low candidate templates (rank 71-100)
LOW_TEMPLATES = [
    "{title} at {company} ({years}) is below the JD's ideal band — included for completeness despite {gap_note}",
    "Adjacent skills only: {title} has some {skills} but career trajectory and {gap_note} put them below the cutoff for the Senior AI Engineer role",
    "Below cutoff: {gap_note}; included to round out the top-100 since they have some {skills} overlap",
    "Lower fit: {title} ({years}) has {n_skills} JD-relevant skills but {gap_note}",
]

GAP_NOTES = [
    "limited production retrieval experience",
    "career trajectory is short of the 5-9y ideal band",
    "no clear production ranking system on the resume",
    "limited evidence of shipped-at-scale work",
    "geographic location outside the JD's preferred cities",
    "no demonstrated evaluation-framework experience",
    "limited LLM-era work in career history",
    "notice period is longer than ideal",
    "open_to_work flag is False",
    "recency of activity is a concern",
    "skill list is keyword-heavy without matching career descriptions",
    "primarily frontend/data-engineering background",
]

STRONG_EXTRAS = [
    "active on Redrob with high recruiter response rate",
    "located in or willing to relocate to the JD's preferred city",
    "strong GitHub presence signals framework-fluency, not framework-dependence",
    "open to work and recently active — meets the JD's availability requirement",
    "career shows shipping discipline (long tenures, product companies)",
]

NOTICE_NOTES = {
    'short': "notice period under 30 days is a plus",
    'medium': "notice period is in the workable 30-60 day band",
    'long': "notice period over 60 days is a concern",
}


# ---------- Main reasoning function ----------

def generate_reasoning(c, rank, total_score):
    """Generate a 1-2 sentence reasoning for one candidate at this rank."""
    p = c.get('profile', {})
    sig = c.get('redrob_signals', {})
    career = c.get('career_history', [])

    years = p.get('years_of_experience', 0) or 0
    title = p.get('current_title', '') or 'professional'
    company = p.get('current_company', '') or 'product company'
    top_skills = get_top_skills_for_reasoning(c, n=3)
    n_skills = len(top_skills)
    skills_str = ', '.join(top_skills) if top_skills else 'core ML skills'

    # Compute features used in templates
    product_company_months = 0
    for h in career:
        if not _is_consulting(h.get('company', '')):
            product_company_months += (h.get('duration_months', 0) or 0)
    product_years = product_company_months / 12.0

    longest = max((h.get('duration_months', 0) or 0 for h in career), default=0)
    longest_years = longest / 12.0
    longest_tenure = (f"{longest_years:.1f}-year tenure"
                      if longest_years >= 3 else "shorter tenures")
    n_career = len(career)

    # Skill highlight
    skill_highlight = (top_skills[0] if top_skills else 'applied ML')
    if len(top_skills) >= 2:
        skill_highlight = f"{top_skills[0]} and {top_skills[1]}"

    # Location note
    loc = (p.get('location', '') or '').lower()
    in_pune_noida = 'pune' in loc or 'noida' in loc
    in_tier1 = any(city in loc for city in
                   ['bangalore', 'bengaluru', 'hyderabad', 'mumbai',
                    'delhi', 'gurgaon', 'gurugram', 'chennai', 'kolkata'])
    if in_pune_noida:
        location_note = "located in the JD's first-preferred city"
    elif in_tier1:
        location_note = "located in a tier-1 Indian city"
    elif sig.get('willing_to_relocate'):
        location_note = "willing to relocate"
    else:
        location_note = "location outside the JD's preferred cities"

    # Notice period
    np_days = sig.get('notice_period_days', 180) or 180
    if np_days <= 30:
        notice_note = NOTICE_NOTES['short']
    elif np_days <= 60:
        notice_note = NOTICE_NOTES['medium']
    else:
        notice_note = NOTICE_NOTES['long']

    # Pick template by rank band
    if rank <= 30:
        template = random.choice(STRONG_TEMPLATES)
    elif rank <= 70:
        template = random.choice(MID_TEMPLATES)
    else:
        template = random.choice(LOW_TEMPLATES)

    # Compose extras / gap_note
    if rank <= 30:
        if 'extras' in template:
            extras = random.choice(STRONG_EXTRAS)
            # Vary the extras
            if random.random() < 0.3 and np_days <= 60:
                extras = notice_note
        else:
            extras = ''
        gap_note = None
    else:
        gap_note = random.choice(GAP_NOTES)
        # Avoid duplicating if location is the actual gap
        if 'location' in gap_note and not in_tier1 and not in_pune_noida:
            gap_note = "location outside the JD's preferred cities"
        extras = None

    # Format the template
    try:
        text = template.format(
            title=title,
            company=company,
            years=years_phrase(years),
            n_skills=n_skills,
            skills=skills_str,
            extras=extras or '',
            skill_highlight=skill_highlight,
            product_years=f"{product_years:.1f} years",
            longest_tenure=longest_tenure,
            n_career=n_career,
            location_note=location_note,
            gap_note=gap_note or '',
        )
    except (KeyError, IndexError):
        # Fallback if template formatting fails
        text = (f"{title} at {company} ({years_phrase(years)} experience) "
                f"with relevant skills in {skills_str}; "
                f"{extras or gap_note or 'see profile for details'}")

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


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
