import pickle
import json
import produce_submission

print("Loading artifacts...")
artifacts = produce_submission.load_artifacts()

print("Retrieving top 2000 candidates...")
indices = produce_submission.retrieve_top_k(artifacts, k=2000)
top_ids = {artifacts['candidate_ids'][idx] for idx in indices}
print(f"Top candidate IDs count: {len(top_ids)}")

print("Extracting profiles from candidates.jsonl...")
profiles = {}
with open("candidates.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
            cid = c['candidate_id']
            if cid in top_ids:
                profiles[cid] = c
        except Exception:
            continue

print(f"Matched {len(profiles)} profiles.")
with open("artifacts/candidate_profiles.pkl", "wb") as f:
    pickle.dump(profiles, f)

print("Saved successfully to artifacts/candidate_profiles.pkl!")
