"""Batch processing orchestrator. Run after collecting matches."""
import sys
from db import get_conn
from features import compute_feature_vectors
from concepts import discover_concepts, save_concepts, find_key_moments, save_key_moments


def process_all(db_path=None):
    """Full pipeline: discover concepts from corpus, find key moments in each match."""
    conn = get_conn(db_path)
    match_ids = [r['match_id'] for r in conn.execute("SELECT match_id FROM games").fetchall()]
    conn.close()

    print(f"Processing {len(match_ids)} matches...")

    # Step 1: Discover concepts from the corpus
    print("Discovering concepts...")
    concept_list = discover_concepts(db_path)
    if not concept_list:
        print("Not enough data to discover concepts (need 20+ feature vectors)")
        return
    save_concepts(concept_list, db_path)
    print(f"Found {len(concept_list)} concepts:")
    for c in concept_list:
        print(f"  {c['name']} — importance {c['importance']:.3f}")

    # Step 2: Find key moments in each match
    print("\nFinding key moments...")
    total_moments = 0
    for mid in match_ids:
        moments = find_key_moments(mid, concept_list, db_path)
        if moments:
            save_key_moments(moments, db_path)
            total_moments += len(moments)

    print(f"Found {total_moments} key moments across {len(match_ids)} matches")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    process_all(db_path)
