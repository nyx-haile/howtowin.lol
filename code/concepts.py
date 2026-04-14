import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from db import get_conn
from features import compute_feature_vectors

# Human-readable names and explanations for each feature.
# These become the "Duolingo lessons" — the teaching content.
FEATURE_META = {
    'team_gold_lead': {
        'name': 'Gold Advantage',
        'explanation': 'Your team has more gold than the enemy. Gold buys items, items win fights. '
                       'Look at which lanes are ahead and help them push that lead.',
        'category': 'economy',
        'phase': 'all',
    },
    'team_xp_lead': {
        'name': 'Experience Lead',
        'explanation': 'Your team has more XP, meaning higher levels and stronger abilities. '
                       'Level advantages are strongest in early fights — a 1-level lead at level 6 is huge.',
        'category': 'economy',
        'phase': 'early',
    },
    'team_kill_lead': {
        'name': 'Kill Pressure',
        'explanation': 'Your team has more kills. Each kill gives gold AND removes an enemy from the map. '
                       'After getting kills, push objectives — don\'t just go back to farming.',
        'category': 'fighting',
        'phase': 'all',
    },
    'team_dragon_diff': {
        'name': 'Dragon Control',
        'explanation': 'Dragons give permanent team buffs and stack toward Dragon Soul. '
                       'Set up vision near dragon pit 60 seconds before it spawns. '
                       'If you can\'t contest, trade for a tower or Rift Herald on the other side.',
        'category': 'objectives',
        'phase': 'mid',
    },
    'team_tower_diff': {
        'name': 'Tower Pressure',
        'explanation': 'Towers give gold, map control, and open paths into the enemy base. '
                       'After winning a fight, always look to take a tower before the enemy respawns. '
                       'An outer tower gives your team safe access to the enemy jungle.',
        'category': 'objectives',
        'phase': 'all',
    },
    'team_baron_diff': {
        'name': 'Baron Control',
        'explanation': 'Baron buff supercharges your minion waves, making them push on their own. '
                       'With Baron, group and siege — don\'t split up. The buff wins by pressure, not by fighting.',
        'category': 'objectives',
        'phase': 'late',
    },
    'team_herald_diff': {
        'name': 'Rift Herald Usage',
        'explanation': 'Rift Herald charges into a tower and deals massive damage. '
                       'Drop it in a lane where the outer tower is already low, or use it to open up mid lane.',
        'category': 'objectives',
        'phase': 'early',
    },
    'team_ward_diff': {
        'name': 'Vision Control',
        'explanation': 'Wards let you see the enemy before they see you. '
                       'More vision means fewer surprises, better objective setups, and safer rotations. '
                       'Place wards where you\'re about to play — near the next objective, '
                       'in the enemy jungle entrance, or in river.',
        'category': 'vision',
        'phase': 'all',
    },
    'team_cs_lead': {
        'name': 'Farming Efficiency',
        'explanation': 'CS (creep score) is the most reliable gold source. '
                       'Missing minions early adds up fast — 10 missed CS is roughly a kill\'s worth of gold. '
                       'Focus on not missing minions before looking for fights.',
        'category': 'economy',
        'phase': 'early',
    },
    'team_spread': {
        'name': 'Team Positioning',
        'explanation': 'How spread out your team is on the map. '
                       'Grouped = strong for teamfights and objectives. '
                       'Spread = strong for pressure but vulnerable to picks. '
                       'Match your positioning to what you want to do next.',
        'category': 'macro',
        'phase': 'mid',
    },
}


def discover_concepts(db_path=None):
    """Train classifier on all matches, extract top features as concepts."""
    conn = get_conn(db_path)
    match_ids = [r['match_id'] for r in conn.execute("SELECT match_id FROM games").fetchall()]
    conn.close()

    all_vectors = []
    for mid in match_ids:
        vecs = compute_feature_vectors(mid, db_path)
        all_vectors.extend(vecs)

    if len(all_vectors) < 20:
        return []

    feature_keys = [
        'team_gold_lead', 'team_xp_lead', 'team_cs_lead', 'team_kill_lead',
        'team_ward_diff', 'team_dragon_diff', 'team_tower_diff',
        'team_baron_diff', 'team_herald_diff', 'team_spread',
    ]

    X = np.array([[v.get(k, 0) for k in feature_keys] for v in all_vectors])
    y = np.array([v['won'] for v in all_vectors])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importances = clf.feature_importances_
    ranked = sorted(zip(feature_keys, importances), key=lambda x: x[1], reverse=True)

    result = []
    for feature_key, importance in ranked:
        if importance < 0.01:
            continue
        meta = FEATURE_META.get(feature_key, {})
        result.append({
            'concept_id': f'concept_{feature_key}',
            'name': meta.get('name', feature_key.replace('_', ' ').title()),
            'explanation': meta.get('explanation', f'This measures {feature_key.replace("_", " ")}.'),
            'importance': float(importance),
            'feature_key': feature_key,
            'category': meta.get('category', 'general'),
            'phase': meta.get('phase', 'all'),
        })

    return result


def find_key_moments(match_id, concept_list, db_path=None):
    """Find timestamps in a match where concepts were most impactful.

    A "key moment" is when a feature changed significantly between frames.
    """
    vectors = compute_feature_vectors(match_id, db_path)
    if not vectors:
        return []

    by_team = {}
    for v in vectors:
        tid = v['team_id']
        if tid not in by_team:
            by_team[tid] = []
        by_team[tid].append(v)

    moments = []
    for concept in concept_list:
        fk = concept['feature_key']

        for tid, team_vecs in by_team.items():
            team_vecs.sort(key=lambda v: v['timestamp_ms'])
            prev_val = None

            for v in team_vecs:
                val = v.get(fk, 0)
                if prev_val is not None:
                    delta = val - prev_val
                    threshold = _significance_threshold(fk)
                    if abs(delta) >= threshold:
                        moments.append({
                            'match_id': match_id,
                            'timestamp_ms': v['timestamp_ms'],
                            'concept_id': concept['concept_id'],
                            'concept_name': concept['name'],
                            'team_id': tid,
                            'description': _describe_moment(concept, delta, v['minutes']),
                            'impact_score': abs(delta) * concept['importance'],
                        })
                prev_val = val

    moments.sort(key=lambda m: m['impact_score'], reverse=True)
    return moments[:10]


def _significance_threshold(feature_key):
    """Minimum change to count as a 'key moment'."""
    thresholds = {
        'team_gold_lead': 1500,
        'team_xp_lead': 1000,
        'team_cs_lead': 15,
        'team_kill_lead': 2,
        'team_ward_diff': 3,
        'team_dragon_diff': 1,
        'team_tower_diff': 1,
        'team_baron_diff': 1,
        'team_herald_diff': 1,
        'team_spread': 1500,
    }
    return thresholds.get(feature_key, 1)


def _describe_moment(concept, delta, minutes):
    """Generate a plain-English description of what happened."""
    minute = int(minutes)
    name = concept['name']
    if delta > 0:
        return f"At {minute} min, your team gained {name.lower()}. {concept['explanation']}"
    else:
        return f"At {minute} min, your team lost {name.lower()}. {concept['explanation']}"


def save_concepts(concept_list, db_path=None):
    """Persist discovered concepts to the database."""
    conn = get_conn(db_path)
    for c in concept_list:
        conn.execute(
            "INSERT OR REPLACE INTO concepts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (c['concept_id'], c['name'], c['explanation'],
             c['feature_key'], c['importance'], c['phase'], c['category'])
        )
    conn.commit()
    conn.close()


def save_key_moments(moments, db_path=None):
    """Persist key moments to the database."""
    conn = get_conn(db_path)
    for m in moments:
        conn.execute(
            "INSERT OR REPLACE INTO key_moments VALUES (?, ?, ?, ?, ?, ?)",
            (m['match_id'], m['timestamp_ms'], m['concept_id'],
             m['team_id'], m['description'], m['impact_score'])
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'discover':
        found = discover_concepts()
        save_concepts(found)
        print(f"Discovered {len(found)} concepts:")
        for c in found:
            print(f"  {c['name']} (importance: {c['importance']:.3f})")
    elif len(sys.argv) > 2 and sys.argv[1] == 'moments':
        conn = get_conn()
        concept_rows = conn.execute("SELECT * FROM concepts ORDER BY importance DESC").fetchall()
        conn.close()
        concept_list = [dict(r) for r in concept_rows]
        moments = find_key_moments(sys.argv[2], concept_list)
        save_key_moments(moments)
        print(f"Found {len(moments)} key moments in {sys.argv[2]}:")
        for m in moments:
            print(f"  [{m['timestamp_ms']}ms] {m['concept_name']}: {m['description'][:80]}")
    else:
        print("Usage: python concepts.py discover | moments <match_id>")
