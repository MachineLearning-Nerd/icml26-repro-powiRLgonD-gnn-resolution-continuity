import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_live_contract_is_anchored_and_six_claims():
    claims=json.loads((ROOT/'contract/live_claims.json').read_text())
    manifest=json.loads((ROOT/'contract/contract_manifest.json').read_text())
    assert len(claims)==6
    assert manifest['claim_count']==6 and manifest['max_points']==12

def test_source_manifest_has_two_pins():
    entries=(ROOT/'evidence/source/SHA256SUMS').read_text().splitlines()
    assert len(entries)==2 and all(len(x.split()[0])==64 for x in entries)
