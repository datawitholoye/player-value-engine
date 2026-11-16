"""
End-to-end test of the scoring API.

Runs the whole pipeline on a small dataset (generate -> SQL features ->
train -> API) and checks the API follows the business rules:
  - safer gambling decisions come first and block offers
  - only eligible players appear in the offer queue, ranked by value at risk
"""
import json

import pytest
from fastapi.testclient import TestClient

from pve import build_features, train_baselines, train_models
from pve.api import create_app
from pve.generate_data import generate, save


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    mp = pytest.MonkeyPatch()
    mp.chdir(root)                                   # the scripts use relative data/ paths
    try:
        save(root / "data/raw", *generate(2500, seed=11))
        build_features.build(verbose=False)
        train_baselines.main()
        train_models.main()
        app = create_app(root / "data/features/player_features.parquet", root / "models")
        yield TestClient(app)
    finally:
        mp.undo()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["snapshot_date"] == "2025-06-30"


def test_unknown_player_is_404(client):
    assert client.get("/players/99999999").status_code == 404


def test_offer_queue_only_has_eligible_players_ranked_by_value_at_risk(client):
    queue = client.get("/offers?limit=50").json()
    assert len(queue) > 0
    assert all(p["safer_gambling"]["decision"] == "eligible" for p in queue)
    ranks = [p["offer"]["offer_rank"] for p in queue]
    assert ranks == sorted(ranks) and ranks[0] == 1
    var = [p["offer"]["value_at_risk"] for p in queue]
    assert var == sorted(var, reverse=True)


def test_flagged_players_never_get_offers_and_always_have_reasons(client, capsys):
    blocked = 0
    for pid in range(1, 400):
        r = client.get(f"/players/{pid}")
        if r.status_code != 200:
            continue
        p = r.json()
        if p["safer_gambling"]["decision"] != "eligible":
            blocked += 1
            assert p["offer"]["eligible"] is False and p["offer"]["offer_rank"] is None
            assert len(p["safer_gambling"]["reasons"]) > 0      # staff need to know why
    assert blocked > 0
