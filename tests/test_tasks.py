from tg_radar.auto_search import auto_policy
from tg_radar.db import SearchTask, task_to_status


def test_task_to_status_maps_fields():
    task = SearchTask(
        name="x",
        keywords=["a"],
        seed_channels=["seed"],
        auto_tune=True,
        depth=1,
        limit=10,
        pages_per_channel=1,
        crawl=True,
        interval_seconds=300,
        enabled=True,
        running=False,
        total_runs=2,
        total_candidates=3,
        total_channels_crawled=4,
        total_messages_saved=5,
    )

    status = task_to_status(task)

    assert status.name == "x"
    assert status.keywords == ["a"]
    assert status.auto_tune is True
    assert status.total_messages_saved == 5


def test_auto_policy_expands_empty_runs():
    task = SearchTask(
        name="x",
        keywords=["web3 job"],
        seed_channels=[],
        auto_tune=True,
        depth=1,
        limit=40,
        pages_per_channel=1,
        crawl=True,
        interval_seconds=180,
        enabled=True,
        running=False,
        total_runs=3,
        last_candidates_found=0,
        last_channels_crawled=0,
        last_messages_saved=0,
    )

    assert auto_policy(task) == (0, 80, 1, 900)
