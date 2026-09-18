from types import SimpleNamespace

from tg_radar.auto_search import auto_policy, empty_run_error
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


def _ingest(candidates: int, errors: list[str]) -> SimpleNamespace:
    return SimpleNamespace(candidates_found=candidates, errors=errors)


def test_a_clean_run_that_found_nothing_is_not_an_error():
    # The healthier the pool, the more often discovery legitimately finds
    # nothing new -- reporting that as failure inverts the signal.
    assert empty_run_error(_ingest(0, [])) is None


def test_a_run_the_source_failed_still_reports_the_error():
    assert empty_run_error(_ingest(0, ["provider timeout"])) == "provider timeout"


def test_errors_are_capped_so_one_bad_run_cannot_flood_the_field():
    ingest = _ingest(0, ["a", "b", "c", "d"])

    assert empty_run_error(ingest) == "a; b; c"


def test_a_run_that_found_candidates_is_never_an_empty_run_error():
    assert empty_run_error(_ingest(3, ["partial failure"])) is None


def test_a_missing_ingest_is_not_reported_as_an_empty_run():
    assert empty_run_error(None) is None
