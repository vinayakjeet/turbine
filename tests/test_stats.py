import pytest

from turbine.stats import cross_rep, mean, percentile, stdev, summarize


def test_percentile_known_values():
    values = list(range(1, 101))
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 95) == pytest.approx(95.05)
    assert percentile(values, 99) == pytest.approx(99.01)
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 100


def test_percentile_interpolates_between_ranks():
    assert percentile([10.0, 20.0], 25) == pytest.approx(12.5)


def test_percentile_rejects_empty_and_out_of_range():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 150)


def test_stdev_is_sample_stdev():
    # Population stdev of this set is exactly 2; sample stdev divides by n-1.
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert stdev(values) == pytest.approx(2.13809, abs=1e-4)
    with pytest.raises(ValueError):
        stdev([1.0])


def test_mean_rejects_nothing():
    with pytest.raises(ValueError):
        mean([])


def test_summarize_shapes_a_distribution():
    s = summarize([float(i) for i in range(1, 101)])
    assert s.n == 100
    assert s.mean == pytest.approx(50.5)
    assert s.p50 == pytest.approx(50.5)
    assert s.p95 > s.p50
    assert s.p99 > s.p95


def test_cross_rep_is_the_table_cell_shape():
    m, sd = cross_rep([1.0, 2.0, 3.0])
    assert m == 2.0
    assert sd == pytest.approx(1.0)
