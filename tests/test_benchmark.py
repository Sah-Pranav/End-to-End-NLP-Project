from summarization.benchmark import calculate_latency_statistics


def test_calculate_latency_statistics():
    mean, median, p95 = calculate_latency_statistics(
        [100.0, 200.0, 300.0, 400.0, 500.0]
    )

    assert mean == 300.0
    assert median == 300.0
    assert p95 == 500.0
