from leapy.dask.transformers.export import TrivialExporter


def test_trivial_export():
    exp = 'foo'
    act = TrivialExporter.to_runtime(exp)
    assert exp == act
