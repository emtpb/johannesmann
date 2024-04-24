"""Tests for `johannesmann` package."""
import numpy as np
import johannesmann


def test_line():
    line = johannesmann.Line(0, 0)
    assert line.slope == 0
    assert line.above(0, 1)
    assert ~line.above(0, -1)


def test_tessellation():
    tessel = johannesmann.Tessellation(4, 4, 2)
    # Manually add a non-random line
    tessel.lines = [johannesmann.Line(0, 0)]
    assert tessel.tile_id(0, 1) == 1
    assert tessel.tile_id(0, -1) == 0
    r = tessel.sample_2d(2, 2)
    assert np.allclose(r, np.array([[0, 0], [1, 1]]))
