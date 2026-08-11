import os
import pytest
from itertools import product

import numpy as np

from deepdrivewe.binners import RectilinearBinner, MultiRectilinearBinner

class TestRectilinearBinner:
    def test1dAssign(self) -> None:
        bounds = [0.0, 1.0, 2.0, 3.0]
        coords = np.array([-1, 0, 0.5, 1.5, 1.6, 2.0, 2.0, 2.9])[:, None]

        assigner = RectilinearBinner(bins=bounds, bin_target_counts=3, target_state_inds=[None], pcoord_idx=0)

        with pytest.warns(UserWarning):
            assert (assigner.assign_bins(coords) == [0, 0, 0, 1, 1, 2, 2, 2]).all()

    def test2dAssign(self) -> None:
        boundaries = [(-1, -0.5, 0, 0.5, 1), (-1, -0.5, 0, 0.5, 1)]
        coords = np.array([(-2, -2), (-0.75, -0.75), (-0.25, -0.25), (0, 0), (0.25, 0.25), (0.75, 0.75), (-0.25, 0.75), (0.25, -0.75)])

        assigner = MultiRectilinearBinner(boundaries, bin_target_counts=3, target_state_inds=[None])

        """bin structure: [(a,b), (c,d)] => x in [a,b), y in [c, d)
        0:[(-1, -0.5), (-1, -0.5)]
        1:[(-1, -0.5), (-0.5, 0)]
        2:[(-1, -0.5), (0, 0.5)]
        3:[(-1, -0.5), (0.5, 1)]
        4:[(-0.5, 0), (-1, -0.5)]
        5:[(-0.5, 0), (-0.5, 0)]
        6:[(-0.5, 0), (0, 0.5)]
        7:[(-0.5, 0), (0.5, 1)]
        8:[(0, 0.5), (-1, -0.5)]
        9:[(0, 0.5), (-0.5, 0)]
        10:[(0, 0.5), (0, 0.5)]
        11:[(0, 0.5), (0.5, 1)]
        12:[(0.5, 1), (-1, -0.5)]
        13:[(0.5, 1), (-0.5, 0)]
        14:[(0.5, 1), (0, 0.5)]
        15:[(0.5, 1), (0.5, 1)]"""

        with pytest.warns(UserWarning):
            assert (assigner.assign_bins(coords) == [0, 0, 5, 10, 10, 15, 7, 8]).all()

    def test2dAssign_v2(self) -> None:
        boundaries = [(0, 1, 2, 3), (0, 1, 2)]
        coords = np.array([(0.5, 0.5), (0.5, 1.5), (1.5, 0.5), (1.5, 1.5), (2.5, 0.5), (2.5, 1.5), (3.5, 1.5)])

        assigner = MultiRectilinearBinner(boundaries, bin_target_counts=3, target_state_inds=[None])

        with pytest.warns(UserWarning):
            # first 6 points are in bins [0, 5]. The last point locate outside the bounds but will be clipped to bin 5
            assert (assigner.assign_bins(coords) == [0, 1, 2, 3, 4, 5, 5]).all()

    def test3dAssign(self) -> None: 
        boundaries = [(0, 1, 2), (0, 1, 2, 3, 4, 5), (0, 1, 2)]
        coords = list(product([0.5, 1.5], [0.5, 1.5, 2.5, 3.5, 4.5], [0.5, 1.5]))  # One point per bin, in row-major order
        coords += [(2.5, 4.5, 1.5), (1.5, 5.5, 1.5)]  # Two points that are located outside the bin boundaries
        coords = np.asarray(coords)
        
        assigner = MultiRectilinearBinner(boundaries, bin_target_counts=3, target_state_inds=[None])

        with pytest.warns(UserWarning):
            # first 20 points are in bins [0, 19]. The last two locate outside the bounds but will be clipped to bin 19
            assert (assigner.assign_bins(coords) == list(range(20)) + [19, 19]).all()


