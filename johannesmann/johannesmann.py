"""Main module."""
import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray

__all__ = ['Line', 'Tessellation']


@dataclass
class Line:
    """Class to represent a line.

    Args:
        y_0: Y-intercept of the line.
        slope: Slope of the line.
    """
    y_0: float
    slope: float

    def above(self, x: float, y: float) -> bool:
        """Check if a given point is above the line.

        Args:
            x: X-Coordinate of the point.
            y: Y-Coordinate of the point.

        Returns:
            True if the point is above the line.
        """
        return y > self.y_0 + x * self.slope


class Tessellation:
    """Class to realise the Johannesmann Spatial Tessellation on a given
    rectangular region.
    """

    def __init__(self, x_range: float, y_range: float, cuts: int) -> None:
        """Class constructor.

        Args:
            x_range: Range to tessellate in x direction.
            y_range: Range to tessellate in y direction.
            cuts: Number of cuts.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.cuts = cuts

        self.lines = []

        for _ in range(cuts):
            # Random point in range
            x = (np.random.rand() - 0.5) * self.x_range
            y = (np.random.rand() - 0.5) * self.y_range
            # Random slope
            slope = np.tan((np.random.rand() - 0.5) * np.pi)
            self.lines.append(Line(
                y_0=y - slope * x,
                slope=slope
            ))

    def tile_id(self, x: float, y: float) -> int:
        """Return a unique id for the tile a point is placed on. This is done
        by checking above which of the lines to point is located and then
        interpreting the list of booleans as an integer.

        Args:
            x: X-Coordinate of the point.
            y: Y-Coordinate of the point.

        Returns:
            Id of the tile the point is placed on.
        """
        aboves = [line.above(x, y) for line in self.lines]
        return sum(2 ** exp * int(bit) for exp, bit in enumerate(aboves))

    def sample_2d(self, x_samples: int, y_samples: int) -> NDArray:
        """Sample the rectangular region in equidistant steps and return an
        array of tile ids.

        Args:
            x_samples: Number of samples in x direction.
            y_samples: Number of samples in y direction.

        Returns:
            Array of tile ids.
        """
        xs, ys = np.meshgrid(
            np.linspace(
                -self.x_range / 2,
                self.x_range / 2,
                x_samples,
                endpoint=True
            ),
            np.linspace(
                -self.y_range / 2,
                self.y_range / 2,
                y_samples,
                endpoint=True
            )
        )

        result = np.zeros((x_samples, y_samples))
        for idx, x in np.ndenumerate(xs):
            result[idx] = self.tile_id(x, ys[idx])

        return result
