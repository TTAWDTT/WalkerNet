from __future__ import annotations

import numpy as np

from scripts.cnop.forecast_field_climatology import monthly_observed_field_climatology


class _DatasetStub:
    def __init__(self) -> None:
        # Two January and two February training samples.  Every variable/grid
        # point has the same value, making the source/month reference explicit.
        data = np.asarray(
            [
                np.full((2, 2, 2), 1.0),
                np.full((2, 2, 2), 10.0),
                np.full((2, 2, 2), 3.0),
                np.full((2, 2, 2), 14.0),
            ],
            dtype=np.float32,
        )
        self.source_payloads = [{"data": data, "years": np.asarray([2000, 2000, 2001, 2001]), "months": np.asarray([1, 2, 1, 2])}]
        self.source_names = ["source-a"]
        self.data_config = {"train_years": (2000, 2001)}


def test_observed_field_climatology_is_source_and_calendar_month_specific() -> None:
    climatology = monthly_observed_field_climatology(_DatasetStub(), 0, np.asarray([2, 1, 2]))

    assert climatology.shape == (3, 2, 2, 2)
    assert np.allclose(climatology[0], 12.0)
    assert np.allclose(climatology[1], 2.0)
    assert np.allclose(climatology[2], 12.0)
