import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class CanolaStageSampler:
    # Expected fraction of the sowing->harvest calendar length spent in each segment:
    # [germination lag, vegetative duration, anthesis duration, remainder to harvest]
    mean_fractions: tuple = (0.09, 0.36, 0.15, 0.40)

    # Concentration for the Dirichlet; higher = tighter around the mean
    kappa: float = 30.0

    # Minimum durations (days) for each segment to enforce realism and ordering
    min_days: tuple = (5, 25, 7, 14)

    # RNG for reproducibility (set to an int or leave None)
    seed: int | None = 123

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        mf = np.asarray(self.mean_fractions, dtype=float)
        if not np.isclose(mf.sum(), 1.0):
            mf = mf / mf.sum()
        self.mean_fractions = tuple(mf)
        self.alpha = mf * float(self.kappa) + 1e-9  # keep strictly positive

    @staticmethod
    def _to_timestamp(x):
        """Accepts pd.Timestamp / datetime / str('YYYY-MM-DD') / int DOY + optional year tuple."""
        if isinstance(x, pd.Timestamp):
            return x.normalize()
        if isinstance(x, datetime):
            return pd.Timestamp(x.date())
        if isinstance(x, str):
            return pd.Timestamp(x)
        # If the user passes (doy, year) as a tuple:
        if isinstance(x, tuple) and len(x) == 2:
            doy, year = int(x[0]), int(x[1])
            jan1 = pd.Timestamp(year=year, month=1, day=1)
            return jan1 + pd.Timedelta(days=doy - 1)
        # If it's just an int (assume DOY of current year)
        if isinstance(x, int):
            today_year = pd.Timestamp.today().year
            jan1 = pd.Timestamp(year=today_year, month=1, day=1)
            return jan1 + pd.Timedelta(days=int(x) - 1)
        raise ValueError(f"Unrecognized date format: {x}")

    @staticmethod
    def _doy(ts: pd.Timestamp) -> int:
        return int(ts.dayofyear)

    def _fit_minima(self, total_days: int, eps_frac: float = 0.02):
        """Ensure minima fit into the season; shrink proportionally if needed, leaving small slack."""
        mins = np.array(self.min_days, dtype=float)
        total_min = mins.sum()
        if total_min >= total_days:
            # shrink all minima to fit, keeping a tiny slack to allow variation
            scale = (1.0 - eps_frac) * (total_days / total_min)
            mins = np.maximum(1.0, np.floor(mins * scale))
        return mins

    def sample_one(self, sowing, harvest):
        """
        Returns a dict with Timestamp and DOY for:
        - vegetative_start
        - flowering_start
        - grainfill_start
        """
        S = self._to_timestamp(sowing)
        H = self._to_timestamp(harvest)
        # If harvest is earlier in calendar than sowing, assume harvest is next year
        if H <= S:
            H = H + pd.DateOffset(years=1)

        D = (H - S).days
        if D < 10:
            raise ValueError(f"Sowing->harvest span ({D} days) too short for realistic staging.")

        # Enforce minima via "slack" Dirichlet: durations = min_days + slack * Dir(alpha)
        mins = self._fit_minima(D)
        slack_days = max(1, D - int(mins.sum()))
        w = self.rng.dirichlet(self.alpha)  # proportions that sum to 1
        durations = mins + (slack_days * w)

        # Build cumulative times (integer days from sowing)
        d1 = int(round(durations[0]))                    # sow -> vegetative start
        d2 = d1 + int(round(durations[1]))               # vegetative -> flowering start
        d3 = d2 + int(round(durations[2]))               # flowering -> grainfill start
        # We don't need the final segment to compute the three starts

        # Clamp to keep inside [S+1, H-1] and strictly increasing
        d1 = max(1, min(d1, D-3))
        d2 = max(d1+1, min(d2, D-2))
        d3 = max(d2+1, min(d3, D-1))

        veg_ts  = S + pd.Timedelta(days=d1)
        flow_ts = S + pd.Timedelta(days=d2)
        gf_ts   = S + pd.Timedelta(days=d3)

        return dict(
            vegetative_start_ts=veg_ts,
            flowering_start_ts=flow_ts,
            grainfill_start_ts=gf_ts,
            vegetative_start_doy=self._doy(veg_ts),
            flowering_start_doy=self._doy(flow_ts),
            grainfill_start_doy=self._doy(gf_ts),
            sowing_ts=S, harvest_ts=H, sowing_doy=self._doy(S), harvest_doy=self._doy(H)
        )

    def sample(self, sowing, harvest, n=1, as_dataframe=True):
        out = [self.sample_one(sowing, harvest) for _ in range(int(n))]
        if as_dataframe:
            return pd.DataFrame(out)[[
                "sowing_ts",
                "harvest_ts",
                "vegetative_start_ts",
                "flowering_start_ts",
                "grainfill_start_ts",
                "sowing_doy",
                "harvest_doy",
                "vegetative_start_doy",
                "flowering_start_doy",
                "grainfill_start_doy"
            ]]
        return out


