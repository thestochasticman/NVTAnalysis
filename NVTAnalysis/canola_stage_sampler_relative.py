
import numpy as np
import pandas as pd
from dataclasses import dataclass

def _to_ts(x):
    if isinstance(x, pd.Timestamp): return x.normalize()
    if isinstance(x, str):          return pd.Timestamp(x)
    if isinstance(x, np.datetime64):return pd.Timestamp(x).normalize()
    raise TypeError("Expected pd.Timestamp or 'YYYY-MM-DD' string.")

def _doy(ts: pd.Timestamp) -> int:
    return int(ts.dayofyear)

@dataclass
class CanolaStageSampler:
    # Target MEAN shares (fractions of season). They will be normalised for the Dirichlet.
    mu_emerg_lag_rel: float = 0.05   # sow -> emergence
    mu_veg_lag_rel: float   = 0.50   # emergence -> end vegetative (SOF boundary)
    mu_anth_lag_rel: float  = 0.20   # end vegetative -> end anthesis (GF boundary)
    mu_grainfill_lag_rel: float = 0.20  # end anthesis -> end grainfill
    mu_mature_lag_rel: float = 0.05     # end grainfill -> maturity

    kappa: float = 30.0

    # How much of S->H the five phases should occupy (1.0 = whole season)
    total_share: float = 1.0

    # RNG seed
    seed: int | None = 123

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

        # Build the Dirichlet mean vector on the simplex
        mu = np.array([
            self.mu_emerg_lag_rel,
            self.mu_veg_lag_rel,
            self.mu_anth_lag_rel,
            self.mu_grainfill_lag_rel,
            self.mu_mature_lag_rel
        ], dtype=float)

        if mu.sum() <= 0:
            raise ValueError("All phase means are zero; please set positive means.")

        self.mu_simplex = mu / mu.sum()        # sums to 1
        self.alpha = self.mu_simplex * self.kappa  # Dirichlet parameters

        # Guard for total_share
        if not (0.0 < float(self.total_share) <= 1.0):
            raise ValueError("total_share must be in (0, 1].")

    def _sample_phase_shares(self) -> np.ndarray:
        """Sample five phase shares from Dirichlet and scale by total_share."""
        p = self.rng.dirichlet(self.alpha)     # sums to 1
        return self.total_share * p            # sums to total_share

    @staticmethod
    def _shares_to_days(total_days: int, shares: np.ndarray) -> np.ndarray:
        """
        Convert shares to integer days. If rounding overshoots, trim from the tail.
        Leaves any shortfall as 'remainder_days' automatically.
        """
        days = np.rint(shares * total_days).astype(int)
        overflow = int(days.sum() - total_days)
        i = len(days) - 1
        while overflow > 0 and i >= 0:
            take = min(overflow, days[i])
            days[i] -= take
            overflow -= take
            i -= 1
        return days  # [dE,dV,dA,dG,dM]

    def _sample_once(self, sowing, harvest):
        S = _to_ts(sowing)
        H = _to_ts(harvest)
        if H <= S:
            H = H + pd.DateOffset(years=1)

        total_days = (H - S).days
        if total_days < 5:
            raise ValueError("Sowing→harvest window too short.")

        shares = self._sample_phase_shares()
        dE, dV, dA, dG, dM = self._shares_to_days(total_days, shares)

        # Build stage boundaries
        Emergence = S + pd.Timedelta(days=int(dE))

        # Start of vegetative: immediately after emergence
        Veg_start = max(Emergence + pd.Timedelta(days=1), Emergence)

        # Start of flowering (SOF) at end of vegetative segment
        SOF = Emergence + pd.Timedelta(days=int(dV))
        SOF = max(SOF, Veg_start + pd.Timedelta(days=1))

        # Grainfill start at end of anthesis
        GF = SOF + pd.Timedelta(days=int(dA))
        GF = max(GF, SOF + pd.Timedelta(days=1))

        # Maturity start at end of grainfill
        Maturity = GF + pd.Timedelta(days=int(dG))
        Maturity = max(Maturity, GF + pd.Timedelta(days=1))

        # Keep maturity strictly before harvest (if rounding overshot)
        Maturity = min(Maturity, H - pd.Timedelta(days=1))
        if Maturity <= GF:
            Maturity = GF + pd.Timedelta(days=1)
            Maturity = min(Maturity, H - pd.Timedelta(days=1))

        remainder_days = max(0, (H - Maturity).days - int(dM))  # slack to harvest beyond the 'mature' segment

        return dict(
            sowing_ts=S, harvest_ts=H, emergence_ts=Emergence,
            vegetative_start_ts=Veg_start, flowering_start_ts=SOF,
            grainfill_start_ts=GF, maturity_start_ts=Maturity,

            sowing_doy=_doy(S), harvest_doy=_doy(H), emergence_doy=_doy(Emergence),
            vegetative_start_doy=_doy(Veg_start), flowering_start_doy=_doy(SOF),
            grainfill_start_doy=_doy(GF), maturity_start_doy=_doy(Maturity),

            # Expose lengths for inspection
            sow_to_em_days=int(dE),
            veg_days=int(dV),
            anthesis_days=int(dA),
            grainfill_days=int(dG),
            mature_lead_days=int(dM),        # intended lead to maturity within total_share
            remainder_days=int(remainder_days)
        )

    def sample(self, sowing, harvest, n: int = 1, as_dataframe: bool = True):
        rows = [self._sample_once(sowing, harvest) for _ in range(int(n))]
        if as_dataframe:
            return pd.DataFrame(rows)[[
                "sowing_ts","harvest_ts","emergence_ts",
                "vegetative_start_ts","flowering_start_ts","grainfill_start_ts","maturity_start_ts",
                "sowing_doy","harvest_doy","emergence_doy",
                "vegetative_start_doy","flowering_start_doy","grainfill_start_doy","maturity_start_doy",
                "sow_to_em_days","veg_days","anthesis_days","grainfill_days","mature_lead_days","remainder_days"
            ]]
        return rows

# Example:
# sampler = CanolaStageSampler(seed=42, kappa=40.0, total_share=1.0)
# df = sampler.sample("2025-05-01", "2025-09-05", n=5)
# print(df)



