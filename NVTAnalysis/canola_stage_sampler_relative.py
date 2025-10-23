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

def _clip01(x, lo=0.0, hi=0.95):
    return float(np.clip(x, lo, hi))

@dataclass
class CanolaStageSampler:
    # --- Relative phase shares (mean sd), all as fractions of (harvest - sowing) ---
    # 1) sow -> emergence
    mu_emerg_lag_rel: float = 0.05
    sd_emerg_lag_rel: float = 0.02

    # 2) emergence -> end of vegetative (vegetative duration)
    mu_veg_lag_rel: float = 0.50
    sd_veg_lag_rel: float = 0.10

    # 3) end vegetative -> end anthesis (flowering/anthesis duration)
    mu_anth_lag_rel: float = 0.20
    sd_anth_lag_rel: float = 0.10

    # 4) end anthesis -> end grainfill (grainfill duration)
    mu_grainfill_lag_rel: float = 0.20
    sd_grainfill_lag_rel: float = 0.10

    # 5) end grainfill -> maturity (maturity lead time before harvest)
    mu_mature_lag_rel: float = 0.05
    sd_mature_lag_rel: float = 0.02

    # RNG
    seed: int | None = 123
    # Soft numeric guard so sum of shares stays < 1 (keeps events before harvest)
    max_sum: float = 0.98  # leave >=2% of the season from maturity->harvest
    clip_hi: float = 0.95  # each individual share clipped to [0, clip_hi]

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def _sample_phase_shares(self):
        # independent normal draws, softly clipped to [0, clip_hi]
        e  = _clip01(self.rng.normal(self.mu_emerg_lag_rel,    self.sd_emerg_lag_rel),    0.0, self.clip_hi)
        v  = _clip01(self.rng.normal(self.mu_veg_lag_rel,      self.sd_veg_lag_rel),      0.0, self.clip_hi)
        a  = _clip01(self.rng.normal(self.mu_anth_lag_rel,     self.sd_anth_lag_rel),     0.0, self.clip_hi)
        g  = _clip01(self.rng.normal(self.mu_grainfill_lag_rel,self.sd_grainfill_lag_rel),0.0, self.clip_hi)
        m  = _clip01(self.rng.normal(self.mu_mature_lag_rel,   self.sd_mature_lag_rel),   0.0, self.clip_hi)
        shares = np.array([e, v, a, g, m], dtype=float)

        s = shares.sum()
        if s > self.max_sum:
            shares *= (self.max_sum / s)  # proportional shrink to fit
        return shares  # sow→em, em→veg_end, veg_end→anth_end, anth_end→gf_end, gf_end→maturity

    def _shares_to_days(self, total_days: int, shares: np.ndarray):
        # integerize with rounding; ensure sum <= total_days-1 (leave >=1 day remainder)
        days = np.rint(shares * total_days).astype(int)
        # If rounding pushes sum too high, trim from last segment(s)
        overflow = int(days.sum() - (total_days - 1))
        i = len(days) - 1
        while overflow > 0 and i >= 0:
            take = min(overflow, days[i])
            days[i] -= take
            overflow -= take
            i -= 1
        return days

    def _sample_once(self, sowing, harvest):
        S = _to_ts(sowing)
        H = _to_ts(harvest)
        if H <= S:
            H = H + pd.DateOffset(years=1)

        total_days = (H - S).days
        if total_days < 10:
            raise ValueError("Sowing→harvest window too short for realistic staging.")

        shares = self._sample_phase_shares()
        dE, dV, dA, dG, dM = self._shares_to_days(total_days, shares)

        # Build boundaries
        Emergence = S + pd.Timedelta(days=dE)

        # "Start of vegetative": the day after emergence (start of that phase)
        Veg_start = max(Emergence + pd.Timedelta(days=1), Emergence)

        # End vegetative = Emergence + dV  → SOF happens at the boundary to anthesis
        SOF = Emergence + pd.Timedelta(days=dV)
        SOF = max(SOF, Veg_start + pd.Timedelta(days=1))  # enforce ordering

        # End anthesis = SOF + dA  → Grainfill start at that boundary
        GF = SOF + pd.Timedelta(days=dA)
        GF = max(GF, SOF + pd.Timedelta(days=1))

        # End grainfill = GF + dG  → Maturity start at that boundary
        Maturity = GF + pd.Timedelta(days=dG)
        Maturity = max(Maturity, GF + pd.Timedelta(days=1))

        # Apply maturity lead-time (dM) to pull maturity before harvest if needed
        # Here, dM is already part of the breakdown; Maturity computed above respects it.
        # Guard to keep everything strictly before harvest:
        Maturity = min(Maturity, H - pd.Timedelta(days=1))

        # Final safety clamps to keep events in order and < H
        Veg_start = min(Veg_start, H - pd.Timedelta(days=3))
        SOF       = min(max(SOF, Veg_start + pd.Timedelta(days=1)), H - pd.Timedelta(days=2))
        GF        = min(max(GF,  SOF + pd.Timedelta(days=1)),       H - pd.Timedelta(days=1))
        if Maturity <= GF:
            Maturity = GF + pd.Timedelta(days=1)
        Maturity = min(Maturity, H - pd.Timedelta(days=1))

        return dict(
            sowing_ts=S, harvest_ts=H, emergence_ts=Emergence,
            vegetative_start_ts=Veg_start, flowering_start_ts=SOF,
            grainfill_start_ts=GF, maturity_start_ts=Maturity,
            sowing_doy=_doy(S), harvest_doy=_doy(H), emergence_doy=_doy(Emergence),
            vegetative_start_doy=_doy(Veg_start), flowering_start_doy=_doy(SOF),
            grainfill_start_doy=_doy(GF), maturity_start_doy=_doy(Maturity),
            # (optional) expose sampled phase-day lengths for inspection
            sow_to_em_days=int(dE),
            veg_days=int(dV),
            anthesis_days=int(dA),
            grainfill_days=int(dG),
            mature_lead_days=int(dM),
            remainder_days=int(max(0, total_days - (dE+dV+dA+dG+dM)))
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
# sampler = CanolaStageSampler(seed=42)
# df = sampler.sample("2025-05-01", "2025-09-05", n=5)
# print(df)
