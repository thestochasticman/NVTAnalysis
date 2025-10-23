# import numpy as np
# import pandas as pd
# from dataclasses import dataclass

# def _to_ts(x):
#     if isinstance(x, pd.Timestamp):
#         return x.normalize()
#     if isinstance(x, str):
#         return pd.Timestamp(x)
#     if isinstance(x, np.datetime64):
#         return pd.Timestamp(x).normalize()
#     raise TypeError("Expected pd.Timestamp or 'YYYY-MM-DD' string.")

# def _doy(ts: pd.Timestamp) -> int:
#     return int(ts.dayofyear)

# def _trunc_normal(rng, mean: float, sd: float, low: float, high: float) -> float:
#     while True:
#         x = rng.normal(mean, sd)
#         if low <= x <= high:
#             return float(x)

# @dataclass
# class CanolaStageSampler:
#     # Emergence lag from sowing (days): TruncNorm(6, 2) within [0, 30]
#     mu_emerg_lag_d: float = 6.0
#     sd_emerg_lag_d: float = 2.0

#     # Vegetative start window after emergence (days): Uniform[7, 14]
#     veg_min_d: int = 7
#     veg_max_d: int = 14

#     # Start of flowering after emergence (days): TruncNorm(50, 3) within [45, 55]
#     sof_mean_d: float = 50.0 # 50
#     sof_sd_d: float = 3.0
#     sof_min_d: int = 45 # 45
#     sof_max_d: int = 55 # 60

#     # Grainfill start after SOF (days): Uniform[12, 18]
#     gf_min_after_sof_d: int = 12
#     gf_max_after_sof_d: int = 18

#     # RNG seed (None = random)
#     seed: int | None = 123

#     def __post_init__(self):
#         self.rng = np.random.default_rng(self.seed)

#     def _sample_once(self, sowing, harvest):
#         S = _to_ts(sowing)
#         H = _to_ts(harvest)
#         if H <= S:
#             H = H + pd.DateOffset(years=1)

#         # Sample emergence lag and compute emergence date
#         emerg_lag = _trunc_normal(self.rng, self.mu_emerg_lag_d, self.sd_emerg_lag_d, 0.0, 30.0)
#         Emergence = S + pd.Timedelta(days=int(round(emerg_lag)))

#         span_after_emerg = (H - Emergence).days
#         if span_after_emerg < 20:
#             raise ValueError("Sowing→harvest window too short after emergence for realistic staging.")

#         # Vegetative start: Uniform[7,14] days after emergence
#         veg_offset = self.rng.uniform(self.veg_min_d, self.veg_max_d)
#         Veg = Emergence + pd.Timedelta(days=int(round(veg_offset)))

#         # Start of flowering: TruncNorm(50,3) in [45,55] days after emergence
#         sof_offset = _trunc_normal(self.rng, self.sof_mean_d, self.sof_sd_d, self.sof_min_d, self.sof_max_d)
#         SOF = Emergence + pd.Timedelta(days=int(round(sof_offset)))

#         # Ensure SOF is after Veg (just in case of rounding ties)
#         if SOF <= Veg:
#             SOF = Veg + pd.Timedelta(days=1)

#         # Grainfill start: Uniform[12,18] days after SOF
#         gf_offset = self.rng.uniform(self.gf_min_after_sof_d, self.gf_max_after_sof_d)
#         GF = SOF + pd.Timedelta(days=int(round(gf_offset)))

#         # Keep everything inside [Emergence+1, H-1]
#         Veg = min(max(Veg, Emergence + pd.Timedelta(days=1)), H - pd.Timedelta(days=3))
#         SOF = min(max(SOF, Veg + pd.Timedelta(days=1)), H - pd.Timedelta(days=2))
#         GF  = min(max(GF,  SOF + pd.Timedelta(days=1)), H - pd.Timedelta(days=1))

#         return dict(
#             sowing_ts=S, harvest_ts=H, emergence_ts=Emergence,
#             vegetative_start_ts=Veg, flowering_start_ts=SOF, grainfill_start_ts=GF,
#             sowing_doy=_doy(S), harvest_doy=_doy(H), emergence_doy=_doy(Emergence),
#             vegetative_start_doy=_doy(Veg), flowering_start_doy=_doy(SOF), grainfill_start_doy=_doy(GF),
#         )

#     def sample(self, sowing, harvest, n: int = 1, as_dataframe: bool = True):
#         rows = [self._sample_once(sowing, harvest) for _ in range(int(n))]
#         if as_dataframe:
#             return pd.DataFrame(rows)[[
#                 "sowing_ts", "harvest_ts", "emergence_ts",
#                 "vegetative_start_ts", "flowering_start_ts", "grainfill_start_ts",
#                 "sowing_doy", "harvest_doy", "emergence_doy",
#                 "vegetative_start_doy", "flowering_start_doy", "grainfill_start_doy",
#             ]]
#         return rows

# # Example:
# # sampler = CanolaStageSampler(seed=42)
# # df = sampler.sample(sowing="2025-05-01", harvest="2025-09-05", n=5)
# # print(df)



import numpy as np
import pandas as pd
from dataclasses import dataclass

def _to_ts(x):
    if isinstance(x, pd.Timestamp):
        return x.normalize()
    if isinstance(x, str):
        return pd.Timestamp(x)
    if isinstance(x, np.datetime64):
        return pd.Timestamp(x).normalize()
    raise TypeError("Expected pd.Timestamp or 'YYYY-MM-DD' string.")

def _doy(ts: pd.Timestamp) -> int:
    return int(ts.dayofyear)

def _trunc_normal(rng, mean: float, sd: float, low: float, high: float) -> float:
    while True:
        x = rng.normal(mean, sd)
        if low <= x <= high:
            return float(x)

@dataclass
class CanolaStageSampler:
    # Emergence lag from sowing (days): TruncNorm(6, 2) within [0, 30]
    mu_emerg_lag_d: float = 6.0
    sd_emerg_lag_d: float = 2.0

    # Vegetative start window after emergence (days): Uniform[7, 14]
    veg_min_d: int = 7
    veg_max_d: int = 14

    # Start of flowering after emergence (days): TruncNorm(50, 3) within [45, 55]
    sof_mean_d: float = 50.0
    sof_sd_d: float = 3.0
    sof_min_d: int = 45
    sof_max_d: int = 55

    # Grainfill start after SOF (days): Uniform[12, 18]
    gf_min_after_sof_d: int = 12
    gf_max_after_sof_d: int = 18

    # Maturity window relative to harvest (days BEFORE harvest)
    maturity_min_before_harvest: int = 7
    maturity_max_before_harvest: int = 14

    # RNG seed (None = random)
    seed: int | None = 123

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def _sample_once(self, sowing, harvest):
        S = _to_ts(sowing)
        H = _to_ts(harvest)
        if H <= S:
            H = H + pd.DateOffset(years=1)

        # Sample emergence lag and compute emergence date
        emerg_lag = _trunc_normal(self.rng, self.mu_emerg_lag_d, self.sd_emerg_lag_d, 0.0, 30.0)
        Emergence = S + pd.Timedelta(days=int(round(emerg_lag)))

        span_after_emerg = (H - Emergence).days
        if span_after_emerg < 20:
            raise ValueError("Sowing→harvest window too short after emergence for realistic staging.")

        # Vegetative start: Uniform[7,14] days after emergence
        veg_offset = self.rng.uniform(self.veg_min_d, self.veg_max_d)
        Veg = Emergence + pd.Timedelta(days=int(round(veg_offset)))

        # Start of flowering: TruncNorm(50,3) in [45,55] days after emergence
        sof_offset = _trunc_normal(self.rng, self.sof_mean_d, self.sof_sd_d, self.sof_min_d, self.sof_max_d)
        SOF = Emergence + pd.Timedelta(days=int(round(sof_offset)))

        # Ensure SOF is after Veg (just in case of rounding ties)
        if SOF <= Veg:
            SOF = Veg + pd.Timedelta(days=1)

        # Grainfill start: Uniform[12,18] days after SOF
        gf_offset = self.rng.uniform(self.gf_min_after_sof_d, self.gf_max_after_sof_d)
        GF = SOF + pd.Timedelta(days=int(round(gf_offset)))

        # Keep everything inside [Emergence+1, H-1]
        Veg = min(max(Veg, Emergence + pd.Timedelta(days=1)), H - pd.Timedelta(days=3))
        SOF = min(max(SOF, Veg + pd.Timedelta(days=1)), H - pd.Timedelta(days=2))
        GF  = min(max(GF,  SOF + pd.Timedelta(days=1)), H - pd.Timedelta(days=1))

        # Maturity start: uniformly 7–14 days BEFORE harvest
        maturity_before = int(self.rng.integers(self.maturity_min_before_harvest,
                                                self.maturity_max_before_harvest + 1))
        Maturity = H - pd.Timedelta(days=maturity_before)
        # (Optional) If you want to force maturity after grainfill start, uncomment:
        # Maturity = max(Maturity, GF + pd.Timedelta(days=1))
        # And then clamp back into [H-14, H-7] if needed:
        # Maturity = min(max(Maturity, H - pd.Timedelta(days=self.maturity_max_before_harvest)),
        #                H - pd.Timedelta(days=self.maturity_min_before_harvest))

        return dict(
            sowing_ts=S, harvest_ts=H, emergence_ts=Emergence,
            vegetative_start_ts=Veg, flowering_start_ts=SOF, grainfill_start_ts=GF,
            maturity_start_ts=Maturity,
            sowing_doy=_doy(S), harvest_doy=_doy(H), emergence_doy=_doy(Emergence),
            vegetative_start_doy=_doy(Veg), flowering_start_doy=_doy(SOF),
            grainfill_start_doy=_doy(GF), maturity_start_doy=_doy(Maturity),
        )

    def sample(self, sowing, harvest, n: int = 1, as_dataframe: bool = True):
        rows = [self._sample_once(sowing, harvest) for _ in range(int(n))]
        if as_dataframe:
            return pd.DataFrame(rows)[[
                "sowing_ts", "harvest_ts", "emergence_ts",
                "vegetative_start_ts", "flowering_start_ts", "grainfill_start_ts", "maturity_start_ts",
                "sowing_doy", "harvest_doy", "emergence_doy",
                "vegetative_start_doy", "flowering_start_doy", "grainfill_start_doy", "maturity_start_doy",
            ]]
        return rows

# Example:
# sampler = CanolaStageSampler(seed=42)
# df = sampler.sample(sowing="2025-05-01", harvest="2025-09-05", n=5)
# print(df)
