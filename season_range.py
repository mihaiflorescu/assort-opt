import numpy as np
import pandas as pd
from typing import List, Union
from sampling import Season


class SeasonRange:
    def __init__(self, seasons: Union[Season, List[Season]]):
        if isinstance(seasons, Season):
            self.seasons = [seasons]
        else:
            self.seasons = list(seasons)

    def create_mask(self, df: pd.DataFrame, date_column: str = "date") -> np.ndarray:
        # Convert to datetime if not already
        dates = pd.to_datetime(df[date_column])

        # Initialize mask with all False
        mask = np.zeros(len(df), dtype=bool)

        # OR together masks for each season
        for season in self.seasons:
            season_start = pd.to_datetime(season.Start)
            season_end = pd.to_datetime(season.End)
            season_mask = (dates >= season_start) & (dates <= season_end)
            mask = mask | season_mask.to_numpy()

        return mask

    def __repr__(self) -> str:
        s = [f"({s.Start}-{s.End})" for s in self.seasons]
        return f"SeasonRange({', '.join(s)})"

    def __len__(self) -> int:
        return len(self.seasons)
