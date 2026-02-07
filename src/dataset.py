from typing import Any, Tuple

import pandas as pd
from torch.utils.data import Dataset


class SentimentDataset(Dataset[Tuple[str, Any]]):
    def __init__(self, file_path: str) -> None:
        self.data = pd.read_csv(file_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, Any]:
        text = str(self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 1]
        return text, label
