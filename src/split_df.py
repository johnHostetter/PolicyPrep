from pathlib import Path

import pandas as pd

from src.utilities.reproducibility import path_to_project_root


def split_based_on_fsa_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the dataframe based on the FSA round.

    Args:
        df: The dataframe to split.

    Returns:
        The dataframe split based on the FSA round.
    """
    return df.groupby("FSA_round")


if __name__ == "__main__":
    data_path: Path = path_to_project_root() / "data" / "raw" / "dataset-train.csv"
    df: pd.DataFrame = pd.read_csv(data_path)
