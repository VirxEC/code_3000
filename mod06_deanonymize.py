import pandas as pd


def load_data(
    anonymized_path: str, auxiliary_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df: pd.DataFrame, aux_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    merged = anon_df.merge(
        aux_df,
        on=["age", "zip3", "gender"],
        how="inner",
    )
    unique_items = merged.groupby("anon_id")["name"].transform("size") == 1
    return merged[unique_items][["anon_id", "name"]].rename(columns={"name": "matched_name"})


def deanonymization_rate(matches_df: pd.DataFrame, anon_df: pd.DataFrame) -> float:
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df)
