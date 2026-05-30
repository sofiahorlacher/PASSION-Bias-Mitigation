from typing import Optional, Sequence, Union

import pandas as pd


def exclude_fitzpatrick_rows(
    df: pd.DataFrame,
    exclude_fitzpatrick_values: Optional[Sequence[Union[str, int]]] = None,
) -> pd.DataFrame:
    """Remove Fitzpatrick groups that should not be part of the pipeline."""
    if not exclude_fitzpatrick_values:
        return df.copy()

    if "fitzpatrick" not in df.columns:
        raise ValueError(
            "Cannot exclude Fitzpatrick values because the 'fitzpatrick' column is missing."
        )

    excluded_values = {str(value).strip() for value in exclude_fitzpatrick_values}
    filtered_df = df[
        ~df["fitzpatrick"].astype(str).str.strip().isin(excluded_values)
    ].copy()
    return filtered_df


def filter_split_to_subjects(
    df_split: pd.DataFrame,
    subject_ids: Sequence[Union[str, int]],
) -> pd.DataFrame:
    """Keep only split rows that still exist after metadata filtering."""
    valid_subject_ids = {str(subject_id).strip() for subject_id in subject_ids}
    filtered_df = df_split[
        df_split["subject_id"].astype(str).str.strip().isin(valid_subject_ids)
    ].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df
