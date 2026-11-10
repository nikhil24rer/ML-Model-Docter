
import pandas as pd


def clean_dataset(df, target_column=None):
    """
    Create a cleaned dataset and a transparent cleaning report.

    Model evaluation must still use its own train/test
    preprocessing pipeline to avoid data leakage.
    """

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    original_rows, original_columns = df.shape
    original_missing = int(df.isna().sum().sum())
    original_duplicates = int(df.duplicated().sum())

    cleaned = df.copy()
    actions = []

    # Remove columns that contain no usable values
    empty_columns = cleaned.columns[
        cleaned.isna().all()
    ].tolist()

    if empty_columns:
        cleaned = cleaned.drop(columns=empty_columns)
        actions.append(
            f"Removed {len(empty_columns)} fully empty columns."
        )

    # Remove completely empty rows
    empty_row_mask = cleaned.isna().all(axis=1)
    empty_rows_removed = int(empty_row_mask.sum())

    if empty_rows_removed:
        cleaned = cleaned.loc[~empty_row_mask].copy()
        actions.append(
            f"Removed {empty_rows_removed} fully empty rows."
        )

    # Remove exact duplicate records
    duplicates_removed = int(cleaned.duplicated().sum())

    if duplicates_removed:
        cleaned = cleaned.drop_duplicates().copy()
        actions.append(
            f"Removed {duplicates_removed} exact duplicate rows."
        )

    # A requested target must exist after empty-column removal
    if target_column is not None:
        if target_column not in cleaned.columns:
            raise ValueError(
                f"Target column '{target_column}' is not available "
                "after cleaning."
            )

        before = len(cleaned)
        cleaned = cleaned.dropna(
            subset=[target_column]
        ).copy()
        target_rows_removed = before - len(cleaned)

        if target_rows_removed:
            actions.append(
                f"Removed {target_rows_removed} rows with missing "
                "target values."
            )
    else:
        target_rows_removed = 0

    if cleaned.empty:
        raise ValueError(
            "No usable rows remain after cleaning."
        )

    # For the downloadable cleaned file only:
    # impute numeric and categorical columns.
    # Never use this export as the evaluation preprocessing pipeline.
    for column in cleaned.columns:

        if column == target_column:
            continue

        series = cleaned[column]

        if series.isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(series):
            median_value = series.median()

            if pd.notna(median_value):
                cleaned[column] = series.fillna(median_value)

        else:
            non_missing = series.dropna()

            if not non_missing.empty:
                mode_values = non_missing.mode()

                if not mode_values.empty:
                    cleaned[column] = series.fillna(
                        mode_values.iloc[0]
                    )

    remaining_missing = int(cleaned.isna().sum().sum())

    actions.append(
        "Numeric feature missing values were filled using medians; "
        "categorical feature missing values using most-frequent values "
        "where possible."
    )

    if remaining_missing:
        actions.append(
            f"{remaining_missing} missing cells remain and require "
            "manual review, including any missing target labels."
        )

    report = {
        "original_rows": int(original_rows),
        "original_columns": int(original_columns),
        "original_missing_values": original_missing,
        "original_duplicate_rows": original_duplicates,
        "empty_rows_removed": int(empty_rows_removed),
        "empty_columns_removed": empty_columns,
        "duplicate_rows_removed": int(duplicates_removed),
        "target_rows_removed": int(target_rows_removed),
        "cleaned_rows": int(cleaned.shape[0]),
        "cleaned_columns": int(cleaned.shape[1]),
        "remaining_missing_values": remaining_missing,
        "actions": actions,
    }

    return cleaned, report
