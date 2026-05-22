import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def canonicalize_cohort_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
        try:
            numeric_value = float(value)
        except ValueError:
            return value
    else:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value).strip()

    if numeric_value.is_integer():
        return str(int(numeric_value))
    return str(numeric_value)


def parse_cohort_spec(spec):
    if spec is None:
        return set()

    cohorts = set()
    for raw_item in str(spec).split(","):
        item = raw_item.strip().replace(" ", "")
        if not item:
            continue
        if "-" in item:
            bounds = item.split("-", 1)
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(f"Invalid cohort range '{raw_item}'.")
            start = int(bounds[0])
            end = int(bounds[1])
            step = 1 if end >= start else -1
            for cohort in range(start, end + step, step):
                cohorts.add(str(cohort))
        else:
            normalized = canonicalize_cohort_value(item)
            if normalized is not None:
                cohorts.add(normalized)
    return cohorts


def resolve_cohort_column(df, cohort_col):
    if cohort_col in df.columns:
        return cohort_col
    if cohort_col == "cohert_num" and "cohort_num" in df.columns:
        return "cohort_num"
    raise KeyError(
        f"Cohort column '{cohort_col}' not found in input CSV. "
        f"Available columns: {list(df.columns)}"
    )


def assign_patient_folds(df, eligible_mask, patient_col, label_col, k_fold, random_state=42):
    fold_assignment = pd.Series(-1, index=df.index, dtype=int)
    if k_fold <= 1:
        return fold_assignment

    train_df = df.loc[eligible_mask].copy()
    pid_labels = (
        train_df.groupby(train_df[patient_col].astype(str))[label_col]
        .max()
        .astype(int)
    )
    uniq_pids = pid_labels.index.to_numpy()
    pid_stratify_labels = pid_labels.values

    sgkf = StratifiedGroupKFold(
        n_splits=k_fold,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(sgkf.split(uniq_pids, pid_stratify_labels, groups=uniq_pids))
    for fold_i, (_, val_idx) in enumerate(splits):
        val_pids = set(uniq_pids[val_idx])
        mask = eligible_mask & df[patient_col].astype(str).isin(val_pids)
        fold_assignment.loc[mask] = fold_i
    return fold_assignment


def build_split_metadata(
    df,
    patient_col,
    label_col,
    split_source="cohort",
    split_col="split",
    train_split_value="training",
    test_split_value="test",
    cohort_col="cohert_num",
    train_cohorts="1-8",
    test_cohorts="9-10",
    k_fold=0,
    random_state=42,
    strict_unassigned=True,
):
    metadata = {
        "actual_cohort_col": None,
        "base_split": pd.Series("train", index=df.index, dtype=object),
        "train_mask": pd.Series(True, index=df.index),
        "test_mask": pd.Series(False, index=df.index),
        "fold_assignment": pd.Series(-1, index=df.index, dtype=int),
    }

    if split_source == "split":
        if split_col not in df.columns:
            raise KeyError(
                f"Split column '{split_col}' not found in input CSV. "
                f"Available columns: {list(df.columns)}"
            )
        split_series = df[split_col].astype(str)
        train_mask = split_series == str(train_split_value)
        test_mask = split_series == str(test_split_value)
        unassigned_mask = ~(train_mask | test_mask)
        if strict_unassigned and unassigned_mask.any():
            extras = sorted(split_series.loc[unassigned_mask].dropna().unique().tolist())
            raise ValueError(
                f"Found rows outside train/test split values for split mode: {extras}"
            )
        base_split = pd.Series("train", index=df.index, dtype=object)
        base_split.loc[test_mask] = "test"
        metadata.update(
            {
                "base_split": base_split,
                "train_mask": train_mask,
                "test_mask": test_mask,
                "fold_assignment": assign_patient_folds(
                    df,
                    train_mask,
                    patient_col,
                    label_col,
                    k_fold,
                    random_state=random_state,
                ),
            }
        )
        return metadata

    if split_source != "cohort":
        raise ValueError(f"Unsupported split_source '{split_source}'.")

    actual_cohort_col = resolve_cohort_column(df, cohort_col)
    cohort_values = df[actual_cohort_col].map(canonicalize_cohort_value)
    train_cohort_values = parse_cohort_spec(train_cohorts)
    test_cohort_values = parse_cohort_spec(test_cohorts)

    overlap = train_cohort_values & test_cohort_values
    if overlap:
        raise ValueError(f"Train/test cohorts overlap: {sorted(overlap)}")

    train_mask = cohort_values.isin(train_cohort_values)
    test_mask = cohort_values.isin(test_cohort_values)
    unassigned_mask = ~(train_mask | test_mask)

    if strict_unassigned and unassigned_mask.any():
        extras = sorted(
            {
                value
                for value in cohort_values.loc[unassigned_mask].tolist()
                if value is not None
            }
        )
        raise ValueError(
            "Found rows whose cohort is not assigned to train/test cohorts: "
            f"{extras}"
        )

    base_split = pd.Series("train", index=df.index, dtype=object)
    base_split.loc[test_mask] = "test"

    metadata.update(
        {
            "actual_cohort_col": actual_cohort_col,
            "base_split": base_split,
            "train_mask": train_mask,
            "test_mask": test_mask,
            "fold_assignment": assign_patient_folds(
                df,
                train_mask,
                patient_col,
                label_col,
                k_fold,
                random_state=random_state,
            ),
        }
    )
    return metadata


def build_output_split_series(base_split, fold_assignment, current_fold=None):
    split_series = pd.Series(base_split, copy=True)
    if current_fold is not None:
        split_series.loc[(split_series == "train") & (fold_assignment == current_fold)] = "val"
    return split_series
