"""Exploratory data analysis of the training set, one function per analysis.

Run with the Run button on this file, or:  python experiments/eda.py
Text tables go to build/eda/ (gitignored).

Structural overview (structural_overview): shapes and ids, constant and near-constant
columns, MICHD signal of the columns dropped for domain reasons (DOMAIN_DROP).

Feature types (feature_types_table): the type of each column in FEATURE_TYPES,
next to an automatic guess from its values, to review the choices made by hand.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# repository root on the import path, so that the Run button works
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import load_data  # noqa: E402
from src.feature_types import (  # noqa: E402
    BINARY,
    CONTINUOUS,
    COUNT,
    DOMAIN_DROP,
    DROP,
    FEATURE_TYPES,
    MIXED_UNITS,
    NOMINAL,
    ORDINAL,
    TYPES,
)
from src.preprocessing import codes_to_nan, special_code_rules  # noqa: E402

EDA_DIR = os.path.join(ROOT, "build", "eda")

# near-constant: one value (or NaN) in more than this share of the rows
NEAR_CONSTANT_SHARE = 0.99
# prevalence checks: one group per value up to this many values, deciles above
MAX_GROUP_VALUES = 60
# smaller groups are left out of the prevalence checks (too noisy)
MIN_GROUP_SIZE = 500
STATUS_ORDER = (
    "all NaN",
    "constant",
    "one value + NaN",
    "near-constant",
    "mostly NaN",
    "",
)

_HOUSEHOLD = "household size (living alone ~ age)"

# Candidates to drop that the overview does not decide: their relation to MICHD does
DOMAIN_UNDECIDED = {
    "_STATE": "geography (one-hot or drop: open team decision)",
    "NUMADULT": _HOUSEHOLD,
    "NUMMEN": _HOUSEHOLD,
    "NUMWOMEN": _HOUSEHOLD,
    "HHADULT": _HOUSEHOLD,
    "MSCODE": "metropolitan status (urban / rural)",
}

# automatic guess for ordinal or nominal: only the codebook tells them apart
CATEGORICAL = "categorical"


def fmt_value(v):
    """Short text for a column value: integers without decimals (1100, not 1100.0)."""
    if v == int(v):
        return str(int(v))
    return f"{v:.4g}"


def pct(v):
    """Fraction as a percentage with one decimal, "-" if undefined."""
    return "-" if np.isnan(v) else f"{100 * v:.1f}"


def save_lines(lines, filename):
    """Write lines to build/eda/filename and return the path."""
    os.makedirs(EDA_DIR, exist_ok=True)
    path = os.path.join(EDA_DIR, filename)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def column_profile(col, y01):
    """Summary of one column, with NaN counted as a value of its own.

    Returns:
        dict with n_values (distinct non-NaN values), nan_frac, top (most frequent
        value as text, "NaN" if NaN is the most frequent), top_share, and the MICHD
        prevalence of the rows with the top value (prev_top) and of the n_rest
        other rows (prev_rest)
    """
    nan = np.isnan(col)
    # NaN counted apart, so that n_values counts real answers only
    values, counts = np.unique(col[~nan], return_counts=True)
    if len(values) == 0 or nan.sum() >= counts.max():
        top, top_mask = "NaN", nan
    else:
        top_value = values[np.argmax(counts)]
        top, top_mask = fmt_value(top_value), col == top_value
    n_rest = int(len(col) - top_mask.sum())
    return dict(
        n_values=len(values),
        nan_frac=nan.mean(),
        top=top,
        top_share=top_mask.mean(),
        prev_top=y01[top_mask].mean(),
        prev_rest=y01[~top_mask].mean() if n_rest > 0 else np.nan,
        n_rest=n_rest,
    )


def constancy_status(profile):
    """Why a column is (almost) constant, one of STATUS_ORDER ("" if it is not)."""
    if profile["n_values"] == 0:
        return "all NaN"
    if profile["n_values"] == 1:
        return "constant" if profile["nan_frac"] == 0 else "one value + NaN"
    if profile["top_share"] > NEAR_CONSTANT_SHARE:
        return "mostly NaN" if profile["top"] == "NaN" else "near-constant"
    return ""


def prevalence_by_group(col, y01, n_bins=10):
    """MICHD prevalence per value of the column, or per decile when it has more than
    MAX_GROUP_VALUES distinct values, plus one group for NaN.

    Returns:
        list of (label, n_rows, prevalence), without the groups smaller than
        MIN_GROUP_SIZE
    """
    nan = np.isnan(col)
    values = np.unique(col[~nan])
    groups = []
    if len(values) <= MAX_GROUP_VALUES:
        for v in values:
            groups.append((fmt_value(v), col == v))
    else:
        # ties can merge quantiles: keep the distinct edges only
        edges = np.unique(np.quantile(col[~nan], np.linspace(0, 1, n_bins + 1)))
        # bin b holds edges[b] <= x < edges[b + 1]; the maximum goes in the last bin
        bins = np.searchsorted(edges, col, side="right") - 1
        bins = np.clip(bins, 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            label = f"{fmt_value(edges[b])}..{fmt_value(edges[b + 1])}"
            groups.append((label, ~nan & (bins == b)))
    groups.append(("NaN", nan))

    result = []
    for label, mask in groups:
        n = int(mask.sum())
        if n >= MIN_GROUP_SIZE:
            result.append((label, n, y01[mask].mean()))
    return result


def sanity_checks(x_train, test_shape, y_train, train_ids, test_ids):
    """Shapes, labels, class balance and ids, as text lines."""
    n_shared = np.intersect1d(train_ids, test_ids).size
    return [
        f"x_train: {x_train.shape[0]} rows x {x_train.shape[1]} columns, "
        f"x_test: {test_shape[0]} rows x {test_shape[1]} columns",
        f"labels: {np.unique(y_train)}, "
        f"positives (MICHD = 1): {np.mean(y_train == 1):.2%}",
        f"ids unique: train {np.unique(train_ids).size == train_ids.size}, "
        f"test {np.unique(test_ids).size == test_ids.size}; "
        f"ids in both train and test: {n_shared}",
    ]


def constant_columns(x_raw, x_clean, y01, names):
    """Table of the constant and near-constant columns, as text lines.

    The status is computed on the raw data and after codes_to_nan: a column can
    become constant only once its special codes are NaN. The MICHD prevalence of
    the top value and of the other rows shows if the rare part carries signal.
    """
    rows = []
    for j, name in enumerate(names):
        profile = column_profile(x_clean[:, j], y01)
        status = constancy_status(profile)
        raw_status = constancy_status(column_profile(x_raw[:, j], y01))
        if status or raw_status:
            rows.append((name, status, raw_status, profile))
    rows.sort(key=lambda r: (STATUS_ORDER.index(r[1]), -r[3]["top_share"], r[0]))

    counts = [sum(r[1] == s for r in rows) for s in STATUS_ORDER[:-1]]
    not_listed = [r[0] for r in rows if r[1] and r[0] not in DOMAIN_DROP]
    lines = [
        "",
        "== Constant and near-constant columns ==",
        f"(near-constant: one value or NaN in more than {NEAR_CONSTANT_SHARE:.0%}"
        " of the rows; prevalences in %)",
        "after codes_to_nan: "
        + ", ".join(f"{c} {s}" for c, s in zip(counts, STATUS_ORDER[:-1])),
        f"flagged and not in DOMAIN_DROP: {len(not_listed)}",
        "",
        f"{'column':10s} {'status':16s} {'raw status':16s} {'domain':6s} "
        f"{'NaN%':>6s} {'values':>6s} {'top':>8s} {'top%':>6s} "
        f"{'prev top':>8s} {'prev rest':>9s} {'n rest':>7s}",
    ]
    for name, status, raw_status, p in rows:
        if name in DOMAIN_DROP:
            domain = "drop"
        elif name in DOMAIN_UNDECIDED:
            domain = "check"
        else:
            domain = ""
        lines.append(
            f"{name:10s} {status or '-':16s} "
            f"{'same' if raw_status == status else raw_status or '-':16s} "
            f"{domain:6s} {pct(p['nan_frac']):>6s} {p['n_values']:6d} "
            f"{p['top']:>8s} {pct(p['top_share']):>6s} {pct(p['prev_top']):>8s} "
            f"{pct(p['prev_rest']):>9s} {p['n_rest']:7d}"
        )
    return lines


def group_summary(name, col, y01, show_groups):
    """One line with the spread of the MICHD prevalence across the groups of a column
    (prevalence_by_group), plus one line per group if show_groups."""
    groups = prevalence_by_group(col, y01)
    head = f"  {name:10s} NaN {pct(np.isnan(col).mean()):>5s}%"
    if len(groups) < 2:
        return [f"{head}  fewer than 2 groups with >= {MIN_GROUP_SIZE} rows"]
    lo = min(groups, key=lambda g: g[2])
    hi = max(groups, key=lambda g: g[2])
    lines = [
        f"{head}  {len(groups):2d} groups  spread {pct(hi[2] - lo[2]):>5s} pts  "
        f"min {pct(lo[2])}% ({lo[0]}, n={lo[1]})  "
        f"max {pct(hi[2])}% ({hi[0]}, n={hi[1]})"
    ]
    if show_groups:
        lines += [f"      {g[0]:>12s}: {pct(g[2]):>5s}%  n={g[1]}" for g in groups]
    return lines


def domain_columns(x_clean, y01, names):
    """MICHD prevalence by value (or decile) of the domain-drop columns and of the
    undecided ones, as text lines: how much univariate signal dropping them loses.

    A large spread is not by itself a reason to keep a column: survey weights, for
    example, are built from age, sex and race, which stay as features anyway.
    """
    index = {name: j for j, name in enumerate(names)}
    lines = [
        "",
        "== Columns dropped for domain reasons ==",
        f"(MICHD prevalence per value, or per decile above {MAX_GROUP_VALUES} values,"
        f" groups with >= {MIN_GROUP_SIZE} rows; spread = max - min, in points)",
    ]
    for title, table, show_groups in (
        ("dropped (src/feature_types.DOMAIN_DROP)", DOMAIN_DROP, False),
        ("undecided (DOMAIN_UNDECIDED in this file)", DOMAIN_UNDECIDED, True),
    ):
        lines += ["", f"--- {title} ---"]
        # distinct reasons, in order of appearance
        for reason in dict.fromkeys(table.values()):
            lines.append(f"[{reason}]")
            for name in [n for n in table if table[n] == reason]:
                col = x_clean[:, index[name]]
                lines += group_summary(name, col, y01, show_groups)
    return lines


def structural_overview(
    x_raw, x_clean, y_train, test_shape, train_ids, test_ids, names
):
    """Shapes and ids, constant columns, domain columns: saves build/eda/overview.txt.

    Args:
        x_raw: raw training data, shape=(N, D)
        x_clean: x_raw after codes_to_nan
        y_train: labels in {-1, 1}, shape=(N,)
        test_shape: shape of x_test
        train_ids, test_ids: ids returned by load_csv_data
        names: the D column names
    """
    names = list(names)
    unknown = [n for n in list(DOMAIN_DROP) + list(DOMAIN_UNDECIDED) if n not in names]
    if unknown:
        raise ValueError(f"column names not in the data: {unknown}")

    y01 = (y_train == 1).astype(float)
    sanity = sanity_checks(x_raw, test_shape, y_train, train_ids, test_ids)
    lines = ["Structural overview of the training set (experiments/eda.py)", ""]
    lines += sanity
    lines += constant_columns(x_raw, x_clean, y01, names)
    lines += domain_columns(x_clean, y01, names)
    path = save_lines(lines, "overview.txt")

    print("\n".join(sanity))
    print(f"full tables: {path}")


def propose_type(name, values, rule):
    """Automatic guess of the type of a column, without the codebook.

    Args:
        name: column name
        values: distinct non-NaN values of the column after codes_to_nan
        rule: (missing codes, none codes) of the column, from special_code_rules

    Returns:
        one of TYPES, or CATEGORICAL for "ordinal or nominal"
    """
    missing, zero = rule
    if name in DOMAIN_DROP:
        return DROP
    # 777 / 7777 / 777777 = "don't know" of the 3-, 4- and 6-digit coded answers
    if missing & {777, 7777, 777777}:
        return MIXED_UNITS
    if 88 in zero:  # "number of days / times", 88 = none
        return COUNT
    if len(values) <= 2:
        return BINARY
    if np.any(values != np.round(values)) or len(values) > MAX_GROUP_VALUES:
        return CONTINUOUS
    return CATEGORICAL


def types_agree(auto, final):
    """True if the automatic guess is compatible with the type chosen by hand."""
    return auto == final or (auto == CATEGORICAL and final in (ORDINAL, NOMINAL))


def value_examples(values, n_show=8):
    """Distinct values as short text: all of them, or the first ones and the last."""
    if len(values) <= n_show:
        return ", ".join(fmt_value(v) for v in values)
    first = ", ".join(fmt_value(v) for v in values[: n_show - 2])
    return f"{first} ... {fmt_value(values[-1])}"


def feature_types_table(x_clean, names):
    """Type of each column (FEATURE_TYPES) next to the automatic guess (propose_type),
    with the values seen after codes_to_nan. Saves build/eda/feature_types.txt.

    Uses no labels: the types are fixed lists that do not bias cross-validation.
    """
    names = list(names)
    if list(FEATURE_TYPES) != names:
        raise ValueError("FEATURE_TYPES must list the data columns in their order")

    rules = special_code_rules(names)
    rows = []
    for j, name in enumerate(names):
        col = x_clean[:, j]
        values = np.unique(col[~np.isnan(col)])
        final = FEATURE_TYPES[name]
        auto = propose_type(name, values, rules[j])
        rows.append((name, final, auto, np.isnan(col).mean(), values))

    n_differ = sum(not types_agree(auto, final) for _, final, auto, _, _ in rows)
    counts = ", ".join(f"{sum(r[1] == t for r in rows)} {t}" for t in TYPES)
    lines = [
        "Feature types (src/feature_types.FEATURE_TYPES, experiments/eda.py)",
        f"(auto: guess from the values and the special codes, {CATEGORICAL} ="
        " ordinal or nominal; * = auto differs from the type chosen by hand)",
        f"types: {counts}",
        f"auto differs: {n_differ}",
        "",
        f"{'column':10s} {'type':12s} {'auto':12s}   {'NaN%':>5s} {'values':>6s}"
        "  examples (after codes_to_nan)",
    ]
    for name, final, auto, nan_frac, values in rows:
        flag = " " if types_agree(auto, final) else "*"
        lines.append(
            f"{name:10s} {final:12s} {auto:12s} {flag} {pct(nan_frac):>5s} "
            f"{len(values):6d}  {value_examples(values)}"
        )
    path = save_lines(lines, "feature_types.txt")
    print(f"types: {counts}")
    print(f"auto differs from the hand-made types: {n_differ}, full table: {path}")


def main():
    x_train, x_test, y_train, train_ids, test_ids, names = load_data()
    test_shape = x_test.shape
    del x_test  # only its shape is needed here
    x_clean = codes_to_nan(x_train, names)
    structural_overview(
        x_train, x_clean, y_train, test_shape, train_ids, test_ids, names
    )
    feature_types_table(x_clean, names)


if __name__ == "__main__":
    main()
