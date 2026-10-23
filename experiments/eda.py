"""Exploratory data analysis of the training set, one function per analysis.

Run with the Run button on this file, or:  python experiments/eda.py
Text tables go to build/eda/, exploratory figures to build/eda/figures/ (both
gitignored), two summary figures (A4, A6) to report/figures/ (tracked in git, not
used by the final report).
The analyses are named A1-A9 here and in the comments of src/; the pipeline steps
they motivate are B1-B13 (listed in src/pipeline.py).

Structural overview (structural_overview, A1): shapes and ids, constant and
near-constant columns, MICHD signal of the columns dropped for domain reasons
(DOMAIN_DROP).

Feature types (feature_types_table, A2): the type of each column in FEATURE_TYPES,
next to an automatic guess from its values, to review the choices made by hand.

Special codes (special_codes_audit, A3): what codes_to_nan changes in each column,
and the code-like values it keeps (real answers that need care, or missing rules).

Missing values (missingness_analysis, A4): how many, label leakage scan, groups of
columns missing together, MICHD prevalence when a value is missing.

A5-A9 look at the values the model sees: after codes_to_nan and the label-free steps
of the pipeline (convert_units, fill_skip_logic), on the feature columns only
(DOMAIN_DROP and the raw columns of CONVERTED left out).

Target (target_overview, A5): class balance, F1 of trivial rules, MICHD prevalence by
age, sex and state (raw and age-adjusted).

Feature-target relation (target_relation, A6): mutual information, |Pearson r|,
shape of the log-odds per value / decile, proposed encoding of each column.

Redundancy (redundancy, A7): pairs of columns where one determines the other,
proposed list of duplicates to drop.

Train vs test (train_test_shift, A8): distance between the train and test
distributions of each column, next to the distance between two random train parts.

Ranges and outliers (ranges_and_outliers, A9): percentiles, implausible values, and
whether clipping or log1p makes a numeric column more linear in the target.

Run only some analyses with:  python experiments/eda.py A6 A7
"""

import os
import sys
import textwrap

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# repository root on the import path, so that the Run button works
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import load_data  # noqa: E402
from src.feature_types import (  # noqa: E402
    BINARY,
    BINNED,
    CONTINUOUS,
    CONVERTED,
    COUNT,
    DOMAIN_DROP,
    DROP,
    FEATURE_TYPES,
    MIXED_UNITS,
    NOMINAL,
    ONE_HOT_ORDINAL,
    ORDINAL,
    REDUNDANT,
    TYPES,
)
from src.plots import (  # noqa: E402
    plot_corr_heatmap,
    plot_dependence_grid,
    plot_mi_vs_corr,
    plot_missing_and_balance,
    plot_ranking,
)
from src.preprocessing import (  # noqa: E402
    codes_to_nan,
    convert_units,
    fill_skip_logic,
    special_code_rules,
)

STEPS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9")
# text tables (gitignored)
EDA_DIR = os.path.join(ROOT, "build", "eda")
# exploratory figures of A6-A7 (gitignored)
EDA_FIGURES_DIR = os.path.join(EDA_DIR, "figures")
# two summary figures (A4, A6), tracked in git; the final report does not use them
FIGURES_DIR = os.path.join(ROOT, "report", "figures")

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

# values that are often BRFSS codes (don't know, refused, none, not applicable)
SUSPICIOUS_CODES = {7, 8, 9, 77, 87, 88, 97, 98, 99, 555, 777, 888, 999, 7777, 9999}

# columns missing together: correlation of their NaN masks above this
MISSING_CORR = 0.9
# leakage scan: a value (or NaN) of at least LEAK_MIN_SIZE rows with a MICHD
# prevalence of LEAK_PREVALENCE or more (the base rate is 8.8%)
LEAK_MIN_SIZE = 30
LEAK_PREVALENCE = 0.5
# upper bounds of the NaN-fraction classes in the missingness summary
NAN_CLASSES = (
    (0.0, "0"),
    (0.05, "<=5%"),
    (0.5, "5-50%"),
    (0.9, "50-90%"),
    (0.99, "90-99%"),
    (1.0, ">99%"),
)

# A5: _AGEG5YR codes 1..13
AGE_GROUPS = (
    "18-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80+",
)
# A6: a line in the column value is enough when it explains this share of the
# variance of the MICHD log-odds across values / deciles (weighted R^2) ...
LINEAR_R2 = 0.8
# ... and the column is curved when a parabola adds at least this much R^2
CURVATURE_GAIN = 0.1
# A6: no signal when the spread of the log-odds is below this many times the one
# expected from sampling noise alone (chi^2 / (groups - 1))
WEAK_SIGNAL = 3.0
# A6: number of features in the figures
TOP_FEATURES = 24
# A6: summary figure, the REPORT_FEATURES columns with the highest MI, skipping a
# column with |Pearson r| > REPORT_MAX_CORR with one already chosen (age variants)
REPORT_FEATURES = 16
REPORT_MAX_CORR = 0.7
# A6, A7: a nominal column with more values (with enough rows) than this may not
# be one-hot encoded as it is (B9): group or drop its rare values
MAX_ONE_HOT = 10
# A7: rows sampled for the pairwise statistics (correlations within +-0.01)
REDUNDANCY_ROWS = 100_000
# A7: quantile bins of the continuous columns, fine enough to see a recoding
REDUNDANCY_BINS = 100
# A7: a determines b when U(b|a) = MI(a, b) / H(b) >= DETERMINED; a and b are the
# same quantity (a linear function of each other, e.g. a unit change) when
# |Pearson r| >= SAME_QUANTITY; pairs above REVIEW (either measure) are listed
DETERMINED = 0.95
SAME_QUANTITY = 0.999
REVIEW = 0.9
# A7: greedy order, blocks of missing fraction (finer columns first inside a
# block); a column is a duplicate only if its missing fraction differs from the
# kept one by at most NAN_TOLERANCE (else its NaN pattern carries information,
# e.g. _HCVU651 = HLTHPLN1 for age 18-64 only)
NAN_BLOCK = 0.05
NAN_TOLERANCE = 0.02
# A9: clipping percentiles to try (a transform that is not a pipeline step)
CLIP_PERCENTILES = (0.5, 99.5)
# A9: clipping or log1p helps when |r| grows by at least this factor and by at
# least R_MIN_GAIN (the standard error of r is ~0.002 with 300k rows)
R_GAIN = 1.1
R_MIN_GAIN = 0.005
# A9: plausible ranges (units as in the data), to count impossible answers
WEEK_MINUTES = 7 * 24 * 60
PLAUSIBLE = {
    "_BMI5": (12, 80),
    "HTM4": (0.9, 2.4),  # meters
    "HTIN4": (36, 95),  # inches
    "WTKG3": (25, 300),  # kg
    "CHILDREN": (0, 20),
    "NUMADULT": (1, 20),
    "HHADULT": (1, 20),
    "PADUR1_": (0, 24 * 60),  # minutes per session
    "PADUR2_": (0, 24 * 60),
    "_MINAC11": (0, WEEK_MINUTES),  # minutes per week
    "_MINAC21": (0, WEEK_MINUTES),
    "PAFREQ1_": (0, 7 * 24),  # times per week: at most once an hour
    "PAFREQ2_": (0, 7 * 24),
    "STRFREQ_": (0, 7 * 24),
}


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


def figure_path(directory, filename):
    """Path of a figure in directory (EDA_FIGURES_DIR or FIGURES_DIR), created if
    needed."""
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


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


def prevalence_by_group(col, y01, n_bins=10, min_size=MIN_GROUP_SIZE):
    """MICHD prevalence per value of the column, or per decile when it has more than
    MAX_GROUP_VALUES distinct values, plus one group for NaN.

    Returns:
        list of (label, n_rows, prevalence), without the groups smaller than
        min_size
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
        if n >= min_size:
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


def special_codes_audit(x_raw, x_clean, y01, names):
    """Check of codes_to_nan, column by column. Saves build/eda/special_codes.txt.

    First table: how many answers codes_to_nan turns into NaN and into 0.
    Second table: values it keeps that are usual BRFSS codes (SUSPICIOUS_CODES) and
    are not next to the other answers (e.g. 8 after 1..4). Each one is either a real
    answer that needs care (an ordinal to reorder) or a rule missing in _RULES.
    The MICHD prevalence is shown to judge them; it is not used by the pipeline.
    """
    names = list(names)
    rules = special_code_rules(names)
    changed, kept = [], []
    for j, name in enumerate(names):
        if FEATURE_TYPES[name] == DROP:
            continue
        raw, clean = x_raw[:, j], x_clean[:, j]
        answered = ~np.isnan(raw)
        n_nan = int(np.sum(answered & np.isnan(clean)))
        n_zero = int(np.sum(answered & (raw != 0) & (clean == 0)))
        missing, zero = rules[j]
        rule = f"NaN {sorted(missing)}, 0 {sorted(zero)}"
        if n_nan or n_zero:
            changed.append(
                f"{name:10s} {FEATURE_TYPES[name]:12s} {n_nan:8d} {n_zero:8d}  {rule}"
            )

        values = np.unique(clean[~np.isnan(clean)])
        n_answered = int(np.sum(~np.isnan(clean)))
        for k in range(1, len(values)):
            v = values[k]
            # a code is far from the previous answer: 8 after 1..4, 98 after 1..76
            if v not in SUSPICIOUS_CODES or v - values[k - 1] <= 1:
                continue
            mask = clean == v
            others = ~np.isnan(clean) & ~mask
            kept.append(
                f"{name:10s} {FEATURE_TYPES[name]:12s} {fmt_value(v):>6s} "
                f"{int(mask.sum()):7d} {pct(mask.sum() / n_answered):>6s} "
                f"{pct(y01[mask].mean()):>6s} {pct(y01[others].mean()):>6s}  {rule}"
            )

    lines = [
        "Special codes: what codes_to_nan does (experiments/eda.py)",
        "",
        f"== Answers changed by codes_to_nan ({len(changed)} columns) ==",
        f"{'column':10s} {'type':12s} {'->NaN':>8s} {'->0':>8s}  rule",
        *changed,
        "",
        f"== Code-like values kept ({len(kept)}): real answers or missing rules ==",
        "(% ans: share of the answered rows; MICHD %: prevalence for the value and"
        " for the other answered rows)",
        f"{'column':10s} {'type':12s} {'value':>6s} {'n':>7s} {'% ans':>6s} "
        f"{'MICHD':>6s} {'others':>6s}  rule",
        *kept,
    ]
    path = save_lines(lines, "special_codes.txt")
    print(f"codes_to_nan changes {len(changed)} columns; {len(kept)} code-like values")
    print(f"kept, to review: {path}")


def count_nan_classes(nan_frac):
    """Number of columns in each class of NAN_CLASSES."""
    counts, low = [], -1.0
    for high, _ in NAN_CLASSES:
        counts.append(int(np.sum((nan_frac > low) & (nan_frac <= high))))
        low = high
    return counts


def connected_groups(linked):
    """Connected components of a symmetric boolean matrix linked, shape (D, D).

    Returns:
        list of groups (sorted lists of indices), largest first
    """
    seen = np.zeros(len(linked), dtype=bool)
    groups = []
    for start in range(len(linked)):
        if seen[start]:
            continue
        seen[start] = True
        stack, group = [start], []
        while stack:
            i = stack.pop()
            group.append(i)
            for k in np.flatnonzero(linked[i] & ~seen):
                seen[k] = True
                stack.append(k)
        groups.append(sorted(group))
    return sorted(groups, key=len, reverse=True)


def missing_together(x_clean, cols):
    """Groups of columns that are missing together (skip logic, optional modules):
    two columns are linked when the correlation of their NaN masks is above
    MISSING_CORR. cols: column indices with both NaN and answered rows.

    Returns:
        list of groups of indices into cols, largest first
    """
    # one float32 column per mask: ~330 MB instead of a float64 copy of x
    masks = np.empty((len(x_clean), len(cols)), dtype=np.float32)
    for k, j in enumerate(cols):
        masks[:, k] = np.isnan(x_clean[:, j])
    p = masks.mean(axis=0)
    cov = masks.T @ masks / len(masks) - np.outer(p, p)
    std = np.sqrt(p * (1 - p))
    corr = cov / np.outer(std, std)
    return connected_groups(corr > MISSING_CORR)


def leakage_scan(x_clean, y01, names):
    """Values (or NaN) of any column, dropped ones included, with a MICHD
    prevalence of LEAK_PREVALENCE or more on at least LEAK_MIN_SIZE rows."""
    lines = []
    for j, name in enumerate(names):
        col = x_clean[:, j]
        for label, n, prev in prevalence_by_group(col, y01, min_size=LEAK_MIN_SIZE):
            if prev >= LEAK_PREVALENCE:
                lines.append(
                    f"{name:10s} {FEATURE_TYPES[name]:12s} {label:>12s} {n:7d} "
                    f"{pct(prev):>6s}"
                )
    return lines


def missingness_analysis(x_raw, x_clean, y01, names):
    """Missing values of the feature columns (DOMAIN_DROP excluded): how many, label
    leakage scan, groups of columns missing together, and MICHD prevalence when a
    value is missing. Saves build/eda/missingness.txt and the figure
    report/figures/missing_balance.png.

    Uses the labels: if is_missing indicators are chosen from this table, the choice
    must be made inside fit_preprocess, on the training part of each fold.
    """
    names = list(names)
    feats = [j for j, name in enumerate(names) if FEATURE_TYPES[name] != DROP]
    frac_raw = np.array([np.isnan(x_raw[:, j]).mean() for j in feats])
    # number of NaN per column after codes_to_nan, by column index
    n_nan = {j: int(np.isnan(x_clean[:, j]).sum()) for j in feats}
    frac = np.array([n_nan[j] for j in feats]) / len(y01)
    header = "  ".join(f"{label:>6s}" for _, label in NAN_CLASSES)
    raw_counts = "  ".join(f"{c:6d}" for c in count_nan_classes(frac_raw))
    clean_counts = "  ".join(f"{c:6d}" for c in count_nan_classes(frac))
    lines = [
        "Missing values (experiments/eda.py)",
        "",
        f"== NaN per column ({len(feats)} feature columns, DOMAIN_DROP excluded) ==",
        f"{'NaN fraction':24s} {header}",
        f"{'raw':24s} {raw_counts}",
        f"{'after codes_to_nan':24s} {clean_counts}",
        f"mean NaN fraction: raw {pct(frac_raw.mean())}%, "
        f"after codes_to_nan {pct(frac.mean())}%",
    ]

    leaks = leakage_scan(x_clean, y01, names)
    lines += [
        "",
        f"== Leakage scan: MICHD >= {LEAK_PREVALENCE:.0%} on >= {LEAK_MIN_SIZE} rows"
        " (all columns, dropped ones included) ==",
        f"{'column':10s} {'type':12s} {'value':>12s} {'n':>7s} {'MICHD':>6s}",
        *leaks,
    ]

    # columns with enough rows both missing and answered
    n_rows = len(y01)
    cols = [
        j
        for j in feats
        if n_nan[j] >= MIN_GROUP_SIZE and n_rows - n_nan[j] >= MIN_GROUP_SIZE
    ]
    groups = [g for g in missing_together(x_clean, cols) if len(g) > 1]
    lines += [
        "",
        "== Groups of columns missing together (NaN-mask correlation >"
        f" {MISSING_CORR}, {len(cols)} columns with >= {MIN_GROUP_SIZE} NaN"
        " and answered rows) ==",
        "(MICHD %: prevalence when the first column is missing / answered)",
    ]
    for g in groups:
        # members from the least to the most missing
        members = sorted((cols[k] for k in g), key=lambda j: n_nan[j])
        first = np.isnan(x_clean[:, members[0]])
        low, high = n_nan[members[0]] / n_rows, n_nan[members[-1]] / n_rows
        lines.append(
            f"{len(members)} columns, NaN {pct(low)}-{pct(high)}%, "
            f"MICHD {pct(y01[first].mean())}% missing / "
            f"{pct(y01[~first].mean())}% answered"
        )
        text = ", ".join(names[j] for j in members)
        lines.append(textwrap.indent(textwrap.fill(text, 84), "    "))

    rows = []
    for j in cols:
        nan = np.isnan(x_clean[:, j])
        p_nan, p_ans = y01[nan].mean(), y01[~nan].mean()
        rows.append((abs(p_nan - p_ans), names[j], nan.mean(), p_nan, p_ans))
    rows.sort(reverse=True)
    lines += [
        "",
        f"== MICHD prevalence when missing vs answered ({len(rows)} columns,"
        " sorted by the absolute difference) ==",
        f"{'column':10s} {'type':12s} {'NaN%':>6s} {'missing':>8s} {'answered':>8s}"
        f" {'diff':>6s}",
    ]
    for _, name, nan_frac, p_nan, p_ans in rows:
        lines.append(
            f"{name:10s} {FEATURE_TYPES[name]:12s} {pct(nan_frac):>6s} "
            f"{pct(p_nan):>8s} {pct(p_ans):>8s} {100 * (p_nan - p_ans):+6.1f}"
        )

    row_nan = np.zeros(len(y01))
    for j in feats:
        row_nan += np.isnan(x_clean[:, j])
    r = np.corrcoef(row_nan, y01)[0, 1]
    lines += [
        "",
        f"== NaN per person (over the {len(feats)} feature columns) ==",
        f"min {int(row_nan.min())}, median {int(np.median(row_nan))},"
        f" max {int(row_nan.max())}; correlation with MICHD r = {r:.3f}",
        "MICHD prevalence by number of NaN (deciles when it has many values):",
    ]
    for label, n, prev in prevalence_by_group(row_nan, y01):
        lines.append(f"    {label:>12s}: {pct(prev):>5s}%  n={n}")

    path = save_lines(lines, "missingness.txt")
    figure = figure_path(FIGURES_DIR, "missing_balance.png")
    plot_missing_and_balance(frac, y01, figure)
    print(f"leakage scan: {len(leaks)} values; {len(groups)} groups missing together")
    print(f"tables: {path}, figure: {figure}")


def feature_columns(names):
    """Indices of the columns the pipeline can use as features: DOMAIN_DROP and the
    raw columns that have a converted version (CONVERTED) left out."""
    return [
        j
        for j, name in enumerate(names)
        if FEATURE_TYPES[name] != DROP and name not in CONVERTED
    ]


def entropy_bits(counts):
    """Entropy in bits of the distribution given by counts (zeros allowed)."""
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def mutual_information(a, b):
    """Mutual information in bits between two arrays of integer codes,
    MI = H(a) + H(b) - H(a, b). A NaN code (-1) counts as a value."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    joint = np.bincount(a * (b.max() + 1) + b)
    h_a, h_b = entropy_bits(np.bincount(a)), entropy_bits(np.bincount(b))
    return h_a + h_b - entropy_bits(joint)


def pearson(a, b):
    """Pearson correlation of two 1D arrays without NaN, NaN if one is constant."""
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def discretize(col, ftype, n_bins=10):
    """Integer codes of a column for counting, NaN = -1: one code per value for the
    binary, ordinal and nominal columns and for those with at most MAX_GROUP_VALUES
    values, one code per quantile bin (n_bins) for the others.

    Returns:
        codes: shape=(N,), 0..K-1, or -1 for NaN
        positions: shape=(K,), the value of each code (mean value for a bin)
        n_values: number of distinct non-NaN values
    """
    answered = ~np.isnan(col)
    v = col[answered]
    codes = np.full(len(col), -1)
    values = np.unique(v)
    if ftype in (BINARY, ORDINAL, NOMINAL) or len(values) <= MAX_GROUP_VALUES:
        codes[answered] = np.searchsorted(values, v)
        return codes, values, len(values)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
    # renumber the non-empty bins 0..K-1
    _, inverse = np.unique(bins, return_inverse=True)
    codes[answered] = inverse
    positions = np.bincount(inverse, weights=v) / np.bincount(inverse)
    return codes, positions, len(values)


def weighted_r2(x, y, w, degree):
    """Weighted R^2 of a polynomial fit of y on x (weights w)."""
    coef = np.polyfit(x, y, degree, w=np.sqrt(w))
    total = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    return 1 - np.sum(w * (y - np.polyval(coef, x)) ** 2) / total


def log_odds_fit(codes, positions, y01):
    """Shape of the MICHD log-odds across the values / bins of a column (codes from
    discretize), using the groups with >= MIN_GROUP_SIZE rows and both classes.

    Returns:
        r2_line, r2_parabola: weighted R^2 (weights = inverse variance of the
            log-odds) of a line / parabola in the value; NaN with < 3 / < 4 groups
        signal: weighted spread of the log-odds divided by (groups - 1); about 1
            when the column carries no signal (chi^2 test of equal prevalences)
        n_groups: number of groups used
    """
    answered = codes >= 0
    n = np.bincount(codes[answered], minlength=len(positions))
    pos = np.bincount(codes[answered], weights=y01[answered], minlength=len(positions))
    ok = (n >= MIN_GROUP_SIZE) & (pos > 0) & (pos < n)
    n_groups = int(ok.sum())
    if n_groups < 3:
        return np.nan, np.nan, np.nan, n_groups
    p = pos[ok] / n[ok]
    lo = np.log(p / (1 - p))
    w = n[ok] * p * (1 - p)
    x = positions[ok]
    signal = np.sum(w * (lo - np.average(lo, weights=w)) ** 2) / (n_groups - 1)
    r2_parabola = weighted_r2(x, lo, w, 2) if n_groups >= 4 else np.nan
    return weighted_r2(x, lo, w, 1), r2_parabola, signal, n_groups


def best_rule_f1(codes, y01):
    """Best F1 of a rule on one column: predict MICHD for its values (NaN included)
    with the highest prevalence. Measured on the same rows: slightly optimistic."""
    _, inverse = np.unique(codes, return_inverse=True)
    n = np.bincount(inverse)
    pos = np.bincount(inverse, weights=y01)
    order = np.argsort(-pos / n)
    tp = np.cumsum(pos[order])
    predicted = np.cumsum(n[order])
    return float(np.max(2 * tp / (predicted + y01.sum())))


def target_overview(x, y01, names):
    """A5: class balance, F1 of trivial rules, MICHD prevalence by age group and sex,
    and by state, raw and age-adjusted. Saves build/eda/target.txt.

    Age-adjusted prevalence of a state: its prevalence in each age group, averaged
    with the age distribution of the whole training set. If the spread between
    states shrinks after the adjustment, part of it is only a difference in age.
    """
    col = {name: j for j, name in enumerate(names)}
    p = y01.mean()
    age = x[:, col["_AGEG5YR"]]
    sex = x[:, col["SEX"]]
    state = x[:, col["_STATE"]]
    age_codes, _, _ = discretize(age, ORDINAL)
    sex_codes, _, _ = discretize(sex, BINARY)
    lines = [
        "Target: MICHD (experiments/eda.py)",
        "",
        f"rows {len(y01)}, MICHD {int(y01.sum())} ({pct(p)}%)",
        "F1 of trivial rules (the model must beat them):",
        f"  everyone MICHD: {2 * p / (1 + p):.3f}",
        f"  random, MICHD with probability {p:.3f}: {p:.3f}",
        f"  best rule on the age group: {best_rule_f1(age_codes, y01):.3f}",
        "  best rule on age group x sex: "
        f"{best_rule_f1(age_codes * 3 + sex_codes, y01):.3f}",
        "",
        "== MICHD % by age group (_AGEG5YR) and sex (SEX 1 = male, 2 = female) ==",
        f"{'age':>6s} {'n':>7s} {'all':>6s} {'male':>6s} {'female':>6s}",
    ]
    for k, label in enumerate(AGE_GROUPS):
        rows = age == k + 1
        male, female = rows & (sex == 1), rows & (sex == 2)
        lines.append(
            f"{label:>6s} {int(rows.sum()):7d} {pct(y01[rows].mean()):>6s} "
            f"{pct(y01[male].mean()):>6s} {pct(y01[female].mean()):>6s}"
        )
    lines.append(
        f"{'all':>6s} {len(y01):7d} {pct(p):>6s} {pct(y01[sex == 1].mean()):>6s} "
        f"{pct(y01[sex == 2].mean()):>6s}"
    )

    # cells state x age group: counts and positives
    known = ~np.isnan(age)
    states = np.unique(state)
    s_idx = np.searchsorted(states, state[known])
    a_idx = age[known].astype(int) - 1
    cell = s_idx * len(AGE_GROUPS) + a_idx
    size = len(states) * len(AGE_GROUPS)
    n = np.bincount(cell, minlength=size).reshape(len(states), -1)
    pos = np.bincount(cell, weights=y01[known], minlength=size).reshape(n.shape)
    weights = n.sum(axis=0) / n.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = pos / n
    present = n > 0
    adjusted = np.nansum(rate * weights, axis=1) / (present * weights).sum(axis=1)
    raw = pos.sum(axis=1) / n.sum(axis=1)
    order = np.argsort(raw)
    lines += [
        "",
        f"== MICHD % by state (_STATE FIPS code, {len(states)} states; rows with a"
        " known age group) ==",
        f"raw: {pct(raw.min())}-{pct(raw.max())}%, std {pct(raw.std())} pts; "
        f"age-adjusted: {pct(adjusted.min())}-{pct(adjusted.max())}%, "
        f"std {pct(adjusted.std())} pts",
        f"{'FIPS':>5s} {'n':>6s} {'raw':>6s} {'age-adj':>7s}",
    ]
    for k in order:
        lines.append(
            f"{fmt_value(states[k]):>5s} {int(n[k].sum()):6d} {pct(raw[k]):>6s} "
            f"{pct(adjusted[k]):>7s}"
        )
    path = save_lines(lines, "target.txt")
    print("\n".join(lines[3:9]))
    print(f"table: {path}")


def feature_stats(x, y01, names, feats):
    """A6: univariate statistics of each feature column with MICHD.

    Returns:
        dict name -> dict with type, nan_frac, n_values, mi (bits, NaN counted as a
        value), r (Pearson on the answered rows), r2_line, r2_parabola, signal,
        n_groups (from log_odds_fit), f1 (best_rule_f1) and kind ("numeric" when
        the column was cut in deciles, else its type)
    """
    y_int = y01.astype(int)
    stats = {}
    for j in feats:
        name, col = names[j], x[:, j]
        ftype = FEATURE_TYPES[name]
        codes, positions, n_values = discretize(col, ftype)
        answered = codes >= 0
        r2_line, r2_parabola, signal, n_groups = log_odds_fit(codes, positions, y01)
        stats[name] = dict(
            type=ftype,
            nan_frac=1 - answered.mean(),
            n_values=n_values,
            mi=mutual_information(codes, y_int),
            r=pearson(col[answered], y01[answered]),
            r2_line=r2_line,
            r2_parabola=r2_parabola,
            signal=signal,
            n_groups=n_groups,
            f1=best_rule_f1(codes, y01),
            kind=ftype if len(positions) == n_values else "numeric",
        )
    return stats


def encoding_hint(s):
    """Proposed encoding of a column, from its type and the shape of its log-odds."""
    if s["type"] == BINARY:
        return "as is"
    if s["type"] == NOMINAL:
        k = s["n_groups"]
        if k < 2:
            return f"rare: < 2 values with >= {MIN_GROUP_SIZE} rows"
        if k > MAX_ONE_HOT:
            return f"one-hot, many values ({k} with >= {MIN_GROUP_SIZE} rows)"
        return f"one-hot ({k} values with >= {MIN_GROUP_SIZE} rows)"
    if np.isnan(s["r2_line"]):
        return "as is (< 3 values with enough rows)"
    if s["signal"] < WEAK_SIGNAL:
        return "weak signal"
    if s["type"] == ORDINAL:
        return "numeric" if s["r2_line"] >= LINEAR_R2 else "one-hot: not monotone"
    if s["r2_parabola"] - s["r2_line"] >= CURVATURE_GAIN:
        return "curved: log / polynomial"
    return "linear" if s["r2_line"] >= LINEAR_R2 else "not linear: bins / one-hot"


def target_relation(x, y01, names, feats, stats):
    """A6: table of the univariate relation of each feature with MICHD, sorted by
    mutual information, with the proposed encoding (encoding_hint). Saves
    build/eda/target_relation.txt and three figures in build/eda/figures/: MI ranking, MI vs
    |r|, log-odds per value / decile of the TOP_FEATURES columns with the highest MI.
    The summary figure report/figures/log_odds.png is the same log-odds grid for
    REPORT_FEATURES columns, without near-duplicates (distinct_top).

    The proposals use the labels of the whole training set: fixed lists chosen from
    them make the cross-validation score slightly optimistic (say so in the report).
    """
    ranked = sorted(stats, key=lambda name: -stats[name]["mi"])
    hints = {name: encoding_hint(stats[name]) for name in ranked}
    lines = [
        "Feature-target relation (experiments/eda.py)",
        f"({len(ranked)} feature columns, after codes_to_nan, convert_units and"
        " fill_skip_logic)",
        "MI: mutual information in millibits, NaN counted as a value; |r|: Pearson on"
        " the answered rows;",
        "R2 line / parab: weighted R^2 of a line / parabola through the log-odds per"
        f" value or decile (groups >= {MIN_GROUP_SIZE} rows);",
        "signal: spread of the log-odds / spread expected from noise (~1 = no signal);"
        " F1: best one-column rule",
        "",
        f"{'column':10s} {'type':11s} {'NaN%':>5s} {'values':>6s} {'MI':>6s} "
        f"{'|r|':>5s} {'R2 line':>7s} {'parab':>6s} {'signal':>7s} {'F1':>5s}  hint",
    ]
    for name in ranked:
        s = stats[name]
        lines.append(
            f"{name:10s} {s['type']:11s} {pct(s['nan_frac']):>5s} {s['n_values']:6d} "
            f"{1000 * s['mi']:6.2f} {abs(s['r']):5.3f} {s['r2_line']:7.2f} "
            f"{s['r2_parabola']:6.2f} {s['signal']:7.1f} {s['f1']:5.3f}  {hints[name]}"
        )

    lines += ["", "== Proposals =="]
    for title, prefix in (
        ("B9 one-hot, nominal", "one-hot ("),
        (f"B9 nominal with > {MAX_ONE_HOT} values (group rare ones?)", "one-hot, many"),
        ("nominal with too few rows per value", "rare"),
        ("B9 one-hot, ordinal not monotone", "one-hot: not monotone"),
        ("curved numeric: log / squared terms (not pipeline steps)", "curved"),
        ("not linear, not curved (bins?)", "not linear"),
        ("weak signal", "weak signal"),
    ):
        chosen = [n for n in ranked if hints[n].startswith(prefix)]
        lines.append(f"{title} ({len(chosen)}):")
        lines.append(textwrap.indent(textwrap.fill(", ".join(chosen), 84), "    "))
    path = save_lines(lines, "target_relation.txt")

    col = {name: j for j, name in enumerate(names)}
    mi = np.array([stats[n]["mi"] for n in ranked])
    abs_r = np.nan_to_num(np.abs([stats[n]["r"] for n in ranked]))
    plot_ranking(
        ranked,
        1000 * mi,
        figure_path(EDA_FIGURES_DIR, "mi_ranking.png"),
        "mutual information with MICHD [millibits]",
        "Features ranked by mutual information",
        top=TOP_FEATURES,
    )
    plot_mi_vs_corr(ranked, mi, abs_r, figure_path(EDA_FIGURES_DIR, "mi_vs_corr.png"))
    top = ranked[:TOP_FEATURES]
    plot_dependence_grid(
        top,
        [x[:, col[n]] for n in top],
        y01,
        [plot_kind(stats[n]) for n in top],
        figure_path(EDA_FIGURES_DIR, f"log_odds_top{TOP_FEATURES}.png"),
        ncols=6,
    )
    report = distinct_top(x, names, ranked)
    figure = figure_path(FIGURES_DIR, "log_odds.png")
    plot_dependence_grid(
        report,
        [x[:, col[n]] for n in report],
        y01,
        [plot_kind(stats[n]) for n in report],
        figure,
        ncols=4,
        panel=(2.3, 1.75),
    )
    print(f"top 5 by MI: {', '.join(ranked[:5])}")
    print(f"table: {path}, figures in {EDA_FIGURES_DIR}, summary figure: {figure}")


def plot_kind(s):
    """How plot_dependence_grid draws a column: a count or continuous column with
    many values in quantile bins with a linear fit ("numeric"), not one point per
    value (the rare values, e.g. 19 or 23 days of PHYSHLTH, are only noise)."""
    if s["type"] in (COUNT, CONTINUOUS) and s["n_values"] > MAX_ONE_HOT:
        return "numeric"
    return s["kind"]


def distinct_top(x, names, ranked):
    """The first REPORT_FEATURES columns of ranked, skipping the columns of
    REDUNDANT and a column with |Pearson r| > REPORT_MAX_CORR (rows where both are answered) with one chosen
    before: one column per quantity (age, blood pressure, ...) in the summary figure.
    """
    col = {name: j for j, name in enumerate(names)}
    chosen = []
    for name in ranked:
        if name in REDUNDANT:  # dropped by the pipeline as a duplicate
            continue
        a = x[:, col[name]]
        distinct = True
        for other in chosen:
            b = x[:, col[other]]
            both = ~np.isnan(a) & ~np.isnan(b)
            if abs(np.nan_to_num(pearson(a[both], b[both]))) > REPORT_MAX_CORR:
                distinct = False
                break
        if distinct:
            chosen.append(name)
            if len(chosen) == REPORT_FEATURES:
                break
    return chosen


def pairwise_correlation(x):
    """Pearson correlation of every pair of columns, each on the rows where both
    are answered, with matrix products instead of a loop over the pairs.

    Returns:
        corr: shape=(D, D), NaN where undefined
        overlap: shape=(D, D), number of rows where both are answered
    """
    answered = (~np.isnan(x)).astype(float)
    mean, std = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    std[std == 0] = 1.0
    # standardized first: the sums below lose precision on large raw values
    z = np.where(np.isnan(x), 0.0, (x - mean) / std)
    overlap = answered.T @ answered
    # s[i, j]: sum of column i over the rows where i and j are answered
    s = z.T @ answered
    ss = (z**2).T @ answered
    cov = overlap * (z.T @ z) - s * s.T
    var = overlap * ss - s**2
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt(var * var.T)
    # a column constant on the common rows (a follow-up and its gate question) has
    # no correlation; var is ~0 there only up to rounding errors
    constant = var <= 1e-8 * overlap**2
    corr[constant | constant.T] = np.nan
    return corr, overlap


def determination(a, b, n_a, n_b):
    """How much each of two coded columns determines the other, on the rows where
    both are answered (codes >= 0; n_a, n_b codes).

    Returns:
        U(b|a) = MI(a, b) / H(b) and U(a|b): 1 when b (a) is a function of a (b)
    """
    both = (a >= 0) & (b >= 0)
    joint = np.bincount(a[both] * n_b + b[both], minlength=n_a * n_b)
    joint = joint.reshape(n_a, n_b)
    h_a, h_b = entropy_bits(joint.sum(axis=1)), entropy_bits(joint.sum(axis=0))
    mi = h_a + h_b - entropy_bits(joint.ravel())
    return (mi / h_b if h_b > 0 else np.nan), (mi / h_a if h_a > 0 else np.nan)


def redundancy(x, names, feats, stats):
    """A7: columns that duplicate other columns. Saves build/eda/redundancy.txt and
    build/eda/figures/corr_top.png (correlations of the TOP_FEATURES columns by MI).

    On the raw columns, before any encoding (one-hot, bins): every pair of the
    feature columns, the columns of REDUNDANT included, plus the raw columns of
    CONVERTED (to check that their converted version is the same answer). On
    REDUNDANCY_ROWS random rows: |Pearson r| of each pair and U(b|a), how much a
    determines b (continuous columns cut in REDUNDANCY_BINS quantile bins, the raw
    mixed-units codes by value).

    Proposal of duplicates to drop (B2), greedy over the feature columns: go through
    them in blocks of NAN_BLOCK missing fraction, the finer ones first (more
    distinct values), and drop a column if a column kept before, with a missing
    fraction within NAN_TOLERANCE, is the same quantity (|r| >= SAME_QUANTITY) or
    determines it (U >= DETERMINED, b coded by value) and the pipeline encodes it
    so that a linear model rebuilds any function of it exactly (rebuilds: one-hot
    or binary). A column determined only by a numeric column (it is a grouping of
    it, e.g. age classes of the age) or by a nominal column whose rare values B9
    merges is kept and listed as conditional: it adds a non-linear term.

    Then: families of linked columns (connected components of the pairs that pass
    one of the two thresholds), the check of every entry of REDUNDANT and of
    CONVERTED (the evidence for the hand-made lists), and all pairs above REVIEW.
    """
    cols = [names[j] for j in feats]
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(x), size=min(REDUNDANCY_ROWS, len(x)), replace=False))
    sub = x[np.ix_(rows, feats)]
    corr, overlap = pairwise_correlation(sub)

    # raw mixed-units codes (1xx per day, 2xx per week ...) by value, not in bins
    coded = [
        discretize(
            sub[:, k], NOMINAL if n in CONVERTED else FEATURE_TYPES[n], REDUNDANCY_BINS
        )
        for k, n in enumerate(cols)
    ]
    codes = [c for c, _, _ in coded]
    sizes = [len(p) for _, p, _ in coded]
    by_value = [len(p) == n_values for _, p, n_values in coded]
    d = len(cols)
    u = np.full((d, d), np.nan)  # u[a, b] = U(b|a)
    for a in range(d):
        for b in range(a + 1, d):
            if overlap[a, b] >= MIN_GROUP_SIZE:
                u[a, b], u[b, a] = determination(codes[a], codes[b], sizes[a], sizes[b])
    enough = overlap >= MIN_GROUP_SIZE
    same = enough & (np.abs(np.nan_to_num(corr)) >= SAME_QUANTITY)
    determined = enough & (np.nan_to_num(u) >= DETERMINED)  # [a, b]: a determines b
    np.fill_diagonal(same, False)
    np.fill_diagonal(determined, False)

    pairs = []
    for a in range(d):
        for b in range(a + 1, d):
            score = max(abs(corr[a, b]), u[a, b], u[b, a])
            if enough[a, b] and score >= REVIEW:
                pairs.append((score, a, b))
    pairs.sort(reverse=True)

    nan_frac = np.isnan(sub).mean(axis=0)
    hints = {name: encoding_hint(stats[name]) for name in cols}
    one_hot = set(ONE_HOT_ORDINAL) | set(BINNED)

    def rebuilds(m):
        """True if a linear model rebuilds exactly any function of column m from
        its encoding in the pipeline: one-hot (nominal, ONE_HOT_ORDINAL, BINNED) or
        binary. Not a nominal column with many values: B9 merges its rare values.
        Not a numeric column: a grouping of it is a step, not linear in it."""
        name = cols[m]
        if hints[name].startswith("one-hot, many"):
            return False
        return name in one_hot or FEATURE_TYPES[name] in (BINARY, NOMINAL)

    features = [k for k in range(d) if cols[k] not in CONVERTED]
    order = sorted(
        features,
        key=lambda k: (
            int(nan_frac[k] / NAN_BLOCK),
            -sizes[k],
            nan_frac[k],
            -stats[cols[k]]["mi"],
        ),
    )
    kept, status = [], {}
    dropped, conditional = [], []
    for k in order:
        reason, note = None, None
        for m in kept:
            if abs(nan_frac[m] - nan_frac[k]) > NAN_TOLERANCE:
                continue
            if same[m, k]:
                reason = f"same quantity as {cols[m]} (r = {corr[m, k]:+.3f})"
                break
            if by_value[k] and determined[m, k]:
                text = f"determined by {cols[m]} (U = {u[m, k]:.3f})"
                if rebuilds(m):
                    reason = text
                    break
                note = note or f"{text}, which is {hints[cols[m]]}"
        if reason:
            dropped.append((cols[k], reason))
            status[k] = "drop"
        else:
            kept.append(k)
            status[k] = "conditional" if note else "keep"
            if note:
                conditional.append((cols[k], note))
    for k in range(d):
        status.setdefault(k, "raw, B3")

    def row(name, text):
        s = stats[name]
        mark = "R" if name in REDUNDANT else " "
        return (
            f"{mark} {name:10s} {pct(s['nan_frac']):>5s} {1000 * s['mi']:6.2f}  {text}"
        )

    lines = [
        "Redundancy (experiments/eda.py)",
        f"({len(rows)} random rows, {len(features)} feature columns + "
        f"{d - len(features)} raw CONVERTED columns, compared before any encoding;",
        " U(b|a) = MI(a, b) / H(b) on the rows where both are answered, 1 = b is a"
        " function of a; R = listed in REDUNDANT)",
        "",
        f"== Proposed duplicates to drop ({len(dropped)}): determined by a kept column"
        f" (U >= {DETERMINED}) or same quantity (|r| >= {SAME_QUANTITY}) ==",
        f"  {'column':10s} {'NaN%':>5s} {'MI':>6s}  reason",
    ]
    lines += [row(name, reason) for name, reason in dropped]
    lines += [
        "",
        f"== Conditional duplicates, kept ({len(conditional)}): determined by a column"
        " that the model cannot use to rebuild them (numeric: they are a grouping of"
        " it; nominal with merged rare values) ==",
        f"  {'column':10s} {'NaN%':>5s} {'MI':>6s}  reason",
    ]
    lines += [row(name, note) for name, note in conditional]

    families = [
        g for g in connected_groups(same | determined | determined.T) if len(g) > 1
    ]
    lines += [
        "",
        f"== Families of linked columns ({len(families)}): each column is the same"
        " quantity as, determines or is determined by another one of the family;"
        " status = greedy proposal, R = dropped by REDUNDANT ==",
    ]
    for g in families:
        lines.append(
            "  "
            + ", ".join(
                f"{cols[k]} ({sizes[k]} val, {pct(nan_frac[k])}% NaN, {status[k]}"
                + (", R)" if cols[k] in REDUNDANT else ")")
                for k in sorted(g, key=lambda k: -sizes[k])
            )
        )

    index = {name: k for k, name in enumerate(cols)}

    def check(b_name, a_name, converted=False):
        """Evidence that column b duplicates column a, with the rules of the greedy
        proposal: same quantity, or a determines b and rebuilds it (one-hot or
        binary), and missing fractions within NAN_TOLERANCE. converted: b is a raw
        mixed-units column and a its converted version, both coded by value; a
        must be a function of b."""
        if b_name not in index or a_name not in index:
            return "column not analysed"
        b, a = index[b_name], index[a_name]
        if not enough[a, b]:
            return f"NO EVIDENCE: {int(overlap[a, b])} rows in common"
        u_ba, u_ab = u[a, b], u[b, a]
        if converted:
            code_a, _, n_a = discretize(sub[:, a], NOMINAL)
            code_b, _, n_b = discretize(sub[:, b], NOMINAL)
            u_ba, u_ab = determination(code_a, code_b, n_a, n_b)
            ok = u_ab >= DETERMINED
            problems = ["converted is not a function of the raw codes"]
        else:
            problems = []
            if not (same[a, b] or determined[a, b]):
                problems.append("not determined")
            elif not (same[a, b] or rebuilds(a)):
                problems.append(f"{a_name} does not rebuild it")
            if abs(nan_frac[a] - nan_frac[b]) > NAN_TOLERANCE:
                problems.append(f"NaN {pct(nan_frac[b])}% vs {pct(nan_frac[a])}%")
            ok = not problems
        verdict = "ok" if ok else "NO (" + ", ".join(problems) + ")"
        return (
            f"{verdict}: |r| = {abs(corr[a, b]):.3f}, "
            f"U(b|a) = {u_ba:.3f}, U(a|b) = {u_ab:.3f}, {int(overlap[a, b])} rows"
        )

    lines += [
        "",
        f"== Check of REDUNDANT (src/feature_types.py, {len(REDUNDANT)} columns):"
        " b dropped as a duplicate of a ==",
    ]
    lines += [f"  {b:10s} <- {a:10s} {check(b, a)}" for b, (a, _) in REDUNDANT.items()]
    lines += [
        "",
        f"== Check of CONVERTED ({len(CONVERTED)} raw columns b, converted version a,"
        f" by value): a must be a function of b, U(a|b) >= {DETERMINED}; U(b|a) < 1"
        " when one quantity has several codes (101 = 207 = 330: once a day) ==",
    ]
    lines += [
        f"  {b:10s} -> {a:10s} {check(b, a, converted=True)}"
        for b, a in CONVERTED.items()
    ]

    lines += [
        "",
        f"== Pairs with |r|, U(b|a) or U(a|b) >= {REVIEW} ({len(pairs)}) ==",
        f"{'a':10s} {'b':10s} {'|r|':>6s} {'U(b|a)':>7s} {'U(a|b)':>7s} {'rows':>7s}",
    ]
    for _, a, b in pairs:
        lines.append(
            f"{cols[a]:10s} {cols[b]:10s} {abs(corr[a, b]):6.3f} {u[a, b]:7.3f} "
            f"{u[b, a]:7.3f} {int(overlap[a, b]):7d}"
        )
    path = save_lines(lines, "redundancy.txt")

    top = sorted(
        (n for n in cols if n not in CONVERTED), key=lambda n: -stats[n]["mi"]
    )[:TOP_FEATURES]
    column = {name: j for j, name in enumerate(names)}
    plot_corr_heatmap(
        top,
        x[:, [column[n] for n in top]],
        figure_path(EDA_FIGURES_DIR, "corr_top.png"),
    )
    print(
        f"{len(pairs)} pairs to review, {len(dropped)} proposed duplicates, "
        f"{len(conditional)} conditional"
    )
    print(f"table: {path}")


def distribution(col, edges, by_value):
    """Share of the rows of col in each category: the values in edges (by_value, one
    more category for values not in edges) or the bins between edges; NaN last."""
    nan = np.isnan(col)
    v = col[~nan]
    if by_value:
        k = np.minimum(np.searchsorted(edges, v), len(edges) - 1)
        k = np.where(edges[k] == v, k, len(edges))
        counts = np.bincount(k, minlength=len(edges) + 1)
    else:
        k = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
        counts = np.bincount(k, minlength=len(edges) - 1)
    return np.append(counts, nan.sum()) / len(col)


def shift_distance(a, b, ftype):
    """Total variation distance between the distributions of a column in two samples
    a and b (values, or deciles of a for numeric columns with many values; NaN is a
    value): 0 = same distribution, 1 = no overlap."""
    values = np.unique(a[~np.isnan(a)])
    by_value = ftype in (BINARY, ORDINAL, NOMINAL) or len(values) <= MAX_GROUP_VALUES
    if not by_value:
        values = np.unique(np.quantile(a[~np.isnan(a)], np.linspace(0, 1, 11)))
    pa, pb = distribution(a, values, by_value), distribution(b, values, by_value)
    return 0.5 * np.abs(pa - pb).sum()


def train_test_shift(x_train, x_test, names, feats):
    """A8: does the test set look like the training set? For each feature column:
    NaN % in train and test, total variation distance (TVD) between train and test,
    and the TVD between two random parts of train of the same sizes (noise level).
    Saves build/eda/shift.txt."""
    rng = np.random.default_rng(0)
    part = rng.permutation(len(x_train))
    ref_a, ref_b = part[len(x_test) :], part[: len(x_test)]
    rows = []
    for j in feats:
        name, ftype = names[j], FEATURE_TYPES[names[j]]
        tr, te = x_train[:, j], x_test[:, j]
        tvd = shift_distance(tr, te, ftype)
        noise = shift_distance(tr[ref_a], tr[ref_b], ftype)
        rows.append((tvd, name, np.isnan(tr).mean(), np.isnan(te).mean(), noise))
    rows.sort(reverse=True)
    tvd = np.array([r[0] for r in rows])
    noise = np.array([r[4] for r in rows])
    above = [r[1] for r in rows if r[0] > 2 * r[4] and r[0] > 0.01]
    lines = [
        "Train vs test (experiments/eda.py)",
        f"(train {len(x_train)} rows, test {len(x_test)} rows, {len(rows)} feature"
        " columns; TVD on values or train deciles, NaN as a value)",
        f"TVD train-test: median {np.median(tvd):.4f}, max {tvd.max():.4f}",
        f"TVD of two random train parts (noise): median {np.median(noise):.4f}, "
        f"max {noise.max():.4f}",
        f"columns with TVD > 2 x noise and > 0.01: {len(above)} {above}",
        "",
        f"{'column':10s} {'NaN% tr':>7s} {'NaN% te':>7s} {'TVD':>7s} {'noise':>7s}",
    ]
    for d, name, nan_tr, nan_te, ref in rows:
        lines.append(
            f"{name:10s} {pct(nan_tr):>7s} {pct(nan_te):>7s} {d:7.4f} {ref:7.4f}"
        )
    path = save_lines(lines, "shift.txt")
    print("\n".join(lines[2:5]))
    print(f"table: {path}")


def ranges_and_outliers(x, y01, names, feats):
    """A9: ranges of the count and continuous columns. Saves build/eda/outliers.txt.

    For each column (answered rows): percentiles, skewness, answers outside the
    plausible range (PLAUSIBLE), MICHD % above the 99th percentile and below it, and
    |r| with MICHD of the raw values, of the values clipped at CLIP_PERCENTILES, and
    of log1p of the clipped values (non-negative columns). A transform is proposed
    when it raises |r| by R_GAIN and by R_MIN_GAIN: a linear model sees the column
    better. Clipping and log1p are not steps of the pipeline (src/pipeline.py).
    """
    lines = [
        "Ranges and outliers of the count and continuous columns (experiments/eda.py)",
        f"(answered rows; clip at percentiles {CLIP_PERCENTILES}; MICHD % above /"
        " below the 99th percentile; out = answers outside the plausible range)",
        "",
        f"{'column':10s} {'type':10s} {'min':>8s} {'p0.5':>8s} {'p50':>8s} "
        f"{'p99.5':>8s} {'max':>8s} {'skew':>6s} {'out':>5s} {'>p99':>5s} "
        f"{'rest':>5s} {'|r|':>6s} {'clip':>6s} {'log':>6s}  hint",
    ]
    hints = {}
    for j in feats:
        name, ftype = names[j], FEATURE_TYPES[names[j]]
        # the mixed-units columns left are numbers after convert_units
        if ftype not in (COUNT, CONTINUOUS, MIXED_UNITS):
            continue
        col = x[:, j]
        answered = ~np.isnan(col)
        v, yv = col[answered], y01[answered]
        q = np.percentile(v, [0, CLIP_PERCENTILES[0], 50, 99, CLIP_PERCENTILES[1], 100])
        skew = np.mean((v - v.mean()) ** 3) / v.std() ** 3 if v.std() > 0 else 0.0
        low, high = PLAUSIBLE.get(name, (-np.inf, np.inf))
        n_out = int(np.sum((v < low) | (v > high)))
        top = v > q[3]
        prev_top = yv[top].mean() if top.any() else np.nan
        clipped = np.clip(v, q[1], q[4])
        r_raw, r_clip = abs(pearson(v, yv)), abs(pearson(clipped, yv))
        r_log = abs(pearson(np.log1p(clipped), yv)) if q[0] >= 0 else np.nan
        hint = ""
        best = max(r_raw, r_clip)
        if r_log >= R_GAIN * best and r_log - best >= R_MIN_GAIN:
            hint = "clip + log1p"
        elif r_clip >= R_GAIN * r_raw and r_clip - r_raw >= R_MIN_GAIN:
            hint = "clip"
        hints[name] = hint
        lines.append(
            f"{name:10s} {ftype:10s} {fmt_value(q[0]):>8s} {fmt_value(q[1]):>8s} "
            f"{fmt_value(q[2]):>8s} {fmt_value(q[4]):>8s} {fmt_value(q[5]):>8s} "
            f"{skew:6.1f} {n_out:5d} {pct(prev_top):>5s} {pct(yv[~top].mean()):>5s} "
            f"{r_raw:6.3f} {r_clip:6.3f} {r_log:6.3f}  {hint}"
        )
    lines += ["", "== Proposed transforms (clip, log1p: not pipeline steps) =="]
    for title in ("clip + log1p", "clip"):
        chosen = [n for n, h in hints.items() if h == title]
        lines.append(f"{title} ({len(chosen)}):")
        lines.append(textwrap.indent(textwrap.fill(", ".join(chosen), 84), "    "))
    path = save_lines(lines, "outliers.txt")
    print(f"{len(hints)} numeric columns; table: {path}")


def main(steps):
    x_train, x_test, y_train, train_ids, test_ids, names = load_data()
    names = list(names)
    y01 = (y_train == 1).astype(float)
    x_clean = codes_to_nan(x_train, names)
    if "A1" in steps:
        structural_overview(
            x_train, x_clean, y_train, x_test.shape, train_ids, test_ids, names
        )
    if "A2" in steps:
        feature_types_table(x_clean, names)
    if "A3" in steps:
        special_codes_audit(x_train, x_clean, y01, names)
    if "A4" in steps:
        missingness_analysis(x_train, x_clean, y01, names)
    del x_train

    # A5-A9: the values the model sees, after the label-free pipeline steps B3, B4
    convert_units(x_clean, names)
    fill_skip_logic(x_clean, names)
    feats = feature_columns(names)
    if "A5" in steps:
        target_overview(x_clean, y01, names)
    if "A6" in steps or "A7" in steps:
        stats = feature_stats(x_clean, y01, names, feats)
    if "A6" in steps:
        target_relation(x_clean, y01, names, feats, stats)
    if "A7" in steps:
        # also the raw columns of CONVERTED, against their converted version
        raw = [j for j, name in enumerate(names) if name in CONVERTED]
        stats_raw = feature_stats(x_clean, y01, names, raw)
        redundancy(x_clean, names, feats + raw, {**stats, **stats_raw})
    if "A8" in steps:
        x_test = codes_to_nan(x_test, names)
        convert_units(x_test, names)
        fill_skip_logic(x_test, names)
        train_test_shift(x_clean, x_test, names, feats)
    if "A9" in steps:
        ranges_and_outliers(x_clean, y01, names, feats)


if __name__ == "__main__":
    chosen = sys.argv[1:] or STEPS
    unknown = set(chosen) - set(STEPS)
    if unknown:
        sys.exit(f"unknown analyses {sorted(unknown)}, choose among {STEPS}")
    main(chosen)
