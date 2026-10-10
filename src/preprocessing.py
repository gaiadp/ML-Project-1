"""Data cleaning: missing values / BRFSS special codes, unit conversion, skip logic,
feature removal, imputation, standardization.

Fit statistics (means, stds, ...) on the training split only, then apply them to validation/test.
src/pipeline.py chains these steps (the B steps of PLAN.md).
"""

import numpy as np

from src.feature_types import BINARY, NOMINAL


# BRFSS special codes (see codebook). "Don't know" / "refused" are usually all-7s /
# all-9s of the field's width, but NOT always, and the same value can be a real answer
# in another column (EMPLOY1=7 "retired", _AGEG5YR=7/9 age groups, _AGE80=77 years,
# _STATE=9 Connecticut). So codes are assigned per column, by name.
# Each rule is (missing codes -> NaN, "none/never" codes -> 0).
_R_1 = ({7, 9}, set())  # default: 1-digit question
_R_9 = ({9}, set())  # calculated variables (_XXX) and 1-digit Qs where 7 is valid
_R_2 = ({77, 99}, set())  # 2-digit categorical question
_R_2DAYS = ({77, 99}, {88})  # "number of days/times", 88 = none
_R_3 = ({777, 999}, set())  # 3-digit coded quantity
_R_3NONE = ({777, 999}, {555, 888})  # 3-digit quantity, 555 = never / 888 = none
_R_4 = ({7777, 9999}, set())  # WEIGHT2, HEIGHT3
_R_6 = ({777777, 999999}, set())  # dates mmyyyy
_R_NONE = (set(), set())  # admin fields, weights, already-computed quantities
_R_1NA = ({7, 8, 9}, set())  # 1-digit question, 8 = not applicable
_R_1NEVER = ({7, 9}, {8})  # 1-digit frequency question, 8 = never / none

_RULES = {
    # admin / design / weights / counts with no special codes
    **dict.fromkeys(
        [
            "_STATE",
            "FMONTH",
            "IDATE",
            "IMONTH",
            "IDAY",
            "IYEAR",
            "DISPCODE",
            "SEQNO",
            "_PSU",
            "CTELENUM",
            "PVTRESD1",
            "COLGHOUS",
            "STATERES",
            "CELLFON3",
            "LADULT",
            "NUMADULT",
            "NUMMEN",
            "NUMWOMEN",
            "CTELNUM1",
            "CELLFON2",
            "CADULT",
            "PVTRESD2",
            "CCLGHOUS",
            "CSTATE",
            "SEX",
            "QSTVER",
            "QSTLANG",
            "MSCODE",
            "_STSTR",
            "_STRWT",
            "_RAWRAKE",
            "_WT2RAKE",
            "_CLLCPWT",
            "_DUALCOR",
            "_LLCPWT",
            "_DRDXAR1",
            "_RACE_G1",
            "_AGE80",
            "_AGE_G",
            "HTIN4",
            "HTM4",
            "WTKG3",
            "_BMI5",
            "_BMI5CAT",
            "FTJUDA1_",
            "FRUTDA1_",
            "BEANDAY_",
            "GRENDAY_",
            "ORNGDAY_",
            "VEGEDA1_",
            "_MISFRTN",
            "_MISVEGN",
            "_FRTRESP",
            "_VEGRESP",
            "_FRUTSUM",
            "_VEGESUM",
            "_FRT16",
            "_VEG23",
            "_FRUITEX",
            "_VEGETEX",
            "METVL11_",
            "METVL21_",
            "ACTIN11_",
            "ACTIN21_",
            "PADUR1_",
            "PADUR2_",
            "_MINAC11",
            "_MINAC21",
            "PAMIN11_",
            "PAMIN21_",
            "PA1MIN_",
            "PAVIG11_",
            "PAVIG21_",
            "PA1VIGM_",
        ],
        _R_NONE,
    ),
    # 1-digit questions where 7 (and 8) are real answers
    **dict.fromkeys(["EMPLOY1", "MARITAL", "EDUCA", "INSULIN", "RCSGENDR"], _R_9),
    # 2-digit categorical questions
    **dict.fromkeys(
        [
            "HHADULT",
            "INCOME2",
            "LASTSMK2",
            "AVEDRNK2",
            "MAXDRNKS",
            "EXRACT11",
            "EXRACT21",
            "JOINPAIN",
            "IMFVPLAC",
            "WHRTST10",
            "CRGVREL1",
            "CRGVPRB1",
            "HPVADSHT",
            "_CRACE1",
            "_CPRACE",
            "_PRACE1",
            "_MRACE1",
        ],
        _R_2,
    ),
    # "how many days/times", 88 = none
    **dict.fromkeys(
        [
            "PHYSHLTH",
            "MENTHLTH",
            "POORHLTH",
            "CHILDREN",
            "DRNK3GE5",
            "DOCTDIAB",
            "FEETCHK",
            "ADPLEASR",
            "ADDOWN",
            "ADSLEEP",
            "ADENERGY",
            "ADEAT1",
            "ADFAIL",
            "ADTHINK",
            "ADMOVE",
        ],
        _R_2DAYS,
    ),
    # 98 = never heard of the A1C test: unknown number of tests
    "CHKHEMO3": ({77, 98, 99}, {88}),
    # visits 1-87 (77 is a real count), 88 = none, 98 = don't know, 99 = refused
    **dict.fromkeys(["ASERVIST", "ASDRVIST", "ASRCHKUP"], ({98, 99}, {88})),
    # ages 1-97: 98 = don't know, 99 = refused
    "DIABAGE2": ({98, 99}, set()),
    # ages 11-96: 97 = "10 or younger", no exact age
    "ASTHMAGE": ({97, 98, 99}, set()),
    # hours of care per week: 97 = don't know, 98 = zero, 99 = refused
    **dict.fromkeys(["SCNTWRK1", "SCNTLWK1"], ({97, 99}, {98})),
    # frequency scales 1..5 (or 1..6) where 8 = never / none comes before 1
    **dict.fromkeys(["ASYMPTOM", "ASNOSLEP", "ASTHMED3", "ASINHALR"], _R_1NEVER),
    # 8 = not applicable (never in a car, blind, ...), not a point of the scale
    **dict.fromkeys(
        ["SEATBELT", "SCNTMNY1", "SCNTMEL1", "VIEYEXM2", "VIINSUR2"], _R_1NA
    ),
    # 6 = unable for other reasons than eyesight, 8 = blind
    **dict.fromkeys(["VIDFCLT2", "VIREDIF3"], ({6, 7, 8, 9}, set())),
    "VINOCRE2": ({8, 77, 99}, set()),
    # 3-digit coded quantities (1xx per day / 2xx per week / 3xx per month ...);
    # LONGWTCH 555 = "all my life" is kept for the unit conversion, it is not 0
    **dict.fromkeys(["EXEROFT1", "EXEROFT2", "EXERHMM1", "EXERHMM2", "LONGWTCH"], _R_3),
    **dict.fromkeys(
        [
            "ALCDAY5",
            "FRUITJU1",
            "FRUIT1",
            "FVBEANS",
            "FVGREEN",
            "FVORANG",
            "VEGETAB1",
            "STRENGTH",
            "BLDSUGAR",
            "FEETCHK2",  # 555 = no feet: 0 checks
            "ASACTLIM",
        ],
        _R_3NONE,
    ),
    **dict.fromkeys(["WEIGHT2", "HEIGHT3"], _R_4),
    **dict.fromkeys(["FLSHTMY2", "HIVTSTD3"], _R_6),
    # calculated variables with their own missing code
    "_AGEG5YR": ({14}, set()),
    "_AGE65YR": ({3}, set()),
    "DROCDY3_": ({900}, set()),
    "_DRNKWEK": ({99900}, set()),
    # codebook codes with the implied decimals already applied in the data:
    # 99900 / 100 = 999, 99000 / 1000 = 99 (legal values stop at 98.999)
    "MAXVO2_": ({999}, set()),
    "FC60_": ({999}, set()),
    **dict.fromkeys(["PAFREQ1_", "PAFREQ2_", "STRFREQ_"], ({99}, set())),
    **dict.fromkeys(
        [
            "_CHISPNC",
            "_DUALUSE",
            "_RFHLTH",
            "_HCVU651",
            "_RFHYPE5",
            "_CHOLCHK",
            "_RFCHOL",
            "_LTASTH1",
            "_CASTHM1",
            "_ASTHMS1",
            "_HISPANC",
            "_RACE",
            "_RACEG21",
            "_RACEGR3",
            "_RFBMI5",
            "_CHLDCNT",
            "_EDUCAG",
            "_INCOMG",
            "_SMOKER3",
            "_RFSMOK3",
            "_RFBING5",
            "_RFDRHV5",
            "_FRTLT1",
            "_VEGLT1",
            "_TOTINDA",
            "PAMISS1_",
            "_PACAT1",
            "_PAINDX1",
            "_PA150R2",
            "_PA300R2",
            "_PA30021",
            "_PASTRNG",
            "_PAREC1",
            "_PASTAE1",
            "_LMTACT1",
            "_LMTWRK1",
            "_LMTSCL1",
            "_RFSEAT2",
            "_RFSEAT3",
            "_FLSHOT6",
            "_PNEUMO2",
            "_AIDTST3",
        ],
        _R_9,
    ),
}


def special_code_rules(names):
    """Column index -> (missing codes, none->0 codes). Unlisted columns get the 1-digit
    default {7, 9}: every such column in the 2015 data is a 1..6 (or 1..8) coded answer."""
    return {j: _RULES.get(n, _R_1) for j, n in enumerate(names)}


def codes_to_nan(x, names):
    """Replace BRFSS special codes with NaN (don't know / refused) or 0 (none / never).

    Parameters
    ----------
    x : np.ndarray, shape (N, D), raw data (Id column already removed)
    names : sequence of str, the D column names (header of x_train.csv without "Id")
    """
    # astype already returns a copy, x is not modified
    x = x.astype(float)
    for col, (missing, zero) in special_code_rules(names).items():
        if missing:
            x[np.isin(x[:, col], list(missing)), col] = np.nan
        if zero:
            x[np.isin(x[:, col], list(zero)), col] = 0.0
    return x


# B3. 3-digit codes with the unit in the first digit (codebook):
# 1xx per day, 2xx per week, 3xx per month, 4xx per year -> times per year
_TIMES_PER_YEAR = {1: 365.0, 2: 52.0, 3: 12.0, 4: 1.0}
# 1xx days, 2xx weeks, 3xx months, 4xx years -> years
_YEARS = {1: 1 / 365, 2: 1 / 52, 3: 1 / 12, 4: 1.0}


def _convert_unit_codes(v, factors):
    """Replace each code u*100 + k of the column v with k * factors[u], in place.
    0 (never / none, set by codes_to_nan) and NaN are not changed."""
    unit = v // 100
    amount = v % 100
    for u, factor in factors.items():
        rows = unit == u
        v[rows] = amount[rows] * factor


def _months_before_interview(v, interview_month):
    """Replace the dates mmyyyy of the column v with the number of months between the
    date and the interview, in place. interview_month = 12 * year + month."""
    month = v // 10000
    year = v % 10000
    month[month == 77] = 6.5  # month unknown: middle of the year
    year[year < 1900] = np.nan  # typing errors such as 80201
    # dates after the interview month (a few flu shots) count as 0 months
    v[:] = np.maximum(interview_month - (12 * year + month), 0)


def convert_units(x, names):
    """B3: put the coded columns that have no converted version on one numeric scale.

    - BLDSUGAR, FEETCHK2 (1xx per day .. 4xx per year): times per year
    - LONGWTCH (1xx days .. 4xx years, 555 = all my life): years; all my life = age
    - FLSHTMY2, HIVTSTD3 (dates mmyyyy): months before the interview (IYEAR, IMONTH)
    - CHECKUP1, EYEEXAM: 8 = never becomes 5, after the last code (4 = longest ago)

    The raw columns with a converted version (CONVERTED in src/feature_types.py) are
    not changed: the pipeline drops them. Call it on the output of codes_to_nan.

    Args:
        x: data after codes_to_nan, shape=(N, D), modified in place
        names: the D column names
    """
    col = {n: j for j, n in enumerate(names)}
    for name in ("BLDSUGAR", "FEETCHK2"):
        if name in col:
            _convert_unit_codes(x[:, col[name]], _TIMES_PER_YEAR)
    if "LONGWTCH" in col:
        v = x[:, col["LONGWTCH"]]
        all_life = v == 555
        _convert_unit_codes(v, _YEARS)
        v[all_life] = x[all_life, col["_AGE80"]]
    for name in ("FLSHTMY2", "HIVTSTD3"):
        if name in col:
            interview_month = 12 * x[:, col["IYEAR"]] + x[:, col["IMONTH"]]
            _months_before_interview(x[:, col[name]], interview_month)
    for name in ("CHECKUP1", "EYEEXAM"):
        if name in col:
            v = x[:, col[name]]
            v[v == 8] = 5


# B4. Skip logic of the questionnaire: each rule is (gate, fills). When every gate
# column has one of the listed answers, the follow-up questions in fills are not
# asked, so their NaN means "no" / "zero" and gets the value in fills (in the coding
# left by codes_to_nan and convert_units). Checked on x_train: for every rule, no row
# that matches the gate has an answer to the follow-ups.
_PA_ZERO = [
    # intensity, MET value, times per week, minutes: 0 when there is no activity
    "ACTIN11_",
    "ACTIN21_",
    "METVL11_",
    "METVL21_",
    "PAFREQ1_",
    "PAFREQ2_",
    "PADUR1_",
    "PADUR2_",
    "_MINAC11",
    "_MINAC21",
    "PAMIN11_",
    "PAMIN21_",
    "PA1MIN_",
    "PAVIG11_",
    "PAVIG21_",
    "PA1VIGM_",
    "EXEROFT1",
    "EXEROFT2",
    "EXERHMM1",
    "EXERHMM2",
]
_SECOND_PA_ZERO = [
    "PAFREQ2_",
    "PADUR2_",
    "EXEROFT2",
    "EXERHMM2",
    "ACTIN21_",
    "PAMIN21_",
    "PAVIG21_",
]
_DIABETES_ZERO = ["DOCTDIAB", "CHKHEMO3", "FEETCHK", "BLDSUGAR", "FEETCHK2"]

_SKIP_LOGIC = [
    # no exercise in the past month: no activity (88 = the EXRACT21 code for
    # "no other activity", used for EXRACT11 too), 0 times, 0 minutes
    ({"EXERANY2": [2]}, {"EXRACT11": 88, "EXRACT21": 88, **dict.fromkeys(_PA_ZERO, 0)}),
    # no second activity (the CDC already sets METVL21_ and _MINAC21 to 0)
    ({"EXRACT21": [88]}, dict.fromkeys(_SECOND_PA_ZERO, 0)),
    # no alcohol in the past 30 days (DRNKANY5 = 2 is ALCDAY5 = 888)
    ({"DRNKANY5": [2]}, dict.fromkeys(["AVEDRNK2", "DRNK3GE5", "MAXDRNKS"], 0)),
    # less than 100 cigarettes in life: smokes "not at all", never smoked regularly
    ({"SMOKE100": [2]}, {"SMOKDAY2": 3, "LASTSMK2": 8}),
    # smokes every day / some days: 0 = smoking now (1 = quit less than a month ago)
    ({"SMOKDAY2": [1, 2]}, {"LASTSMK2": 0}),
    # no high blood pressure (no, only in pregnancy, borderline): no medication
    ({"BPHIGH4": [2, 3, 4]}, {"BPMEDS": 2}),
    # no day of bad physical or mental health: no day with activities limited
    ({"PHYSHLTH": [0], "MENTHLTH": [0]}, {"POORHLTH": 0}),
    # cholesterol never checked: never told it is high (_RFCHOL 1 = no);
    # last check "never" = 5, after 4 = 5+ years ago (as _CHOLCHK = 3)
    ({"BLOODCHO": [2]}, {"TOLDHI2": 2, "_RFCHOL": 1, "CHOLCHK": 5}),
    # no arthritis: no limitation, interference "not at all" (3), joint pain 0
    ({"HAVARTH3": [2]}, {"LMTJOIN3": 2, "ARTHDIS2": 2, "ARTHSOCL": 3, "JOINPAIN": 0}),
    # not diabetic (no, only in pregnancy, pre-diabetes): no insulin, no eye damage,
    # no course, 0 visits / tests / checks. DIABAGE2 and EYEEXAM have no value for
    # "not diabetic" and stay NaN.
    (
        {"DIABETE3": [2, 3, 4]},
        {"INSULIN": 2, "DIABEYE": 2, "DIABEDU": 2, **dict.fromkeys(_DIABETES_ZERO, 0)},
    ),
    # never told to have asthma: no asthma now
    ({"ASTHMA3": [2]}, {"ASTHNOW": 2}),
    # never had a stroke: no stroke rehabilitation
    ({"CVDSTRK3": [2]}, {"STREHAB1": 2}),
]


def fill_skip_logic(x, names):
    """B4: fill the NaN of the follow-up questions that were not asked (_SKIP_LOGIC).

    Only NaN entries change, and only in the rows that match the gate (a NaN in a gate
    column is not a match). Rules whose gate columns are missing are skipped.

    Args:
        x: data after codes_to_nan and convert_units, shape=(N, D), modified in place
        names: the D column names
    """
    col = {n: j for j, n in enumerate(names)}
    for gate, fills in _SKIP_LOGIC:
        if not all(n in col for n in gate):
            continue
        not_asked = np.ones(x.shape[0], dtype=bool)
        for name, answers in gate.items():
            not_asked &= np.isin(x[:, col[name]], answers)
        for name, value in fills.items():
            if name in col:
                j = col[name]
                x[not_asked & np.isnan(x[:, j]), j] = value


def drop_columns(x, keep_mask):
    """Keep only the columns where keep_mask is True. keep_mask: bool array, shape (D,)."""
    return x[:, keep_mask]


def drop_high_missing_columns(x, max_missing_frac=0.5):
    """Return a boolean keep-mask dropping columns with too many NaNs.

    Threshold is a knob, not a fixed rule: tune it against what the EDA finds.
    """
    missing_frac = np.isnan(x).mean(axis=0)
    return missing_frac <= max_missing_frac


def missing_rate_difference(nan, y, min_rows=500):
    """B5/B6: MICHD prevalence when a column is NaN minus when it is answered.

    Args:
        nan: np.isnan of the training data, shape=(N, D)
        y: training labels, shape=(N,), 1 = MICHD (-1/1 or 0/1)
        min_rows: a column with fewer NaN rows or fewer answered rows gets 0

    Returns:
        diff: shape=(D,)
    """
    min_rows = max(min_rows, 1)
    positive = y == 1
    n_nan = nan.sum(axis=0)
    n_answered = len(y) - n_nan
    pos_nan = nan[positive].sum(axis=0)
    pos_answered = positive.sum() - pos_nan
    enough = (n_nan >= min_rows) & (n_answered >= min_rows)
    diff = np.zeros(nan.shape[1])
    diff[enough] = (
        pos_nan[enough] / n_nan[enough] - pos_answered[enough] / n_answered[enough]
    )
    return diff


def one_per_group(nan, cols, max_corr=0.9):
    """B6: keep one column per group of columns that are missing together.

    Goes through cols in the given order and keeps a column only if the |correlation|
    between its NaN mask and the mask of every column kept before is <= max_corr
    (the same threshold used to find the groups in experiments/eda.py). Opposite
    masks (e.g. a landline-only and a cell-only question) count as the same group.

    Args:
        nan: np.isnan of the training data, shape=(N, D)
        cols: candidate column indices, most important first; their masks must not
            be constant
        max_corr: correlation above which two masks are the same group

    Returns:
        the kept column indices, shape=(K,)
    """
    cols = np.asarray(cols, dtype=int)
    if len(cols) == 0:
        return cols
    masks = nan[:, cols].astype(np.float32)
    mean = masks.mean(axis=0)
    cov = masks.T @ masks / len(masks) - np.outer(mean, mean)
    sd = np.sqrt(np.diag(cov))
    corr = np.abs(cov / np.outer(sd, sd))
    kept = []
    for i in range(len(cols)):
        if all(corr[i, k] <= max_corr for k in kept):
            kept.append(i)
    return cols[kept]


def fit_impute_median(x_train):
    """Compute per-column medians on train, ignoring NaN. Returns a (D,) array to reuse at apply time."""
    return np.nanmedian(x_train, axis=0)


def apply_impute(x, medians):
    x = x.copy()
    nan_mask = np.isnan(x)
    cols = np.where(nan_mask.any(axis=0))[0]
    for col in cols:
        x[nan_mask[:, col], col] = medians[col]
    return x


def fit_impute_by_type(x_train, types):
    """B7: fill value of each column, to use with apply_impute.

    Binary and nominal columns get their most frequent value (the median of unordered
    codes can be a meaningless category), the others the median.

    Args:
        x_train: shape=(N, D), with NaN
        types: the D column types (FEATURE_TYPES values, None if unknown)

    Returns:
        fill: shape=(D,)
    """
    fill = fit_impute_median(x_train)
    for j, t in enumerate(types):
        if t in (BINARY, NOMINAL):
            v = x_train[:, j]
            values, counts = np.unique(v[~np.isnan(v)], return_counts=True)
            if len(values) > 0:
                fill[j] = values[np.argmax(counts)]
    return fill


def fit_binary_recode(x_train, types):
    """B8: mask of the binary columns coded 1/2, to recode with apply_binary_recode.
    Binary columns already coded 0/1 (_FRTRESP, _VEGRESP, PAMISS1_) are left out."""
    recode = np.zeros(len(types), dtype=bool)
    for j, t in enumerate(types):
        if t == BINARY:
            v = x_train[:, j]
            recode[j] = np.isin(v[~np.isnan(v)], [1, 2]).all()
    return recode


def apply_binary_recode(x, recode):
    """B8: 1/2 -> 1/0 in the columns where recode is True, in place.

    1 = code 1 of the codebook: "yes" for the questions, but "no risk factor" for
    the calculated _RF* variables (e.g. _RFHYPE5 1 = no high blood pressure).
    After standardization this only flips the sign of the weight: it is for
    readable weights, the predictions do not change.
    """
    x[:, recode] = 2 - x[:, recode]


def fit_standardize(x_train):
    """Compute per-column mean/std on train. Returns (mean, std) to reuse at apply time."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_standardize(x, mean, std):
    return (x - mean) / std
