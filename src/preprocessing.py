"""Data cleaning: missing values / BRFSS special codes, feature removal, standardization.

Fit statistics (means, stds, ...) on the training split only, then apply them to validation/test.
"""

import numpy as np


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
            "PAFREQ1_",
            "PAFREQ2_",
            "_MINAC11",
            "_MINAC21",
            "STRFREQ_",
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
            "VINOCRE2",
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
            "CHKHEMO3",
            "FEETCHK",
            "ASERVIST",
            "ASDRVIST",
            "ASRCHKUP",
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
    # ages 1-97: 98 = don't know, 99 = refused
    **dict.fromkeys(["DIABAGE2", "ASTHMAGE"], ({98, 99}, set())),
    # hours of care per week: 97 = don't know, 98 = zero, 99 = refused
    **dict.fromkeys(["SCNTWRK1", "SCNTLWK1"], ({97, 99}, {98})),
    # 3-digit coded quantities (1xx per day / 2xx per week / 3xx per month ...)
    **dict.fromkeys(["EXEROFT1", "EXEROFT2", "EXERHMM1", "EXERHMM2"], _R_3),
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
            "FEETCHK2",
            "LONGWTCH",
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
    "MAXVO2_": ({999}, set()),
    "FC60_": ({999}, set()),
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


def drop_columns(x, keep_mask):
    """Keep only the columns where keep_mask is True. keep_mask: bool array, shape (D,)."""
    return x[:, keep_mask]


def drop_high_missing_columns(x, max_missing_frac=0.5):
    """Return a boolean keep-mask dropping columns with too many NaNs.

    Threshold is a knob, not a fixed rule: tune it against what the EDA finds.
    """
    missing_frac = np.isnan(x).mean(axis=0)
    return missing_frac <= max_missing_frac


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


def fit_standardize(x_train):
    """Compute per-column mean/std on train. Returns (mean, std) to reuse at apply time."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_standardize(x, mean, std):
    return (x - mean) / std
