"""Column lists from the BRFSS 2015 codebook, shared by the EDA and the pipeline.

DOMAIN_DROP: columns that do not describe the respondent (interview ids and dates,
phone screening, survey design and weights). They are dropped for domain reasons;
experiments/eda.py reports how much MICHD signal each of them carries.
"""

_ID_DATE = "interview id or date"
_PROCESS = "interview process (complete/partial, questionnaire version)"
_SCREENING = "phone screening (eligibility, phone type)"
_DESIGN = "survey design or sampling weight"

# column name -> reason it is not used as a feature
DOMAIN_DROP = {
    "SEQNO": _ID_DATE,
    "_PSU": _ID_DATE,  # primary sampling unit, equal to SEQNO in BRFSS
    "IDATE": _ID_DATE,
    "IMONTH": _ID_DATE,
    "IDAY": _ID_DATE,
    "IYEAR": _ID_DATE,
    "FMONTH": _ID_DATE,
    "DISPCODE": _PROCESS,
    "QSTVER": _PROCESS,
    "CTELENUM": _SCREENING,
    "PVTRESD1": _SCREENING,
    "COLGHOUS": _SCREENING,
    "STATERES": _SCREENING,
    "CELLFON3": _SCREENING,
    "LADULT": _SCREENING,
    "CTELNUM1": _SCREENING,
    "CELLFON2": _SCREENING,
    "CADULT": _SCREENING,
    "PVTRESD2": _SCREENING,
    "CCLGHOUS": _SCREENING,
    "CSTATE": _SCREENING,
    "LANDLINE": _SCREENING,
    "_STSTR": _DESIGN,
    "_STRWT": _DESIGN,
    "_RAWRAKE": _DESIGN,
    "_WT2RAKE": _DESIGN,
    "_DUALUSE": _DESIGN,
    "_DUALCOR": _DESIGN,
    "_LLCPWT": _DESIGN,
    "_CLLCPWT": _DESIGN,  # weight of the random child module
}
