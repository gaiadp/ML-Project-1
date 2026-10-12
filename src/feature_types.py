"""Column lists from the BRFSS 2015 codebook, shared by the EDA and the pipeline.

DOMAIN_DROP: columns that are not used as features, with the reason (interview ids
and dates, phone screening, survey design and weights, a child instead of the
respondent, label leakage). experiments/eda.py reports how much MICHD signal each
of them carries.

FEATURE_TYPES: the type of every column (one of TYPES), from the codebook. It decides
how the pipeline encodes a column. Comments mark the codes that need care.

CONVERTED: mixed-units columns that the dataset also provides already converted to
one unit; the pipeline keeps only the converted column.
"""

_ID_DATE = "interview id or date"
_PROCESS = "interview process (complete/partial, questionnaire version)"
_SCREENING = "phone screening (eligibility, phone type)"
_DESIGN = "survey design or sampling weight"
_CHILD = "random child module: describes a child, not the respondent"
_QUALITY_FLAG = "data-quality flag, 99.98% one value"

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
    "QSTLANG": "interview language, covered by _HISPANC",
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
    "RCSGENDR": _CHILD,
    "RCSRLTN2": _CHILD,
    "CASTHDX2": _CHILD,
    "CASTHNO2": _CHILD,
    "_CHISPNC": _CHILD,
    "_CRACE1": _CHILD,
    "_CPRACE": _CHILD,
    "_FRT16": _QUALITY_FLAG,
    "_VEG23": _QUALITY_FLAG,
    # asked only if CVDINFR4 = 1 (heart attack), which defines MICHD: 100% positives
    "HAREHAB1": "label leakage: asked only after a heart attack",
}

DROP = "drop"  # not used as a feature (listed in DOMAIN_DROP)
BINARY = "binary"  # two answers, mostly 1 = yes, 2 = no
ORDINAL = "ordinal"  # ordered codes, e.g. GENHLTH 1 excellent .. 5 poor
NOMINAL = "nominal"  # unordered codes, e.g. MARITAL, EMPLOY1
COUNT = "count"  # number of days / times / people
CONTINUOUS = "continuous"  # measured or computed quantity (age, BMI, minutes)
MIXED_UNITS = "mixed_units"  # unit in the first digit (1xx per day, ...) or a date
TYPES = (DROP, BINARY, ORDINAL, NOMINAL, COUNT, CONTINUOUS, MIXED_UNITS)

# column name -> type, in the order of the columns of x_train.csv
FEATURE_TYPES = {
    # record identification and phone screening
    "_STATE": NOMINAL,  # 53 states and territories (FIPS code)
    "FMONTH": DROP,
    "IDATE": DROP,
    "IMONTH": DROP,
    "IDAY": DROP,
    "IYEAR": DROP,
    "DISPCODE": DROP,
    "SEQNO": DROP,
    "_PSU": DROP,
    "CTELENUM": DROP,
    "PVTRESD1": DROP,
    "COLGHOUS": DROP,
    "STATERES": DROP,
    "CELLFON3": DROP,
    "LADULT": DROP,
    "NUMADULT": COUNT,  # landline sample only
    "NUMMEN": COUNT,  # landline sample only
    "NUMWOMEN": COUNT,  # landline sample only
    "CTELNUM1": DROP,
    "CELLFON2": DROP,
    "CADULT": DROP,
    "PVTRESD2": DROP,
    "CCLGHOUS": DROP,
    "CSTATE": DROP,
    "LANDLINE": DROP,
    "HHADULT": COUNT,  # cell sample only, same question as NUMADULT
    # core questionnaire: health, conditions, demographics, habits
    "GENHLTH": ORDINAL,  # 1 excellent .. 5 poor
    "PHYSHLTH": COUNT,  # days in the past 30
    "MENTHLTH": COUNT,
    "POORHLTH": COUNT,
    "HLTHPLN1": BINARY,
    "PERSDOC2": NOMINAL,  # 1 one doctor, 2 more than one, 3 none
    "MEDCOST": BINARY,
    "CHECKUP1": ORDINAL,  # 1 < 1 year .. 4 = 5+ years ago; 8 = never, after 4
    "BPHIGH4": NOMINAL,  # 1 yes, 2 only in pregnancy, 3 no, 4 borderline
    "BPMEDS": BINARY,
    "BLOODCHO": BINARY,
    "CHOLCHK": ORDINAL,  # 1 < 1 year .. 4 = 5+ years ago
    "TOLDHI2": BINARY,
    "CVDSTRK3": BINARY,
    "ASTHMA3": BINARY,
    "ASTHNOW": BINARY,
    "CHCSCNCR": BINARY,
    "CHCOCNCR": BINARY,
    "CHCCOPD1": BINARY,
    "HAVARTH3": BINARY,
    "ADDEPEV2": BINARY,
    "CHCKIDNY": BINARY,
    "DIABETE3": NOMINAL,  # 1 yes, 2 only in pregnancy, 3 no, 4 pre-diabetes
    "DIABAGE2": CONTINUOUS,  # age when told diabetic
    "SEX": BINARY,
    "MARITAL": NOMINAL,
    "EDUCA": ORDINAL,
    "RENTHOM1": NOMINAL,  # own, rent, other
    "NUMHHOL2": BINARY,  # landline sample: more than one phone number
    "NUMPHON2": COUNT,  # landline sample: number of residential phone numbers
    "CPDEMO1": BINARY,  # landline sample: has a cell phone
    "VETERAN3": BINARY,
    "EMPLOY1": NOMINAL,
    "CHILDREN": COUNT,
    "INCOME2": ORDINAL,  # 1 < $10k .. 8 = $75k+
    "INTERNET": BINARY,
    "WEIGHT2": MIXED_UNITS,  # pounds, or 9xxx = kg: WTKG3 is the converted value
    "HEIGHT3": MIXED_UNITS,  # 510 = 5 ft 10 in, or 9xxx = cm: HTM4 is converted
    "PREGNANT": BINARY,
    "QLACTLM2": BINARY,
    "USEEQUIP": BINARY,
    "BLIND": BINARY,
    "DECIDE": BINARY,
    "DIFFWALK": BINARY,
    "DIFFDRES": BINARY,
    "DIFFALON": BINARY,
    "SMOKE100": BINARY,
    "SMOKDAY2": ORDINAL,  # 1 every day, 2 some days, 3 not at all
    "STOPSMK2": BINARY,
    "LASTSMK2": ORDINAL,  # 1 < 1 month .. 7 = 10+ years, 8 = never regularly
    "USENOW3": ORDINAL,  # 1 every day, 2 some days, 3 not at all
    "ALCDAY5": MIXED_UNITS,  # 1xx days per week, 2xx per month: see DROCDY3_
    "AVEDRNK2": COUNT,
    "DRNK3GE5": COUNT,
    "MAXDRNKS": COUNT,
    "FRUITJU1": MIXED_UNITS,  # 1xx per day, 2xx per week, 3xx per month: FTJUDA1_
    "FRUIT1": MIXED_UNITS,  # converted: FRUTDA1_
    "FVBEANS": MIXED_UNITS,  # converted: BEANDAY_
    "FVGREEN": MIXED_UNITS,  # converted: GRENDAY_
    "FVORANG": MIXED_UNITS,  # converted: ORNGDAY_
    "VEGETAB1": MIXED_UNITS,  # converted: VEGEDA1_
    "EXERANY2": BINARY,
    "EXRACT11": NOMINAL,  # about 75 types of activity
    "EXEROFT1": MIXED_UNITS,  # 1xx per week, 2xx per month: PAFREQ1_
    "EXERHMM1": MIXED_UNITS,  # hours and minutes (130 = 1 h 30): PADUR1_
    "EXRACT21": NOMINAL,
    "EXEROFT2": MIXED_UNITS,  # converted: PAFREQ2_
    "EXERHMM2": MIXED_UNITS,  # converted: PADUR2_
    "STRENGTH": MIXED_UNITS,  # 1xx per week, 2xx per month: STRFREQ_
    "LMTJOIN3": BINARY,
    "ARTHDIS2": BINARY,
    "ARTHSOCL": ORDINAL,  # 1 a lot, 2 a little, 3 not at all
    "JOINPAIN": ORDINAL,  # pain scale 0..10
    "SEATBELT": ORDINAL,  # 1 always .. 5 never; 8 = never in a car
    "FLUSHOT6": BINARY,
    "FLSHTMY2": MIXED_UNITS,  # date of last flu shot, mmyyyy
    "IMFVPLAC": NOMINAL,
    "PNEUVAC3": BINARY,
    "HIVTST6": BINARY,
    "HIVTSTD3": MIXED_UNITS,  # date of last HIV test, mmyyyy
    "WHRTST10": NOMINAL,
    # optional modules (asked only in some states, or only to a subgroup)
    "PDIABTST": BINARY,
    "PREDIAB1": NOMINAL,  # 1 yes, 2 only in pregnancy, 3 no
    "INSULIN": BINARY,
    "BLDSUGAR": MIXED_UNITS,  # 1xx per day .. 4xx per year
    "FEETCHK2": MIXED_UNITS,  # 1xx per day .. 4xx per year; 555 = no feet
    "DOCTDIAB": COUNT,
    "CHKHEMO3": COUNT,  # 98 = never heard of the test
    "FEETCHK": COUNT,
    "EYEEXAM": ORDINAL,  # 1 < 1 month .. 4 = 2+ years ago; 8 = never, after 4
    "DIABEYE": BINARY,
    "DIABEDU": BINARY,
    "CAREGIV1": NOMINAL,  # 1 yes, 2 no, 8 = the person cared for died
    "CRGVREL1": NOMINAL,
    "CRGVLNG1": ORDINAL,
    "CRGVHRS1": ORDINAL,
    "CRGVPRB1": NOMINAL,
    "CRGVPERS": BINARY,
    "CRGVHOUS": BINARY,
    "CRGVMST2": NOMINAL,
    "CRGVEXPT": BINARY,
    "VIDFCLT2": ORDINAL,  # 1 no difficulty .. 5 unable; 6 other reasons, 8 blind
    "VIREDIF3": ORDINAL,  # as VIDFCLT2
    "VIPRFVS2": ORDINAL,  # 1 < 1 month .. 5 never
    "VINOCRE2": NOMINAL,
    "VIEYEXM2": ORDINAL,  # 1 < 1 month .. 5 never; 8 = blind
    "VIINSUR2": BINARY,
    "VICTRCT4": NOMINAL,  # 1 yes, 2 removed, 3 no
    "VIGLUMA2": BINARY,
    "VIMACDG2": BINARY,
    "CIMEMLOS": BINARY,
    "CDHOUSE": ORDINAL,  # 1 always .. 5 never
    "CDASSIST": ORDINAL,
    "CDHELP": ORDINAL,
    "CDSOCIAL": ORDINAL,
    "CDDISCUS": BINARY,
    "WTCHSALT": BINARY,
    "LONGWTCH": MIXED_UNITS,  # 1xx days .. 4xx years; 555 = all my life
    "DRADVISE": BINARY,
    "ASTHMAGE": CONTINUOUS,  # age at asthma diagnosis; 97 = 10 or younger
    "ASATTACK": BINARY,
    "ASERVIST": COUNT,  # 98 = don't know
    "ASDRVIST": COUNT,  # 98 = don't know
    "ASRCHKUP": COUNT,  # 98 = don't know
    "ASACTLIM": COUNT,  # days in the past year
    "ASYMPTOM": ORDINAL,  # 1 < once a week .. 5 all the time; 8 = never, before 1
    "ASNOSLEP": ORDINAL,  # 8 = none, before 1
    "ASTHMED3": ORDINAL,  # 8 = never, before 1
    "ASINHALR": ORDINAL,  # 8 = never, before 1
    "HAREHAB1": DROP,
    "STREHAB1": BINARY,
    "CVDASPRN": BINARY,
    "ASPUNSAF": NOMINAL,  # 1 yes, 2 yes (stomach), 3 no
    "RLIVPAIN": BINARY,
    "RDUCHART": BINARY,
    "RDUCSTRK": BINARY,
    "ARTTODAY": ORDINAL,
    "ARTHWGT": BINARY,
    "ARTHEXER": BINARY,
    "ARTHEDU": BINARY,
    "TETANUS": NOMINAL,  # 1-3 yes (type of shot), 4 no
    "HPVADVC2": NOMINAL,  # 1 yes, 2 no, 3 doctor refused
    "HPVADSHT": ORDINAL,  # 1, 2 shots, 3 = all shots
    "SHINGLE2": BINARY,
    "HADMAM": BINARY,
    "HOWLONG": ORDINAL,  # 1 < 1 year .. 5 = 5+ years ago
    "HADPAP2": BINARY,
    "LASTPAP2": ORDINAL,
    "HPVTEST": BINARY,
    "HPLSTTST": ORDINAL,
    "HADHYST2": BINARY,
    "PROFEXAM": BINARY,
    "LENGEXAM": ORDINAL,
    "BLDSTOOL": BINARY,
    "LSTBLDS3": ORDINAL,
    "HADSIGM3": BINARY,
    "HADSGCO1": BINARY,  # 1 sigmoidoscopy, 2 colonoscopy
    "LASTSIG3": ORDINAL,
    "PCPSAAD2": BINARY,
    "PCPSADI1": BINARY,
    "PCPSARE1": BINARY,
    "PSATEST1": BINARY,
    "PSATIME": ORDINAL,
    "PCPSARS1": NOMINAL,
    "PCPSADE1": NOMINAL,
    "PCDMDECN": NOMINAL,  # several answers are coded as one number (12 .. 4328)
    "SCNTMNY1": ORDINAL,  # 1 always .. 5 never; 8 = not applicable
    "SCNTMEL1": ORDINAL,  # as SCNTMNY1
    "SCNTPAID": NOMINAL,
    "SCNTWRK1": COUNT,  # hours of work per week
    "SCNTLPAD": NOMINAL,
    "SCNTLWK1": COUNT,
    "SXORIENT": NOMINAL,
    "TRNSGNDR": NOMINAL,
    "RCSGENDR": DROP,
    "RCSRLTN2": DROP,
    "CASTHDX2": DROP,
    "CASTHNO2": DROP,
    "EMTSUPRT": ORDINAL,  # 1 always .. 5 never
    "LSATISFY": ORDINAL,  # 1 very satisfied .. 4 very dissatisfied
    "ADPLEASR": COUNT,  # days in the past 2 weeks
    "ADDOWN": COUNT,
    "ADSLEEP": COUNT,
    "ADENERGY": COUNT,
    "ADEAT1": COUNT,
    "ADFAIL": COUNT,
    "ADTHINK": COUNT,
    "ADMOVE": COUNT,
    "MISTMNT": BINARY,
    "ADANXEV": BINARY,
    # questionnaire version and weighting
    "QSTVER": DROP,
    "QSTLANG": DROP,
    "MSCODE": ORDINAL,  # 1 center city .. 5 outside metropolitan areas
    "_STSTR": DROP,
    "_STRWT": DROP,
    "_RAWRAKE": DROP,
    "_WT2RAKE": DROP,
    "_CHISPNC": DROP,
    "_CRACE1": DROP,
    "_CPRACE": DROP,
    "_CLLCPWT": DROP,
    "_DUALUSE": DROP,
    "_DUALCOR": DROP,
    "_LLCPWT": DROP,
    # calculated variables (mostly recodings of the answers above). Units as in the
    # data: the implied decimals of the codebook are already applied, except for
    # DROCDY3_ and _DRNKWEK
    "_RFHLTH": BINARY,
    "_HCVU651": BINARY,  # NaN also for age 65+
    "_RFHYPE5": BINARY,
    "_CHOLCHK": ORDINAL,  # 1 checked < 5 years ago, 2 5+ years ago, 3 never
    "_RFCHOL": BINARY,
    "_LTASTH1": BINARY,
    "_CASTHM1": BINARY,
    "_ASTHMS1": ORDINAL,  # 1 current, 2 former, 3 never
    "_DRDXAR1": BINARY,
    "_PRACE1": NOMINAL,
    "_MRACE1": NOMINAL,
    "_HISPANC": BINARY,
    "_RACE": NOMINAL,
    "_RACEG21": BINARY,
    "_RACEGR3": NOMINAL,
    "_RACE_G1": NOMINAL,
    "_AGEG5YR": ORDINAL,  # 1 = 18-24 .. 13 = 80+
    "_AGE65YR": BINARY,
    "_AGE80": CONTINUOUS,  # age in years, 80 = 80 or older
    "_AGE_G": ORDINAL,  # 1 = 18-24 .. 6 = 65+
    "HTIN4": CONTINUOUS,  # inches
    "HTM4": CONTINUOUS,  # meters
    "WTKG3": CONTINUOUS,  # kg
    "_BMI5": CONTINUOUS,
    "_BMI5CAT": ORDINAL,  # 1 underweight .. 4 obese
    "_RFBMI5": BINARY,
    "_CHLDCNT": ORDINAL,  # 1 no children .. 6 = 5 or more
    "_EDUCAG": ORDINAL,
    "_INCOMG": ORDINAL,
    "_SMOKER3": ORDINAL,  # 1 every day, 2 some days, 3 former, 4 never
    "_RFSMOK3": BINARY,
    "DRNKANY5": BINARY,
    "DROCDY3_": CONTINUOUS,  # drink occasions per day x 100
    "_RFBING5": BINARY,
    "_DRNKWEK": CONTINUOUS,  # drinks per week x 100
    "_RFDRHV5": BINARY,
    "FTJUDA1_": CONTINUOUS,  # times per day
    "FRUTDA1_": CONTINUOUS,
    "BEANDAY_": CONTINUOUS,
    "GRENDAY_": CONTINUOUS,
    "ORNGDAY_": CONTINUOUS,
    "VEGEDA1_": CONTINUOUS,
    "_MISFRTN": COUNT,  # number of missing fruit answers
    "_MISVEGN": COUNT,  # number of missing vegetable answers
    "_FRTRESP": BINARY,
    "_VEGRESP": BINARY,
    "_FRUTSUM": CONTINUOUS,
    "_VEGESUM": CONTINUOUS,
    "_FRTLT1": BINARY,
    "_VEGLT1": BINARY,
    "_FRT16": DROP,
    "_VEG23": DROP,
    "_FRUITEX": NOMINAL,  # 0 included, 1 missing answers, 2 out of range
    "_VEGETEX": NOMINAL,
    "_TOTINDA": BINARY,
    "METVL11_": CONTINUOUS,
    "METVL21_": CONTINUOUS,
    "MAXVO2_": CONTINUOUS,  # estimated from age and sex
    "FC60_": CONTINUOUS,  # estimated from age and sex
    "ACTIN11_": ORDINAL,  # 0 none, 1 moderate, 2 vigorous
    "ACTIN21_": ORDINAL,
    "PADUR1_": CONTINUOUS,  # minutes
    "PADUR2_": CONTINUOUS,
    "PAFREQ1_": CONTINUOUS,  # times per week
    "PAFREQ2_": CONTINUOUS,
    "_MINAC11": CONTINUOUS,
    "_MINAC21": CONTINUOUS,
    "STRFREQ_": CONTINUOUS,  # times per week
    "PAMISS1_": BINARY,
    "PAMIN11_": CONTINUOUS,
    "PAMIN21_": CONTINUOUS,
    "PA1MIN_": CONTINUOUS,
    "PAVIG11_": CONTINUOUS,
    "PAVIG21_": CONTINUOUS,
    "PA1VIGM_": CONTINUOUS,
    "_PACAT1": ORDINAL,  # 1 highly active .. 4 inactive
    "_PAINDX1": BINARY,
    "_PA150R2": ORDINAL,
    "_PA300R2": ORDINAL,
    "_PA30021": BINARY,
    "_PASTRNG": BINARY,
    "_PAREC1": NOMINAL,  # 1 both guidelines, 2 aerobic, 3 strength, 4 neither
    "_PASTAE1": BINARY,
    "_LMTACT1": NOMINAL,  # arthritis and limitation combined
    "_LMTWRK1": NOMINAL,
    "_LMTSCL1": NOMINAL,
    "_RFSEAT2": BINARY,
    "_RFSEAT3": BINARY,
    "_FLSHOT6": BINARY,  # age 65+ only
    "_PNEUMO2": BINARY,  # age 65+ only
    "_AIDTST3": BINARY,
}

# B10 binning: group the values of a count column into a few classes (bins).
# src/preprocessing.bin_columns replaces the value v with the bin index k
# (edges[k-1] <= v < edges[k]), the pipeline one-hot encodes it.
# PHYSHLTH (days of bad physical health in the past 30): the raw column has every
# number of days, 0 to 30. We group them because single days are noisy: answers heap
# on round numbers (0, 7, 14, 15, 30) and days such as 19 or 23 have only a few hundred
# rows, so their MICHD prevalence jumps around. The bins are weeks, all of the same
# width, plus the two answers that stand apart, "none" and "every day":
#   days    0      1-7    8-14   15-21  22-29  30
#   rows    63.6%  20.2%  3.8%   3.6%   0.9%   7.9%
#   MICHD   5.9%   8.4%   14.2%  18.6%  22.3%  22.9%
# Balanced bins are not possible (0 days alone is 63.6% of the rows); narrower bins
# (5 days) leave a few thousand rows per bin and the noise comes back. Switch off with
# binning=False to use the raw number of days (to be decided in cross-validation).
BINNED = {
    "PHYSHLTH": (1, 8, 15, 22, 30),  # bins 0, 1-7, 8-14, 15-21, 22-29, 30
}

# B9: ordinal columns one-hot encoded like the nominal ones (one_hot_ordinal=True):
# the answers have an order, but the MICHD prevalence goes up and down along it, and
# the distance between two codes is not a quantity, so one weight per code does not fit
ONE_HOT_ORDINAL = {
    # 1 every day, 2 some days, 3 former, 4 never: former smokers 13.5% (older, and
    # some quit after the heart attack), never 6.2%, daily 10.4%
    "_SMOKER3": "smoking status, former smokers highest",
    # 0 smoking now (skip logic), 1-7 time since quitting in intervals of very
    # different widths (< 1 month .. 10+ years), 8 never smoked: not a time scale
    "LASTSMK2": "time since quitting, uneven intervals + now / never",
    # 1 highly active, 2 active, 3 insufficiently active, 4 inactive: U-shaped
    "_PACAT1": "activity category, U-shaped",
    # 1 within a year .. 4 five+ years, 5 never: last year 10.2%, the others 4-6%
    # (people with heart disease have regular check-ups)
    "CHECKUP1": "last check-up, highest in the last year",
    # the one-hot of these two rebuilds exactly the columns of REDUNDANT that group
    # their classes. CHOLCHK: 1 < 1 year .. 4 = 5+ years, 5 never (skip logic),
    # curved log-odds (R^2 0.90 for a line, 0.99 for a parabola)
    "CHOLCHK": "last cholesterol check, rebuilds BLOODCHO and _CHOLCHK",
    # _ASTHMS1: 1 current, 2 former, 3 never
    "_ASTHMS1": "asthma status, rebuilds ASTHMA3, ASTHNOW, _LTASTH1, _CASTHM1",
}

# B2b: column name -> (kept column it duplicates, reason) (drop_redundant=True).
# A column is listed only with three pieces of evidence:
#  - data: experiments/eda.py A7 compares every pair of raw columns (before one-hot
#    and bins) and checks each entry here: the kept column determines it
#    (U >= 0.95) or is the same quantity (|r| >= 0.999); build/eda/redundancy.txt,
#    section "Check of REDUNDANT";
#  - codebook (2015 BRFSS codebook, section in brackets): the column is computed
#    from the kept one, or is the same question;
#  - encoding: a linear model rebuilds the dropped column exactly from the kept one
#    (kept one-hot or binary, or both a linear function of each other), and the two
#    have the same missing values (within 2 points).
# In each family the kept column is the finest one (all the others are groupings
# of it) and, at equal detail, the more complete one or the raw question.
# Not listed, although A7 links them to a kept column (they add information that a
# linear model cannot rebuild; dropping them lowered the validation F1 from 0.4215
# to 0.4193, while dropping the 30 below left it at 0.4215):
#  - groupings of a numeric column, i.e. step functions of it: _AGEG5YR, _AGE_G,
#    _AGE65YR (of _AGE80), _CHLDCNT (CHILDREN), _EDUCAG (EDUCA), _INCOMG (INCOME2),
#    _RFHLTH (GENHLTH), _RFSEAT2, _RFSEAT3 (SEATBELT), _VEGLT1 (_VEGESUM),
#    _RFBMI5 (_BMI5CAT) [7.11-7.14, 7.20-7.23, 1.1, 13.1-13.2, 10.14 calc.];
#  - the same answer for one age range only, blank for the others, so that their
#    missing values encode the age: _HCVU651 (HLTHPLN1, 18-64), _FLSHOT6
#    (FLUSHOT6, 65+), _PNEUMO2 (PNEUVAC3, 65+) [3.1, 14.1, 14.2 calc.].
_PACAT1_GROUPS = "groups of the 4 classes of _PACAT1 (one-hot) [11.22-11.26]"
REDUNDANT = {
    # MAXVO2_ = f(age, sex), kept (works as an age x sex term); FC60_ linear in it
    "FC60_": ("MAXVO2_", "functional capacity, linear in MAXVO2_ [11.4, 11.5]"),
    # height: HTM4 also has the metric answers, HTIN4 is blank for them
    "HTIN4": ("HTM4", "same height in inches [7.15, 7.16]"),
    # race and ethnicity: _RACE (8 classes, one-hot) [7.7] combines _HISPANC and
    # _MRACE1; the others group its classes
    "_RACEGR3": ("_RACE", "_RACE 3-6 merged [7.9]"),
    "_RACE_G1": ("_RACE", "_RACEGR3 with another coding [7.10]"),
    "_RACEG21": ("_RACE", "non-Hispanic white or not [7.8]"),
    "_HISPANC": ("_RACE", "Hispanic = _RACE 8 [7.6]"),
    # blood pressure
    "_RFHYPE5": ("BPHIGH4", "BPHIGH4 1 / 2-4, BPHIGH4 one-hot [4.1 calc.]"),
    # cholesterol: CHOLCHK one-hot, 5 = never checked after the skip logic
    "_CHOLCHK": ("CHOLCHK", "CHOLCHK 1-3 / 4 / never [5.1 calc.]"),
    "BLOODCHO": ("CHOLCHK", "ever checked = CHOLCHK not 5 (skip logic) [5.1]"),
    "_RFCHOL": ("TOLDHI2", "TOLDHI2 recoded, BLOODCHO = 1 [5.2 calc.]"),
    # asthma: _ASTHMS1 current / former / never, one-hot [6.4 calc.]
    "ASTHMA3": ("_ASTHMS1", "ever asthma = _ASTHMS1 1-2 [6.4]"),
    "_LTASTH1": ("_ASTHMS1", "ASTHMA3 recoded [6.2 calc.]"),
    "ASTHNOW": ("_ASTHMS1", "still asthma = _ASTHMS1 1 (skip logic) [6.5]"),
    "_CASTHM1": ("_ASTHMS1", "current asthma = _ASTHMS1 1 [6.3 calc.]"),
    # arthritis: the _LMT* (one-hot) combine HAVARTH3 with a follow-up question
    "_DRDXAR1": ("HAVARTH3", "HAVARTH3 recoded [6.5 calc.]"),
    "LMTJOIN3": ("_LMTACT1", "limited by arthritis, within _LMTACT1 [12.1 calc.]"),
    "ARTHDIS2": ("_LMTWRK1", "work limited, within _LMTWRK1 [12.2 calc.]"),
    "ARTHSOCL": ("_LMTSCL1", "social limits, within _LMTSCL1 [12.3 calc.]"),
    # smoking: _SMOKER3 (daily / some days / former / never) and LASTSMK2, one-hot
    "SMOKDAY2": ("_SMOKER3", "every day / some days / not at all [8.1 calc.]"),
    "SMOKE100": ("_SMOKER3", "never smoked = _SMOKER3 4 [8.1 calc.]"),
    "_RFSMOK3": ("_SMOKER3", "current smoker = _SMOKER3 1-2 [8.2 calc.]"),
    # exercise: same answer [11.1 calc.]. EXERANY2, METVL11_, METVL21_ are also
    # functions of EXRACT11/21, but B9 merges the rare activities: kept
    "_TOTINDA": ("EXERANY2", "EXERANY2 recoded [11.1 calc.]"),
    # physical activity, from the minutes per week PA1MIN_ (kept as a number).
    # _PACAT1: 1 > 300 min, 2 150-300, 3 1-149, 4 none
    "_PA150R2": ("_PACAT1", _PACAT1_GROUPS),  # 150+ (1, 2), 1-149 (3), none (4)
    "_PA300R2": ("_PACAT1", _PACAT1_GROUPS),  # 300+ (1), 1-300 (2, 3), none (4)
    "_PA30021": ("_PACAT1", _PACAT1_GROUPS),  # 300+ (1) or not
    "_PAINDX1": ("_PACAT1", _PACAT1_GROUPS),  # 150+ (1, 2) or not
    "_PASTAE1": ("_PAREC1", "both guidelines = _PAREC1 1, one-hot [11.28, 11.29]"),
    # fruit and vegetables
    "_FRTRESP": ("_FRUITEX", "missing fruit answers = _FRUITEX 1 [10.9, 10.17]"),
    "_VEGRESP": ("_VEGETEX", "missing vegetable answers = _VEGETEX 1 [10.10, 10.18]"),
    # HIV test
    "_AIDTST3": ("HIVTST6", "HIVTST6 recoded [15.1 calc.]"),
}

# raw mixed-units column -> the same answer converted to one unit by the CDC.
# The other mixed-units columns (BLDSUGAR, FEETCHK2, LONGWTCH, FLSHTMY2, HIVTSTD3)
# have no converted version: src/preprocessing.convert_units converts them.
CONVERTED = {
    "WEIGHT2": "WTKG3",  # kg
    "HEIGHT3": "HTM4",  # meters
    "ALCDAY5": "DROCDY3_",  # drink occasions per day x 100
    "FRUITJU1": "FTJUDA1_",  # times per day
    "FRUIT1": "FRUTDA1_",
    "FVBEANS": "BEANDAY_",
    "FVGREEN": "GRENDAY_",
    "FVORANG": "ORNGDAY_",
    "VEGETAB1": "VEGEDA1_",
    "EXEROFT1": "PAFREQ1_",  # times per week
    "EXEROFT2": "PAFREQ2_",
    "EXERHMM1": "PADUR1_",  # minutes
    "EXERHMM2": "PADUR2_",
    "STRENGTH": "STRFREQ_",  # times per week
}
