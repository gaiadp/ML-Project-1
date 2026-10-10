"""Column lists from the BRFSS 2015 codebook, shared by the EDA and the pipeline.

DOMAIN_DROP: columns that are not used as features, with the reason (interview ids
and dates, phone screening, survey design and weights, a child instead of the
respondent, label leakage). experiments/eda.py reports how much MICHD signal each
of them carries.

FEATURE_TYPES: the type of every column (one of TYPES), from the codebook. It decides
how the pipeline encodes a column. Comments mark the codes that need care.
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
