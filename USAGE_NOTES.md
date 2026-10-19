# Accortezze per usare il codice e costruire i modelli

Note per il gruppo: cose non ovvie da tenere a mente quando si usano le funzioni in
`src/` e `implementations.py`. Ogni punto indica dove si trova il codice interessato.

**Indice**

1. [Dati ed etichette](#1-dati-ed-etichette)
2. [Codici speciali BRFSS](#2-codici-speciali-brfss)
3. [Ordine del preprocessing](#3-ordine-del-preprocessing)
4. [One-hot encoding](#4-one-hot-encoding)
5. [Espansioni di feature](#5-espansioni-di-feature)
6. [I 6 metodi](#6-i-6-metodi)
7. [Valutazione e submission](#7-valutazione-e-submission)
8. [Metodi extra](#8-metodi-extra)

---

## 1. Dati ed etichette

- **Etichette**: `y_train.csv` e `load_csv_data` (`helpers.py`) danno `y` in `{-1, 1}`.
  - `logistic_regression` / `reg_logistic_regression` vogliono `y` in `{0, 1}`:

    ```python
    y01 = (y == 1).astype(float)
    ```

  - la submission (`create_csv_submission`) vuole di nuovo `{-1, 1}`, altrimenti `ValueError`:

    ```python
    y_pred = np.where(p >= soglia, 1, -1)
    ```

- **Classi sbilanciate**: solo ~8.8% di positivi (MICHD). Un modello che dice sempre
  "no" ha ~91% di accuracy → valutare con F1, pesare le classi o regolare la soglia di
  decisione (0.5 è quasi sempre troppo alta). La soglia va scelta su validation, **mai**
  sul test.
- **Caricamento lento**: `np.genfromtxt` su `x_train` impiega minuti. `src.data.load_data()`
  salva gli array una volta in `build/data.npz` (`np.savez`) e poi li ricarica con `np.load`.
- **Memoria**: `x_train` è 328135 × 321 float64 = ~840 MB. Espansioni polinomiali o
  one-hot su molte colonne la moltiplicano: selezionare le feature **prima** di espandere
  (oppure usare float32).
- `load_csv_data` toglie la colonna `Id`: i nomi delle colonne corrispondenti sono
  l'header di `x_train.csv` **senza** il primo elemento (`"Id"`).

---

## 2. Codici speciali BRFSS

*Codice: `src/preprocessing.py` → `codes_to_nan`*

- Il significato di un codice cambia da colonna a colonna: `EMPLOY1 = 7` è "pensionato",
  `_AGEG5YR = 7/9` sono fasce d'età, `_AGE80 = 77` è un'età, `_STATE = 9` è il Connecticut.
  Per questo `codes_to_nan(x, names)` usa regole per **nome** di colonna (`_RULES`).
  - passare sempre i nomi nello stesso ordine delle colonne di `x`;
  - se si aggiunge o toglie una colonna prima di chiamarla, i nomi devono seguire.
- I codici "nessuno / mai" (88, 555, 888, per alcune colonne 98) diventano **0, non NaN**:
  "88 giorni di cattiva salute" in realtà significa 0 giorni.
- Lo stesso codice 8 ha significati diversi (regole controllate col codebook, 10 ottobre):

  | Significato di 8 | Trattamento | Esempi |
  |---|---|---|
  | "mai / nessuno" | → 0 | `ASYMPTOM`, `ASTHMED3`, ... |
  | "non applicabile" | → NaN | `SEATBELT`, `SCNTMNY1`, ... |
  | risposta vera fuori ordine ("mai", va dopo 4) | `convert_units` lo cambia in 5 | `CHECKUP1`, `EYEEXAM` |

  Anche 98 cambia: "non so" nelle visite per asma (`ASERVIST`, ...), "zero" in `SCNTWRK1`.
- Nei dati i decimali impliciti del codebook sono **già** applicati (`WTKG3` in kg, `HTM4` in
  metri, `_BMI5`, `PAFREQ1_` volte/settimana), tranne `DROCDY3_` e `_DRNKWEK` (ancora ×100).
  Quindi anche i codici sono divisi: `PAFREQ1_ = 99` è il codice 99000 "non so".
- Le colonne a 3 cifre hanno **unità miste**:
  - `ALCDAY5`: 1xx = a settimana, 2xx = al mese;
  - `FRUIT1`, `FVBEANS`, `VEGETAB1`, ...: 1xx / 2xx / 3xx = al giorno / settimana / mese;
  - `EXEROFT1/2`, `STRENGTH`;
  - `WEIGHT2` (9xxx = kg), `HEIGHT3` (piedi/pollici oppure 9xxx = cm).

  Non usarle come numeri così come sono. Con `convert_units=True` la pipeline:
  - toglie quelle che hanno una versione già calcolata (`CONVERTED` in
    `src/feature_types.py`: `WEIGHT2` → `WTKG3`, `ALCDAY5` → `DROCDY3_`, ...);
  - converte le altre (`convert_units` in `src/preprocessing.py`): `BLDSUGAR` / `FEETCHK2`
    in volte all'anno, `LONGWTCH` in anni, le date `FLSHTMY2` / `HIVTSTD3` in mesi prima
    dell'intervista.
- Molte domande sì/no sono codificate 1 = sì, 2 = no: il segno del peso appare
  "invertito". Non è un errore, ma va ricordato quando si interpretano i coefficienti.
- **Colonne da non usare come feature**: identificativi e variabili di disegno/peso del
  sondaggio (`SEQNO`, `_PSU`, `IDATE`, `_STSTR`, `_STRWT`, `_LLCPWT`, `_WT2RAKE`, ...).
  Non descrivono la persona e aggiungono solo rumore.
- **Missing informativi**: molte domande vengono poste solo a un sottogruppo (skip logic,
  es. `BPMEDS` solo a chi ha la pressione alta). In quei casi il fatto che il dato manchi
  è predittivo. La pipeline lo gestisce in due modi:
  - dove il NaN vuol dire "no / zero" (domanda non posta perché la risposta di filtro lo
    esclude) lo riempie con quel valore: `fill_skip_logic`, regole in `_SKIP_LOGIC`
    (`src/preprocessing.py`), verificate sui dati (nessuna riga che soddisfa il filtro ha
    una risposta alla domanda successiva);
  - per gli altri NaN informativi aggiunge colonne indicatrici `XXX_isnan` (B6).

---

## 3. Ordine del preprocessing

*Codice: `src/pipeline.py`, `src/preprocessing.py`, `src/features.py`*

Ordine consigliato:

```
codes_to_nan → scelta/rimozione colonne → imputazione → encoding/espansioni
→ standardizzazione → add_bias
```

### 3.1 La pipeline (`src/pipeline.py`)

`src/pipeline.py` concatena i passi in quest'ordine (passi B1–B13, elencati nella tabella
qui sotto e nella docstring di `src/pipeline.py`):

```python
state = fit_preprocess(x_train, y_train, names, config)
tx = apply_preprocess(x, names, state)   # train, validation o test
state["out_names"]                       # nomi delle colonne di tx ("bias" prima)
```

| Passo | Flag in `config` | Cosa fa |
|---|---|---|
| B1 | — | `codes_to_nan`: codici speciali → NaN o 0 |
| B3 | `convert_units` | una scala numerica per colonna; tolte le colonne grezze a unità miste |
| B4 | `skip_logic_fill` | NaN delle domande non poste → "no" / 0 |
| B10 | `binning` | le colonne di `BINNED` (`PHYSHLTH`) → indice della classe, poi one-hot (B9) |
| B2 | `domain_drop`, `drop_redundant`, `select_features` | toglie id / disegno del sondaggio / leakage (`DOMAIN_DROP`), i 30 duplicati esatti (`REDUNDANT`) e, solo in `V1_CONFIG`, le 112 colonne non scelte dall'EDA (`NOT_SELECTED`) |
| B5 | `max_missing_frac`, `keep_informative` | toglie le colonne con troppi NaN, tranne quelle con NaN informativo |
| B6 | `missing_indicators` | una colonna `isnan` per gruppo di colonne informative |
| B7 | `impute` | mediana, oppure moda per binarie / nominali (`"by_type"`) |
| B8 | `recode_binary` | 1/2 → 1/0 |
| B9 | `one_hot`, `one_hot_min_rows`, `one_hot_ordinal` | one-hot delle nominali (e delle classi di B10); con `one_hot_ordinal` anche le ordinali di `ONE_HOT_ORDINAL` |
| B12, B13 | — | standardizzazione (media/std del training) e colonna di bias |

- `config` è un dict di flag, uno per passo (`DEFAULT_CONFIG` = pipeline completa,
  `V0_CONFIG` = baseline: codici → NaN, colonne con > 50% NaN tolte, mediana, one-hot
  delle nominali). Per l'ablation: `dict(DEFAULT_CONFIG, flag=False)`.
  Le liste fisse usate dai passi (`REDUNDANT`, `ONE_HOT_ORDINAL`, `BINNED`, `NOT_SELECTED`
  in `src/feature_types.py`) vengono dall'EDA (`experiments/eda.py`), che guarda le etichette
  di tutto il training: il punteggio in CV è un po' ottimista (dirlo nel report).
  `run.py` usa `V0_CONFIG` finché la pipeline completa non è validata in CV. La prima
  submission era `dict(V0_CONFIG, one_hot=False)` (test in `tests/test_pipeline.py`).
- **`V1_CONFIG`** = `DEFAULT_CONFIG` + selezione delle feature dall'EDA
  (`select_features=True`): usa 124 colonne e toglie le 112 di `NOT_SELECTED`, ognuna con
  la sua regola (R1: MI < 1 mbit; R2: moduli opzionali risposti da < 15%, se ne tiene una
  colonna per modulo; R3: domande solo-fisso con risposte senza segnale; R4: `STREHAB1`,
  copia di `CVDSTRK3` dopo B4). La selezione sostituisce B5 (`max_missing_frac=1.0`): una
  colonna scelta resta anche con tanti NaN (es. `CVDASPRN`); B6 aggiunge comunque gli
  indicatori. In CV dà lo stesso F1 di `DEFAULT_CONFIG` con il 20% di colonne in meno.
  `select_features` accetta anche l'elenco delle regole da applicare, per esempio
  `("R1", "R2", "R4")` per togliere solo la R3 (`not_selected` in `src/pipeline.py`):
  lo usa l'ablation.
- `y_train` serve perché `keep_informative` / `missing_indicators` scelgono le colonne in
  base alla differenza di prevalenza MICHD tra righe NaN e righe con risposta (≥ 3 punti,
  ≥ 500 righe per lato). Con entrambi a `False` si può passare `None`.
  Usare le etichette per scegliere le colonne è un "fit": in CV va fatto sulla sola parte
  di training di ogni fold, come le mediane.
- **Indicatori**: uno per **gruppo** di colonne che mancano insieme (stesso modulo), non uno
  per colonna: `one_per_group` tiene quello con la differenza più grande e salta gli altri
  con maschera correlata > 0.9 (anche in negativo: domande solo-fisso vs solo-cellulare).
- `convert_units` e `fill_skip_logic` modificano `x` **in place** (per non copiare ~800 MB):
  chiamarle sull'output di `codes_to_nan`, che è già una copia.
- **One-hot (B9)**: le nominali (`NOMINAL` in `src/feature_types.py`) diventano colonne
  `"NOME=valore"` (`one_hot=True`, anche in `V0_CONFIG`). Solo in `DEFAULT_CONFIG` anche:
  - le ordinali di `ONE_HOT_ORDINAL` (`one_hot_ordinal=True`): `_SMOKER3`, `LASTSMK2`,
    `_PACAT1`, `CHECKUP1` hanno una prevalenza che sale e scende lungo i codici; `CHOLCHK`
    e `_ASTHMS1` sono one-hot perché così ricostruiscono le colonne tolte come duplicati
    (`REDUNDANT`);
  - le colonne di `BINNED`, raggruppate in classi (`PHYSHLTH`: 0, 1–7, 8–14, 15–21, 22–29,
    30 giorni; la colonna originale ha tutti i giorni, raggruppati per togliere il rumore
    dei valori rari; B10, `binning=True`).
- Le binarie 1/2 diventano 1/0 (B8; 1 = codice 1 del codebook: "sì" per le domande, ma
  "nessun fattore di rischio" per le `_RF*`).

### 3.2 Regole generali

- **Fit solo sul training.** Tutte le statistiche (mediane, media/std, maschera delle
  colonne, categorie del one-hot) si calcolano sul training e si **riusano** su validation
  e test. Calcolarle su tutto il dataset = data leakage (stime troppo ottimistiche).
- In cross-validation vale lo stesso: il preprocessing va rifittato **dentro** ogni fold,
  solo sulla parte di training del fold.
- `drop_high_missing_columns` restituisce una maschera: calcolarla sul training e applicare
  la **stessa** maschera (`drop_columns`) a validation e test.
- `fit_impute_median`: va chiamata **dopo** la rimozione delle colonne, così gli indici
  delle mediane corrispondono. Una colonna tutta NaN ha mediana NaN e resta NaN.
- `fit_standardize`: con NaN presenti la media diventa NaN → imputare **prima**.
  Le colonne costanti (std = 0) vengono lasciate con std = 1 (diventano tutte 0).
- `add_bias` va chiamata **dopo** la standardizzazione: la colonna di 1 ha std = 0, quindi
  standardizzandola diventerebbe una colonna di 0 e il modello perderebbe l'intercetta.

---

## 4. One-hot encoding

*Codice: `src/features.py` → `one_hot_encode`, `fit_one_hot_categories`*

- Serve per le variabili categoriche: i codici numerici (es. `EMPLOY1` = 1..8) non sono
  quantità, e trattarli come numeri impone un andamento lineare che non esiste.
  Con il one-hot ogni categoria ha il suo peso.
- Funzionamento: `(x_col[:, None] == categories[None, :])` crea una matrice (N, K) con 1
  nella colonna della categoria della riga e 0 altrove.
- **Nella pipeline (B9)** lo fa `fit_one_hot_categories`: categorie fissate sul training, il
  valore più frequente è il riferimento (riga tutta a 0, niente collinearità col bias); i
  valori con meno di `one_hot_min_rows` (500) righe, i NaN (già imputati) e i valori mai
  visti finiscono nel riferimento.
- **A mano**, le categorie vanno fissate sul training e riusate, altrimenti, se una
  categoria manca in uno split, le colonne non corrispondono:

  ```python
  enc_tr, cats = one_hot_encode(x_train[:, j])
  enc_te, _ = one_hot_encode(x_test[:, j], categories=cats)
  ```

- NaN e categorie mai viste nel training → riga tutta a 0 (`NaN == x` è sempre `False`).
  Il NaN diventa quindi la "categoria di riferimento" e il suo effetto finisce nel bias.
  Se si vuole un peso proprio per "mancante", aggiungere una colonna `np.isnan(x_col)`.
  Se invece si imputa prima con la mediana, i NaN finiscono nella categoria mediana.
- **Collinearità**: senza NaN ogni riga ha esattamente un 1, quindi le K colonne sommano
  alla colonna del bias. Con ridge / logistica regolarizzata va bene; con `least_squares`
  (o GD non regolarizzato) togliere una colonna di riferimento: `enc[:, 1:]`.
- Non usarlo su colonne con tanti valori distinti (età, peso, giorni): troppe colonne e
  pochi esempi per categoria. Per quelle: termini lineari/polinomiali o binning.

---

## 5. Espansioni di feature

*Codice: `src/features.py` → `build_poly`, `pairwise_products`*

- `build_poly` restituisce `[x, x^2, ..., x^d]` **senza** la colonna di 1: il bias va
  aggiunto a parte con `add_bias` (una volta sola per tutta la matrice).
- Applicarla a feature già standardizzate: le potenze di valori grandi (es. 200⁴)
  esplodono e rendono XᵀX mal condizionata. Dopo l'espansione ri-standardizzare.
- Inutile su colonne binarie o one-hot: x² = x, si creano solo colonne duplicate.
- `pairwise_products`: il numero di coppie cresce come D² → scegliere poche coppie
  sensate, non tutte.
- **Feature ridondanti** (es. 5 versioni dell'età: `_AGE80`, `_AGEG5YR`, `_AGE_G`,
  `_AGE65YR`, e `MAXVO2_` / `FC60_` calcolate dall'età; `_RFCHOL` = `TOLDHI2` ricodificata;
  `_DRDXAR1` = `HAVARTH3`) creano collinearità: tenerne una per gruppo.
  - `experiments/eda.py` A7 confronta tutte le coppie di colonne grezze (prima di one-hot
    e classi) → `build/eda/redundancy.txt`: proposta automatica, famiglie di colonne
    collegate e verifica di ogni voce di `REDUNDANT` (`src/feature_types.py`), la lista
    che la pipeline toglie con `drop_redundant=True` (la pipeline non legge il `.txt`).
  - Una colonna va in `REDUNDANT` solo se: è funzione di quella tenuta (U ≥ 0.95) o la
    stessa grandezza (|r| ≥ 0.999), il codebook lo conferma, quella tenuta è one-hot o
    binaria (il modello lineare la ricostruisce esattamente) e hanno gli stessi NaN.
  - I raggruppamenti di una colonna numerica (`_AGE65YR`, `_RFHLTH`, ...) e le versioni per
    una sola fascia d'età (`_HCVU651`, `_FLSHOT6`, `_PNEUMO2`) restano: toglierli
    abbassava l'F1 di validazione.

---

## 6. I 6 metodi

*Codice: `implementations.py`*

- Le firme **non** vanno cambiate: il grader le importa direttamente.
- Ogni funzione restituisce `(w, loss)` con `w` = **ultimo** vettore di pesi (non il
  migliore) e `loss` calcolata su quel `w`. Vettori 1D di forma `(D,)`, non `(D, 1)`.
- MSE con fattore 1/2: `loss = mean((y - tx @ w)**2) / 2`.
- Nei metodi regolarizzati la loss restituita **non** include il termine di penalità.
- `ridge_regression` usa la convenzione `(XᵀX + 2 N λ I) w = Xᵀy`: i valori di λ non sono
  confrontabili con quelli di altre convenzioni (es. `λ I`).
  La penalità si applica a **tutti** i pesi, compreso il bias (`np.eye` su tutte le
  colonne): con features standardizzate l'effetto è piccolo, ma va ricordato.
- La penalità ridge/L2 tratta tutte le feature allo stesso modo solo se hanno la stessa
  scala → standardizzare prima.
- `least_squares` usa una pseudo-inversa (autovalori ~0 ignorati): funziona anche con
  colonne collineari, ma in quel caso la soluzione non è unica (minima norma).
  `numpy.linalg.lstsq` è vietato.
- GD/SGD: il passo `gamma` dipende dalla scala delle feature; senza standardizzazione
  diverge o è lentissimo. Con la logistica: calcolare sigmoide e log-loss in modo
  numericamente stabile (evitare `log(0)` ed `exp` di numeri grandi, es. `np.logaddexp`).
- Verificare con `tests/test_implementations.py` prima di consegnare.

---

## 7. Valutazione e submission

- Split / k-fold **stratificati** (stessa percentuale di positivi in ogni fold), con seed
  fisso per rendere i risultati riproducibili.
- `src/cross_validation.py`: `cross_validate(x, y, names, config, method, ...)` e
  `cross_validation_ridge(x, y, names, config, lambdas)` vogliono i dati **grezzi** e
  fittano la pipeline dentro ogni fold, solo sulla parte di training.
  - La soglia di decisione si sceglie sugli score di **training** del fold.
  - `f1` (con media `f1_mean` e deviazione standard `f1_std` sui fold) è il numero da
    riportare; `f1_val_best` (soglia scelta sulla validazione stessa) è ottimista come il
    singolo split di `run.py`.
  - `f1_default`: F1 con la soglia naturale del metodo (0.5 per le probabilità, 0 per i
    metodi lineari), senza ottimizzarla: è la riga "soglia 0.5" dell'ablation.
  - Ablation: `experiments/ablation.py` (spiegata in `experiments/README.md`).
  - `f1_rule`: F1 dopo la regola dei positivi noti (`src/rules.py`: le righe in cui
    `HAREHAB1` è stata chiesta, anche con risposta 7/9, messe a 1 dopo la soglia).
- **`HAREHAB1`** (leakage dell'etichetta): tolta da ogni config. `keep_leakage=True` la
  tiene, solo per misurarne l'effetto (`experiments/leakage.py`: modello B con la colonna,
  modello C con la regola). La regola guarda i dati GREZZI: dopo `codes_to_nan` le risposte
  7/9 diventerebbero NaN e 15 positivi certi andrebbero persi.
- `run.py` deve riprodurre **esattamente** la nostra migliore submission: fissare i seed e
  usare lo stesso preprocessing (con statistiche fittate sul training completo).
- Gli `Id` della submission sono quelli di `x_test` (`test_ids` di `load_csv_data`), nello
  stesso ordine delle predizioni.

---

## 8. Metodi extra

`src/extra_implementations/` (solo NumPy), usati con `fit` / `predict_scores` / `predict` di
`src/models.py` come i 6 metodi (`ALL_METHODS` = `METHODS` + `EXTRA_METHODS`).

- **4 ottimizzatori della stessa loss** di `reg_logistic_regression` (logistica media +
  `lambda_ * ||w||^2`, penalità anche sul bias): `reg_logistic_adaptive_gd` (passo di
  Barzilai-Borwein + backtracking), `reg_logistic_newton` (IRLS), `reg_logistic_sophia`
  (minibatch, diagonale dell'Hessiana, passo clippato), `reg_logistic_unixgrad`
  (extra-gradient con passo adattivo).
  - A convergenza danno **lo stesso `w`** di `reg_logistic_regression` con abbastanza
    iterazioni: l'F1 cambia solo se il GD a passo fisso non era convergente. Il guadagno è
    in velocità e nel non dover scegliere `gamma`.
  - Significato dei parametri di `fit`: `gamma` = primo passo di prova per l'adaptive GD,
    passo massimo per coordinata per Sophia, ignorato da Newton e UniXGrad.
  - Parametri propri, passati a `fit` come argomenti in più (`**options`): Sophia
    `batch_size`, `beta1`, `beta2`, `rho`, `hess_every`; UniXGrad `radius` (deve contenere
    l'ottimo: se troppo piccolo, il risultato è l'ottimo vincolato alla palla); Newton `tol`.
- **2 ensemble di alberi**: `random_forest` e `gradient_boosting` (alberi a istogrammi, 32
  bin per colonna ai quantili del training).
  - `fit` restituisce un **dict** (bordi dei bin + alberi), non un vettore: non fare
    `tx @ w`, usare sempre `predict_scores`. Gli score sono probabilità (soglia 0.5).
  - Parametri propri: `n_trees`, `max_depth`, `min_samples_leaf`, `subsample`, `n_bins`;
    RF anche `max_features` (default `sqrt(D)`); boosting `learning_rate`, `reg_lambda`,
    `colsample`. `lambda_`, `gamma`, `max_iters`, `initial_w` sono ignorati.
  - Non serve standardizzare (conta solo l'ordine dei valori), ma l'output della pipeline
    va bene così com'è. Le potenze / interazioni di B11 non servono: gli alberi le
    costruiscono da soli.
  - Lenti: ~3 min per fit (RF, 100 alberi) e ~6 min (boosting, 200 alberi) su un fold
    di CV, quindi decine di minuti per riga di griglia.
- **Ibrido `logistic_boosting`**: logistica regolarizzata (Newton, `lambda_`, `max_iters`,
  `initial_w`) e poi gradient boosting **sui suoi residui**: gli alberi partono dal margine
  `tx @ w` invece che dalla prevalenza, quindi imparano solo quello che il modello lineare non
  vede (interazioni, effetti curvi). Con `n_trees=0` è esattamente la logistica di Newton.
  Parametri degli alberi come per `gradient_boosting` (default `learning_rate=0.05`,
  `max_depth=3`, `n_trees=100`); `fit` restituisce un dict (`w` + alberi).
- Con i metodi base, un parametro di `**options` dà `ValueError`; un parametro sbagliato per
  un metodo extra dà `TypeError`. `pos_weight` resta solo per le 2 logistiche base.
- Esperimenti: `experiments/<metodo>/01_baseline.py` e `04_final.py` per ogni metodo extra
  (stessi fold e seed degli altri metodi).
