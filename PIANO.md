# Piano del progetto

Programma di lavoro per il Project 1 (scadenza: giovedì 29 ottobre 2026, ore 16:00), con lo stato di ogni punto.
Le caselle spuntate sono fatte; tra parentesi chi se ne è occupato e dove si trova il codice.

**Indice**

- [Settimana 1 (7–12 ott): toolbox funzionante e prima submission](#settimana-1-712-ott-toolbox-funzionante-e-prima-submission)
- [Settimana 2 (13–19 ott): EDA, feature e cross-validation](#settimana-2-1319-ott-eda-feature-e-cross-validation)
- [Settimana 3 (20–25 ott): esperimenti per il report](#settimana-3-2025-ott-esperimenti-per-il-report)
- [Ultimi giorni (26–29 ott): report e consegna](#ultimi-giorni-2629-ott-report-e-consegna)

---

## Settimana 1 (7–12 ott): toolbox funzionante e prima submission

### 1. Finire `implementations.py` ✅

- [x] `logistic_regression`, con σ(z) = 1 / (1 + e^{-z}):
  - loss = mean(log(1 + e^{z}) − y·z) con z = tx @ w
  - gradiente = tx.T @ (σ(z) − y) / N
- [x] `reg_logistic_regression`: stesso gradiente + 2·λ·w. La loss restituita **non** include la penalità.
- [x] In tutte le funzioni iterative la loss va calcolata sul w finale. Con `max_iters=0` devono restituire `initial_w` e la sua loss.
- [x] Per evitare overflow si usa `np.logaddexp(0, z)` al posto di `log(1 + exp(z))`.

### 2. Testare le implementazioni ✅

- [x] Test ufficiali (`conftest.py` e `test_project1_public.py` da
  `github.com/epfml/ML_course/tree/main/projects/project1/grading_tests`), lanciati con
  `pytest --github_link <path-o-url-del-repo> .` dalla cartella `grading_tests`.
- [x] Test nostri in `tests/test_implementations.py`:
  - [x] GD con tante iterazioni converge allo stesso w di `least_squares`
  - [x] `ridge_regression` con λ = 0 coincide con `least_squares`
  - [x] `reg_logistic_regression` con λ = 0 coincide con `logistic_regression`
  - [x] gradiente controllato numericamente con differenze finite
  - [x] tutti gli output hanno forma (D,), non (D, 1)
  - [x] la loss dei metodi regolarizzati non contiene la penalità

### 3. Infrastruttura ✅

- [x] Cache dei dati con `np.savez` in `build/` (`src/data.py`): il caricamento con `genfromtxt` richiede circa un minuto.
- [x] `src/metrics.py`: accuracy, precision, recall, **F1** (la metrica di AIcrowd; con l'8.8% di positivi l'accuracy non dice nulla) e `best_threshold`, che sceglie la soglia sul validation set.
- [x] `src/models.py`: wrapper `fit` / `predict` uniformi attorno ai 6 metodi, più la conversione delle etichette {-1, 1} ↔ {0, 1}.

### 4. Prima pipeline end-to-end ✅

- [x] Preprocessing minimo, poi ridge, poi soglia, poi submission su AIcrowd (`run.py`, submission del 10/10).
- [x] Serve soprattutto a verificare che il formato della submission sia corretto.

---

## Settimana 2 (13–19 ott): EDA, feature e cross-validation

### 5. `experiments/eda.py` ✅ (Paolo)

- [x] Grafici per il report: missing per colonna, sbilanciamento delle classi, feature più correlate con il target (con le funzioni di `src/plots.py`).
  Figure in `report/figures/`: `missing_balance.png`, `log_odds.png`.
- [x] Bastano 1–2 figure davvero informative.

### 6. Feature engineering ✅ (Paolo, Gaia — `src/preprocessing.py`, `src/features.py`, `src/feature_types.py`, `src/pipeline.py`)

- [x] Convertire le colonne con unità miste (`ALCDAY5`, `FRUIT1`, …) oppure usare le versioni già calcolate (`_BMI5`, `WTKG3`, …): `CONVERTED` e `convert_units`.
- [x] One-hot per le categoriche (B9).
- [x] Indicatori di missing dove il dato mancante è informativo (B5/B6).
- [x] Rimozione degli ID e delle variabili di peso del sondaggio (`DOMAIN_DROP`); le colonne di label leakage (`LABEL_LEAKAGE`) sono tolte con qualunque config.
- [x] Eventualmente polinomi o interazioni sulle continue più importanti: **provati, nessun guadagno.**
  Quadrati / log1p (Paolo, poi tolti in `b411321`); quadrati delle 79 colonne numeriche e interazioni età × feature
  (controllo in CV a 3 fold su 60k righe): F1 0.4166 / 0.4188 contro 0.4175 ± 0.002 della pipeline attuale. Risultato negativo da citare nel report.

### 7. Cross-validation generica ✅ (Gaia — `src/cross_validation.py`)

- [x] k-fold (k = 5, stratificata, seed fisso) che accetta qualunque modello: `cross_validate`.
- [x] **Il preprocessing va rifittato dentro ogni fold**, per evitare il data leakage.
- [x] Restituisce F1 medio e deviazione standard sui fold (più gli score out-of-fold).

### 8. Gestire lo sbilanciamento delle classi ✅ (Claudia — `src/models.py`, `experiments/imbalance.py`)

- [x] Tuning della soglia, pesi di classe nella loss logistica, undersampling dei negativi: confrontate tutte e tre le strade.

  Risultati (CV a 5 fold, `DEFAULT_CONFIG`, `reg_logistic_regression` con λ = 1e-4, γ = 0.3, 500 iterazioni):

  | Variante | F1 medio ± std | Soglia media |
  |---|---|---|
  | soglia 0.5 | 0.2396 ± 0.0076 | 0.500 |
  | soglia ottimizzata | **0.4313 ± 0.0025** | 0.200 |
  | `pos_weight` = 3.2 | 0.4311 ± 0.0022 | 0.444 |
  | `pos_weight` = 10.3 | 0.4302 ± 0.0018 | 0.723 |
  | `neg_per_pos` = 1 | 0.4279 ± 0.0022 | 0.728 |
  | `neg_per_pos` = 3 | 0.4310 ± 0.0028 | 0.455 |

  Conta la soglia. Con la soglia ottimizzata, pesi e undersampling spostano solo il log-odds di circa log(peso)
  (le soglie ottimali lo confermano) e non migliorano l'F1. Scelta: modello finale senza pesi, soglia scelta in CV.

---

## Settimana 3 (20–25 ott): esperimenti per il report

### 9. `experiments/baselines.py`

- [ ] Tutti e 6 i metodi sulla stessa pipeline, con F1 in CV.
- [ ] Esempio: classe maggioritaria → least squares → ridge → logistica → logistica regolarizzata.
- Sono i "solid comparison baselines" che la consegna richiede esplicitamente.

### 10. `experiments/tuning.py`

- [ ] Grid search su λ, γ, `max_iters` (e grado polinomiale, se serve), con un grafico CV vs λ (utile anche per discutere over/underfitting).

### 11. `experiments/ablation.py`

- [ ] Partendo dal modello finale, si toglie una modifica alla volta (one-hot, indicatori di missing, soglia, …) e si misura il calo di F1:
  `cross_validate` con `dict(DEFAULT_CONFIG, flag=False)`.
- Il risultato diventa una tabella nel report e pesa molto nella valutazione.

### 12. `run.py`

- [ ] Riproduce **esattamente** la miglior submission: seed fissati, iperparametri hardcoded, output in `submissions/`.
- [ ] Va controllato rilanciandolo e confrontando il CSV con quello inviato (per esempio con `diff`).

---

## Ultimi giorni (26–29 ott): report e consegna

### 13. Report LaTeX (`report/report.tex`)

- [ ] Massimo 2 pagine, riferimenti a parte: introduzione, preprocessing (descritto in modo riproducibile), modelli, setup di CV,
  tabella baseline, tabella ablation, 1–2 figure, discussione, punteggio AIcrowd.
- [ ] Vanno indicati tutti i valori usati: λ, γ, numero di fold, soglia.

### 14. README

- [ ] Struttura del repo, come ottenere i dati, come lanciare `run.py`, test ed esperimenti.

### 15. Controllo finale

- [ ] Repo pubblico.
- [ ] Nessuna libreria vietata: solo NumPy, più matplotlib/seaborn per i grafici.
- [ ] `implementations.py` importabile senza effetti collaterali.
- [ ] Dataset non committato.
- [ ] Codice commentato, formattato con black 22.6.0, nessun "todo" nei `.py`.
- [ ] Link a un commit preciso (`.../tree/<hash>`) e test ufficiali passati con quel link.
