# Interazioni, ensemble e feature screening (21–22 ottobre)

Esperimenti sui due metodi con l'F1 più alto: `reg_logistic_newton` e `logistic_boosting`.
Tutti gli F1 sono in CV a 5 fold con gli stessi fold (`K_FOLD = 5`, `SEED = 1`), quindi si possono confrontare fold per fold.
La deviazione standard tra fold è circa 0.002: differenze più piccole sono rumore, a meno che non vadano nella stessa direzione in tutti i fold.
HAREHAB1 non è mai usata come feature. Dove compare la regola "HAREHAB1 risposto → 1", è indicato.

## In sintesi

| Modello | F1 CV | Rispetto a |
|---|---|---|
| Newton, pipeline v1 (`04_final` attuale) | 0.4323 | — |
| Newton + grado 2 + log + top 20 interazioni | 0.4339 | +0.0016 su Newton v1 |
| Logistic boosting (`04_final` attuale) | 0.4338 | — |
| Logistic boosting + top 20 interazioni | 0.4333 | −0.0005 sul boosting |
| Newton + feature cliniche fatte a mano | 0.4326–0.4335 | da −0.0004 a −0.0013 su Newton con interazioni |
| **Ensemble Newton (interazioni) + boosting** | **0.4356** | **+0.0018 sul boosting, meglio in 5/5 fold** |
| Ensemble + regola HAREHAB1 | 0.4414 | +0.0058 sull'ensemble |

1. **Polinomi e log/sqrt non servono**, nemmeno al modello lineare. Le **interazioni** aiutano Newton (+0.0016) ma non il boosting, perché gli alberi le catturavano già.
2. **Le feature costruite a mano** (comorbidità, fattori di rischio, limitazioni, interazioni cliniche) **non migliorano l'F1**. L'abbiamo verificato prima con uno score test e poi in CV.
3. **L'ensemble Newton + boosting ha l'F1 in CV più alto**, con +0.0018 in tutti i fold. Le submission sono pronte, con e senza regola.
4. Modelli e ottimizzatori molto diversi si fermano tutti a circa 0.43–0.435. **Il limite è l'informazione nelle feature, non il modello.**

---

## 1. Newton: grado, trasformazioni, interazioni (fasi 02 → 03)

Script: `experiments/reg_logistic_newton/02_degree.py`, `02b_transforms.py`, `03_interactions.py`.
Risultati: `results/reg_logistic_newton/02_degree.json`, `02b_transforms.json`, `03_interactions.json`.
Griglia: λ ∈ {1e-6, 1e-4, 1e-2}, `max_iters = 20`. Nelle tabelle c'è il λ migliore per ogni riga.

| Fase | Variante | F1 | Colonne |
|---|---|---|---|
| 02 grado | 1 (= baseline v1) | 0.4323 ± 0.0020 | 334 |
| | **2** | **0.4323 ± 0.0022** | 390 |
| | 3 | 0.4321 ± 0.0017 | 444 |
| 02b trasformazioni (su grado 2) | nessuna | 0.4323 | 390 |
| | sqrt | 0.4322 | |
| | **log** | **0.4324 ± 0.0018** | 427 |
| | sqrt + log | 0.4319 | |
| 03 interazioni (su grado 2 + log) | top 0 | 0.4324 | 427 |
| | top 10 | 0.4336 ± 0.0012 | 472 |
| | **top 20** | **0.4339 ± 0.0016** | 617 |
| | top 30 | 0.4336 ± 0.0027 | 862 |

- **Polinomi e log non danno niente** (+0.0001). I log-odds sono già quasi lineari nelle feature, coerente con l'EDA (A6). Inoltre molti numerici sono già discretizzati (B10) e codificati one-hot, quindi la curvatura è già in parte presente.
- **Le interazioni** tra le 20 colonne più correlate con l'etichetta (190 prodotti) danno **+0.0016**, con varianza più bassa. Con 30 colonne (435 prodotti) si peggiora un po': 20 è il punto giusto.
- Newton è lineare, quindi **da solo non vede le interazioni**. Bisogna costruirle come colonne.

**Final con queste espansioni** (`experiments/reg_logistic_newton/extra_final_interactions.py`). Lo script è separato e non sovrascrive `04_final.json`, che resta il final su v1:

- CV 0.4339 ± 0.0016; soglia out-of-fold 0.196 (F1 out-of-fold 0.4344).
- `results/reg_logistic_newton/extra/04_final_interactions.json`, submission `submissions/reg_logistic_newton_interactions.csv` (14.7% di positivi).

`04_final.py` di Newton ha `FROM_PHASE = "01_baseline"`, cioè riproduce il JSON esistente. Il final con le espansioni sta solo nello script `extra_`.

## 2. Logistic boosting + interazioni

Script: `experiments/logistic_boosting/extra_interactions.py`.
Risultati: `results/logistic_boosting/extra/03_interactions.json` e `04_final_interactions.json`.
Lo script non è una fase, quindi `run_phases.py` non lo esegue e `04_final.json` non viene toccato.

Setting: pipeline v1 + top 20 interazioni, λ = 1e-4, depth 3, learning rate 0.05. Il confronto è con le righe di `01_baseline` che hanno lo stesso fit (stessi fold):

| Alberi | Con interazioni | Senza (01_baseline) | Diff | Meglio in |
|---|---|---|---|---|
| 200 | 0.4330 | 0.4335 | −0.0006 | 1/5 |
| 400 | 0.4333 | 0.4338 | −0.0005 | 1/5 |

Final con interazioni: CV 0.4333, F1 out-of-fold 0.4347, `submissions/logistic_boosting_interactions.csv`. **Non conviene usarla.**

**Perché niente polinomi né log per il boosting.** Gli alberi decidono con soglie (`x > 3`). Una soglia su x² o su log x divide i dati esattamente come una soglia su x, quindi le trasformazioni monotone non aggiungono niente agli alberi. Sulla parte lineare le avevamo già testate con Newton, e non servivano.

**Perché provare le interazioni anche se gli alberi le imparano già.** Gli alberi le imparano a scalini e spendendo profondità e alberi. Con le interazioni come colonne esplicite, la parte logistica le fitta in modo esatto e gli alberi restano liberi per il resto.

**Risultato:** i due contributi non si sommano. Le interazioni danno a Newton +0.0016, e il boosting dà a Newton +0.0015: è lo stesso guadagno visto da due lati. **Il vantaggio del boosting su Newton viene proprio dalle interazioni.**

## 3. Ensemble Newton + logistic boosting

Script: `experiments/ensemble.py`. Risultati: `results/ensemble/ensemble.json`.

    p = a · p_newton + (1 − a) · p_boosting,   a ∈ {0, 0.05, ..., 1}

- **Componenti:** Newton con interazioni (`extra/04_final_interactions`: grado 2 + log + top 20) e boosting `04_final` (v1, senza espansioni). Per ciascuno abbiamo preso la versione migliore.
- **CV senza barare:** per ogni fold, il peso `a` e la soglia sono scelti sulle probabilità out-of-fold degli altri 4 fold, poi l'F1 si misura sul fold rimasto. I due modelli da soli sono valutati con lo stesso protocollo, quindi le righe sono confrontabili.
- **Cache:** le probabilità out-of-fold e di test dei due modelli sono in `results/ensemble/scores_<metodo>.npz`. Con questi file `python experiments/ensemble.py` rigenera le submission in pochi secondi, e l'ho verificato: il file è identico byte per byte. Senza la cache lo script riaddestra tutto: circa 5 minuti per Newton, 1–2.5 ore per il boosting.

| Modello | F1 | Diff vs boosting | Meglio in | F1 + regola | Peso `a` per fold |
|---|---|---|---|---|---|
| Newton (interazioni) da solo | 0.4336 ± 0.0015 | −0.0001 | 2/5 | 0.4390 | 1 |
| Boosting da solo | 0.4338 ± 0.0021 | — | — | 0.4399 | 0 |
| **Ensemble** | **0.4356 ± 0.0020** | **+0.0018** | **5/5** | **0.4414** | 0.4, 0.4, 0.35, 0.35, 0.35 |

- **È meglio in tutti i fold**, e il peso è stabile (circa 35–40% Newton, 60–65% boosting). Se il miglioramento fosse dovuto al caso, il peso cambierebbe molto da un fold all'altro.
- **Perché funziona:** i due modelli hanno lo stesso F1 ma sbagliano su persone in parte diverse. Newton usa interazioni esplicite, il boosting alberi sui residui, e la media compensa una parte degli errori.
- **Final:** a = 0.35, soglia 0.206, F1 out-of-fold 0.4360. Produce due file:
  - `submissions/ensemble_newton_boosting.csv`: **senza** HAREHAB1, 13.60% di positivi. Si confronta direttamente con `logistic_boosting.csv`.
  - `submissions/ensemble_newton_boosting_rule.csv`: stesse predizioni + regola. Nel test HAREHAB1 ha una risposta in 238 righe, e la regola ne cambia 74 da −1 a 1 (13.67% di positivi).

## 4. Feature screening: feature costruite a mano

Script: `experiments/feature_screening.py`. Risultati: `results/reg_logistic_newton/extra/feature_screening.json`.
`src/` non è stato toccato: le feature nuove sono aggiunte come colonne extra ai dati grezzi.

### Metodo: uno screening a livelli, dal più economico al più costoso

**Livello 1, score test (secondi per gruppo).** Il modello attuale (Newton con interazioni) viene fittato una volta. Al minimo della loss, i residui `y − p` sono scorrelati da tutte le colonne del modello. Una nuova colonna aiuta solo se è correlata con quello che resta nei residui. Lo score test (Rao / moltiplicatore di Lagrange) misura proprio questo, senza rifittare:

    g = X_new^T (p − y) / N             gradiente della loss nel nuovo peso, a w_new = 0
    I = H_nn − H_nb H_bb⁻¹ H_bn         curvatura che resta dopo le colonne già presenti
    χ² = N gᵀ I⁻¹ g   ~ χ²(q) se X_new non aggiunge niente

χ²/(2N) stima di quanto scenderebbe la loss logistica media rifittando con X_new. L'ho verificato su dati sintetici:
- colonna ridondante → χ² = 0;
- rumore → 1.7;
- colonna utile → 150.6, contro un valore vero di 150.9 rifittando.

Come **riferimento di scala** si usa lo stesso test per le top 20 interazioni rispetto al modello senza (fase 02b), di cui sappiamo che valgono +0.0016 di F1.

**Livello 2, CV con Newton (circa 4 minuti per gruppo; oggi 12–19 minuti perché girava insieme ad altri processi).** Riguarda solo i gruppi significativi al livello 1, con soglia di Bonferroni p < 0.05/6. Vengono aggiunti alla pipeline di `03_interactions` e valutati in CV fold per fold contro 0.4339.

**Livello 3, boosting / ensemble:** non fatto, perché nessun gruppo ha superato il livello 2.

### Gruppi provati

Indicatori costruiti dai codici grezzi: 1 = sì; NaN, 7/9 e domanda non posta contano come no.

| Gruppo | Contenuto | Colonne |
|---|---|---|
| Comorbidità (numero) | numero di condizioni tra diabete, ipertensione, colesterolo, ictus, reni, BPCO, artrite | 1 |
| Comorbidità (one-hot) | stesso numero, come indicatori ≥2, ≥3, ≥4 | 3 |
| Fattori di rischio (one-hot) | età di rischio (uomini 55+, donne 65+), fumatore attuale, ipertensione, colesterolo, diabete, obesità (BMI ≥ 30), inattività → ≥2..≥5 | 4 |
| Limitazione funzionale (one-hot) | DIFFWALK, USEEQUIP, QLACTLM2, DIFFDRES, DIFFALON, non in grado di lavorare, ≥14 giorni di cattiva salute fisica → ≥1..≥4 | 4 |
| Contatti sanitari (one-hot) | check-up nell'ultimo anno, vaccino antipneumococco, antinfluenzale, aspirina quotidiana, farmaci per la pressione → ≥1..≥4 | 4 |
| Interazioni cliniche | età×sesso, età×diabete, età×ipertensione, età×colesterolo, età×fumo (attuale ed ex), età×ictus, sesso×fumo, sesso×diabete, diabete×ipertensione, ipertensione×colesterolo, obesità×diabete, diabete×reni, età×comorbidità | 14 |

Gli indicatori "≥k" danno, insieme alle colonne già presenti, le stesse informazioni di un one-hot del conteggio, ma ogni colonna rappresenta il passaggio da k−1 a k.

### Livello 1

| Gruppo | Colonne | χ² | p | Δ loss stimata | Rispetto al riferimento |
|---|---|---|---|---|---|
| Riferimento: top 20 interazioni | 190 | 1152.9 | ~0 | 1.8e-3 | 1× |
| Comorbidità (numero) | 1 | 0.0 | 0.82 | ~0 | controllo ok |
| Comorbidità (one-hot) | 3 | 17.7 | 6e-4 | 2.7e-5 | circa 1/65 |
| Fattori di rischio (one-hot) | 4 | 18.2 | 1e-3 | 2.8e-5 | circa 1/65 |
| Limitazione funzionale (one-hot) | 4 | 21.8 | 3e-4 | 3.3e-5 | circa 1/55 |
| Contatti sanitari (one-hot) | 4 | 6.7 | 0.15 | 1.0e-5 | non passa |
| Interazioni cliniche | 14 | 54.8 | 1e-6 | 8.4e-5 | circa 1/20 |

- Il **numero** di comorbidità dà esattamente 0, come previsto: è una somma di colonne che il modello ha già, e la logistica fa già una somma pesata. Per un modello lineare una feature nuova è utile solo se è **non lineare** nelle colonne esistenti.
- Tra le interazioni cliniche porta informazione nuova solo **età×sesso** (χ² = 34.6), più un po' età×fumo (≈ 10). Molte danno esattamente 0, probabilmente perché quelle coppie sono già tra le top 20 interazioni automatiche.
- Quattro gruppi sono significativi, ma con un effetto 20–65 volte più piccolo del riferimento. Con 328 000 righe anche effetti minuscoli risultano significativi.

### Livello 2 (riferimento 0.4339 ± 0.0016)

| Gruppo | F1 | Diff | Meglio in |
|---|---|---|---|
| Comorbidità (one-hot) | 0.4330 ± 0.0019 | −0.0009 | 2/5 |
| Fattori di rischio (one-hot) | 0.4335 ± 0.0021 | −0.0004 | 2/5 |
| Limitazione funzionale (one-hot) | 0.4328 ± 0.0017 | −0.0011 | 2/5 |
| Interazioni cliniche | 0.4327 ± 0.0009 | −0.0012 | 1/5 |
| Tutti insieme | 0.4326 ± 0.0018 | −0.0013 | 1/5 |

**Nessun gruppo migliora l'F1.** Un calo della loss dell'ordine di 3e-5 è troppo piccolo per spostare l'ordinamento attorno alla soglia, quindi l'F1 non cambia, e le differenze osservate sono nel rumore. È coerente con quello che prevedeva il livello 1.

C'è un effetto collaterale: i gruppi one-hot entrano tra le 20 colonne più correlate con l'etichetta e cambiano quindi le interazioni scelte da `interaction_top`. Le interazioni cliniche però non entrano nella top 20 e non migliorano comunque, quindi questo non spiega il risultato.

**Per il report:** le feature costruite con conoscenza clinica non aggiungono niente a un modello che ha già le variabili originali in one-hot e le interazioni principali. L'informazione era già nei dati.

---

## 5. Altre conclusioni emerse

**L'ottimizzatore non conta più.** Adaptive GD (0.4322), Newton (0.4323) e UniXGrad (0.4318) ottimizzano tutti la stessa loss logistica regolarizzata, che è convessa e ha un unico minimo. Ci arrivano più velocemente del GD classico, ma non arrivano a un punto migliore. Newton raggiunge il minimo esatto in circa 10 passi.

**La selezione dei dati non conta più.** Dall'ablation (`results/ablation.md`): togliere una regola di selezione o usare tutte le colonne cambia l'F1 di ±0.0005. Contano solo il one-hot (−0.007 se lo togli) e gli indicatori di missing (−0.0015).

**Metodi lineari contro logistica.** Least squares, ridge e MSE-GD in baseline fanno circa 0.421, la logistica circa 0.431. Il distacco è piccolo perché con la soglia tarata least squares su etichette 0/1 trova comunque una buona direzione (è equivalente a un discriminante di Fisher). Con grado 3 + log + top 20 interazioni, least squares e ridge salgono a **0.4306** (risultato non salvato in `results/`), cioè al livello della logistica in baseline: conta più lo spazio delle feature che la loss. Con le stesse feature anche la logistica sale, quindi il distacco resta.

**Tutti i modelli convergono a circa 0.43–0.435:** lineare (0.4323), lineare con interazioni (0.4339), solo alberi (gradient boosting 0.4298), ibrido (0.4338). Il collo di bottiglia è l'informazione contenuta nelle risposte al questionario.

**HAREHAB1 è label leakage, anche se parziale.** MICHD = 1 se si ha avuto un infarto *oppure* una diagnosi coronarica *oppure* l'angina. HAREHAB1 viene chiesta solo dopo un infarto, quindi:
- HAREHAB1 risposto ⇒ infarto ⇒ MICHD = 1, con certezza (773 righe su 773 nel training);
- HAREHAB1 vuoto ⇏ MICHD = 0, perché si può avere MICHD senza infarto.

Le altre definizioni rompono solo la seconda direzione. Per un sottoinsieme di positivi la variabile non predice l'etichetta, la *rivela*. In più è un'informazione successiva all'evento, quindi non disponibile nell'uso del modello per la prevenzione, e gli organizzatori hanno già tolto le colonne che definiscono l'etichetta (CVDINFR4, CVDCRHD4). Per questo il modello principale non la usa, e la regola va riportata a parte.

**CV contro AIcrowd.** Il punteggio sul test di AIcrowd e quello in CV sono stime diverse: un altro campione, e una soglia che si trasferisce in modo un po' diverso. Le CV vanno confrontate tra loro (per esempio ensemble 0.4356 contro boosting 0.4338). Per confrontare su AIcrowd bisogna caricare le submission.

---

## 6. Da fare / da decidere

- [ ] Caricare su AIcrowd `ensemble_newton_boosting.csv` e `ensemble_newton_boosting_rule.csv` (al massimo 5 submission al giorno per team).
- [ ] Decidere se la submission finale usa la regola HAREHAB1. Nel report vanno comunque riportati entrambi i numeri.
- [ ] **`run.py`**: oggi riproduce il modello A (`submissions/logistic_boosting.csv`). Deve produrre *esattamente* il CSV della submission scelta: se sarà l'ensemble, va aggiornato. Ci sono due strade:
  - **solo inference:** legge `results/ensemble/scores_*.npz` e il peso e la soglia da `ensemble.json`, e il README spiega che il training è in `experiments/ensemble.py`;
  - **training completo:** `python experiments/ensemble.py --force`, 1–2.5 ore. Non ho verificato che il riaddestramento completo dia esattamente lo stesso CSV; l'ho verificato solo per la versione da cache.

## Comandi per riprodurre

    python experiments/reg_logistic_newton/02_degree.py            # poi 02b_transforms.py, 03_interactions.py
    python experiments/reg_logistic_newton/extra_final_interactions.py
    python experiments/logistic_boosting/extra_interactions.py     # ~3 h
    python experiments/feature_screening.py                        # --level 1 per il solo score test
    python experiments/ensemble.py                                 # secondi dalla cache, --force per riaddestrare
