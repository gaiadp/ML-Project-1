# Predicting Coronary Heart Disease Risk from BRFSS Lifestyle Data

Berra Claudia, Cermelli Paolo, Di Paolo Gaia

> **Bozza v1 (in inglese), scritta per essere copiata in LaTeX.** Le note tra `[CHECK ...]` o in blocchi come questo sono per voi, vanno tolte prima di consegnare. In fondo al file c'è una sezione "Note per il gruppo" con le cose da controllare e i tagli possibili. Ogni numero viene da `results/`, da `experiments/EDA.md` o dal codice; i punteggi AIcrowd (0.445, 0.446, 0.449, 0.452) non sono nel repository, li ha dati Claudia.

## Abstract

Heart attacks and coronary heart disease are among the leading causes of death worldwide, and a model that spots people at risk from a short health questionnaire could help early prevention. We predict whether a person reports coronary heart disease (MICHD) from the 321 answers of 328,135 people in the BRFSS 2015 survey, where only 8.8% of the people are positive. We build a careful cleaning pipeline for the survey data, compare the six required methods with seven more, and combine two models: a logistic regression with interactions solved with Newton's method, and a logistic regression followed by boosted decision trees. The final ensemble reaches F1 = 0.4356 ± 0.0020 in 5-fold cross-validation and 0.446 on AIcrowd. Very different models all end up near 0.43 to 0.435, which suggests that the limit is the information in the questionnaire and not the model. We also found one survey question that reveals the label, and we left it out.

## 1 Introduction

Cardiovascular diseases such as heart attacks are one of the leading causes of death worldwide. [CHECK: add a reference for this sentence (WHO) or remove it.] Many of the risk factors (age, blood pressure, smoking, general health) can be asked in a short questionnaire, so a model that ranks people by risk is cheap to apply.

Our task is to predict, from the answers of a person to the 2015 BRFSS telephone survey of the US Centers for Disease Control and Prevention \cite{brfss2015}, whether that person has MICHD, i.e. reported a heart attack, coronary heart disease or angina. The classes are very unbalanced: 8.8% of the people are positive. Accuracy would be useless (answering "healthy" for everybody is 91% accurate), so the challenge uses the F1 score, the harmonic mean of precision (of the people we flag, how many are really positive) and recall (of the positive people, how many we find). Flagging everybody gives F1 = 0.162, and the best single rule we found, based on the answer to the general health question, gives 0.309 (5-fold cross-validation, Table 1).

Our contributions are:

1. a cleaning pipeline made for survey data (special answer codes, skipped questions, missing answers that carry information), with an ablation study that measures what each step is worth;
2. a comparison of the six required methods with seven further methods that we wrote ourselves, and a combination of two of them that gives our best honest score;
3. the discovery of a column that gives away the label, and a measurement of how much it would add if we used it.

## 2 Data and exploratory analysis

The training set has 328,135 people and 321 columns, the test set 109,379 people. We looked at the data with nine small analyses (`experiments/eda.py`); these are the findings that changed what we did.

**Answer codes are not values.** In the codebook, numbers such as 7, 9 or 88 often mean "don't know", "refused" or "none", but the same number can be a real answer in another column (7 is "retired" in the employment question). We wrote the rule for each of the 237 columns affected, turning those codes into "missing" or 0.

**Missing values are everywhere, and they can carry information.** After the recoding, 47% of the values of an average column are missing. Many questions belong to optional modules, and follow-up questions are skipped when they do not apply (nobody who had no stroke is asked about stroke rehabilitation). Skipped questions are filled with "no". The other missing values can still be informative: the people who were not asked the diabetes module have 7.9% MICHD, those who were asked have 21.9%. For this reason we add an "is missing" indicator for groups of columns whose missingness changes the MICHD rate by at least 3 points.

**Which columns carry signal.** We ranked the columns by mutual information with the label, which tells how much knowing a column reduces the uncertainty on the label. General health and age are at the top, followed by employment status and the use of blood pressure medication. MICHD goes from 0.5% in the group aged 18 to 24 up to 22.2% for people aged 80 and more, and it is 11.2% for men against 7.1% for women. 89 of the 236 candidate columns carry almost no information (less than 1 millibit, against 431 millibits of uncertainty in the label itself).

**Duplicates.** The survey contains many calculated variables that repeat other answers (for example four race variables), and 14 answers that are stored twice, in raw and in converted units. We found 30 columns that are duplicates of another kept column and removed them.

**Train and test look alike.** For every column, the difference between the distribution in the training and in the test set is as small as between two random halves of the training set. We therefore trust cross-validation as an estimate of the test score, as far as the inputs are concerned.

**One column gives away the label.** `HAREHAB1` asks whether the person did rehabilitation after a heart attack. The question is asked only to people who had a heart attack, so any answer means MICHD = 1: in the training set all 773 people who were asked it are positive. It is the only column of the 321 with this behaviour. We discuss in Section 5 why we do not use it.

[Optional figure, if there is space: `figures/log_odds.png` (log-odds of MICHD for the 16 most informative columns), caption: "MICHD rate (log-odds scale) for the 16 most informative columns. Many curves are close to straight lines, which is why a linear model on the right encoding works well." Do not write the second sentence unless you have looked at the figure and agree.]

## 3 Models and methods

### 3.1 Data preparation

The preparation is a chain of steps. Every number it needs (medians, categories, which columns are informative) is computed on the training part only and then applied unchanged to validation and test rows. In cross-validation this happens again inside each fold, so the validation fold never influences the preparation.

1. Recode the special answer codes (Section 2) and convert the answers given in different units to one scale.
2. Drop what is not a property of the person: 41 columns with ids, interview dates, survey weights and the child module, plus the 30 duplicates.
3. Fill the skipped questions, add the "is missing" indicators, and impute the other missing values (median for numbers, most frequent value for yes/no and category columns).
4. Encode: yes/no columns as 0/1, category columns as one 0/1 column per frequent value (one-hot), the days of bad physical health in a few bins, and a few ordinal columns where more is not always worse (for example smoking status) also one-hot. Then standardize every column.
5. Keep 124 of the 236 remaining columns with four rules based on the exploratory analysis (very low mutual information, small optional modules, questions asked only to landline phones, a copy of another column).

This gives 336 input columns on the whole training set. The rules of step 5 were chosen by looking at the labels of all the training people, so the validation folds already influenced which columns we keep, and the cross-validation score is slightly optimistic. Section 4 shows that switching the selection off changes F1 by only 0.0002.

For the best linear model we add to these columns the squares of the columns that take more than two values, the logarithm of the continuous and count columns, and the products of the 20 columns most correlated with the label (190 products). These columns are used only by the logistic regression in the ensemble (619 columns in total).

### 3.2 Models

We implemented the six required methods in `implementations.py`. For least squares we use a pseudo-inverse built from an eigendecomposition, because the matrix X^T X can be singular when columns are collinear. All other methods are in `src/` and use only NumPy.

**Logistic regression.** It gives a probability for each person and is the base of our best models. Plain gradient descent needs hundreds of steps and a step size to tune. We also use **Newton's method**, which uses the curvature of the loss to choose the direction and the length of each step: it needs about 10 steps instead of hundreds, and has no step size to choose (if a full step does not lower the loss, it is halved). We add a small penalty λ‖w‖² on the weights (λ = 10^-4 for the first model and 10^-6 for the one with interactions).

**Boosted trees.** A decision tree asks yes/no questions about the columns. Boosting builds many small trees one after the other, each one fitted to the mistakes that the previous ones left \cite{friedman2001,chen2016xgboost}. We wrote the trees ourselves; to be fast, the values of each column are first cut into 32 bins \cite{ke2017lightgbm}.

**Logistic boosting (our hybrid).** We first train the logistic regression, then we add 400 trees (depth 3, learning rate 0.05) that correct its mistakes. The idea is that the logistic part captures the large additive effects (age, general health) cheaply and exactly, and the trees only have to learn what is left: thresholds and interactions between columns.

**Ensemble.** The final probability is p = 0.35 · p_Newton + 0.65 · p_boosting, where p_Newton comes from the logistic regression with interactions and p_boosting from logistic boosting.

**Decision threshold.** With only 8.8% positives, cutting the probability at 0.5 flags almost nobody. We cut at the value that maximizes F1 (0.206 for the ensemble). This matters a lot: for logistic boosting F1 is 0.2395 with 0.5 and 0.4338 with the tuned threshold.

### 3.3 Evaluation

All the experiments use the same 5 stratified folds (each has 8.8% positives, seed 1). We report the mean and standard deviation of F1 over the validation folds, with the threshold chosen on the training part of each fold. The standard deviation between folds is about 0.002, so a difference smaller than that is noise unless it goes in the same direction in all folds; for this reason we also report in how many of the 5 folds a variant is better. For the ensemble, the weight and the threshold are chosen on the out-of-fold probabilities of the other four folds. For the final submission, the threshold is chosen on the out-of-fold probabilities of all the training people, then the models are trained on the whole training set. `run.py` reproduces our submission (seed 1; Python standard library and NumPy only) in about 11.5 minutes (689 s on a laptop) [CHECK: add the machine, CPU and RAM].

## 4 Results

**Table 1.** Cross-validated F1 (mean ± std over 5 folds). The baseline pipeline (v0) only recodes the special codes, drops the columns with more than 50% missing values, imputes the median, one-hot encodes and standardizes. Ours is the pipeline of Section 3.1.

| Method | Baseline pipeline | Our pipeline |
|---|---|---|
| Flag everybody (no model) | 0.1623 ± 0.0000 | – |
| Best rule on general health | 0.3088 ± 0.0022 | – |
| Linear regression, gradient descent | 0.4184 ± 0.0024 | 0.4211 ± 0.0018 |
| Linear regression, SGD | 0.3996 ± 0.0111 | 0.4084 ± 0.0028 |
| Least squares | 0.4179 ± 0.0023 | 0.4212 ± 0.0025 |
| Ridge regression | 0.4181 ± 0.0028 | 0.4216 ± 0.0026 |
| Logistic regression | 0.4270 ± 0.0029 | 0.4311 ± 0.0017 |
| Regularized logistic regression | 0.4270 ± 0.0025 | 0.4308 ± 0.0013 |
| Regularized logistic regression, Newton | – | 0.4323 ± 0.0020 |
| Random forest | – | 0.4158 ± 0.0018 |
| Gradient boosting | – | 0.4298 ± 0.0024 |
| Logistic boosting | – | 0.4338 ± 0.0020 |
| Newton + interactions | – | 0.4339 ± 0.0016 |
| **Ensemble (submitted)** | – | **0.4356 ± 0.0020** |

[CHECK: the first two rows are the trivial rules of `results/baselines.md`, computed with the same folds; "Best rule on general health" is the one-column rule on `GENHLTH`. The row "Newton + interactions" is `reg_logistic_newton/extra/04_final_interactions.json` (its pipeline is not "our pipeline" of Section 3.1 alone: it has the extra columns of the last paragraph of 3.1). Consider a footnote or a separate column.]

**The preparation matters more than the loss.** Our pipeline improves every method over the baseline pipeline, by 0.0027 to 0.0088 in F1. The logistic regressions are about 0.010 above the linear regressions (0.4311 against 0.4212), and tuning the threshold is worth far more than any of these steps (Table 2).

**Table 2.** What each ingredient is worth. Top: the data preparation, switching one step off at a time (Newton logistic regression, λ tuned again for each variant). Bottom: the model (logistic boosting). "Folds" is the number of folds in which the variant is better than the reference.

| Variant | F1 | Δ F1 | Folds |
|---|---|---|---|
| *Reference (Newton)* | 0.4323 ± 0.0020 | – | – |
| no one-hot encoding | 0.4255 ± 0.0016 | −0.0068 | 0/5 |
| no one-hot of the ordinal columns | 0.4306 ± 0.0016 | −0.0017 | 0/5 |
| no "is missing" indicators | 0.4307 ± 0.0016 | −0.0015 | 1/5 |
| days of bad health not binned | 0.4313 ± 0.0025 | −0.0010 | 1/5 |
| no unit conversion | 0.4317 ± 0.0023 | −0.0006 | 2/5 |
| no feature selection | 0.4321 ± 0.0025 | −0.0002 | 3/5 |
| median imputation for all columns | 0.4323 ± 0.0020 | +0.0000 | 3/5 |
| *Reference (logistic boosting)* | 0.4338 ± 0.0020 | – | – |
| no trees (logistic regression, Newton) | 0.4323 ± 0.0020 | −0.0015 | 0/5 |
| no trees, gradient descent instead of Newton | 0.4308 ± 0.0013 | −0.0029 | 0/5 |
| threshold 0.5 instead of tuned | 0.2395 ± 0.0061 | −0.1943 | 0/5 |

[CHECK: in the table above, the last row of the top block is "median imputation (B7)" in `results/ablation.md`: the reference imputes by type (median for numbers, most frequent value for yes/no and categories), the variant uses the median for everything. Keep only the rows you want to discuss; the full table has 16 rows.]

The one-hot encoding is the step that matters most (−0.0068 without it), followed by the indicators of missing answers (−0.0015). The feature selection and the other cleaning steps are within the noise. Adding trees to the logistic regression gains 0.0015 and is better in all 5 folds, and Newton's method is 0.0015 better than gradient descent on the same model (0.4323 against 0.4308). We did not check whether gradient descent simply had not fully converged.

**What did not help.** Squares and logarithms of the numeric columns change nothing for the logistic regression (0.4323 to 0.4324). Products of the 20 columns most correlated with the label help it (0.4339, +0.0016); with 10 products we get 0.4336 and with 30 also 0.4336. The same products do not help the boosted trees (−0.0005), which can find interactions by themselves. We also built columns by hand from clinical knowledge (number of conditions, risk factors, functional limitations, age × sex and other products): a score test said that they carry almost no new information, and in cross-validation none of the groups improved F1 (differences from −0.0004 to −0.0013). Three different ways to minimize the same logistic loss (gradient descent with adaptive steps, Newton, UniXGrad) end at the same F1 (0.4322, 0.4323, 0.4318); a fourth optimizer (Sophia) reached only 0.3928 in our setting.

**The ensemble.** Logistic regression with interactions (0.4336 ± 0.0015) and logistic boosting (0.4338 ± 0.0021), evaluated with the same protocol as the ensemble (weight 1 or 0), have the same F1, but their average has 0.4356 ± 0.0020, better than boosting in all 5 folds (+0.0018). The weight chosen on the other folds is stable (0.35 to 0.40 for Newton).

**Table 3.** The four predictions we sent to AIcrowd. The first is our submission. "Rule" sets to positive the people who were asked the `HAREHAB1` question.

| Prediction | Uses HAREHAB1 | CV F1 | AIcrowd F1 |
|---|---|---|---|
| **Ensemble (submitted)** | no | 0.4356 ± 0.0020 | 0.446 |
| Logistic boosting | no | 0.4338 ± 0.0020 | 0.445 |
| Ensemble + rule | yes | 0.4414 | 0.449 |
| Logistic boosting, `HAREHAB1` as a feature | yes | 0.4413 ± 0.0032 | 0.452 |

## 5 Discussion

**What worked.** Most of the gain comes from representing the survey well (one-hot encoding, missing-answer indicators, a tuned threshold). After that every model we tried ends between 0.43 and 0.435, from the plain logistic regression to trees, a hybrid and the ensemble. This suggests that the limit is the information contained in the answers and not the model, and it is also why the ensemble gains only 0.0018, even if it does so in all 5 folds. We think the two models make partly different mistakes (one has explicit interactions, the other trees), but we did not measure how different their errors are.

**The column that gives away the label.** Using `HAREHAB1` as a feature raises F1 by 0.0075 in cross-validation (better in 5 of 5 folds), and applying the rule to our ensemble gives 0.4414. We do not use it in the submitted model for two reasons: the answer exists only after a heart attack, so in a prevention setting it is not available when the prediction is needed, and the dataset already leaves out the columns that define the label (the heart attack and coronary disease questions are not among the 321 columns). Our submitted ensemble scores 0.446 on AIcrowd, lower than the 0.452 of the version with `HAREHAB1` as a feature, which is our best online score, and this is why we report 0.446 here as the test performance of the submitted predictions.

**Limitations.** (i) The hyperparameters and the choice of rows of the tables were selected on the same folds that we report, without a nested cross-validation, so the scores are a little optimistic. (ii) The column selection used the labels of all the training people (Section 3.1). (iii) The data preparation ablation was done with the Newton logistic regression, not with the final ensemble. (iv) For the logistic regression with interactions, the best λ (10^-6) is at the edge of the grid we tried. (v) Cross-validation and AIcrowd are different samples with different thresholds, so only the order of the models is comparable, not the values. (vi) Differences smaller than 0.002 between variants are inside the noise between folds.

## 6 Summary

We predicted coronary heart disease from a health survey with a pipeline that treats special codes, skipped questions and missing answers with care, and an ensemble of a logistic regression with interactions and a logistic regression followed by boosted trees, all written in NumPy. The ensemble gets F1 = 0.4356 in cross-validation and 0.446 on AIcrowd. The ablation shows that the encoding of the survey and the choice of the threshold matter much more than the model, and that a survey column which reveals the label would give a higher score but not a useful model.

## References

(BibTeX keys used above; the entries are in the next section.)

---

# Note per il gruppo (togliere prima di consegnare)

## Da compilare o verificare

1. **Tempo di esecuzione di `run.py`**: Claudia lo sta misurando; va nella fine di 3.3.
2. **Riferimento per la frase sulle malattie cardiovascolari** (introduzione): il testo del progetto cita l'OMS, ma non dà un riferimento bibliografico. Aggiungere una citazione oppure togliere la frase.
3. **Punteggi AIcrowd** (0.445, 0.446, 0.449, 0.452): non sono nel repository, li ha riportati Claudia dalla pagina delle submission. Verificare che i file caricati siano quelli in `submissions/` (Paolo ha confermato 0.445 e 0.452 per `logistic_boosting.csv` e `logistic_boosting_harehab1.csv`).
4. **Voci bibliografiche**: sono scritte a memoria, controllare titoli, pagine e anno su Google Scholar prima di consegnare.
5. **Frase "il progetto ha già tolto le colonne che definiscono l'etichetta"** (Discussione): ho controllato che `CVDINFR4` e `CVDCRHD4` non sono tra le 321 colonne di `x_train.csv`; resta un'affermazione sul modo in cui il dataset è stato preparato, che non possiamo dimostrare, quindi eventualmente ammorbiditela in "do not appear among the 321 columns".
6. **Le due frasi con "suggests / we think"** (limite nell'informazione; i due modelli sbagliano su persone diverse) sono ipotesi del gruppo, non misure. Restano scritte come ipotesi: non toglierle dal tono.

## Tre punti che nella bozza sono dichiarati ma sono scomodi

- La tabella 1, riga "Newton + interactions": non usa la pipeline v1 da sola (ha anche grado 2, log e 20 interazioni).
- La CV dei modelli con HAREHAB1 e della regola non è "pulita" come le altre: servono solo a misurare il leakage.
- Newton batte GD su uguale modello di 0.0014: la causa (GD non convergito?) non è verificata, e il testo lo dice.

## Lunghezza

Il limite è 2 pagine IEEE a due colonne (riferimenti esclusi). Questa bozza ha circa 2800 parole di testo (tabelle escluse) e 3 tabelle, quindi **è troppo lunga di circa il 40–50%**: serve tagliare. (Ordine di grandezza: due pagine IEEE a due colonne con tabelle contengono circa 1500–1800 parole di testo.) Tagli possibili, dal meno doloroso:

1. Tabella 2: tenere solo 6–7 righe (one-hot, indicatori, selezione, alberi, GD contro Newton, soglia 0.5).
2. Tabella 1: togliere SGD, random forest, la colonna del baseline v0 per i modelli lineari (lasciare solo la media dell'aumento).
3. Sezione 2: ridurre "Duplicates" e "Train and test look alike" a una frase ciascuna.
4. Paragrafo "What did not help": lasciare solo le interazioni, la frase sulle feature cliniche e Sophia/optimizer in una riga.
5. Figura `log_odds.png`: occupa circa un terzo di colonna, includerla solo se resta spazio.

## Cosa è stato lasciato fuori di proposito

Le fasi 01–04 per metodo (`run_phases.py`), la tabella delle regole R1–R4 con le colonne tolte, i dettagli di UniXGrad e Sophia, il test di score, la differenza tra le due procedure per la soglia (CV e submission: è nel testo in 3.3), le cifre sulla cache dell'ensemble. Sono nei README e in `experiments/EDA.md` e `results/`.

## BibTeX (da copiare in `references.bib`)

```bibtex
@misc{brfss2015,
  author       = {{Centers for Disease Control and Prevention}},
  title        = {Behavioral Risk Factor Surveillance System, 2015 Annual Data},
  year         = {2016},
  howpublished = {\url{https://www.cdc.gov/brfss/annual_data/annual_2015.html}}
}

@inproceedings{chen2016xgboost,
  author    = {Chen, Tianqi and Guestrin, Carlos},
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  year      = {2016}
}

@article{friedman2001,
  author  = {Friedman, Jerome H.},
  title   = {Greedy function approximation: A gradient boosting machine},
  journal = {The Annals of Statistics},
  volume  = {29},
  number  = {5},
  pages   = {1189--1232},
  year    = {2001}
}

@inproceedings{ke2017lightgbm,
  author    = {Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  title     = {{LightGBM}: A Highly Efficient Gradient Boosting Decision Tree},
  booktitle = {Advances in Neural Information Processing Systems 30},
  year      = {2017}
}
```
