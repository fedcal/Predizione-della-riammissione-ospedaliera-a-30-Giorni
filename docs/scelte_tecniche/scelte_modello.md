---
layout: default
title: Scelte di modellazione
parent: Scelte tecniche
nav_order: 2
math: mathjax
description: >-
  Decisioni di modellazione del progetto Hospital Readmission 30d: LogReg
  vs ensemble, gestione dello sbilanciamento, ottimizzazione soglia su
  matrice costi e integrazione del fairness audit.
---

# Scelte di modellazione: razionale
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

Documenta le decisioni "perche' cosi' e non cosa'" di livello modellazione. Per dettagli teorici sulle tecniche, vedi `docs/teoria/`.

## 1. Famiglie di modelli scelte

### 1.1 LogisticRegression (baseline interpretabile)

**Perche'**: piu' interpretabile baseline possibile su tabular structured data. I coefficienti sulle feature standardizzate sono direttamente comunicabili a clinici e governance.

**Perche' L2, non L1**: in Diabetes-130 molte feature minori contribuiscono cumulativamente. L1 (Lasso) azzererebbe le categoriche rare (es. `medical_specialty=Urology`) perdendo segnale clinico utile. L2 mantiene tutto, regolarizzando i coefficienti grandi.

**Perche' `class_weight='balanced'`**: il dataset ha 11% di positivi. Senza bilanciamento, la LogReg "dimentica" la classe minoritaria. `balanced` calcola pesi inversamente proporzionali alla frequenza ($w_0 \approx 0.56$, $w_1 \approx 4.5$). E' la tecnica piu' semplice e robusta; SMOTE e undersampling sono alternative esplorabili.

**Trade-off**: cattura male le interazioni non lineari (es. `age * num_medications`). Su Diabetes-130 questo limita l'AUC-PR a ~0.20.

### 1.2 RandomForest (ensemble)

**Perche'**: cattura interazioni non lineari senza tuning fine. Su questo dataset porta tipicamente AUC-PR da ~0.20 (LR) a ~0.22-0.25 (RF).

**Perche' RF e non solo XGBoost**: confronto utile dal punto di vista didattico (RF mostra il guadagno del bagging su decision tree singolo; XGBoost mostra il guadagno aggiuntivo del boosting). RF e' anche piu' interpretabile in termini di feature_importance e piu' veloce in training.

**Perche' `class_weight='balanced'`** (idem LR): bilancia internamente i pesi senza generare dati sintetici.

**Trade-off**:
- `feature_importances_` impurity-based ha bias verso variabili continue ad alta cardinalita'. Per analisi serie usare `permutation_importance`.
- Latenza inferenza ~50 ms (su 400 alberi) vs <1 ms di LR.
- Probabilita' meno calibrate: per output che vanno comunicati come "rischio %" considerare `CalibratedClassifierCV(rf, method='isotonic')`.

### 1.3 XGBoost (opzionale, dipendenza extra)

**Perche' opzionale**: il guadagno tipico su Diabetes-130 e' marginale (+1-2 pt AUC-PR su RF) e richiede tuning piu' fine. La dipendenza pesa ~50MB di binari C++. Lasciato come `pip install -e ".[advanced]"`.

**Quando usarlo**: se serve squeezing dell'ultimo punto percentuale di performance (es. competition Kaggle) oppure se gia' presente nello stack di produzione.

## 2. Iperparametri "saggi" — perche' questi e non altri

### 2.1 LogReg `C = 1.0`

Range provato: `[0.01, 0.1, 1.0, 10.0]`. Logiche:

- **Scala log**: $C$ entra inversamente -> step esponenziali coprono bene.
- **`C = 0.01`**: regolarizzazione forte, predizione quasi costante.
- **`C = 10`**: regolarizzazione debole, vicino a OLS.
- **Punto ottimo `C = 1.0`**: in pratica vince sempre o quasi su questo dataset. La regolarizzazione moderata e' coerente con il fatto che $n \gg p$ (80k righe vs ~200 colonne post-OHE).

### 2.2 RF `n_estimators = 400, max_features = 'sqrt'`

- `n_estimators`: 200 in baseline -> 400 in tuning. Oltre c'e' ROI marginale ma costo crescente. Differenza fra 400 e 800 alberi: <0.5 pt AUC-PR.
- `max_depth = None`: alberi completamente sviluppati. Decorrelazione + bagging compensano l'overfit del singolo albero (Breiman 2001).
- `max_features = 'sqrt'`: classico Breiman. Decorrela alberi diversi.
- `min_samples_leaf = 1` (default): nessuna sottostima del segnale rare.

### 2.3 Cross-validation `K = 5`, group-aware, stratificata

- **K = 5**: punto di equilibrio standard. K = 3 -> varianza alta, K = 10 -> tempi 2x. K = 5 sweet spot.
- **`StratifiedGroupKFold`**: bilancia il target *e* mantiene `patient_nbr` disgiunti. Sklearn>=1.0.
- **`shuffle=True, random_state=42`**: il dataset originale e' ordinato per `encounter_id`, correlato con tempo. Senza shuffle i fold avrebbero distribuzioni diverse di periodo -> varianza inflated.

### 2.4 Scoring `average_precision` (AUC-PR)

Su classi sbilanciate AUC-PR e' piu' informativa di AUC-ROC. Il tuning ottimizza direttamente la metrica che ci interessa nel contesto clinico (recall sulla classe positiva). Vedi `docs/teoria/05_metriche_classi_sbilanciate.md`.

## 3. Strategia di sbilanciamento: confronto

Tre strategie principali sul dataset Diabetes-130:

| Strategia | AUC-PR (CV) | Pro | Contro |
|---|---|---|---|
| **`class_weight='balanced'`** | ~0.20-0.22 | Semplice, robusto, no overfit | Trasformazione "soft" |
| `class_weight='balanced'` + soglia ottimizzata | ~0.20-0.22 (stesso AUC-PR, recall cambia) | Riusa fit, sposta solo il decision boundary | Soglia da ricalibrare in produzione |
| SMOTE + `class_weight=None` | ~0.22-0.25 | Piu' aggressivo sulla minoritaria | Genera dati sintetici clinicamente impossibili (interpolazione fra `gender='M'` e `gender='F'`) |
| Undersampling random della maggioritaria | ~0.18-0.20 | Veloce, modello bilanciato | Butta via dati informativi |

**Scelta progettuale**: `class_weight='balanced'` come default + ottimizzazione soglia. SMOTE come opzione esplorabile (dipendenza extra `imbalanced-learn`).

## 4. Soglia decisionale: 0.5 vs ottima

Default: 0.5. Ma su classi sbilanciate la soglia 0.5 e' raramente la migliore.

**Soglia ottima** sotto matrice costi (default `cost_fn_over_fp = 5`):

$$
\tau^* = \arg\min_\tau \big[\text{FP}(\tau) + 5 \cdot \text{FN}(\tau)\big]
$$

Su Diabetes-130 la soglia ottima cade tipicamente in [0.15, 0.30]: il modello "alerta" piu' generosamente per non perdere FN.

Implementato in `threshold.find_optimal_threshold`. La pipeline salva entrambe le metriche (soglia 0.5 e soglia ottima) nel `holdout_metrics.json`.

## 5. Selezione del miglior modello

Non scegliamo "il modello con AUC-PR piu' alta punto e basta". Criteri composti:

1. **AUC-PR su CV** (group-aware): metrica primaria.
2. **Stabilita'** (std fra fold).
3. **Interpretabilita'**: a parita' di performance, preferiamo LR (coefficienti) a RF (importance richiede SHAP per essere veramente leggibile).
4. **Latenza inferenza**: LR <1 ms; RF/XGB ~50 ms su 400 alberi. In sanita' su flussi batch non importa, ma e' un fattore in produzione real-time.
5. **Tempo di training**: rilevante per il retraining periodico.
6. **Performance fairness**: a parita' di AUC-PR, preferiamo il modello con gap di equalized_odds piu' basso.

Su test set tipici (con tuning standard):

- LogReg: AUC-PR ~0.20, recall@thr=0.5 ~0.55, training 3s, latenza <1ms.
- RandomForest: AUC-PR ~0.22, recall@thr=0.5 ~0.60, training 60s, latenza ~50ms.

Differenza marginale (~2 pt) su un problema dove l'AUC-ROC plateau in letteratura e' 0.60-0.67. **In produzione opterei per LogReg** per:

- Interpretabilita' nativa (cruciale per comunicare al clinico).
- Velocita' (deploy on-edge possibile).
- Robustezza al drift (modelli lineari degradano piu' lentamente quando la distribuzione cambia).

La pipeline seleziona automaticamente il modello con il miglior AUC-PR CV come `best_model.joblib`. In esperimenti reali la scelta finale va negoziata con la direzione clinica.

## 6. Cosa NON ho fatto (e perche')

- **Calibrazione esplicita** (`CalibratedClassifierCV`): le LR sono gia' ben calibrate; per RF migliorerebbe leggermente. Estensione semplice.
- **Stacking ensemble** (LR + RF + XGB con meta-learner): aggiungerebbe ~1 pt AUC-PR. Trade-off complessita'/beneficio negativo per il PW didattico.
- **Hyperparameter tuning bayesiano** (Optuna): piu' efficiente di GridSearch su grid grandi (>1000 combinazioni). Su Diabetes-130 le grid sono gestibili.
- **Mitigazione di fairness in-processing** (`ExponentiatedGradient`, `EqualizedOddsConstraint`): trade-off accuracy-fairness piu' efficiente del post-processing. Estensione esplorabile.
- **Time-aware split** (train sui primi anni, test sugli ultimi): piu' realistico ma il dataset non ha timestamp espliciti coerenti per supportarlo.
- **Modelli deep learning**: tabular structured data di queste dimensioni (80k righe, 50 feature) non beneficia di MLP/Transformer. Boosting/RF sono lo stato dell'arte.

## 7. Quale sarebbe il next step?

Per portare il modello in produzione clinica:

1. **Calibrazione**: `CalibratedClassifierCV(rf, method='isotonic')` se RF e' il modello scelto.
2. **Drift detection**: monitor sulle distribuzioni delle feature in input (KS-test mensile). Alert se KS > 0.15.
3. **Retraining schedule**: ogni 6-12 mesi su dati nuovi.
4. **Shadow mode** prima del go-live: il modello gira ma le predizioni non vengono usate; si confrontano con le decisioni dei clinici.
5. **A/B test** dell'intervento attivato: gruppo A = follow-up sui pazienti predetti a rischio, gruppo B = follow-up randomizzato. Misurare riduzione delle readmission.
6. **Audit fairness ricorrente**: ogni retraining, ricalcolare i gap DP/EO per sottogruppo. Se peggiorano, indagare.
7. **Logging delle predizioni** per debug e per costruire il dataset di follow-up.
8. **API REST** che espone `predict_readmission_risk()` (FastAPI + Docker, ~50 righe).
9. **Test di stabilita'**: se il modello viene ri-tunato, AUC-PR deve restare entro $\pm$5%.

Tutto fuori scope per il PW didattico, ma il codice e' gia' strutturato per supportarlo.
