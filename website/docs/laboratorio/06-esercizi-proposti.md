---
sidebar_position: 6
title: "Esercizi proposti"
description: "Cinque esercizi graduati per consolidare le competenze: dal facile (modifica matrice costi) all'avanzato (introduzione di un terzo modello + ablation study)."
---

# Esercizi proposti

:::tip Come usarli
Ogni esercizio ha **obiettivo**, **passi indicativi**, **output atteso** e **criterio di "fatto"**. Lavora in un branch git separato (`exercise/01-…`) per non sporcare `main`. Apri una PR a te stesso/a per rivedere il diff prima di chiudere.
:::

## Esercizio 1 — Re-tunare la soglia su una nuova matrice costi

**Difficoltà**: ⭐ (facile)
**Tempo stimato**: 30 minuti.

### Obiettivo

Capire come la **matrice costi** plasma la soglia ottima e tutte le metriche operative.

### Passi

1. Apri `src/readmit_pipeline/config.py` e modifica `COST_FN_OVER_FP` da `5.0` a `10.0`.
2. Rilancia `readmit-train`.
3. Confronta `holdout_metrics.json` con la versione precedente: come cambia `tau`, precision, recall, F2?
4. Documenta in mezza pagina **perché** la soglia si abbassa (o si alza).

### Output atteso

| Cost ratio | $\tau^*$ | Recall | Precision | FN | FP |
|---|---|---|---|---|---|
| 5 | ~0.27 | ~0.71 | ~0.14 | ~420 | ~6300 |
| 10 | ~0.20 | ~0.82 | ~0.11 | ~260 | ~9800 |

### "Fatto" se…

Sei in grado di rispondere in 2 frasi a: *"Cosa succede a $\tau^*$ se il costo del falso negativo cresce? Perché?"*

## Esercizio 2 — Aggiungere una nuova feature

**Difficoltà**: ⭐⭐
**Tempo stimato**: 1–2 ore.

### Obiettivo

Esercitare il **feature engineering clinico** e misurarne l'impatto.

### Passi

1. Crea una nuova feature `prior_utilization_score`:

   $$
   \text{score} = w_1 \cdot \text{n\_outpatient} + w_2 \cdot \text{n\_emergency} + w_3 \cdot \text{n\_inpatient}
   $$

   con pesi $w_1 = 1, w_2 = 2, w_3 = 4$ (motivazione: i ricoveri pesano più dell'ambulatoriale).
2. Aggiungila in `src/readmit_pipeline/features.py` (o equivalente).
3. Rilancia il training. AUC-PR migliora? Di quanto?
4. Verifica con permutation importance: la nuova feature è realmente informativa?

### Output atteso

- Tabella confronto: AUC-PR baseline vs con nuova feature.
- Feature importance plot.

### "Fatto" se…

L'AUC-PR cambia in modo misurabile (anche peggiorando!) e sai spiegare perché.

## Esercizio 3 — Modello alternativo e ablation

**Difficoltà**: ⭐⭐⭐
**Tempo stimato**: 2–3 ore.

### Obiettivo

Esercitare il confronto fra modelli e l'**ablation study** (rimuovere componenti per misurarne il contributo).

### Passi

1. Aggiungi un terzo classifier (es. `GradientBoostingClassifier` o `HistGradientBoostingClassifier`).
2. Esegui il training con tutti e tre i modelli.
3. Costruisci una **tabella di ablation**:

| Configurazione | AUC-PR (CV) | Recall @ τ\* | Precision @ τ\* |
|---|---|---|---|
| LogReg | 0.232 | … | … |
| LogReg + class_weight | 0.245 | … | … |
| RF | 0.247 | … | … |
| RF + class_weight | 0.255 | … | … |
| HistGB | 0.260 | … | … |
| HistGB + class_weight | 0.265 | … | … |

4. Decidi (motivando) quale modello finale "porteresti in produzione".

### Output atteso

- Tabella di ablation.
- 1 paragrafo che motiva la scelta finale (e che peso ha l'interpretabilità nel verdetto).

### "Fatto" se…

La scelta finale è motivata da **più di un criterio** (performance, interpretabilità, complessità di tuning, comportamento in fairness).

## Esercizio 4 — Mitigazione di fairness

**Difficoltà**: ⭐⭐⭐⭐
**Tempo stimato**: 3–4 ore.

### Obiettivo

Applicare una tecnica di mitigazione di fairness e quantificare il trade-off.

### Passi

1. Identifica il **gap più grande** dal `fairness_summary.csv` (es. equalized_odds su `race`).
2. Applica `ThresholdOptimizer` di Fairlearn con `constraints="equalized_odds"`:

   ```python
   from fairlearn.postprocessing import ThresholdOptimizer
   to = ThresholdOptimizer(
       estimator=trained_model,
       constraints="equalized_odds",
       prefit=True,
       predict_method="predict_proba",
   )
   to.fit(X_train, y_train, sensitive_features=A_train)
   y_pred_mitigated = to.predict(X_test, sensitive_features=A_test)
   ```

3. Ricalcola le metriche per gruppo **e** quelle aggregate.
4. Confronto:

| Metrica | Senza mitigazione | Con mitigazione |
|---|---|---|
| AUC-PR globale | … | … |
| Equalized odds diff (race) | 0.14 | … |
| Recall medio | … | … |

5. Scrivi 1 pagina di **discussione**: il trade-off è accettabile? Da chi viene presa la decisione, tecnicamente e organizzativamente?

### Output atteso

- Codice della mitigazione.
- Tabella confronto.
- Discussione critica.

### "Fatto" se…

La discussione tocca esplicitamente: *chi* decide il trade-off, *con quali stakeholder*, e *quale rischio etico* corre la scelta di **non** mitigare.

## Esercizio 5 — Report finale

**Difficoltà**: ⭐⭐⭐⭐⭐ (richiede tutti i precedenti)
**Tempo stimato**: 4–6 ore.

### Obiettivo

Scrivere un **report 4–5 pagine** che potresti consegnare a una direzione clinica.

### Struttura consigliata

1. **Executive summary** (½ pagina): problema, metrica primaria, risultato chiave, principale rischio.
2. **Contesto clinico e formulazione** (½ pagina): perché 30 gg, perché binarizzazione, matrice costi.
3. **Dati e preprocessing** (1 pagina): EDA, ICD-9 grouping, gestione missing, esclusioni.
4. **Modellazione e validazione** (1 pagina): modelli, CV, threshold tuning, soglia scelta, performance test.
5. **Fairness audit** (1 pagina): tabelle per gruppo, gap, scelta della definizione di equità, mitigazione (se applicata).
6. **Interpretabilità, limiti e raccomandazioni** (½ pagina): feature importanti, casi di fallimento, limiti del dataset, implicazioni operative.

### "Fatto" se…

- Un revisore non-tecnico (es. amministratore ospedaliero) capisce il messaggio chiave nei primi 2 paragrafi.
- Un revisore tecnico (es. data scientist senior) può rifare le tue scelte leggendo il documento.
- La discussione critica **non maschera i limiti** del modello. Anzi, li mette in evidenza.

## Quando hai finito

Apri una pull request sulla tua repository (o invia il branch) con:

- Il codice modificato.
- I file di output aggiornati (`reports/`).
- Il report in PDF o Markdown.
- Un changelog: cosa hai cambiato rispetto alla baseline.

E festeggia: hai fatto un progetto ML reale **end-to-end**, con consapevolezza etica. È più di quello che fanno il 90% dei corsi.

## Prossimo passo

Consulta le [**Appendici → Glossario**](/docs/appendici/glossario) per il vocabolario completo, e le [**FAQ**](/docs/appendici/faq) per le domande ricorrenti.
