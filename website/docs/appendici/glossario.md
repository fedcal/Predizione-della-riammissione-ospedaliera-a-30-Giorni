---
sidebar_position: 1
title: "Glossario"
description: "Glossario di tutti i termini tecnici (ML) e clinici usati nel progetto Hospital Readmission 30d, con definizioni brevi e collegamenti ai capitoli di approfondimento."
---

# Glossario

:::tip Come usarlo
Cerca col `Ctrl/Cmd + F`. Quando un termine ha più aspetti, la definizione punta al capitolo di teoria che lo approfondisce.
:::

## Termini di Machine Learning

### AUC-PR
Area sotto la **precision-recall curve**. Metrica primaria del progetto su classi sbilanciate. Baseline = prevalenza positiva (~0.11). Vedi *Teoria → 05 Metriche*.

### AUC-ROC
Area sotto la **ROC curve** (TPR vs FPR). Misura la qualità del ranking. Su classi sbilanciate **può ingannare**: la usiamo come riferimento secondario.

### Class weighting
Tecnica per gestire lo sbilanciamento: si dà più peso agli errori sulla classe minoritaria durante il training (`class_weight="balanced"` in sklearn).

### ColumnTransformer
Componente sklearn che applica trasformazioni **diverse** a colonne **diverse** (es. scaling sulle numeriche, one-hot sulle categoriche).

### Confusion matrix
Tabella 2×2 con TN, FP, FN, TP. Da qui derivano tutte le metriche.

### Cross-validation (CV)
Stima della performance tramite ripetizione del fit/test su fold diversi. Qui usiamo `GroupKFold` per non spezzare i pazienti fra fold.

### Data leakage
Contaminazione del test con informazione del training. Spesso causato da preprocessing fatto **prima** dello split, o da `patient_nbr` non gestito.

### Demographic parity
Definizione di equità: $P(\hat{Y}=1 \mid A=a)$ uguale fra gruppi. Vedi *Teoria → 06 Fairness*.

### Equalized odds
Definizione di equità: $P(\hat{Y}=1 \mid Y=y, A=a)$ uguale fra gruppi. Spesso la più rilevante in sanità.

### F-β
$F_\beta = (1+\beta^2)\cdot\frac{P \cdot R}{\beta^2 \cdot P + R}$. Con $\beta > 1$ enfatizza la recall.

### Fairlearn
Libreria Microsoft per audit e mitigazione di fairness in modelli ML. [fairlearn.org](https://fairlearn.org/)

### False negative (FN)
Paziente realmente riammesso che il modello classifica come "non a rischio". **Errore clinicamente più grave** nel nostro setup.

### False positive (FP)
Paziente non riammesso che il modello classifica come "a rischio". Errore meno costoso, ma non a costo zero.

### Group-aware split
Split che garantisce che tutti i record di un gruppo (qui `patient_nbr`) finiscano nello stesso fold. Indispensabile per evitare leakage da encounter multipli.

### Holdout
Set di test tenuto da parte, mai visto durante training/tuning. Usato **una sola volta** per stimare la performance finale.

### Imputation
Sostituzione dei valori mancanti con valori plausibili (media, mediana, "Unknown"…).

### Logistic Regression (LogReg)
Modello lineare per classificazione binaria. Output = probabilità tramite funzione sigmoide. **Interpretabile** (coefficienti leggibili).

### MetricFrame
Oggetto Fairlearn che calcola un dizionario di metriche **per sottogruppo** e ne riporta gap aggregati.

### One-hot encoding
Trasformazione di una colonna categorica in N colonne binarie (una per categoria).

### Pipeline
Concatenazione di step (preprocessing + modello) in un unico oggetto sklearn. Garantisce assenza di leakage e riproducibilità.

### Precision (PPV)
$\frac{TP}{TP+FP}$. Frazione di allarmi corretti.

### Predictive parity
Definizione di equità: $P(Y=1 \mid \hat{Y}=1, A=a)$ uguale fra gruppi.

### Random Forest
Ensemble di decision tree con bootstrap e selezione casuale di feature. Robusto, gestisce non linearità.

### Recall (TPR, sensibilità)
$\frac{TP}{TP+FN}$. Frazione di positivi correttamente intercettati.

### ROC curve
Plot di TPR vs FPR al variare della soglia.

### SHAP
*SHapley Additive exPlanations*: tecnica di interpretabilità che assegna a ogni feature un contributo per ciascuna predizione.

### SMOTE
*Synthetic Minority Over-sampling Technique*: genera esempi sintetici della classe minoritaria interpolando fra vicini.

### Soglia (threshold) $\tau$
Valore sopra il quale una probabilità viene classificata come positiva. Tunata su matrice costi, **non** lasciata a 0.5 per default.

### Stratified split
Split che preserva la proporzione delle classi in train e test.

### True negative (TN) / True positive (TP)
Predizione corretta (negativa / positiva).

## Termini clinici

### A1Cresult / HbA1c
Emoglobina glicata: glicemia media dei 3 mesi precedenti. Target `<7%` per diabetici. Nel dataset valori `None`, `Norm`, `>7`, `>8`.

### CMS
*Centers for Medicare & Medicaid Services* (USA). Definisce gli standard di rimborso e le metriche di qualità ospedaliera.

### Comorbidità
Presenza simultanea di più condizioni patologiche. Misurata nel progetto tramite le 3 colonne `diag_*`.

### Discharge disposition
Modalità di dimissione (codice in `discharge_disposition_id`): home, home_health_care, hospice, expired, ecc.

### Encounter
Singolo ricovero ospedaliero (= una riga del dataset). Identificato da `encounter_id`.

### HRRP
*Hospital Readmissions Reduction Program*: programma USA che penalizza ospedali con tassi di readmission a 30 giorni superiori alla media.

### ICD-9
*International Classification of Diseases, 9th revision*: sistema di codifica delle diagnosi usato nel dataset (~700 codici unici).

### Patient_nbr
ID anonimo del paziente. **Può comparire più volte** (encounter multipli) → richiede group-aware split.

### Riammissione (readmission)
Nuovo ricovero dopo una dimissione precedente, di solito non programmato.

### Riconciliazione farmacologica
Processo strutturato di revisione della terapia del paziente al momento di una transizione di cura. Riduce errori di prescrizione.

### Strack et al. 2014
Paper di riferimento per il dataset: *Impact of HbA1c Measurement on Hospital Readmission Rates*. Definisce la classificazione delle diagnosi in macro-categorie.

### Variabile sensibile (attributo protetto)
Caratteristica demografica su cui non vogliamo discriminare: `race`, `age`, `gender` nel nostro caso.

## Termini di engineering del software

### CLI (entry-point)
Comando da terminale registrato nel `pyproject.toml`: qui `readmit-train` e `readmit-predict`.

### Editable install
`pip install -e .`: installa il package in modo che le modifiche al sorgente siano immediatamente attive.

### joblib
Libreria Python per serializzare oggetti (pipeline, modelli) su disco.

### pyproject.toml
File di configurazione del package Python (sostituto moderno di `setup.py`).

### Virtualenv (venv)
Ambiente Python isolato. Da usare **sempre**.

## Riferimenti bibliografici

- **Strack, B. et al.** (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International, 2014:781670.
- **Leppin, A. L. et al.** (2014). *Preventing 30-day hospital readmissions: a systematic review and meta-analysis*. JAMA Internal Medicine.
- **Chouldechova, A.** (2017). *Fair prediction with disparate impact: A study of bias in recidivism prediction instruments*. Big Data, 5(2), 153–163.
- **Kleinberg, J., Mullainathan, S., Raghavan, M.** (2016). *Inherent trade-offs in the fair determination of risk scores*. arXiv:1609.05807.
- **Obermeyer, Z. et al.** (2019). *Dissecting racial bias in an algorithm used to manage the health of populations*. Science, 366(6464), 447–453.
- **Fairlearn User Guide**: [fairlearn.org/main/user_guide](https://fairlearn.org/main/user_guide/index.html)
- **scikit-learn API**: [scikit-learn.org/stable/api](https://scikit-learn.org/stable/api/)
