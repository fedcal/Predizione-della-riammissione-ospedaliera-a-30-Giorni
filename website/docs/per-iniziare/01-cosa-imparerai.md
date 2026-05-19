---
sidebar_position: 1
title: "Cosa imparerai"
description: "Panoramica didattica del progetto Hospital Readmission 30d: obiettivi formativi, competenze ML e cliniche che acquisirai, traguardi misurabili a fine percorso."
---

# Cosa imparerai

:::tip Benvenuto in classe
Questa pagina è il **patto formativo** del progetto. Spiega cosa porterai a casa al termine del percorso, e cosa **non** è obiettivo (così non ti senti in colpa se qualcosa resta fuori).
:::

## Il problema in una frase

> **Dato un paziente diabetico al momento della dimissione, qual è la probabilità che torni in ospedale entro 30 giorni?**

Questo è un problema reale, costoso (~15.000 $ a riammissione negli USA), e — soprattutto — **ricco di insidie didattiche**: dati sporchi, classi sbilanciate, attributi sensibili, scelte etiche. È il classico progetto in cui il "modello che ha l'AUC più alta" **non è** la risposta giusta.

## Obiettivi di apprendimento

Al termine del progetto saprai:

### Sul piano del Machine Learning

1. **Formulare** un problema clinico come classificazione binaria, motivando ogni scelta (binarizzazione del target, definizione di "successo", scelta delle metriche).
2. **Gestire dati sporchi reali**: missing value codificati in modo non standard (`?` invece di `NaN`), feature ad altissima cardinalità (codici ICD-9 con 700+ valori unici), colonne a varianza zero, pazienti che compaiono più volte.
3. **Costruire una pipeline end-to-end** in scikit-learn (`ColumnTransformer` + `Pipeline`) che resti coerente fra training e inferenza, senza data leakage.
4. **Confrontare modelli** di natura diversa (regressione logistica vs ensemble) capendo perché ognuno sbaglia in modo diverso.
5. **Scegliere metriche appropriate** quando la classe positiva è all'11% (perché l'accuracy ti tradisce, perché AUC-PR è meglio di AUC-ROC, cos'è la F-β).
6. **Ottimizzare una soglia** decisionale rispetto a una **matrice dei costi**, non al default 0.5.

### Sul piano dell'equità algoritmica

7. **Eseguire un fairness audit** con [Fairlearn](https://fairlearn.org/) su attributi sensibili come `race` e `age`.
8. **Distinguere fra definizioni di equità**: demographic parity, equalized odds, predictive parity — e capire perché in genere **non si possono soddisfare tutte insieme**.
9. **Discutere criticamente** se e quando includere la razza in un modello clinico.

### Sul piano dell'ingegneria del software

10. **Strutturare un progetto Python** professionale: package installabile, CLI, test, configurazione esplicita.
11. **Documentare le scelte** in modo che un collega (o tu fra sei mesi) possa rifare il tuo ragionamento.
12. **Pubblicare** il lavoro su GitHub Pages tramite un workflow CI/CD.

## Cosa **non** è obiettivo

Per onestà didattica, ecco cosa **non** vedrai in questo progetto:

| Fuori scope | Perché |
|---|---|
| Deep learning su tabular | Su 50 feature strutturate, RF/XGBoost sono migliori e più semplici. |
| Tuning iperparametri estremo | L'AUC su questo dataset è notoriamente bassa (0.60–0.67 in letteratura). Inseguire +0.005 non è didattico. |
| Deploy in produzione (Kubernetes, FastAPI, monitoring) | Il focus è sulla pipeline ML, non sull'infrastruttura. |
| Time series / cox regression / survival analysis | Sono approcci validi per la readmission, ma fuori dal perimetro del corso. |

## Traguardi misurabili a fine progetto

Saprai rispondere — senza esitare — a queste domande:

1. *Perché hai trasformato un target a tre classi in binario?*
2. *Cos'è il group-aware split e perché qui è obbligatorio?*
3. *Perché AUC-ROC inganna su una classe positiva al 5%? Cosa usi al suo posto?*
4. *Cos'è la matrice dei costi e come la traduci in una soglia decisionale?*
5. *Cos'è la demographic parity? E l'equalized odds? Quale ha più senso in un programma di follow-up post-dimissione?*
6. *Il tuo modello ha recall più bassa sui pazienti afroamericani: cosa fai?*

Se a fine percorso queste domande non ti fanno più paura, l'obiettivo didattico è raggiunto.

## Il "perché" prima del "come"

Una raccomandazione che vale per tutto il progetto:

:::warning Regola d'oro
**Nessuna scelta tecnica senza una motivazione scritta.**
:::

Vale per la binarizzazione del target, per la strategia di imputazione dei missing, per la metrica primaria, per la soglia, per il modello finale. Un revisore — o un comitato etico — non valuta la tua AUC: valuta **la qualità del tuo ragionamento**.

## Prossimo passo

Vai a [**Prerequisiti di Machine Learning**](./02-prerequisiti-ml.md) per assicurarti di avere le basi necessarie, oppure salta direttamente al [**Percorso di studio**](./04-percorso-studio.md) se conosci già scikit-learn.
