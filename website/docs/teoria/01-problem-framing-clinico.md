---
sidebar_position: 1
title: "Problem framing clinico"
description: "Capitolo: Problem framing clinico. Progetto Hospital Readmission 30d."
---

# Problem framing clinico

## Indice

---

## 1. Cosa misura la "readmission a 30 giorni"

La **riammissione ospedaliera entro 30 giorni dalla dimissione** e' un indicatore di qualita' assistenziale internazionale. Non e' un evento "naturale" ma un costrutto clinico-organizzativo: misura quanto bene la **transizione ospedale-territorio** ha funzionato per quel paziente.

Negli Stati Uniti, il programma **Hospital Readmissions Reduction Program (HRRP)** — istituito dal Patient Protection and Affordable Care Act del 2010 — penalizza economicamente gli ospedali con tassi di riammissione superiori alla media nazionale per condizioni specifiche (tra cui il diabete). Una riammissione non pianificata costa in media **$15.000-20.000 per episodio** (CMS data, 2018) e segnala una **gestione subottimale** della transizione di cura: educazione paziente insufficiente, mancanza di follow-up territoriale, riconciliazione farmacologica errata.

### Perche' 30 giorni e non 90 o un anno

La finestra dei **30 giorni** e' uno standard CMS (Centers for Medicare & Medicaid Services). E' clinicamente significativa perche':

- I primi 30 giorni post-dimissione sono il periodo di **massima fragilita'**: il paziente e' uscito dall'ambiente protetto, deve gestire autonomamente nuovi farmaci, ed e' soggetto a complicazioni dell'episodio acuto.
- Riammissioni a >30 giorni sono spesso eventi clinici **nuovi**, scollegati dal ricovero originale. Difficile attribuirle alla qualita' del ricovero indice.
- E' la finestra in cui un **intervento di prevenzione** (telefonata di follow-up, visita ambulatoriale anticipata) puo' ancora cambiare l'esito.

## 2. Binarizzazione del target: `<30 days` come classe positiva

Il target originale `readmitted` ha **tre valori**:

| Valore | Significato |
|---|---|
| `<30` | Riammesso entro 30 giorni dalla dimissione |
| `>30` | Riammesso oltre 30 giorni |
| `NO`  | Non riammesso (entro l'orizzonte di osservazione) |

**Convenzione adottata** (allineata al tutorial di Microsoft Fairlearn e al paper Strack 2014):

$$
y_{\text{30d}} = \begin{cases} 1 & \text{se } \texttt{readmitted} = \text{`&lt;30'} \\ 0 & \text{altrimenti} \end{cases}
$$

### Perche' binarizzare

- **Coerenza con HRRP**: l'indicatore penalizzato e' la riammissione a 30 giorni, non a 60 o 90.
- **Azionabilita'**: l'output del modello deve guidare un intervento (follow-up telefonico, visita ambulatoriale entro 7 giorni). Una probabilita' singola e' immediatamente utilizzabile da un clinico, una distribuzione su tre classi non lo e'.
- **Statistica**: la classificazione binaria con `<30` come classe positiva ha un trattamento metodologico standard (AUC-PR, recall, soglie di costo). La classificazione multiclasse 1-vs-rest aggiungerebbe complessita' senza valore clinico.

### Caveat importante

> Un paziente riammesso al **giorno 31** e' considerato "non a rischio" da questa formulazione. Clinicamente la differenza fra giorno 29 e giorno 31 e' minima: il modello ne paghera' un piccolo costo predittivo. E' un trade-off accettabile per la chiarezza della formulazione.

## 3. Matrice dei costi asimmetrica

In sanita', i due tipi di errore non hanno costo simmetrico.

### Falso negativo (FN): paziente ad alto rischio non intercettato

Il modello predice `readmission = 0` ma il paziente viene riammesso entro 30 giorni. Costo:

- **Costo diretto**: l'ospedale paga il ricovero ($15-20k) + penalita' HRRP.
- **Costo indiretto**: peggiore esito clinico per il paziente (eventuali complicanze evolute mentre era a casa).
- **Costo etico**: il paziente avrebbe beneficiato di un follow-up che non e' arrivato.

### Falso positivo (FP): paziente a basso rischio classificato come "a rischio"

Il modello predice `readmission = 1` ma il paziente sta bene. Costo:

- **Costo diretto**: telefonata di follow-up + eventuale visita ambulatoriale (~$50-200 per episodio).
- **Costo indiretto**: tempo di un infermiere/case-manager sottratto a chi ne avrebbe piu' bisogno.
- **Costo emotivo**: ansia indotta nel paziente.

### Quantificazione del rapporto

Una stima conservativa, supportata dalla letteratura health economics:

$$
\frac{\text{costo}(\text{FN})}{\text{costo}(\text{FP})} \in [5, 20]
$$

Default in `config.py`: **`COST_FN_OVER_FP = 5.0`** (estremo conservativo). Il modulo `threshold.py` ottimizza la soglia decisionale minimizzando:

$$
\mathcal{L}(\tau) = \text{FP}(\tau) + 5 \cdot \text{FN}(\tau)
$$

dove $\text{FP}(\tau)$, $\text{FN}(\tau)$ sono i conteggi alla soglia $\tau$.

## 4. Quali interventi attiva il modello

Dal punto di vista operativo, l'output del modello (`probability >= threshold`) attiva un **bundle di follow-up post-dimissione**:

| Intervento | Costo unitario | Evidenza efficacia (RR riduzione readmission) |
|---|---|---|
| Telefonata di follow-up (48-72h) | ~$15 | 0.75-0.85 |
| Visita ambulatoriale anticipata (entro 7 giorni) | ~$150 | 0.65-0.80 |
| Coinvolgimento case-manager / caregiver | ~$200 | 0.70-0.85 |
| Riconciliazione farmacologica strutturata | ~$80 | 0.80-0.90 |

Fonte: meta-analisi Leppin et al. (2014, JAMA Intern Med). I numeri sono indicativi; il punto e' che **l'intervento ha un costo finito** e va distribuito sui pazienti **piu' a rischio** per massimizzare l'impatto.

## 5. Vincoli operativi che derivano dal contesto

### 5.1 Capacita' del programma di follow-up

Il numero di follow-up giornalieri che un ospedale puo' eseguire e' **finito** (vincolato dal personale infermieristico). Se il modello "alerta" il 30% dei dimessi, ma il programma puo' gestire solo il 15%, serve **ranking** non solo classificazione.

Implicazione: la **probabilita' (`predict_proba`)** e' piu' utile della classe binaria. La soglia diventa un parametro **negoziabile** con la direzione clinica in base alla capacita' operativa.

### 5.2 Latenza del modello

Il modello deve produrre la predizione **al momento della dimissione**, in finestra di pochi minuti (mentre il clinico compila la lettera di dimissione). Non e' un problema computazionale per LogReg/RF su 50 feature, ma e' un vincolo da menzionare in produzione.

### 5.3 Esplicabilita' obbligatoria

In sanita' non basta una predizione: serve una **giustificazione**. Per un paziente classificato "ad alto rischio", il clinico vuole sapere **perche'**:

- "13 farmaci diversi, 4 ricoveri nell'anno precedente, HbA1c >8%."

Questa lista guida il bundle di intervento (es. focus sulla riconciliazione farmacologica). Implicazione architetturale: si privilegiano modelli con **interpretabilita' nativa** (LogReg) o si aggiunge SHAP/feature_importance.

## 6. Riferimenti

- **Strack, B. et al.** (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International, 2014:781670.
- **Leppin, A. L. et al.** (2014). *Preventing 30-day hospital readmissions: a systematic review and meta-analysis of randomized trials*. JAMA Internal Medicine, 174(7), 1095-1107.
- **Centers for Medicare & Medicaid Services**. *Hospital Readmissions Reduction Program (HRRP)*. https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp
- **Kansagara, D. et al.** (2011). *Risk prediction models for hospital readmission: a systematic review*. JAMA, 306(15), 1688-1698.