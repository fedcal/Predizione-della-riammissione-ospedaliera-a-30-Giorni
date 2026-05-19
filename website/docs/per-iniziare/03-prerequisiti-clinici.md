---
sidebar_position: 3
title: "Prerequisiti clinici"
description: "Vocabolario sanitario di base per affrontare il dataset Diabetes 130-US Hospitals: riammissione, ICD-9, HbA1c, HRRP, programmi di follow-up post-dimissione."
---

# Prerequisiti clinici

:::info Perché questa pagina
Non serve essere medici per fare questo progetto, ma servono **alcuni concetti minimi** per capire cosa significano le colonne del dataset e perché certe scelte di preprocessing sono motivate clinicamente.
:::

## 1. Cos'è una "riammissione a 30 giorni"

**Riammissione (readmission)** = ricovero ospedaliero che avviene **dopo** una dimissione precedente, di solito **non programmato** e per cause cliniche.

**A 30 giorni** = se il nuovo ricovero avviene **entro 30 giorni** dalla dimissione precedente.

Perché 30 e non 60 o 90?

- È lo standard usato dai **CMS** (Centers for Medicare & Medicaid Services, USA).
- I primi 30 giorni post-dimissione sono il periodo di **massima fragilità** clinica: il paziente è uscito dall'ambiente protetto, deve gestire nuovi farmaci, può avere complicazioni dell'episodio acuto.
- È la finestra in cui **un intervento di prevenzione può ancora fare la differenza** (telefonata, visita anticipata, riconciliazione farmacologica).

## 2. HRRP: il contesto economico

**Hospital Readmissions Reduction Program (HRRP)** è un programma del CMS introdotto nel 2010. Gli ospedali con tassi di readmission a 30 giorni **superiori alla media nazionale** per certe patologie (tra cui il diabete) ricevono **penalità economiche** sui rimborsi Medicare.

Implicazione: prevedere correttamente chi tornerà entro 30 giorni non è un esercizio accademico. Per un ospedale è **un problema da milioni di dollari l'anno**.

## 3. Il diabete in due frasi

- **Diabete tipo 1**: il pancreas non produce insulina. Insorgenza tipicamente giovanile, gestione con insulina dall'inizio.
- **Diabete tipo 2**: l'insulina viene prodotta ma il corpo non la usa efficacemente (insulino-resistenza). Tipicamente adulto/anziano, gestito con dieta, farmaci orali (metformina, sulfaniluree…), eventualmente insulina.

Il dataset contiene **ricoveri di pazienti diabetici**, indipendentemente dal tipo. Le 23 colonne sui farmaci coprono i principali principi attivi della terapia diabetica.

## 4. HbA1c: la "glicata"

**Emoglobina glicata (HbA1c)** = misura della glicemia media degli ultimi ~3 mesi. Espressa in percentuale.

| HbA1c | Interpretazione |
|---|---|
| `< 5.7%` | Normale |
| `5.7% – 6.4%` | Pre-diabete |
| `≥ 6.5%` | Diabete |
| `> 7%` | Diabete **mal controllato** (target terapeutico generale) |
| `> 8%` | Diabete molto mal controllato |

Nel dataset la colonna `A1Cresult` ha valori:

- `None` — il test **non è stato eseguito** durante il ricovero (~83% dei casi).
- `Norm` — eseguito, valore normale.
- `>7` — eseguito, valore alto.
- `>8` — eseguito, valore molto alto.

:::tip Insight clinico
Il paper originale (Strack et al., 2014) mostra che **misurare l'HbA1c durante il ricovero** è associato a tassi di readmission **inferiori**. Intuizione: chi misura, agisce sul controllo glicemico → migliore gestione post-dimissione.
:::

## 5. ICD-9: il "vocabolario" delle diagnosi

**ICD-9** = International Classification of Diseases, 9° revisione. Sistema di codifica numerica delle diagnosi.

Esempi:

| Codice | Descrizione |
|---|---|
| `250.xx` | Diabete mellito |
| `428` | Insufficienza cardiaca |
| `486` | Polmonite |
| `V58.6` | Uso a lungo termine di farmaci |

Nel dataset ci sono **tre colonne** di diagnosi (`diag_1`, `diag_2`, `diag_3`) per ogni ricovero. Ogni colonna può contenere **uno qualsiasi degli oltre 700** codici diversi.

:::warning Cardinalità esplosiva
Se fai un one-hot encoding "ingenuo" su `diag_1`, ottieni 700 nuove colonne. Il modello diventa ingestibile, fragile, lentissimo. **Soluzione**: raggruppare i codici in 9–15 **macro-categorie cliniche** (circolatorie, respiratorie, diabete, ecc.) seguendo la classificazione di Strack et al. (2014). Vedi *Teoria → 03 Preprocessing*.
:::

> Curiosità: oggi negli USA si usa l'**ICD-10**, ma il dataset (1999–2008) è in ICD-9. È uno dei limiti storici da menzionare nella discussione finale.

## 6. Tipi e fonti di ricovero

Tre colonne descrivono le **circostanze del ricovero**:

| Colonna | Cosa contiene |
|---|---|
| `admission_type_id` | Tipo: emergenza, urgenza, elettivo, neonato, trauma… |
| `admission_source_id` | Da dove arriva il paziente: pronto soccorso, ambulatorio, trasferimento, casa di cura… |
| `discharge_disposition_id` | Dove viene dimesso: casa, struttura riabilitativa, hospice, **deceduto**… |

Sono codificate con numeri (`1`, `2`, `3`…) che si mappano a descrizioni testuali tramite il file **`IDS_mapping.csv`** fornito con il dataset.

:::warning Casi da escludere
I pazienti dimessi come `Expired` (deceduti) o trasferiti in `Hospice` **non possono essere riammessi**: vanno **rimossi** dal dataset prima del training. Altrimenti il modello impara che "morto = non riammesso", il che è vero ma non è quello che vogliamo prevedere.
:::

## 7. Il programma di follow-up: cosa "attiva" il modello

Lo scopo finale della predizione è **assegnare risorse di follow-up** ai pazienti più a rischio. Le opzioni tipiche:

| Intervento | Costo unitario indicativo | Quando si attiva |
|---|---|---|
| Telefonata di follow-up (entro 48–72h) | ~15 $ | Tutti i flagged. |
| Visita ambulatoriale anticipata (entro 7 gg) | ~150 $ | Flag + criteri clinici. |
| Case-manager / coinvolgimento caregiver | ~200 $ | Casi complessi. |
| Riconciliazione farmacologica strutturata | ~80 $ | Politerapia. |

Capacità giornaliera limitata → il modello deve **classificare bene**, ma anche **ordinare** (chi è più a rischio per primo).

## 8. Variabili sensibili: race, age, gender

Il dataset contiene **attributi protetti** che richiedono attenzione etica:

- **`race`**: Caucasian, AfricanAmerican, Hispanic, Asian, Other, `?` (sconosciuto, ~2%).
- **`age`**: codificata in fasce di 10 anni (`[0-10)`, `[10-20)`, …, `[90-100)`).
- **`gender`**: Female, Male, Unknown/Invalid.

Usare la **razza** in un modello clinico è una scelta carica di implicazioni: la razza è un **costrutto sociale**, non biologico, ma riflette **disparità strutturali** che impattano sugli esiti di salute. Non c'è una risposta unica — il progetto richiede di prenderla esplicitamente, e di **misurare** le conseguenze tramite **fairness audit**.

Vedi *Teoria → 06 Fairness audit in sanità* per la discussione completa.

## Glossario rapido

| Termine | Definizione minima |
|---|---|
| **Readmission** | Nuovo ricovero dopo una dimissione precedente. |
| **HRRP** | Programma USA che penalizza ospedali con alte readmission a 30 gg. |
| **HbA1c** | Glicemia media a 3 mesi. Target `<7%` per diabetici. |
| **ICD-9** | Sistema di codifica diagnosi (qui ~700 codici). |
| **Encounter** | Singolo ricovero ospedaliero (riga del dataset). |
| **Patient_nbr** | ID paziente. **Può comparire più volte** (encounter multipli). |
| **Discharge disposition** | Modalità di dimissione (casa, hospice, deceduto…). |

## Prossimo passo

[**Percorso di studio consigliato**](./04-percorso-studio.md): l'ordine in cui leggere la documentazione e affrontare i notebook.
