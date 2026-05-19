---
sidebar_position: 2
title: "FAQ"
description: "Domande frequenti su Hospital Readmission 30d: scelte di progetto, errori comuni di setup, dubbi su fairness e metriche."
---

# FAQ — Domande frequenti

## Sul setup e il dataset

### Perché il dataset non è nella repository?

Per due motivi:

1. **Policy UCI / Kaggle**: la redistribuzione non è incoraggiata.
2. **Dimensioni**: ~20 MB di CSV non sono un problema, ma vogliamo che lo studente faccia *l'esperienza* di scaricare i dati da una fonte autorevole e capire la licenza.

### Posso usare il dataset Kaggle invece di UCI?

Sì, sono lo stesso dataset. Kaggle è più comodo da scaricare (1 clic dopo login), UCI è la fonte ufficiale.

### `pip install -e ".[notebooks]"` mi dà errore. Come faccio?

Quasi sempre serve un compilatore C per `numpy`/`scipy`. Su Linux: `sudo apt install build-essential python3-dev`. Su macOS: `xcode-select --install`. Su Windows: installa MS C++ Build Tools, oppure (più semplice) usa WSL2.

### Quanto spazio occupa l'installazione completa?

~500 MB inclusi virtualenv, dipendenze ML e dataset.

## Sul problem framing

### Perché binarizzare il target invece di tenerlo a 3 classi?

Tre ragioni:

1. **Coerenza con HRRP**: l'indicatore penalizzato dal sistema sanitario USA è la readmission a 30 giorni, non a 60 o 90.
2. **Azionabilità clinica**: il modello deve dire "questo paziente ha bisogno di follow-up" o "no". Una distribuzione su 3 classi è ambigua.
3. **Allineamento con la letteratura**: Strack 2014 e il tutorial Fairlearn usano la stessa binarizzazione → risultati confrontabili.

### Un paziente riammesso al giorno 31 è davvero "non a rischio"?

No, clinicamente è ambiguo. È un **costo accettato** della binarizzazione. Va menzionato esplicitamente nel report.

### Perché 5× come rapporto costi FN/FP e non 10× o 20×?

È una scelta **conservativa**. La letteratura health economics suggerisce un range 5–20× (un FN costa $15–20k, un FP $50–200). Cominciare da 5 mantiene il modello più "selettivo"; valori più alti aumentano i falsi positivi. È un parametro **da discutere con la committenza clinica**.

## Sul preprocessing

### Perché eliminare `weight` invece di imputarlo?

Con il 97% di missing, qualunque imputazione **inventa dati**, non li ricostruisce. La feature non è informativa e introduce solo rumore.

### Cosa fare con `medical_specialty` (49% missing)?

Tre opzioni, tutte legittime se motivate:

1. **Eliminarla** (perdiamo segnale clinico).
2. **Imputare** con "Unknown" (la categoria "Unknown" diventa essa stessa informativa: chi sono i pazienti senza specialità medica registrata?).
3. **Raggruppare** in macro-categorie cliniche (Internal Medicine, Cardiology, Surgery, …, Other, Unknown).

Nel progetto scegliamo la (3). Le altre sono valide se documentate.

### Perché raggruppare i codici ICD-9 in 9 macro-categorie?

700 valori unici → 700 colonne one-hot → modello che overfitta e impossibile da interpretare. Le macro-categorie (circolatorie, respiratorie, diabete, ecc.) preservano il segnale clinico riducendo drasticamente la dimensionalità.

## Sui modelli e le metriche

### Perché AUC-PR e non AUC-ROC come metrica primaria?

Su classi al 11%, l'AUC-ROC è gonfiato dalla presenza massiccia di negativi. L'AUC-PR misura **solo** la performance sulla classe positiva, che è quella che ci interessa.

### Random Forest mi dà AUC-PR solo 1–2% meglio di LogReg. Conviene?

Probabilmente no. Considerazioni:

- Il gap è dentro la std della CV → **non significativo**.
- LogReg è **interpretabile** (clinico legge i coefficienti).
- RF richiede tuning più attento.

In un setting clinico, LogReg + class_weight è una scelta **legittima e difendibile** anche se RF è marginalmente migliore.

### La mia AUC-PR è 0.65! Che ho sbagliato?

Probabilmente hai **data leakage**. Controlla:

- Stai usando `GroupKFold` su `patient_nbr`? Se no, lo stesso paziente è in train e test.
- Hai fittato il preprocessing **dopo** lo split, dentro la `Pipeline`? Se no, il preprocessing ha visto il test set.
- Hai usato la target (o feature derivate dalla target, come `readmitted`) come feature?

In letteratura, AUC-PR sopra 0.30 su questo dataset è **sospetto**.

### Devo provare XGBoost / LightGBM / CatBoost / DL?

XGBoost / LightGBM sono ragionevoli, ma il guadagno marginale è piccolo e introducono dipendenze esterne. Deep learning su 50 feature tabulari è **sconsigliato** (gli ensemble di alberi sono in genere superiori e più stabili).

## Sulla fairness

### Devo includere `race` nel modello?

È **la** domanda etica del progetto. Non c'è una risposta unica:

- **Includerla**: il modello la usa direttamente, rischio di **discriminazione diretta**, ma il fairness audit è più trasparente.
- **Escluderla**: il modello può comunque **inferirla** indirettamente da altre feature (es. ZIP code, certe diagnosi), e tu perdi la possibilità di misurare il gap.
- **Compromesso**: escluderla dal training, **usarla solo** in fairness audit. Spesso preferito in setting clinici.

La scelta va **dichiarata** e **motivata**.

### Demographic parity o equalized odds?

In sanità, in genere **equalized odds**. Motivo: la prevalenza vera della readmission **varia** fra gruppi (è un fatto biologico/sociale, non un bug). Forzare la stessa selection rate per tutti significherebbe sotto-allertare gruppi a rischio più alto. Equalized odds dice: *"a parità di vero stato, il modello tratta tutti uguali"*.

### Cos'è il "teorema di impossibilità" di Chouldechova?

In sintesi: se le **prevalenze sono diverse** fra gruppi, **non puoi soddisfare contemporaneamente** demographic parity, equalized odds e predictive parity. È un risultato matematico, non un'opinione. Devi **scegliere**.

### Cosa faccio se il mio gap di equalized odds è 0.10?

Dipende dal contesto:

- Per un programma di follow-up post-dimissione, **0.10 è notevole**: significa che un gruppo riceve il 10% in più (o in meno) di intercettazioni a parità di rischio.
- Le opzioni: (a) accettare e documentare; (b) applicare mitigazione (es. `ThresholdOptimizer`); (c) raccogliere più dati per il gruppo svantaggiato.

La decisione è etica, non tecnica.

## Sull'interpretabilità

### Come spiego il modello a un clinico?

Tre livelli, dal globale all'individuale:

1. **Globale**: feature importance (RF) o coefficienti standardizzati (LogReg).
2. **Locale**: SHAP value per singola predizione → "questo paziente è ad alto rischio perché ha 13 farmaci e 4 ricoveri precedenti".
3. **Counterfactual**: "se il paziente avesse fatto l'HbA1c, il rischio scenderebbe del 15%".

### Coefficienti LogReg negativi: cosa significano?

Una feature con coefficiente negativo **riduce** la probabilità di essere positivo. Esempio: se `A1Cresult_Norm` ha coefficiente negativo, significa che valori normali di HbA1c sono associati a minore probabilità di readmission a 30 gg → coerente con la letteratura.

## Sulla pubblicazione e il workflow

### Come pubblico la documentazione su GitHub Pages?

Tre step:

1. Push del codice su GitHub (branch `main` o `master`).
2. *Settings → Pages → Build and deployment → Source* = **GitHub Actions** (una sola volta).
3. Il workflow `.github/workflows/deploy-docs.yml` si occupa di tutto: build di Docusaurus + deploy.

Ogni push successivo aggiorna il sito.

### Posso usare il sito per il mio progetto del corso?

Sì, è MIT. Puoi fork-arlo, modificarlo, pubblicarlo con il tuo nome. Apprezzata (non obbligatoria) un'attribuzione.

### Non vedo i miei cambiamenti su GitHub Pages

- Hai aspettato 1–2 minuti dopo il push?
- *Actions* mostra il workflow verde?
- *Settings → Pages* punta a "GitHub Actions" e non a una branch specifica?

### Come testo localmente la documentazione?

```bash
cd website
npm install
npm run start    # apre http://localhost:3000 con hot reload
npm run build    # verifica che il build di produzione passi
```

## Altre domande

### Posso usare questo progetto come portfolio?

Sì, è il senso. Includi nel tuo portfolio:

- Link al repo GitHub.
- Link alla documentazione pubblicata.
- Il report finale (PDF).
- Una breve discussione del **risultato di fairness** (è quello che distingue il progetto).

### Dove trovo l'autore?

[**federicocalo.dev**](https://federicocalo.dev) per portfolio, altri progetti e contatti.

### Non ho trovato la mia domanda

Apri una issue su GitHub: [github.com/fedcal/Predizione-della-riammissione-ospedaliera-a-30-Giorni/issues](https://github.com/fedcal/Predizione-della-riammissione-ospedaliera-a-30-Giorni/issues)
