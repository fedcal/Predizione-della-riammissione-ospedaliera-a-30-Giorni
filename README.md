# Predizione della riammissione ospedaliera a 30 Giorni


La riammissione  ospedaliera entro 30 giorni dalla dimissione è un indicatore di qualità assistenziale riconosciuto a livello internazionale. L'obiettivo è simulare il lavoro di un team dati che supporta un ospedale nella costruzione di un modello predittivo capace di identificare, al momento della dimissione, i pazienti ad alto rischio di riammissione.

## Dataset
Userete il dataset “Diabetes 130-US Hospitals for Years 1999–2008”, disponibile nel UCI Machine Learning Repository (id 296) e su Kaggle. Il dataset raccoglie dieci anni di dati clinici (1999–2008) provenienti da 130 ospedali statunitensi e reti di assistenza integrata. Contiene 101.766 righe, ciascuna corrispondente a un ricovero ospedaliero di un paziente diabetico, e circa 50 feature.

## Colonne disponibili
Le feature principali includono:

- Identificativi: encounter_id (ID del ricovero), patient_nbr (ID del paziente — attenzione: uno stesso paziente può comparire più volte con ricoveri distinti).
- Dati demografici e sensibili: race (Caucasian, AfricanAmerican, Hispanic, Asian, Other, sconosciuto), gender (Female, Male, Unknown/Invalid), age (fasce decennali: [0–10), [10–20), …, [90–100)).
- Informazioni sul ricovero: admission_type_id, discharge_disposition_id, admission_source_id, time_in_hospital (giorni di degenza, da 1 a 14), payer_code, medical_specialty del medico responsabile.
- Indicatori di utilizzo del sistema sanitario: number_outpatient, number_emergency, number_inpatient (visite ambulatoriali, accessi al pronto soccorso e ricoveri nell’anno precedente).
- Dati clinici quantitativi: num_lab_procedures, num_procedures (procedure non di laboratorio), num_medications, number_diagnoses.
- Diagnosi: diag_1, diag_2, diag_3 (diagnosi primaria, secondaria e terziaria come codici ICD-9, ciascuna con oltre 700 valori unici).
- Esiti di laboratorio: max_glu_serum (risultato del test glicemico: >200, >300, Normal, None), A1Cresult (risultato dell’emoglobina glicata: >7, >8, Normal, None).
- Farmaci per il diabete: 23 colonne relative a singoli farmaci (metformin, insulin, glipizide, glyburide, pioglitazone, rosiglitazone, ecc.), ciascuna con valori che indicano se il dosaggio è stato aumentato (Up), ridotto (Down), mantenuto stabile (Steady) o se il farmaco non è stato prescritto (No).
- Indicatori riassuntivi sui farmaci: change (se c’è stata una variazione nella terapia diabetica: Yes/No), diabetesMed (se è stato prescritto un farmaco per il diabete: Yes/No).
- Variabile target: readmitted, con tre valori: <30 (riammesso entro 30 giorni), >30 (riammesso dopo 30 giorni), NO (non riammesso).

## Qualità dei dati
Il dataset presenta criticità rilevanti che fanno parte del lavoro richiesto:

- I valori mancanti sono codificati come ? (non come NaN), concentrati soprattutto in weight (circa 97% mancante), payer_code (circa 40%), medical_specialty (circa 49%) e race (circa 2%).
- Due colonne (examide, citoglipton) contengono un solo valore e sono prive di varianza.
- Le diagnosi (diag_1, diag_2, diag_3) hanno altissima cardinalità e richiedono raggruppamento in macro-categorie cliniche (es. circolatorie, respiratorie, diabete, ecc.).
- La variabile target è fortemente sbilanciata: la classe di interesse (<30) rappresenta circa l’11% dei ricoveri.

## Contesto e problema

La riammissione ospedaliera entro 30 giorni dalla dimissione è un indicatore di qualità assistenziale riconosciuto a livello internazionale. Negli Stati Uniti, il programma Hospital Readmissions Reduction Program (HRRP) penalizza economicamente gli ospedali con tassi di riammissione superiori alla media nazionale per determinate patologie, tra cui il diabete. Una riammissione non pianificata ha un costo stimato di decine di migliaia di dollari per episodio e rappresenta un segnale di gestione subottimale della transizione ospedale-territorio.

L’obiettivo è simulare il lavoro di un team dati che supporta la direzione clinica di un ospedale nella costruzione di un modello predittivo capace di identificare, al momento della dimissione, i pazienti diabetici ad alto rischio di riammissione entro 30 giorni. Il modello dovrà essere non solo accurato, ma anche equo: un sistema che predice sistematicamente peggio per determinati gruppi demografici (es. pazienti afroamericani o anziani) rischia di esacerbare disuguaglianze sanitarie già esistenti.

## Obiettivi
Alla fine del progetto, si dovrebbe essere in grado di:

- Impostare un problema di classificazione binaria su dati clinici reali, definendo la variabile target a partire da una colonna multiclasse e scegliendo metriche coerenti con il contesto sanitario.
- Gestire un dataset con missing values codificati in modo non standard, feature ad alta cardinalità (codici ICD-9), colonne a varianza zero, e pazienti ripetuti.
- Confrontare almeno due modelli di classificazione, valutandone sia la performance predittiva sia il comportamento differenziale rispetto a sottogruppi demografici.
- Condurre un’analisi di equità (fairness audit) utilizzando metriche di disparità e strumenti dedicati (es. Fairlearn), discutendone i risultati in chiave clinica e organizzativa.
- Produrre una lettura critica dei risultati che integri considerazioni predittive, etiche e operative.

## Obiettivo tecnico del progetto
Progettare, implementare e documentare una pipeline completa che:

- Carica e prepara i dati, gestendo i pazienti con ricoveri multipli in modo esplicito e motivato.
- Esegue preprocessing e feature engineering adeguati alla natura clinica dei dati.
- Addestra almeno due tipologie di modelli di classificazione.
- Valida i modelli con metriche orientate al contesto clinico e allo sbilanciamento.
- Conduce un’analisi di equità rispetto ad almeno due attributi protetti (race e age come minimo), misurando disparità nelle metriche predittive tra sottogruppi.
- Fornisce una discussione critica che colleghi i risultati tecnici a implicazioni operative ed etiche concrete.

### Fasi di lavoro consigliate

#### Fase 1: Comprensione del contesto clinico e definizione del problema
Prima di toccare il codice, è necessario comprendere il dominio.

Definire il significato clinico della readmission a 30 giorni: perché è diversa dalla readmission a 90 giorni o a un anno? Quale tipo di intervento potrebbe essere attivato sulla base della predizione (es. follow-up telefonico, visita ambulatoriale anticipata, coinvolgimento del caregiver)?

Decidere come trattare la variabile target originale a tre classi. La scelta raccomandata è la binarizzazione: <30 diventa la classe positiva (readmitted = 1), le altre due (>30 e NO) vengono aggregate nella classe negativa (readmitted = 0). Questa è la formulazione usata anche dal team Fairlearn di Microsoft nel loro tutorial sulla fairness. Motivare esplicitamente questa scelta e discutere le implicazioni (ad esempio: un paziente riammesso al giorno 31 è davvero “non a rischio”?).

Ragionare su una matrice dei costi asimmetrica: qual è il costo di un falso negativo (paziente ad alto rischio non intercettato, che viene riammesso) rispetto a un falso positivo (paziente a basso rischio che riceve un follow-up non necessario)? Nell’ambito sanitario, il costo del falso negativo è generalmente molto più alto, ma il falso positivo ha comunque un costo (risorse limitate, possibile sovra-trattamento). Questa matrice guiderà le scelte sulla soglia di decisione nelle fasi successive.

#### Fase 2: Analisi esplorativa (EDA)
Esplorare il dataset con attenzione alla struttura dei dati clinici:

- Distribuzione della variabile target (nelle tre classi originali e nella versione binarizzata).
- Distribuzione demografica: razza, genere, fasce d’età. Verificare se la distribuzione della target varia significativamente tra sottogruppi (ad esempio: il tasso di readmission a 30 giorni è diverso per pazienti afroamericani rispetto a caucasici?).
- Analisi dei pattern di utilizzo sanitario: pazienti con molte visite al pronto soccorso o ricoveri precedenti hanno tassi di readmission più elevati?
- Distribuzione dei giorni di degenza e del numero di farmaci e procedure.
- Analisi delle diagnosi primarie: raggruppare i codici ICD-9 in macro-categorie cliniche (circolatorie, respiratorie, diabete, digestive, muscoloscheletriche, neoplasie, ecc.) e verificare se il tasso di readmission varia per categoria diagnostica.
- Verifica del risultato dell’HbA1c e del test glicemico: la letteratura di riferimento (Strack et al., 2014) suggerisce che la misurazione dell’HbA1c è associata a tassi di readmission inferiori. Verificare questa ipotesi nel dataset.
- Analisi dei pazienti con encounter multipli: quanti pazienti compaiono più di una volta? Come si distribuiscono le readmission tra primo e successivi ricoveri?

#### Fase 3: Preprocessing e feature engineering
Gestione dei missing: i valori mancanti sono codificati come ?. Decidere strategia per colonna: weight è quasi interamente mancante e andrà probabilmente eliminata; race ha pochi missing e può essere imputata o mantenuta come categoria “Unknown”; medical_specialty e payer_code richiedono una scelta esplicita (eliminazione, imputazione, raggruppamento in macro-categorie). Documentare ogni scelta.

Gestione dei pazienti multipli: lo stesso patient_nbr può comparire più volte. Questo introduce correlazione tra osservazioni. Opzioni possibili: mantenere solo il primo ricovero per paziente; mantenere solo l’ultimo; mantenere tutti ma usare un group-aware split (cioè assicurarsi che tutti i ricoveri dello stesso paziente finiscano nello stesso fold). Motivare la scelta.

Eliminazione di record non informativi: rimuovere i ricoveri terminati con decesso del paziente (discharge_disposition_id corrispondenti a “Expired” o “Hospice”), dato che questi pazienti non possono essere riammessi.

Encoding delle variabili categoriche:

- Diagnosi (diag_1, diag_2, diag_3): raggruppare i codici ICD-9 in macro-categorie cliniche (9–15 gruppi), seguendo la classificazione proposta da Strack et al. (2014, Tabella 2 del paper originale).
- Farmaci per il diabete: le 23 colonne con valori (Up, Down, Steady, No) possono essere trattate come categoriche ordinali oppure semplificate in indicatori binari (farmaco prescritto sì/no, dosaggio modificato sì/no).
- admission_type_id, discharge_disposition_id, admission_source_id: mappare i codici numerici alle descrizioni cliniche (usando il file IDS_mapping.csv fornito con il dataset), poi raggruppare in macro-categorie significative.

Feature engineering:

- Creare un indicatore di complessità farmacologica: numero totale di farmaci diabetici diversi prescritti durante il ricovero.
- Creare un indicatore di intensità di utilizzo sanitario pregresso: somma pesata di visite ambulatoriali, accessi al pronto soccorso e ricoveri precedenti.
- Creare un indicatore di comorbidità: basato sulla presenza di specifiche macro-categorie diagnostiche nelle diagnosi secondaria e terziaria.
- Valutare se creare interazioni tra feature cliniche e demografiche (es. combinazione di fascia d’età e numero di farmaci).

Scaling: standardizzare o normalizzare le feature numeriche dove necessario per modelli sensibili alla scala (es. regressione logistica).

#### Fase 4: Modellazione
Addestrare almeno due modelli di classificazione:

- Un modello interpretabile come baseline: (ad es. Logistic Regression).
- Un modello ensemble: (ad es. Random Forest, XGBoost o LightGBM).

Almeno una strategia per affrontare lo sbilanciamento della classe positiva (~11%): class_weight=“balanced”, SMOTE, undersampling della classe maggioritaria, o calibrazione della soglia di decisione.

 Split dei dati: usare un holdout stratificato (es. 80/20) oppure una stratified k-fold cross-validation.

#### Fase 5: Validazione e ottimizzazione della soglia

Valutare i modelli con metriche adeguate al contesto:

- Metriche primarie: AUC-ROC, AUC-PR (precision-recall), recall sulla classe positiva (<30).
- Metriche secondarie: precision sulla classe positiva, F1 per la classe positiva, F-beta con beta > 1 (se si vuole privilegiare la recall).
- Non usare l’accuracy come metrica primaria: con l’89% di classe negativa, un modello che predice sempre “non riammesso” avrebbe circa l’89% di accuracy.

Costruire la curva precision-recall e la curva ROC. Studiare l’effetto della soglia di decisione sulla matrice dei costi definita in Fase 1: esiste un punto di lavoro che intercetta una quota sufficiente di pazienti a rischio senza generare troppi falsi positivi per il programma di follow-up?

#### Fase 6: Analisi di equità (Fairness Audit)
Questa fase è il tratto distintivo del progetto. L’analisi di equità deve essere condotta in modo strutturato.

Definire gli attributi sensibili (protetti): come minimo race e age. Opzionalmente gender.

Per ciascun attributo protetto, calcolare le metriche predittive (recall, precision, false positive rate, false negative rate) separatamente per ogni sottogruppo. Ad esempio: qual è la recall del modello sui pazienti afroamericani rispetto ai caucasici? Il modello “manca” più readmission per un gruppo che per un altro?

Applicare metriche di fairness standard:

- Demographic Parity (o Statistical Parity): la probabilità di essere classificati come “a rischio” è uguale tra i gruppi?
- Equalized Odds: dato lo stesso vero stato (readmitted o no), la probabilità di essere classificati correttamente è uguale tra i gruppi?
- Predictive Parity: dato che il modello predice “a rischio”, la probabilità che il paziente sia effettivamente riammesso è uguale tra i gruppi?

Discutere criticamente i risultati. Quale definizione è più rilevante nel contesto clinico? Ad esempio, in un programma di follow-up post-dimissione, è più importante che tutti i gruppi abbiano la stessa probabilità di essere intercettati se a rischio (equalized odds), o che tutti i gruppi ricevano lo stesso tasso di intervento (demographic parity)?

#### Fase 7: Interpretabilità e analisi degli errori

Studiare i falsi negativi (pazienti riammessi entro 30 giorni che il modello non ha intercettato): hanno caratteristiche comuni? Appartengono a specifici sottogruppi demografici o diagnostici?

Collegare i risultati a possibili azioni cliniche: se il modello identifica che i pazienti con molti ricoveri precedenti e nessuna misurazione dell’HbA1c sono ad alto rischio, questo suggerisce un intervento specifico (richiedere sistematicamente il test HbA1c).

#### Fase 8: Discussione critica e limiti
Discutere esplicitamente:

- Limiti del dataset: i dati coprono il periodo 1999–2008 e le pratiche cliniche possono essere cambiate; il peso corporeo è quasi interamente mancante; le diagnosi sono codificate in ICD-9 (oggi si usa ICD-10); non sono disponibili informazioni su condizioni socio-economiche, supporto familiare, aderenza terapeutica post-dimissione.
- Limiti del modello: le performance predittive su questo dataset sono notoriamente moderate (AUC-ROC tipicamente tra 0.60 e 0.67 in letteratura). Discutere perché: il problema è intrinsecamente difficile perché molti fattori determinanti della readmission non sono catturati nel dataset (condizioni abitative, accesso ai servizi territoriali, eventi acuti post-dimissione).
- Implicazioni etiche dell’uso di variabili come razza in un modello predittivo clinico: la razza è un costrutto sociale, non una variabile biologica. Include la razza nel modello perché è predittiva? O la si esclude per evitare discriminazione diretta, rischiando però di perdere informazione su disparità strutturali? Non esiste una risposta unica: l’obiettivo è dimostrare consapevolezza del dilemma.
- Considerazioni sulla messa in produzione: come si integrerebbe questo modello nel flusso di lavoro ospedaliero? Chi riceve l’output? Qual è il rischio di automation bias (il clinico che si fida ciecamente della predizione)?

### Deliverable richiesti

- Notebook o script in Python che copra l’intera pipeline: EDA, preprocessing, feature engineering, modellazione, validazione, analisi di equità, interpretabilità. Il codice deve essere leggibile, con sezioni ben separate, uso di seed per la riproducibilità, e commenti che motivino le scelte.
- Report di 4–5 pagine che descriva:
  - contesto clinico e formulazione del problema,
  - scelte di preprocessing e feature engineering, con motivazioni,
  - modelli provati e metriche ottenute,
  - risultati dell’analisi di equità, con discussione delle definizioni di fairness adottate e dei risultati osservati per sottogruppo,
  - analisi degli errori e delle feature più influenti,
  - discussione critica su limiti del dataset, del modello e implicazioni etiche.

### Criteri di valutazione

- Solidità dell’impostazione: chiarezza nella definizione del problema, scelta motivata della binarizzazione della target, definizione della matrice dei costi.
- Qualità dell’EDA e delle feature: capacità di estrarre insight clinicamente rilevanti, gestione corretta dei codici ICD-9, dei missing non standard e dei pazienti con ricoveri multipli.
- Rigorosità nella validazione: uso di metriche appropriate per dataset sbilanciati, gestione corretta dello split (senza leakage tra pazienti), studio della soglia di decisione.
- Profondità dell’analisi di equità: non è sufficiente calcolare una metrica di fairness; occorre interpretarla nel contesto clinico, discutere i trade-off tra definizioni di fairness incompatibili e collegare i risultati a implicazioni operative reali.
- Qualità della discussione critica: consapevolezza dei limiti, capacità di distinguere tra ciò che il modello può e non può catturare, riflessione etica non superficiale sull’uso di variabili sensibili.
- Qualità del codice e del report: struttura, leggibilità, capacità di comunicare sia a colleghi tecnici sia a stakeholder clinici e di governance.

### Risorse consigliate

- Dataset: https://archive.ics.uci.edu/dataset/296 oppure https://www.kaggle.com/datasets/brandao/diabetes
- Fairlearn (Microsoft): https://fairlearn.org — in particolare la documentazione sul Diabetes 130-Hospitals Dataset come caso studio.
- Paper originale: Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. BioMed Research International, 2014:781670.
- Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. Big Data, 5(2), 153–163.
- Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. Science, 366(6464), 447–453.

---

## Repository GitHub

**Nome del repository pubblico raccomandato**: `Predizione-della-riammissione-ospedaliera-a-30-Giorni`
URL: <https://github.com/fedcal/Predizione-della-riammissione-ospedaliera-a-30-Giorni>

Il sito di documentazione è costruito con **Docusaurus 3 + TypeScript + KaTeX** a partire dai sorgenti in [`website/docs/`](website/docs/) ed è pubblicato dal workflow [`.github/workflows/deploy-docs.yml`](.github/workflows/deploy-docs.yml) (build con `npm ci && npm run build`, deploy via `actions/deploy-pages@v4`). **Setup richiesto una volta sola**: *Settings → Pages → Build and deployment → Source = **GitHub Actions***. Ogni push su `main` aggiorna il sito.

## Documentazione completa

Il sito Docusaurus (italiano, mobile-first, SEO-ready, KaTeX per la matematica) copre teoria clinica, scelte tecniche e razionale del modello:

- **[Documentazione online](https://fedcal.github.io/Predizione-della-riammissione-ospedaliera-a-30-Giorni/)**

Sezioni principali:

- **Teoria** — problem framing clinico, EDA, preprocessing dati clinici, modelli per classificazione binaria sbilanciata, metriche AUC-PR / F-beta, fairness audit in sanita', interpretabilita' & limiti.
- **Scelte tecniche** — architettura modulare, decisioni di modellazione (LogReg vs Ensemble), trade-off espliciti.

## Quick start

### 1. Setup

```bash
git clone https://github.com/fedcal/Predizione-della-riammissione-ospedaliera-a-30-Giorni.git
cd Predizione-della-riammissione-ospedaliera-a-30-Giorni

python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[notebooks]"
```

### 2. Scaricare il dataset

Il dataset **Diabetes 130-US Hospitals for Years 1999-2008** (UCI ML Repository, id 296) va scaricato manualmente — non e' incluso nella repo per motivi di licenza/dimensione.

1. Vai su <https://archive.ics.uci.edu/dataset/296>.
2. Scarica lo ZIP, estrai `diabetic_data.csv` e `IDS_mapping.csv`.
3. Copiali in `data/raw/`.

### 3. Pipeline completa

```bash
readmit-train              # full tuning, ~5-15 min su laptop
readmit-train --quick      # smoke test, ~1 min
```

Output:

- `reports/models/best_model.joblib` — pipeline serializzata pronta per l'inferenza.
- `reports/cv_summary.csv` — risultati K-fold (AUC-PR per modello).
- `reports/holdout_metrics.json` — metriche test (incluse soglia 0.5 e soglia ottima sotto matrice costi).
- `reports/fairness_summary.csv` — gap demographic_parity ed equalized_odds per `race`/`age`.
- `reports/fairness_report.csv` — metriche per sottogruppo (selection_rate, TPR, FPR, precision).
- `reports/figures/*.png` — curve ROC, PR, confusion matrix.

### 4. Inferenza

```python
from readmit_pipeline.inference import predict_readmission_risk

paziente = {
    "race": "Caucasian", "gender": "Female", "age": "[60-70)",
    "time_in_hospital": 4, "num_medications": 15,
    "A1Cresult": ">7", "diag_1": "428", "insulin": "Steady",
}
proba = predict_readmission_risk(paziente)
print(f"Rischio readmission 30d: {proba:.1%}")
```

Oppure da CLI:

```bash
echo '{"race":"Caucasian","age":"[60-70)","num_medications":15,"insulin":"Steady"}' > paziente.json
readmit-predict --input paziente.json --threshold 0.3
```

### 5. Notebook didattici

```bash
python scripts/build_notebooks.py
jupyter lab notebooks/
```

I 5 notebook sono pensati per essere letti in sequenza:

1. `01_eda_demographics_target.ipynb` — distribuzione target, demografia, missing values, encounter multipli.
2. `02_preprocessing_icd9_grouping.ipynb` — feature engineering, mapping ICD-9, ColumnTransformer.
3. `03_models_logreg_vs_ensemble.ipynb` — group-aware split, tuning AUC-PR, confronto modelli.
4. `04_fairness_audit.ipynb` — Fairlearn `MetricFrame`, demographic_parity, equalized_odds, predictive_parity.
5. `05_interpretability_and_errors.ipynb` — coefficienti LR, feature importance, analisi errori, limiti.

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso *Machine Learning Engineer* di **DataMasters/Skiller**.

Per altri progetti, contatti e portfolio: <https://federicocalo.dev>.

## Licenza

[MIT License](LICENSE) © 2026 Federico Calò.

Il dataset *Diabetes 130-US Hospitals for Years 1999-2008* e' di pubblico dominio (Strack et al., 2014; UCI ML Repository, id 296).