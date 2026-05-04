---
layout: default
title: Scelte tecniche
nav_order: 3
has_children: true
permalink: /scelte_tecniche/
description: >-
  Decisioni architetturali e di modellazione del progetto Hospital
  Readmission 30d, con trade-off espliciti su LogReg vs ensemble, gestione
  dello sbilanciamento, soglia di decisione e fairness.
---

# Scelte tecniche

Questa sezione documenta **come** è costruito il progetto e **perché** ogni
componente è stata progettata in un certo modo. È pensata per chi vuole
estendere o adattare la pipeline a un dominio clinico simile (altri
indicatori HRRP, altre coorti, altri dataset sbilanciati con attributi
sensibili).

## Capitoli

| Capitolo | Titolo | Cosa contiene |
|:--|:--|:--|
| 1 | [Architettura](architettura/) | Moduli `src/readmit_pipeline/`, flusso dati end-to-end, CLI (`readmit-train`, `readmit-predict`), dipendenze fra componenti. |
| 2 | [Scelte di modellazione](scelte_modello/) | Selezione famiglie di modelli (LogReg vs RandomForest), gestione sbilanciamento, ottimizzazione soglia su matrice costi, integrazione fairness audit. |

{: .tip }
> Per la teoria sottostante (problem framing clinico, metriche, fairness)
> consulta la sezione **[Teoria](../teoria/)**. Le scelte tecniche
> documentate qui presuppongono familiarità con i capitoli 1, 4, 5 e 6
> della sezione Teoria.
