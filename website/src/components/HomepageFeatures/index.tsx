import type { ReactNode } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
  to: string;
  cta: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: "Teoria, spiegata bene",
    emoji: "📚",
    description: (
      <>
        Sette articoli su problem framing clinico, EDA, preprocessing, modelli,
        metriche, fairness audit, interpretabilità & limiti — con riferimenti
        alla letteratura medica.
      </>
    ),
    to: "/docs/category/teoria",
    cta: "Esplora la teoria",
  },
  {
    title: "Pipeline production-ready",
    emoji: "⚙️",
    description: (
      <>
        Codice modulare in Python con scikit-learn + Fairlearn, group-aware
        split su patient_nbr, ICD-9 grouping, fairness audit per race/age, CLI
        readmit-train.
      </>
    ),
    to: "/docs/scelte-tecniche/architettura",
    cta: "Vedi l'architettura",
  },
  {
    title: "Trade-off etici documentati",
    emoji: "🎯",
    description: (
      <>
        Ogni scelta motivata: perché LogReg+RF, come definire fairness, come
        gestire imbalance, dove sono i limiti del dataset 1999-2008 e
        dell'analisi di equità.
      </>
    ),
    to: "/docs/scelte-tecniche/scelte-modello",
    cta: "Leggi le decisioni",
  },
];

function Feature({ title, emoji, description, to, cta }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className={styles.featureCard}>
        <div className={styles.featureEmoji} role="img" aria-label={title}>
          {emoji}
        </div>
        <Heading as="h3" className={styles.featureTitle}>
          {title}
        </Heading>
        <p className={styles.featureDesc}>{description}</p>
        <Link
          className={clsx(
            "button button--primary button--sm",
            styles.featureCta,
          )}
          to={to}
        >
          {cta}
        </Link>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="text--center" style={{ marginBottom: "3rem" }}>
          <Heading as="h2">Cosa trovi in questa documentazione</Heading>
          <p
            style={{
              color: "var(--ifm-color-emphasis-700)",
              maxWidth: 700,
              margin: "0 auto",
            }}
          >
            Ogni sezione è autocontenuta. Leggi nell'ordine se vuoi una
            progressione didattica, oppure salta direttamente all'argomento che
            ti serve.
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
