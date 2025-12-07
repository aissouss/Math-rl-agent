# Math-RL-Agent


**Apprentissage par Renforcement appliqué au Calcul Mental (Additions 0–10)**
Projet universitaire — Implémentation d’un agent DQN apprenant à résoudre des additions simples par essais-erreurs.

---

## 📄 Description du projet

Ce projet met en œuvre un agent d’apprentissage par renforcement (**Deep Q-Learning**) chargé de résoudre des opérations d’addition aléatoires.
Aucun dataset n’est fourni : toutes les données sont générées dynamiquement par l’environnement **MathEnv** pendant l'entraînement.

Le projet contient :

* l’environnement RL générant les exercices,
* un modèle DQN simple (MLP),
* un algorithme d'entraînement avec replay buffer,
* une fonction de test pour évaluer l’agent,
* une fonction EDA permettant d’analyser les récompenses obtenues,
* un **rapport complet PDF** décrivant le contexte théorique et les résultats.

---

## 📁 Arborescence du projet

```
Math-rl-agent/
│
├── main.py                # Script principal : train -> test -> EDA
├── requirements.txt       # Dépendances du projet
├── README.md              # Documentation du projet
├── MATHAGENT.pdf          # Rapport complet
│
└── src/
    ├── env.py             # Environnement MathEnv
    ├── model.py           # Réseau DQN + ReplayBuffer + policy
    ├── train.py           # Fonction d'entraînement
    ├── test.py            # Fonction de test de l'agent
    └── eda.py             # Analyse simple (récompenses & rolling mean)
```

---

## 🚀 Exécution du projet

### 1️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2️⃣ Lancer l’entraînement + test + EDA

```bash
python main.py
```

Le script réalisera automatiquement :

* l’entraînement du modèle (300 épisodes)
* l’affichage de prédictions sur des additions aléatoires
* une petite analyse EDA en console et graphiques Matplotlib

---

## 📌 Fonctionnement général

### 🔹 **1. Environnement (MathEnv)**

Génère des additions aléatoires entre 0 et 10.
L’agent propose une réponse → reçoit +2 si correct, -1 si incorrect.

### 🔹 **2. Agent DQN**

Un MLP simple (2 → 64 → 64 → 21 actions).
Apprend une politique via l’algorithme Q-Learning.

### 🔹 **3. Mémoire Replay**

Permet de stabiliser l’apprentissage en réutilisant d’anciennes transitions.

### 🔹 **4. EDA**

Affiche :

* histogramme des récompenses
* reward moyen glissant
* taux de réponses correctes

---

## 📊 Rapport PDF

Ce document présente :

* le cadre théorique (RL, MDP, DQN)
* la modélisation
* la méthodologie
* les résultats
* une conclusion académique propre pour ton dossier ou CV

---

## 🧑‍💻 Auteure

**Aissya BOUKRAA** — Étudiante en L3 Informatique
Projet personnel + renforcement des connaissances en Machine Learning & RL.


