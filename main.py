from src.train import train
from src.test import test_agent
from src.eda import eda

if __name__ == "__main__":
    # 1 — Entraînement
    qnet, rewards, df_transitions = train(300)

    # 2 — Test de l'agent après entraînement
    test_agent(qnet, 10)

    # 3 — Analyse EDA complète (plots + stats)
    eda(df_transitions, rewards)



