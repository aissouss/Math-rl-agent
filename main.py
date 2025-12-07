from src.train import train
from src.test import test_agent
from src.eda import eda
import torch
import os

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("\n===== Entraînement de l'agent DQN =====")
    qnet, rewards, df = train(episodes=300)

    # Sauvegarde du modèle
    torch.save(qnet.state_dict(), "results/qnet.pth")
    print("Modèle sauvegardé dans results/qnet.pth")

    print("\n===== Test de l'agent après entraînement =====")
    test_agent(qnet, n=10)

    print("\n===== Analyse exploratoire (EDA) =====")
    eda(df)

    print("\nFin du pipeline — résultats disponibles dans le dossier /results/")
