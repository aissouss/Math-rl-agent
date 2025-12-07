import os
import matplotlib.pyplot as plt

def eda(df):
    # Créer le dossier results/ s'il n'existe pas
    os.makedirs("results", exist_ok=True)

    # 1 — Courbe reward par épisode
    plt.figure(figsize=(8,4))
    df["reward"].plot()
    plt.title("Reward par épisode")
    plt.xlabel("Épisodes")
    plt.ylabel("Reward total")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/reward_curve.png")
    plt.close()

    # 2 — Distribution des récompenses
    plt.figure(figsize=(5,4))
    df["reward"].hist(bins=10)
    plt.title("Distribution des récompenses")
    plt.xlabel("Reward")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig("results/reward_distribution.png")
    plt.close()

    # 3 — Reward moyen glissant
    df["rolling"] = df["reward"].rolling(window=40).mean()
    plt.figure(figsize=(8,4))
    plt.plot(df["rolling"])
    plt.title("Reward moyen (fenêtre glissante)")
    plt.xlabel("Transitions")
    plt.ylabel("Reward moyen")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/rolling_reward.png")
    plt.close()

    print("✔️ Graphiques EDA enregistrés dans /results/")

