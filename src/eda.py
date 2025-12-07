import matplotlib.pyplot as plt

def eda(df, rewards):
    # --- Courbe reward par épisode ---
    plt.figure(figsize=(8,4))
    plt.plot(rewards)
    plt.title("Reward par épisode (DQN — Additions 0–10)")
    plt.xlabel("Épisodes")
    plt.ylabel("Reward total")
    plt.grid(True)
    plt.show()

    # Aperçu dataset
    print(df.head())

    # --- Distribution des rewards ---
    plt.figure(figsize=(5,4))
    df["reward"].hist(bins=10)
    plt.title("Distribution des récompenses")
    plt.xlabel("Reward")
    plt.ylabel("Fréquence")
    plt.grid(False)
    plt.show()

    # --- Taux de réponses positives ---
    correct_rate = (df["reward"] > 0).mean()
    print(f"Taux de réponses positives : {correct_rate*100:.2f}%")

    # --- Reward moyen (fenêtre glissante) ---
    df["rolling_reward"] = df["reward"].rolling(window=40).mean()

    plt.figure(figsize=(8,4))
    plt.plot(df["rolling_reward"])
    plt.title("Reward moyen (fenêtre glissante)")
    plt.xlabel("Transitions")
    plt.ylabel("Reward moyen")
    plt.grid(True)
    plt.show()



