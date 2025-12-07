import matplotlib.pyplot as plt

def eda(df):
    # Distribution des récompenses
    plt.figure(figsize=(5,4))
    df["reward"].hist(bins=10)
    plt.title("Distribution des récompenses")
    plt.xlabel("Reward")
    plt.ylabel("Fréquence")
    plt.grid(False)
    plt.show()

    # Reward glissant
    df["rolling_reward"] = df["reward"].rolling(window=40).mean()

    plt.figure(figsize=(8,4))
    plt.plot(df["rolling_reward"])
    plt.title("Reward moyen (fenêtre glissante)")
    plt.xlabel("Transitions")
    plt.ylabel("Reward moyen")
    plt.grid(True)
    plt.show()

    # Taux de réponses correctes
    print("Taux de réponses positives :", (df["reward"] > 0).mean())


