import matplotlib.pyplot as plt

def eda(df):
    plt.figure(figsize=(5,4))
    df["reward"].hist(bins=10)
    plt.title("Distribution des récompenses")
    plt.show()

    df["rolling"] = df["reward"].rolling(window=40).mean()

    plt.figure(figsize=(8,4))
    plt.plot(df["rolling"])
    plt.title("Reward moyen (fenêtre glissante)")
    plt.show()

    print("Taux de réponses positives :", (df["reward"] > 0).mean())
