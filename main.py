from src.train import train
from src.test import test_agent
from src.eda import eda

if __name__ == "__main__":
    # Entraînement
    qnet, rewards, df = train(300)

    # Test après entraînement
    test_agent(qnet, 10)

    # Analyse EDA (plots + stats)
    eda(df)

