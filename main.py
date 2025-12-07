from src.train import train
from src.test import test_agent
from src.eda import eda

if __name__ == "__main__":
    qnet, rewards, df = train(300)
    test_agent(qnet, 10)
    eda(df)
