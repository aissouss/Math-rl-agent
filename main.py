def test_agent(model, n=10):
    env = MathEnv()
    for _ in range(n):
        s = env.reset()
        x, y = s
        correct = int(x + y)

        with torch.no_grad():
            pred = model(torch.tensor(s, dtype=torch.float32)).argmax().item()

        print(f"{x} + {y} = ?  → Agent : {pred} | Correct : {correct}")


