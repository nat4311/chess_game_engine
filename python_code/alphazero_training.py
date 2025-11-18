import subprocess

t = 1
while True:
    print("======================================")
    log = f"Training Loop: {t}"
    t+=1
    print(log)
    print("----------------------------------------")
    subprocess.run(["python3", "python_code/alphazero.py"])
