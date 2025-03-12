import os, pickle, sympy


results = dict()

for roots, dirs, files in os.walk("."):
    for file in files:
        if ".p" in file:
            with open(file, "rb") as f:
                results[file] = pickle.load(f)

print(results)