import itertools
import json
import numpy as np
import pandas as pd

M = json.load(open("pcos_model.json"))
FEATURES = M["features"]
LOW_CUT, HIGH_CUT = M["low_cut"], M["high_cut"]


def score(answers):
    x = np.array([answers.get(f, np.nan) for f in FEATURES], dtype=float)
    x = np.where(np.isnan(x), np.array(M["medians"]), x)
    z = (x - np.array(M["means"])) / np.array(M["scales"])
    return 1 / (1 + np.exp(-(M["intercept"] + float(z @ np.array(M["coefficients"])))))


def band(p):
    return "Low" if p < LOW_CUT else ("Moderate" if p < HIGH_CUT else "High")


binary = ["hair growth(Y/N)", "Skin darkening (Y/N)", "Weight gain(Y/N)",
          "Pimples(Y/N)", "Fast food (Y/N)"]

rows = []
for combo in itertools.product([0, 1], repeat=len(binary)):
    for cyc in (2, 4):                      
        for age in (18, 25, 32, 40):
            for clen in (3, 5, 8):
                a = dict(zip(binary, combo))
                a["Cycle(R/I)"] = cyc
                a["Age (yrs)"] = age
                a["Cycle length(days)"] = clen
                p = score(a)
                rows.append({**a, "p": round(p, 3), "band": band(p)})

df = pd.DataFrame(rows)
print(f"Total combinations tested: {len(df)}\n")

print("\nScore range per level:")
print(df.groupby("level")["p"].agg(["min", "max", "count"]).round(3))

mod = df[df.band == "Moderate"].sort_values("p")
print(f"\n--- {len(mod)} Moderate combinations. Ten examples: ---")
show = ["hair growth(Y/N)", "Skin darkening (Y/N)", "Weight gain(Y/N)", "Pimples(Y/N)",
        "Fast food (Y/N)", "Cycle(R/I)", "Age (yrs)", "p"]
print(mod[show].head(10).to_string(index=False))