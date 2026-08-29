# /// notebook
# # ch09 · Migraines – TMS vs Placebo
#
# 2×2 contingency table: TMS treatment for migraines.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import marimo as mo
    from stat2lib.data import load_rda, Stat2Table
    return Path, Stat2Table, load_rda, mo, pl, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(1122, "Migraines", "tms treatment of migraines", ["Group", "Yes", "No", "Trials"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda):
    data = load_rda(desc.name)
    print("TMS vs Placebo migraine relief:")
    data
    return data

@app.cell
def _(data, pl):
    tms = data.filter(pl.col("Group") == "TMS")
    placebo = data.filter(pl.col("Group") == "Placebo")
    tms_yes = tms["Yes"].item(); tms_no = tms["No"].item()
    plc_yes = placebo["Yes"].item(); plc_no = placebo["No"].item()
    print(f"TMS:     {tms_yes}/{tms_yes + tms_no} = {tms_yes/(tms_yes + tms_no):.1%}")
    print(f"Placebo: {plc_yes}/{plc_yes + plc_no} = {plc_yes/(plc_yes + plc_no):.1%}")
    print(f"\nOdds ratio: {(tms_yes/tms_no)/(plc_yes/plc_no):.2f}")
    return

if __name__ == "__main__": app.run()