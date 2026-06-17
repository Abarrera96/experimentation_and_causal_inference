import pandas as pd
import urllib.request
import zipfile
import shutil
import os


# ================================================================ #
#  DATASET 1 — Cookie Cats (Notebook 02: A/B Test)                 #
#  Real randomized experiment from a mobile puzzle game            #
#  Source: Tactile Entertainment / DataCamp                        #
#                                                                  #
#  Context:                                                        #
#  Cookie Cats is a hugely popular mobile puzzle game. As players  #
#  progress, they hit "gates" that force them to wait or pay.      #
#  Tactile Entertainment ran an A/B test to decide whether to      #
#  move the gate from level 30 to level 40.                        #
#                                                                  #
#  Key variables:                                                   #
#    userid        — unique player ID                              #
#    version       — gate_30 (control) or gate_40 (treatment)      #
#    sum_gamerounds— rounds played in first 14 days                #
#    retention_1   — came back 1 day after install? (binary)       #
#    retention_7   — came back 7 days after install? (binary)      #
#                                                                  #
#  Why it's a real A/B test:                                       #
#    Players were randomly assigned to gate_30 or gate_40          #
#    when they installed the game. True randomization.             #
# ================================================================ #

def download_cookie_cats():

    os.makedirs("data", exist_ok=True)

    urls = [
        "https://raw.githubusercontent.com/ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats/master/cookie_cats.csv",
        "https://raw.githubusercontent.com/ileanadatamania/Data-Science-Portfolio/master/cookie_cats.csv",
    ]

    df = None
    for i, url in enumerate(urls):
        try:
            print(f"Trying URL {i+1}/{len(urls)}...")
            df = pd.read_csv(url)
            print(f"Success.")
            break
        except Exception as e:
            print(f"Failed: {e}")

    if df is None:
        print("\nAll URLs failed. Download manually:")
        print("  https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats")
        print("  Save as: data/cookie_cats.csv")
        raise RuntimeError("Could not download Cookie Cats dataset.")

    df.to_csv("data/cookie_cats.csv", index=False)
    print(f"\nCookie Cats dataset saved: data/cookie_cats.csv")
    print(f"  Rows         : {df.shape[0]:,}")
    print(f"  Control      : {(df['version']=='gate_30').sum():,} players (gate at level 30)")
    print(f"  Treatment    : {(df['version']=='gate_40').sum():,} players (gate at level 40)")
    print(f"\n  Retention D1 (control)  : {df[df['version']=='gate_30']['retention_1'].mean():.4f}")
    print(f"  Retention D1 (treatment): {df[df['version']=='gate_40']['retention_1'].mean():.4f}")
    print(f"\n  Retention D7 (control)  : {df[df['version']=='gate_30']['retention_7'].mean():.4f}")
    print(f"  Retention D7 (treatment): {df[df['version']=='gate_40']['retention_7'].mean():.4f}")


# ================================================================ #
#  DATASET 2 — Card & Krueger 1994 (Notebooks 03 + 04)            #
#  Natural experiment: NJ raised minimum wage, PA did not          #
#  Source: David Card, UC Berkeley                                 #
#                                                                  #
#  Context:                                                        #
#  In April 1992, New Jersey raised its minimum wage from $4.25    #
#  to $5.05. Pennsylvania did not change its minimum wage.        #
#  Card & Krueger surveyed 410 fast-food restaurants in both       #
#  states before (Feb 1992) and after (Nov 1992) the change.      #
#                                                                  #
#  Key variables:                                                   #
#    nj         — 1 if New Jersey (treated), 0 if Pennsylvania    #
#    fte        — full-time equivalent employees (pre)             #
#    fte2       — full-time equivalent employees (post)            #
#    fte_change — fte2 - fte (outcome for Double ML)              #
#    bk/kfc/roys/wendys — chain dummies                           #
#    co_owned   — 1 if company-owned (not franchise)              #
#                                                                  #
#  TWO formats created:                                            #
#                                                                  #
#  card_krueger_wide.csv — ONE row per restaurant                  #
#    Used in: 04_double_ml_dowhy.ipynb                            #
#    Why: Double ML needs fte_change as a single outcome column.   #
#    One row per unit, no time dimension in the model.            #
#                                                                  #
#  card_krueger_long.csv — TWO rows per restaurant (pre + post)   #
#    Used in: 03_difference_in_differences.ipynb                  #
#    Why: DiD regression needs a `post` column to estimate        #
#    the interaction term nj:post. This requires one row per      #
#    restaurant per time period:                                   #
#      fte_obs ~ nj + post + nj:post                              #
# ================================================================ #

def download_card_krueger():

    os.makedirs("data", exist_ok=True)

    print("Downloading Card & Krueger (1994) dataset from Berkeley...")
    url = "http://davidcard.berkeley.edu/data_sets/njmin.zip"

    try:
        urllib.request.urlretrieve(url, "njmin.zip")
        print("Download complete.")
    except Exception as e:
        print(f"Primary source failed: {e}")
        print("Trying fallback (Google Drive mirror)...")
        fallback = "https://docs.google.com/uc?id=10h_5og14wbNHU-lapQaS1W6SBdzI7W6Z&export=download"
        try:
            df_fallback = pd.read_csv(fallback)
            df_fallback.to_csv("data/card_krueger_long.csv", index=False)
            print(f"Fallback succeeded: {df_fallback.shape[0]} rows saved.")
            return
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            raise

    # Unzip
    print("Unzipping...")
    with zipfile.ZipFile("njmin.zip", "r") as z:
        z.extractall("njmin_raw")

    # Read raw data and assign column names from codebook
    print("Reading raw data and assigning column names...")
    df = pd.read_csv(
        "njmin_raw/public.dat",
        sep=r'\s+',
        header=None,
        na_values="."
    )

    columns = [
        "sheet", "chain", "co_owned", "state",
        "southj", "centralj", "northj", "pa1", "pa2", "shore",
        "ncalls", "empft", "emppt", "nmgrs", "wage_st",
        "inctime", "firstinc", "bonus", "pctaff", "meal",
        "open", "hrsopen", "psoda", "pfry", "pentree", "nregs", "nregs11",
        "type2", "status2", "date2", "ncalls2", "empft2", "emppt2",
        "nmgrs2", "wage_st2", "inctime2", "firstinc2", "special2",
        "meals2", "open2r", "hrsopen2", "psoda2", "pfry2", "pentree2",
        "nregs2", "nregs112"
    ]
    df.columns = columns

    # ── Key variables ──────────────────────────────────────── #

    # Treatment: 1 = New Jersey, 0 = Pennsylvania
    df["nj"] = (df["state"] == 1).astype(int)

    # FTE = full-time + 0.5 * part-time + managers
    df["fte"]        = df["empft"]  + 0.5 * df["emppt"]  + df["nmgrs"]
    df["fte2"]       = df["empft2"] + 0.5 * df["emppt2"] + df["nmgrs2"]
    df["fte_change"] = df["fte2"] - df["fte"]

    # Chain dummies
    df["bk"]     = (df["chain"] == 1).astype(int)
    df["kfc"]    = (df["chain"] == 2).astype(int)
    df["roys"]   = (df["chain"] == 3).astype(int)
    df["wendys"] = (df["chain"] == 4).astype(int)

    # ── WIDE FORMAT (one row per restaurant) ───────────────── #
    # Used in: 04_double_ml_dowhy.ipynb
    df.to_csv("data/card_krueger_wide.csv", index=False)

    # ── LONG FORMAT (two rows per restaurant) ──────────────── #
    # Used in: 03_difference_in_differences.ipynb
    shared = ["sheet", "nj", "bk", "kfc", "roys", "wendys",
              "co_owned", "southj", "centralj", "northj", "pa1", "pa2"]

    pre = df[shared + ["wage_st", "fte"]].copy()
    pre["post"] = 0
    pre.rename(columns={"fte": "fte_obs", "wage_st": "wage"}, inplace=True)

    post = df[shared + ["wage_st2", "fte2"]].copy()
    post["post"] = 1
    post.rename(columns={"fte2": "fte_obs", "wage_st2": "wage"}, inplace=True)

    df_long = pd.concat([pre, post], ignore_index=True)
    df_long.to_csv("data/card_krueger_long.csv", index=False)

    # ── Summary ────────────────────────────────────────────── #
    print(f"\nCard & Krueger datasets saved:")
    print(f"  data/card_krueger_wide.csv — {df.shape[0]} restaurants (one row each)")
    print(f"  data/card_krueger_long.csv — {df_long.shape[0]} rows (two per restaurant)")

    print("\nMean FTE by state and period:")
    summary = df_long.groupby(["nj", "post"])["fte_obs"].mean().unstack()
    summary.index   = ["Pennsylvania (control)", "New Jersey (treated)"]
    summary.columns = ["Pre (Feb 1992)", "Post (Nov 1992)"]
    print(summary.round(2))

    nj_change = summary.loc["New Jersey (treated)", "Post (Nov 1992)"] - summary.loc["New Jersey (treated)", "Pre (Feb 1992)"]
    pa_change = summary.loc["Pennsylvania (control)", "Post (Nov 1992)"] - summary.loc["Pennsylvania (control)", "Pre (Feb 1992)"]
    print(f"\nRaw DiD estimate: {nj_change - pa_change:.2f} FTE employees")

    # Cleanup
    os.remove("njmin.zip")
    shutil.rmtree("njmin_raw")
    print("Temp files cleaned up.")


# ================================================================ #
#  MAIN                                                            #
# ================================================================ #

if __name__ == "__main__":

    print("=" * 60)
    print("DATASET 1 — Cookie Cats (A/B Test)")
    print("=" * 60)
    download_cookie_cats()

    print()
    print("=" * 60)
    print("DATASET 2 — Card & Krueger 1994 (DiD + Double ML)")
    print("=" * 60)
    download_card_krueger()

    print("\nAll datasets ready. Files in data/:")
    for f in sorted(os.listdir("data")):
        size_kb = os.path.getsize(f"data/{f}") / 1024
        print(f"  {f:<40} {size_kb:>8.1f} KB")