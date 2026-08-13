import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(PRJ_DIR))

import numpy as np, pandas as pd
from dataloader import MCPDataLoader
from eventweights import computeDeltaGameWinExpectancy, computeEventWeights
from expectancy import computeGameWinExpectancy
from constants import EVENT_TYPES, GAME_STATES, OUTPUT_DIR, RNG_SEED

# Exploratory analysis               500–1,000
# Standard errors                    1,000–2,000
# 95% confidence intervals           2,000–5,000
# Publication-quality tail estimates 5,000–10,000+
# Very small p-values                10,000+
NUM_BOOTSTRAP_SAMPLES = 5000
CONFIDENCE_LEVEL = 0.95

# Summarize bootstrap estimates for game-win expectancy values (for different
# score states) or event weights (for different event types). The original
# full-dataset estimates are also reported.
#
def summarizeBootstrap(
    originalValues:  dict,
    bootstrapValues: pd.DataFrame,
    orderedLabels:   list[str],
) -> pd.DataFrame:
    
    alpha = 1.0 - CONFIDENCE_LEVEL

    bootstrapValues = bootstrapValues.reindex(columns=orderedLabels)
    originalSeries  = pd.Series(originalValues, dtype=float).reindex(orderedLabels)
    bootstrapMean   = bootstrapValues.mean(axis=0)

    summaryDf = pd.DataFrame( {
        "original":         originalSeries,
        "bootstrap_mean":   bootstrapMean,
        "bootstrap_median": bootstrapValues.median(axis=0),
        "bootstrap_bias":   bootstrapMean - originalSeries,
        "bootstrap_se":     bootstrapValues.std(axis=0, ddof=1),
        "ci_lower":         bootstrapValues.quantile(alpha/2.0, axis=0),
        "ci_upper":         bootstrapValues.quantile(1.0 - alpha/2.0, axis=0),
        "valid_replicates": bootstrapValues.notna().sum(axis=0)
    } )
    return summaryDf


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dl = MCPDataLoader("w")
    originalGweDict, _, originalPts = computeGameWinExpectancy(dl.points)
    originalWeightDict, _ = computeEventWeights(
        computeDeltaGameWinExpectancy(originalPts, originalGweDict) )

    rng = np.random.default_rng(RNG_SEED)
    gweDictList = []
    weightDictList = []

    for iteration in range(NUM_BOOTSTRAP_SAMPLES):
        bootstrappedPoints, _ = dl.bootstrap(rng)
        gweDict, _, pts = computeGameWinExpectancy(bootstrappedPoints)
        weightDict, _ = computeEventWeights(
            computeDeltaGameWinExpectancy(pts, gweDict) )
        gweDictList.append(gweDict)
        weightDictList.append(weightDict)

        if (iteration + 1) % 100 == 0:
            print(f"Completed {iteration + 1}/{NUM_BOOTSTRAP_SAMPLES} samples")

    gweBootstrapDf    = pd.DataFrame(gweDictList).reindex(columns=GAME_STATES)
    weightBootstrapDf = pd.DataFrame(weightDictList).reindex(columns=EVENT_TYPES)

    gweBootstrapDf.index    = range(1, NUM_BOOTSTRAP_SAMPLES + 1)
    weightBootstrapDf.index = range(1, NUM_BOOTSTRAP_SAMPLES + 1)

    gweBootstrapDf.index.name    = "bootstrap_iteration"
    weightBootstrapDf.index.name = "bootstrap_iteration"

    # Produce mean, median, standard error (SE), bias, and percentile CI.
    gweSummaryDf = summarizeBootstrap(
        originalGweDict,
        gweBootstrapDf,
        GAME_STATES
    )
    gweSummaryDf.index.name = "score_state"

    weightSummaryDf = summarizeBootstrap(
        originalWeightDict,
        weightBootstrapDf,
        EVENT_TYPES
    )
    weightSummaryDf.index.name = "event"

    # Warn if a score state or event was absent from any sample.
    missingGweCounts    = NUM_BOOTSTRAP_SAMPLES - gweSummaryDf["valid_replicates"]
    missingWeightCounts = NUM_BOOTSTRAP_SAMPLES - weightSummaryDf["valid_replicates"]

    if (missingGweCounts > 0).any():
        print("\nWarning: missing GWE estimates:")
        print(missingGweCounts[missingGweCounts > 0])
    if (missingWeightCounts > 0).any():
        print("\nWarning: missing event-weight estimates:")
        print(missingWeightCounts[missingWeightCounts > 0])

    print("\nGWE bootstrap summary")
    print(gweSummaryDf)

    print("\nEvent-weight bootstrap summary")
    print(weightSummaryDf)

    gweBootstrapPath    = OUTPUT_DIR / "bootstrap-gwe-replicates.csv"
    weightBootstrapPath = OUTPUT_DIR / "bootstrap-event-weight-replicates.csv"
    gweSummaryPath      = OUTPUT_DIR / "bootstrap-gwe-summary.csv"
    weightSummaryPath   = OUTPUT_DIR / "bootstrap-event-weight-summary.csv"

    gweBootstrapDf.to_csv(gweBootstrapPath)
    weightBootstrapDf.to_csv(weightBootstrapPath)
    gweSummaryDf.to_csv(gweSummaryPath)
    weightSummaryDf.to_csv(weightSummaryPath)

    print("\nOutput written to:")
    print(gweBootstrapPath)
    print(weightBootstrapPath)
    print(gweSummaryPath)
    print(weightSummaryPath)

