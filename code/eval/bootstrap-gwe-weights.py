import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(PRJ_DIR))

import numpy as np, pandas as pd
from dataloader import MCPDataLoader
from eventweights import computeDeltaGameWinExpectancy, computeEventWeights
from expectancy import computeGameWinExpectancy
from constants import EVENT_TYPES, GAME_STATES, OUTPUT_DIR, RNG_SEED


NUM_BOOTSTRAP_SAMPLES = 2000
CONFIDENCE_LEVEL = 0.95

def summarizeBootstrap(
    originalValues: dict,
    bootstrapValues: pd.DataFrame,
    orderedLabels: list[str],
) -> pd.DataFrame:
    """
    Summarize bootstrap estimates for GWE states or event weights.

    The original full-dataset estimates remain the reported point
    estimates. Bootstrap samples provide uncertainty statistics.
    """
    alpha = 1.0 - CONFIDENCE_LEVEL

    bootstrapValues = bootstrapValues.reindex(
        columns=orderedLabels
    )

    originalSeries = pd.Series(
        originalValues,
        dtype=float,
    ).reindex(orderedLabels)

    bootstrapMean = bootstrapValues.mean(axis=0)

    summaryDf = pd.DataFrame({
        "estimate": originalSeries,
        "bootstrap_mean": bootstrapMean,
        "bootstrap_median": bootstrapValues.median(axis=0),
        "bootstrap_bias": bootstrapMean - originalSeries,
        "bootstrap_se": bootstrapValues.std(
            axis=0,
            ddof=1,
        ),
        "ci_lower": bootstrapValues.quantile(
            alpha / 2.0,
            axis=0,
        ),
        "ci_upper": bootstrapValues.quantile(
            1.0 - alpha / 2.0,
            axis=0,
        ),
        "valid_replicates": bootstrapValues.notna().sum(
            axis=0
        ),
    })

    return summaryDf


if __name__ == "__main__":
    dl = MCPDataLoader("w")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Calculate the point estimates from the complete dataset.
    originalGweDict, _, originalPts = (
        computeGameWinExpectancy(dl.points)
    )

    originalDeltaPts = computeDeltaGameWinExpectancy(
        originalPts,
        originalGweDict,
    )

    originalWeightDict, _ = computeEventWeights(
        originalDeltaPts
    )

    # Create the generator once. Its state advances after each
    # call to dl.bootstrap(), producing distinct samples.
    rng = np.random.default_rng(RNG_SEED)

    gweDictList = []
    weightDictList = []

    for iteration in range(NUM_BOOTSTRAP_SAMPLES):
        bootstrappedPoints = dl.bootstrap(rng)

        # Re-estimate GWE from this bootstrap sample.
        gweDict, _, pts = computeGameWinExpectancy(
            bootstrappedPoints
        )

        # Use this sample's GWE estimates to calculate its
        # point-level changes in expectancy.
        deltaPts = computeDeltaGameWinExpectancy(
            pts,
            gweDict,
        )

        # Estimate the event weights for this sample.
        weightDict, _ = computeEventWeights(deltaPts)

        gweDictList.append(gweDict)
        weightDictList.append(weightDict)

        if (iteration + 1) % 100 == 0:
            print(
                f"Completed {iteration + 1}/"
                f"{NUM_BOOTSTRAP_SAMPLES} samples"
            )

    # Rows are bootstrap iterations; columns are quantities.
    gweBootstrapDf = pd.DataFrame(
        gweDictList
    ).reindex(columns=GAME_STATES)

    weightBootstrapDf = pd.DataFrame(
        weightDictList
    ).reindex(columns=EVENT_TYPES)

    gweBootstrapDf.index = range(
        1,
        NUM_BOOTSTRAP_SAMPLES + 1,
    )
    weightBootstrapDf.index = range(
        1,
        NUM_BOOTSTRAP_SAMPLES + 1,
    )

    gweBootstrapDf.index.name = "bootstrap_iteration"
    weightBootstrapDf.index.name = "bootstrap_iteration"

    # Produce mean, median, SE, bias, and percentile CI.
    gweSummaryDf = summarizeBootstrap(
        originalGweDict,
        gweBootstrapDf,
        GAME_STATES,
    )
    gweSummaryDf.index.name = "score_state"

    weightSummaryDf = summarizeBootstrap(
        originalWeightDict,
        weightBootstrapDf,
        EVENT_TYPES,
    )
    weightSummaryDf.index.name = "event"

    # Warn if a score state or event was absent from any sample.
    missingGweCounts = (
        NUM_BOOTSTRAP_SAMPLES
        - gweSummaryDf["valid_replicates"]
    )
    missingWeightCounts = (
        NUM_BOOTSTRAP_SAMPLES
        - weightSummaryDf["valid_replicates"]
    )

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

    outputPrefix = f"bootstrap-{TOUR}"

    # Save the raw replicates for reproducibility.
    gweBootstrapPath = (
        OUTPUT_DIR
        / f"{outputPrefix}-gwe-replicates.csv"
    )
    weightBootstrapPath = (
        OUTPUT_DIR
        / f"{outputPrefix}-event-weight-replicates.csv"
    )

    # Save the final summary tables.
    gweSummaryPath = (
        OUTPUT_DIR
        / f"{outputPrefix}-gwe-summary.csv"
    )
    weightSummaryPath = (
        OUTPUT_DIR
        / f"{outputPrefix}-event-weight-summary.csv"
    )

    gweBootstrapDf.to_csv(gweBootstrapPath)
    weightBootstrapDf.to_csv(weightBootstrapPath)
    gweSummaryDf.to_csv(gweSummaryPath)
    weightSummaryDf.to_csv(weightSummaryPath)

    print("\nOutput written to:")
    print(gweBootstrapPath)
    print(weightBootstrapPath)
    print(gweSummaryPath)
    print(weightSummaryPath)

