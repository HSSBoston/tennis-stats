import pathlib
import sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(PRJ_DIR))

import numpy as np, pandas as pd
from constants import OUTPUT_DIR

CONFIDENCE_LEVEL = 0.95

# Import CSV files
eligibilityDf     = pd.read_csv(OUTPUT_DIR / "bootstrap-edge-eligibility.csv")
globalStabilityDf = pd.read_csv(OUTPUT_DIR / "bootstrap-edge-global-stability.csv",
                                index_col="bootstrap_iteration")
playerSummaryDf   = pd.read_csv(OUTPUT_DIR / "bootstrap-edge-player-summary.csv",
                                index_col="player")
edgeBootstrapDf   = pd.read_csv(OUTPUT_DIR / "bootstrap-edge-value-replicates.csv",
                                index_col="bootstrap_iteration")
rankBootstrapDf   = pd.read_csv(OUTPUT_DIR / "bootstrap-edge-rank-replicates.csv",
                                index_col="bootstrap_iteration")

# Preliminary consistency checks for the imported CSV files
eligiblePlayers = eligibilityDf.loc[
    eligibilityDf["eligible"] == True,
    "player",
].tolist()

numBootstrapSamples = len(edgeBootstrapDf)

if numBootstrapSamples == 0:
    raise ValueError("No EDGE bootstrap replicates were found.")

if len(rankBootstrapDf) != numBootstrapSamples:
    raise ValueError(
        "EDGE-value and rank files have different numbers of bootstrap iterations.")

if len(globalStabilityDf) != numBootstrapSamples:
    raise ValueError(
        "EDGE-value and global-stability files have different numbers of bootstrap iterations.")

if not edgeBootstrapDf.index.equals(rankBootstrapDf.index):
    raise ValueError("EDGE-value and rank bootstrap iteration IDs differ.")

if not edgeBootstrapDf.index.equals(globalStabilityDf.index):
    raise ValueError("EDGE-value and global-stability iteration IDs differ.")

for name, dfPlayers in {
    "EDGE replicate file": edgeBootstrapDf.columns,
    "rank replicate file": rankBootstrapDf.columns,
    "player-summary file": playerSummaryDf.index,
}.items():
    if set(dfPlayers) != set(eligiblePlayers):
        raise ValueError(f"The eligible-player cohort differs in the {name}.")

# Return descriptive statistics for a given data
#
def summarizeDistribution(values):
    return {
        "mean":     values.mean(),
        "median":   values.median(),
        "ci_lower": values.quantile(0.025),
        "ci_upper": values.quantile(0.975) }


if __name__ == "__main__":
    edgeBootstrapDf = edgeBootstrapDf.reindex(columns=eligiblePlayers)
    rankBootstrapDf = rankBootstrapDf.reindex(columns=eligiblePlayers)
    playerSummaryDf = playerSummaryDf.reindex(eligiblePlayers)

    alpha         = 1.0 - CONFIDENCE_LEVEL
    lowerQuantile = alpha / 2.0
    upperQuantile = 1.0 - alpha / 2.0

    # ---------------------------------------------------------------
    # 1. Number of eligible WTA top-100 players
    # ---------------------------------------------------------------

    numEligiblePlayers = len(eligiblePlayers)

    # ---------------------------------------------------------------
    # 2. Percentage of bootstrap iterations in which each player
    #    has a valid (numerical, non-NaN) EDGE value
    # ---------------------------------------------------------------

    # notna() generates:
    #          Sabalenka  Swiatek  Player C
    #   itr 1  True       True     True
    #   itr 2  True       True     False
    #   itr 3  True       True     True
    # sum(axis=0) generates:
    #   Sabalenka 3
    #   Swiatek   3
    #   Player C  2
    validEdgeCounts  = edgeBootstrapDf.notna().sum(axis=0)
    validEdgePercent = validEdgeCounts / numBootstrapSamples * 100.0

    playerAvailabilityDf = pd.DataFrame( {
        "player":                     eligiblePlayers,
        "valid_edge_replicates":      validEdgeCounts.reindex(eligiblePlayers).to_numpy(),
        "total_bootstrap_replicates": numBootstrapSamples,
        "valid_edge_percentage":      validEdgePercent.reindex(eligiblePlayers).to_numpy()
    } )

    # ---------------------------------------------------------------
    # 4. Median bootstrap EDGE standard error across eligible players
    # ---------------------------------------------------------------

    # std() generates:
    #  Sabalenka  std of her 5,000 EDGE values
    #  Swiatek    std of her 5,000 EDGE values
    playerEdgeSe = edgeBootstrapDf.std(axis=0, ddof=1)
    medianEdgeSe = playerEdgeSe.median()

    # ---------------------------------------------------------------
    # 5. Median and 90th-percentile EDGE confidence-interval width
    # ---------------------------------------------------------------

    edgeCiLower = edgeBootstrapDf.quantile(lowerQuantile, axis=0)
    edgeCiUpper = edgeBootstrapDf.quantile(upperQuantile, axis=0)
    edgeCiWidth = edgeCiUpper - edgeCiLower
      # edgeCiWidth:
      #   Sabalenka  her CI width
      #   Swiatek    her CI width

    medianEdgeCiWidth       = edgeCiWidth.median()
    percentile90EdgeCiWidth = edgeCiWidth.quantile(0.90)

    # ---------------------------------------------------------------
    # 6. Median width of the bootstrap rank interval
    # ---------------------------------------------------------------

    rankCiLower = rankBootstrapDf.quantile(lowerQuantile, axis=0)
    rankCiUpper = rankBootstrapDf.quantile(upperQuantile, axis=0)
    rankCiWidth = rankCiUpper - rankCiLower
      # rankCiWidth:
      #   Sabalenka  her rank interval
      #   Swiatek    her rank interval

    medianRankCiWidth = rankCiWidth.median()

    # ---------------------------------------------------------------
    # 7. Stability of the overall ranking
    # ---------------------------------------------------------------

    overallRankingStability = summarizeDistribution(
        globalStabilityDf["rank_correlation_with_original"])

    # ---------------------------------------------------------------
    # 8. Stability of the top-10 and top-20 sets
    # ---------------------------------------------------------------

    top10Stability = summarizeDistribution( globalStabilityDf["top_10_overlap_rate"])
    top20Stability = summarizeDistribution( globalStabilityDf["top_20_overlap_rate"])

    # ---------------------------------------------------------------
    # 9. Scale-relative EDGE confidence-interval width
    # ---------------------------------------------------------------

    originalEdge = playerSummaryDf["original_edge"].dropna()
    crossPlayerOriginalEdgeSd = originalEdge.std(ddof=1)
    relativeEdgeCiWidth = edgeCiWidth / crossPlayerOriginalEdgeSd
    medianRelativeEdgeCiWidth = relativeEdgeCiWidth.median()
    percentile90RelativeEdgeCiWidth = relativeEdgeCiWidth.quantile(0.90)

    # ---------------------------------------------------------------
    # Reconcile recalculated player quantities with the summary file
    # ---------------------------------------------------------------

    np.testing.assert_allclose(
        playerEdgeSe,
        playerSummaryDf["bootstrap_edge_se"],
        equal_nan=True)

    np.testing.assert_allclose(
        edgeCiWidth,
        playerSummaryDf["edge_ci_width"],
        equal_nan=True)

    np.testing.assert_allclose(
        rankCiWidth,
        playerSummaryDf["rank_ci_width"],
        equal_nan=True)

    np.testing.assert_allclose(
        validEdgePercent / 100.0,
        playerSummaryDf["availability_rate"],
        equal_nan=True)

    # ---------------------------------------------------------------
    # Assemble the overall results
    # ---------------------------------------------------------------

    overallResults = pd.Series( {
        "bootstrap_replicates":          numBootstrapSamples,
        "number_eligible_players":        numEligiblePlayers,
        "median_bootstrap_edge_se":       medianEdgeSe,
        "median_edge_ci_width":           medianEdgeCiWidth,
        "90th_percentile_edge_ci_width":  percentile90EdgeCiWidth,
        "median_relative_edge_ci_width":  medianRelativeEdgeCiWidth,
        "90th_percentile_relative_edge_ci_width": percentile90RelativeEdgeCiWidth,
        "median_rank_ci_width":           medianRankCiWidth,
        "mean_spearman_rank_correlation":       overallRankingStability["mean"],
        "median_spearman_rank_correlation":     overallRankingStability["median"],
        "spearman_correlation_2.5_percentile":  overallRankingStability["ci_lower"],
        "spearman_correlation_97.5_percentile": overallRankingStability["ci_upper"],
        "mean_top_10_overlap_rate":       top10Stability["mean"],
        "median_top_10_overlap_rate":     top10Stability["median"],
        "top_10_overlap_2.5_percentile":  top10Stability["ci_lower"],
        "top_10_overlap_97.5_percentile": top10Stability["ci_upper"],
        "mean_top_20_overlap_rate":       top20Stability["mean"],
        "median_top_20_overlap_rate":     top20Stability["median"],
        "top_20_overlap_2.5_percentile":  top20Stability["ci_lower"],
        "top_20_overlap_97.5_percentile": top20Stability["ci_upper"]
        } )

    print("\nOverall bootstrap diagnostics")
    print(overallResults.to_string())

    minAvailabilityPercent = 99.0
    minPlayerProportion = 0.90

    meetsAvailabilityRequirement = (
        playerAvailabilityDf["valid_edge_percentage"] >= minAvailabilityPercent)

    numMeetingAvailRequirement        = int(meetsAvailabilityRequirement.sum())
    proportionMeetingAvailRequirement = meetsAvailabilityRequirement.mean()
    
    passesRequirement = proportionMeetingAvailRequirement >= minPlayerProportion
    
    if passesRequirement:
        print("\nAvailability requirement satisfied:")
    else:
        print("\nAvailability requirement NOT satisfied:")
    print(
        f"{numMeetingAvailRequirement} of "
        f"{len(playerAvailabilityDf)} eligible players "
        f"({proportionMeetingAvailRequirement:.1%}) had valid EDGE "
        f"estimates in at least "
        f"{minAvailabilityPercent:.0f}% of bootstrap iterations." )

    # tolerance 1.0: stricter; current cohort fails;
    # tolerance 2.0: current cohort passes;
    # tolerance 3.0: relatively permissive.
    MAX_MEDIAN_RELATIVE_CI_WIDTH = 2.0

    passesCiRequirement = medianRelativeEdgeCiWidth <= MAX_MEDIAN_RELATIVE_CI_WIDTH
    if passesCiRequirement:
        print("\nPrecision requirement satisfied.")
    else:
        print("\nPrecision requirement NOT satisfied.")
    print(
        "Median relative EDGE CI width: "
        f"{medianRelativeEdgeCiWidth:.3f}")

    print("\nPlayer-level EDGE availability")
    print(playerAvailabilityDf.to_string(index=False,
                                         formatters={"valid_edge_percentage": "{:.2f}".format}))
    
    