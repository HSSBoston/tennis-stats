# EDGE - Event-Driven Gain in Expectancy

import pandas as pd


def computeEdge(
    playerNname: str,
    df: pd.DataFrame,
    matches: pd.DataFrame,
    w_dict: dict)
-> Optional[dict]:

    in_matches = matches[
        (matches["Player 1"] == player_name) |
        (matches["Player 2"] == player_name)
    ]
    if in_matches.empty:
        return None

    p1_map = dict(zip(in_matches["match_id"], in_matches["Player 1"]))
    pnum = {mid: (1 if p1_map[mid] == player_name else 2)
            for mid in in_matches["match_id"]}

    sub = df[df["match_id"].isin(pnum)].copy()
    if sub.empty:
        return None
    sub["player_num"] = sub["match_id"].map(pnum).astype("Int64")
    sub["is_server"] = sub["player_num"] == sub["Svr"]

    valid = sub.dropna(subset=["event", "perspective"])
    owned = (
        ((valid["perspective"] == "server")   &  valid["is_server"]) |
        ((valid["perspective"] == "returner") & ~valid["is_server"])
    )
    player_pts = valid[owned]
    N = len(player_pts)
    if N == 0:
        return None
    counts = player_pts["event"].value_counts().to_dict()
    X = sum(w_dict.get(e, 0.0) * c for e, c in counts.items()) / N
    return {"player": player_name, "X": X, "N": N, "counts": counts,
            "matches": int(in_matches.shape[0])}



if __name__ == "__main__":
    from constants import playersW, playersM
    from dataloader import MCPDataLoader
    from winprob import computeV
    from eventweights import computeDeltaV, computeW

    dl = MCPDataLoader("w", playersW)
    points  = dl.points
    matches = dl.matches
    
    vDict, vDf, pts = computeV(points)
    vDfSorted = vDf.sort_values(["game win prob"])
    print(vDfSorted)
    
    weights = computeW( computeDeltaV(pts, vDict) )
    print(weights)
    
    wDict = weights["w"].to_dict()
#     print( wDict )

    res = computeEdge(name, pts, matches, w_dict)
