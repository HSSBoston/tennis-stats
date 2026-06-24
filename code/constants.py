from pathlib import Path

PRJ_DIR    = Path(__file__).parent
MCP_DIR    = PRJ_DIR / "data" / "tennis_MatchChartingProject"
OUTPUT_DIR = PRJ_DIR / "output"

# 18 non-terminal game states (server's point first)
GAME_STATES = [
    "0-0",  "0-15",  "0-30",  "0-40",
    "15-0", "15-15", "15-30", "15-40",
    "30-0", "30-15", "30-30", "30-40",
    "40-0", "40-15", "40-30", "40-40",
    "AD-40", "40-AD",
]

EVENT_TYPES = [
    "ace",         # service ace or winner — server perspective
    "double_fault",          #                       — server perspective
    "forced_return_error_drawn", # forced return error drawn - server perspective
    "unforced_return_error", #                               — returner perspective
    "winner",                # rally winner (incl. return winner) — hitter perspective
    "forced_error_drawn",    # forced error drawn          — drawer (winner) perspective
    "unforced_error",        # unforced error during rally — errorer perspective
]
