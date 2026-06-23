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

        
        
