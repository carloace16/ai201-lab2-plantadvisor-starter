import json
import os
from datetime import datetime
from config import DATA_PATH

# Plant database and seasonal data are loaded once at module load.
# This mirrors how a real service would cache its data source in memory.
with open(os.path.join(DATA_PATH, "plants.json"), encoding="utf-8") as f:
    _plant_db = json.load(f)

with open(os.path.join(DATA_PATH, "seasons.json"), encoding="utf-8") as f:
    _season_data = json.load(f)

# Maps calendar months to seasons for auto-detection.
_MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall",  10: "fall",  11: "fall",
}


def lookup_plant(plant_name: str) -> dict:

    target = plant_name.strip().lower()

    target_key = target.replace(" ","_")
    if target_key in _plant_db:
        return {"found": True, "plant": _plant_db[target_key]}
    
    for key, data in _plant_db.items():
        if target == data.get("display_name", "").lower():
            return {"found": True, "plant": data}

        if target == data.get("scientific_name", "").lower():
            return {"found": True, "plant": data}

        for item in data.get("aliases", []):
            if target == item.lower():
                return {"found": True, "plant": data}

    return {
        "found": False,
        "name": plant_name,
        "message": (
            f"'{plant_name}' is not in the local plant database. Do not invent "
            f"specific care numbers as if you had real data. Instead, acknowledge "
            f"it isn't in the database, then offer general guidance based on the "
            f"plant type if you can infer it, and suggest a reliable source for "
            f"specific care details."
        ),
    }


def get_seasonal_conditions(season: str | None = None) -> dict:
    """
    Return current seasonal care context for houseplants.

    If season is provided and valid, returns that season's data.
    If season is None (or invalid), auto-detects from the current calendar month.

    Pre-implemented — read through this and the spec before working on lookup_plant().
    """
    VALID_SEASONS = {"spring", "summer", "fall", "winter"}

    if season and season.lower() in VALID_SEASONS:
        # Caller specified a valid season — use it directly
        season_key = season.lower()
        detected = False
    else:
        # Auto-detect from the current month using the _MONTH_TO_SEASON mapping
        current_month = datetime.now().month
        season_key = _MONTH_TO_SEASON[current_month]
        detected = True

    # Copy the season dict so we don't mutate the cached data
    result = dict(_season_data[season_key])
    result["detected_season"] = detected
    return result
