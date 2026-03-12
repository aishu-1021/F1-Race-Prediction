"""
🏎️ F1 Data Fetcher — Step 1 of your F1 Predictor Project
==========================================================
This script pulls real F1 race data using the FastF1 library.
It fetches:
  - Race results (finishing positions, points)
  - Qualifying results (grid positions)
  - Lap times
  - Driver & team info

Run: python fetch_data.py
Output: Saves CSV files to the /data folder for use in the model
"""

import fastf1
import pandas as pd
import os
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SEASONS = [2022, 2023, 2024, 2025]  # 4 seasons of training data (2022–2025)
                                    # 2022–2024 already cached, only 2025 fetched fresh
CACHE_DIR = "./cache"              # FastF1 caches data locally (saves time)
OUTPUT_DIR = "./"                  # Where to save the CSVs

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def fetch_season_results(year):
    """
    Fetches all race results for a given season.
    Returns a DataFrame with one row per driver per race.
    """
    print(f"\n Fetching {year} season...")
    season_data = []

    # Get the schedule for the year
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        gp_name = event["EventName"]

        print(f" Round {round_num}: {gp_name}...")

        try:
            time.sleep(3)  # ⏳ Wait 3 seconds between races to avoid rate limits

            # Load the race session
            race = fastf1.get_session(year, round_num, "R")
            race.load(telemetry=False, weather=True, messages=False)

            # Load qualifying session (for grid positions)
            quali = fastf1.get_session(year, round_num, "Q")
            quali.load(telemetry=False, weather=False, messages=False)

            # Get race results
            race_results = race.results[["DriverNumber", "Abbreviation",
                                         "FullName", "TeamName",
                                         "GridPosition", "Position",
                                         "Points", "Status"]].copy()

            # Get qualifying results
            quali_results = quali.results[["DriverNumber", "Q1", "Q2", "Q3"]].copy()

            # Merge race + qualifying
            merged = race_results.merge(quali_results, on="DriverNumber", how="left")

            # Add metadata
            merged["Year"] = year
            merged["Round"] = round_num
            merged["GrandPrix"] = gp_name
            merged["TrackName"] = event["Location"]
            merged["Country"] = event["Country"]

            # Add weather info (average during race)
            if race.weather_data is not None and not race.weather_data.empty:
                merged["AirTemp"] = race.weather_data["AirTemp"].mean()
                merged["Rainfall"] = race.weather_data["Rainfall"].any()
            else:
                merged["AirTemp"] = None
                merged["Rainfall"] = False

            season_data.append(merged)
            print(f" Done - {len(merged)} drivers")

        except Exception as e:
            print(f"  Skipped (error: {e})")
            continue

    if not season_data:
        print(f" No data found for {year}")
        return pd.DataFrame()

    return pd.concat(season_data, ignore_index=True)


def fetch_lap_times(year, round_num):
    """
    Fetches lap-by-lap times for a specific race.
    Useful for visualizing race pace and strategy.
    """
    print(f"\n Fetching lap times: {year} Round {round_num}...")
    session = fastf1.get_session(year, round_num, "R")
    session.load(telemetry=False)

    laps = session.laps[["Driver", "LapNumber", "LapTime",
                           "Compound", "TyreLife", "Stint",
                           "PitInTime", "PitOutTime"]].copy()

    # Convert LapTime to seconds for easier analysis
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

    return laps


def add_driver_form(df):
    """
    Adds a 'rolling form' feature — average points over last 3 races.
    This captures whether a driver is on a hot streak or struggling.
    """
    print("\n Calculating driver form (rolling average)...")
    df = df.sort_values(["Year", "Round"])

    df["DriverForm_Last3"] = (
        df.groupby("Abbreviation")["Points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    return df


def add_team_form(df):
    """
    Adds team-level rolling form — average points of both drivers combined.
    """
    print(" Calculating team form...")
    team_points = df.groupby(["Year", "Round", "TeamName"])["Points"].sum().reset_index()
    team_points.rename(columns={"Points": "TeamPointsThisRace"}, inplace=True)

    df = df.merge(team_points, on=["Year", "Round", "TeamName"], how="left")

    df["TeamForm_Last3"] = (
        df.groupby("TeamName")["TeamPointsThisRace"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    return df


def add_track_history(df):
    """
    Adds each driver's historical average finish position at each track.
    Some drivers consistently perform well at certain circuits.
    """
    print(" Calculating track history...")
    track_history = (
        df.groupby(["Abbreviation", "TrackName"])["Position"]
        .mean()
        .reset_index()
        .rename(columns={"Position": "AvgFinishAtTrack"})
    )

    df = df.merge(track_history, on=["Abbreviation", "TrackName"], how="left")
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_seasons = []

    # Fetch each season
    for year in SEASONS:
        df = fetch_season_results(year)
        if not df.empty:
            all_seasons.append(df)

    if not all_seasons:
        print("\n No data fetched. Check your internet connection.")
        exit()

    # Combine all seasons
    full_df = pd.concat(all_seasons, ignore_index=True)
    print(f"\n Total records fetched: {len(full_df)}")

    # Add engineered features
    full_df = add_driver_form(full_df)
    full_df = add_team_form(full_df)
    full_df = add_track_history(full_df)

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, "f1_race_data.csv")
    full_df.to_csv(output_path, index=False)
    print(f"\n Data saved to: {output_path}")
    print(f" Columns: {list(full_df.columns)}")
    print(f"\n Done! You're ready for Step 2 — training the model.")
    print("   Next: cd ../models && python train_model.py")

    # Quick preview — one race winner from each season
    print("\n Sample data (first race winner per season):")
    winners = full_df[full_df["Position"] == 1].groupby("Year").first().reset_index()
    print(winners[["Year", "Round", "GrandPrix", "Abbreviation",
                    "TeamName", "GridPosition", "Position", "Points"]].to_string(index=False))