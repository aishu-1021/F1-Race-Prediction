"""
📊 F1 Visualizations — Step 3: See Your Data & Predictions
===========================================================
This script generates charts and visualizations including:
  1. Driver championship standings over the season
  2. Grid position vs Finish position (does pole = win?)
  3. Team performance comparison
  4. Prediction probability chart
  5. Track position map (simplified circuit visualization)

Run: python visualize.py
Input: ../data/f1_race_data.csv
Output: Saves PNG charts to ./charts/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ─── SETUP ────────────────────────────────────────────────────────────────────
CHARTS_DIR = "./charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# F1-inspired color palette
TEAM_COLORS = {
    "Red Bull Racing":  "#3671C6",
    "Mercedes":         "#27F4D2",
    "Ferrari":          "#E8002D",
    "McLaren":          "#FF8000",
    "Aston Martin":     "#229971",
    "Alpine":           "#0093CC",
    "Williams":         "#64C4FF",
    "AlphaTauri":       "#5E8FAA",
    "Alfa Romeo":       "#C92D4B",
    "Haas F1 Team":     "#B6BABD",
}

plt.style.use("dark_background")

# ─── CHART 1: Championship Standings ──────────────────────────────────────────

def plot_championship_standings(df, year=2024):
    print(" Chart 1: Championship standings...")
    season = df[df["Year"] == year].copy()

    # Cumulative points per driver per round
    standings = (
        season.groupby(["Round", "Abbreviation"])["Points"]
        .sum()
        .groupby(level=1)
        .cumsum()
        .reset_index()
    )
    standings.columns = ["Round", "Driver", "CumulativePoints"]

    # Get top 8 drivers only
    final_standings = standings.groupby("Driver")["CumulativePoints"].max()
    top_drivers = final_standings.nlargest(8).index.tolist()
    standings = standings[standings["Driver"].isin(top_drivers)]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_drivers)))

    for i, driver in enumerate(top_drivers):
        d = standings[standings["Driver"] == driver]
        ax.plot(d["Round"], d["CumulativePoints"],
                marker="o", markersize=4, linewidth=2,
                label=driver, color=colors[i])

    ax.set_title(f"🏆 {year} F1 Championship — Points Progression",
                 fontsize=16, fontweight="bold", color="white", pad=20)
    ax.set_xlabel("Race Round", color="white")
    ax.set_ylabel("Cumulative Points", color="white")
    ax.tick_params(colors="white")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.3)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "01_championship_standings.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── CHART 2: Grid Position vs Finish ─────────────────────────────────────────

def plot_grid_vs_finish(df):
    print(" Chart 2: Grid vs Finish position...")

    df = df.copy()
    df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df.dropna(subset=["GridPosition", "Position"])
    df = df[(df["GridPosition"] <= 20) & (df["Position"] <= 20)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    # Left: scatter plot
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    ax.scatter(df["GridPosition"], df["Position"],
               alpha=0.2, s=15, color="#FF8000")
    ax.plot([1, 20], [1, 20], "--", color="white", alpha=0.4, label="No change")
    ax.set_title("Grid vs Finish Position\n(below line = gained places)",
                 color="white", fontsize=12)
    ax.set_xlabel("Grid Position (Qualifying)", color="white")
    ax.set_ylabel("Finish Position (Race)", color="white")
    ax.tick_params(colors="white")
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(alpha=0.15)

    # Right: win rate from each grid position
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")

    win_rate = (
        df[df["GridPosition"] <= 10]
        .groupby("GridPosition")
        .apply(lambda x: (x["Position"] == 1).mean())
        .reset_index()
    )
    win_rate.columns = ["GridPosition", "WinRate"]

    bars = ax2.bar(win_rate["GridPosition"], win_rate["WinRate"] * 100,
                   color=["#E8002D" if g == 1 else "#FF8000" if g <= 3 else "#3671C6"
                          for g in win_rate["GridPosition"]])
    ax2.set_title("Win Rate by Grid Position\n(historical)", color="white", fontsize=12)
    ax2.set_xlabel("Grid Position", color="white")
    ax2.set_ylabel("Win Rate (%)", color="white")
    ax2.tick_params(colors="white")
    ax2.grid(alpha=0.15, axis="y")

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "02_grid_vs_finish.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {path}")


# ─── CHART 3: Team Performance ────────────────────────────────────────────────

def plot_team_performance(df, year=2024):
    print(" Chart 3: Team performance...")
    season = df[df["Year"] == year].copy()

    team_stats = (
        season.groupby("TeamName")
        .agg(
            TotalPoints=("Points", "sum"),
            Wins=("Position", lambda x: (x == 1).sum()),
            Podiums=("Position", lambda x: (x <= 3).sum()),
            AvgFinish=("Position", "mean"),
        )
        .reset_index()
        .sort_values("TotalPoints", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    colors = [TEAM_COLORS.get(t, "#666666") for t in team_stats["TeamName"]]
    bars = ax.barh(team_stats["TeamName"], team_stats["TotalPoints"], color=colors)

    for bar, wins, podiums in zip(bars, team_stats["Wins"], team_stats["Podiums"]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f"{int(bar.get_width())} pts | {wins}W {podiums}P",
                va="center", color="white", fontsize=8)

    ax.set_title(f"🏎️ {year} Constructor Standings", color="white",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Points", color="white")
    ax.tick_params(colors="white")
    ax.set_xlim(0, team_stats["TotalPoints"].max() * 1.25)
    ax.grid(alpha=0.15, axis="x")

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "03_team_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── CHART 4: Prediction Probability ──────────────────────────────────────────

def plot_prediction(predictions, race_name="Next Race"):
    """
    Visualizes win probabilities for each driver.
    predictions: DataFrame with Driver, Team, TopNProbability columns
    """
    print(" Chart 4: Prediction chart...")

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    colors = [TEAM_COLORS.get(t, "#888888") for t in predictions["Team"]]
    bars = ax.barh(predictions["Driver"][::-1],
                   predictions["TopNProbability"][::-1] * 100,
                   color=colors[::-1], edgecolor="white", linewidth=0.5)

    for bar, prob in zip(bars, predictions["TopNProbability"][::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{prob*100:.1f}%", va="center", color="white", fontsize=10,
                fontweight="bold")

    ax.set_title(f"🏁 Predicted Podium Probabilities — {race_name}",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Podium Probability (%)", color="white")
    ax.set_xlim(0, 110)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.15, axis="x")

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "04_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── CHART 5: Simplified Track Layout ─────────────────────────────────────────

def plot_track_layout(cars=None, track="Monaco"):
    """
    Draws a simplified Monaco-style track with car positions.
    In the real version, FastF1 telemetry gives actual X/Y coordinates.
    """
    print(" Chart 5: Track visualization...")

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Draw simplified Monaco track shape (parametric curve)
    t = np.linspace(0, 2 * np.pi, 500)
    # Custom shape to resemble a circuit
    x = 3 * np.cos(t) + 0.5 * np.cos(2*t) + 0.3 * np.cos(3*t)
    y = 2 * np.sin(t) + 0.3 * np.sin(2*t)

    ax.plot(x, y, color="#888888", linewidth=20, solid_capstyle="round",
            solid_joinstyle="round", zorder=1)
    ax.plot(x, y, color="#333333", linewidth=16, solid_capstyle="round",
            solid_joinstyle="round", zorder=2)
    ax.plot(x, y, color="#555555", linewidth=1, linestyle="--",
            alpha=0.5, zorder=3)  # center line

    # Start/Finish line
    ax.axvline(x=3, ymin=0.48, ymax=0.53, color="white", linewidth=3, zorder=5)
    ax.text(3.3, 0, "START/FINISH", color="white", fontsize=8,
            va="center", fontweight="bold")

    # Place cars on track
    if cars is None:
        cars = [
            {"driver": "NOR", "team": "McLaren",  "position": 0.05},
            {"driver": "VER", "team": "Red Bull Racing", "position": 0.12},
            {"driver": "LEC", "team": "Ferrari",  "position": 0.19},
            {"driver": "RUS", "team": "Mercedes", "position": 0.27},
            {"driver": "HAM", "team": "Mercedes", "position": 0.35},
        ]

    for car in cars:
        t_pos = car["position"] * 2 * np.pi
        cx = 3 * np.cos(t_pos) + 0.5 * np.cos(2*t_pos) + 0.3 * np.cos(3*t_pos)
        cy = 2 * np.sin(t_pos) + 0.3 * np.sin(2*t_pos)

        color = TEAM_COLORS.get(car["team"], "#FFFFFF")
        ax.scatter(cx, cy, s=200, color=color, zorder=10, edgecolors="white",
                   linewidths=1.5)
        ax.annotate(car["driver"], (cx, cy), textcoords="offset points",
                    xytext=(10, 5), fontsize=8, color="white", fontweight="bold",
                    zorder=11)

    ax.set_title(f"🏎️ Car Positions — {track} Circuit",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend
    legend_handles = [
        mpatches.Patch(color=TEAM_COLORS.get(c["team"], "#FFF"),
                       label=f"{c['driver']} ({c['team'][:10]})")
        for c in cars
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8, framealpha=0.3, labelcolor="white")

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "05_track_visualization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(" Generating F1 visualizations...\n")

    # Try to load real data — check multiple possible locations
    _paths = ["f1_race_data.csv", "../f1_race_data.csv",
              "data/f1_race_data.csv", "../data/f1_race_data.csv"]
    data_path = next((p for p in _paths if os.path.exists(p)), None)
    if data_path:
        df = pd.read_csv(data_path)
        latest_year = df["Year"].max()
        print(f"  Data loaded ({len(df)} records, latest year: {latest_year})\n")

        plot_championship_standings(df, year=latest_year)
        plot_grid_vs_finish(df)
        plot_team_performance(df, year=latest_year)
    else:
        print("   No data file found. Run fetch_data.py first.")
        print("   Generating sample charts with mock data...\n")

    # These work without real data
    sample_predictions = pd.DataFrame([
        {"Driver": "NOR", "Team": "McLaren",         "TopNProbability": 0.72},
        {"Driver": "VER", "Team": "Red Bull Racing", "TopNProbability": 0.68},
        {"Driver": "LEC", "Team": "Ferrari",         "TopNProbability": 0.55},
        {"Driver": "RUS", "Team": "Mercedes",        "TopNProbability": 0.48},
        {"Driver": "HAM", "Team": "Mercedes",        "TopNProbability": 0.40},
        {"Driver": "SAI", "Team": "Ferrari",         "TopNProbability": 0.35},
        {"Driver": "PIA", "Team": "McLaren",         "TopNProbability": 0.30},
        {"Driver": "ALO", "Team": "Aston Martin",    "TopNProbability": 0.20},
    ])
    plot_prediction(sample_predictions, race_name="2026 Chinese GP")
    plot_track_layout(track="Monaco")

    print(f"\n🎉 All charts saved to: {CHARTS_DIR}/")
    print("   Open the PNG files to see your visualizations!")