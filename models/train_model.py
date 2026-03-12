"""
🧠 F1 Race Winner Predictor — Step 2: Train the Model
=======================================================
This script trains a Random Forest classifier to predict race winners.

It uses these features to make predictions:
  - Grid position (qualifying result)
  - Driver recent form (rolling avg points)
  - Team recent form
  - Driver's track history
  - Weather (air temp, rainfall)
  - Team name (encoded)

Run: python train_model.py
Input: ../data/f1_race_data.csv
Output: Saves the trained model as f1_model.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Auto-detect the CSV path whether you run from /models or from project root
_possible_paths = [
    "f1_race_data.csv",           # CSV is in the project root folder
    "data/f1_race_data.csv",      # CSV is inside /data subfolder
    "../data/f1_race_data.csv",   # Running from /models folder
    "../f1_race_data.csv",        # CSV in root, running from /models
]
DATA_PATH = next((p for p in _possible_paths if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "   Could not find f1_race_data.csv!\n"
        "   Make sure you ran: python data/fetch_data.py first.\n"
        "   Then run this script from the project root:\n"
        "   python models/train_model.py"
    )
print(f" Found data at: {DATA_PATH}")
MODEL_PATH = "./f1_model.pkl"
TOP_N = 3   # Predict if driver finishes in top N (podium)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    print(" Loading race data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} records loaded across {df['Year'].nunique()} seasons")
    return df

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def prepare_features(df):
    """
    Converts raw race data into ML-ready features.
    """
    print("\n🔧 Preparing features...")
    df = df.copy()

    # ── Target variable: Did this driver finish in the Top N? ──
    # We predict podium (top 3) rather than exact winner for better accuracy
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["IsTopN"] = (df["Position"] <= TOP_N).astype(int)

    # ── Encode team names as numbers (ML needs numbers, not text) ──
    le_team = LabelEncoder()
    df["TeamEncoded"] = le_team.fit_transform(df["TeamName"].fillna("Unknown"))

    # ── Encode driver abbreviations ──
    le_driver = LabelEncoder()
    df["DriverEncoded"] = le_driver.fit_transform(df["Abbreviation"].fillna("Unknown"))

    # ── Grid position (most important feature!) ──
    df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce").fillna(15)

    # ── Fill missing form values with average ──
    df["DriverForm_Last3"] = df["DriverForm_Last3"].fillna(df["DriverForm_Last3"].median())
    df["TeamForm_Last3"] = df["TeamForm_Last3"].fillna(df["TeamForm_Last3"].median())
    df["AvgFinishAtTrack"] = df["AvgFinishAtTrack"].fillna(10)  # Default: midfield

    # ── Weather features ──
    df["AirTemp"] = df["AirTemp"].fillna(25)
    df["Rainfall"] = df["Rainfall"].astype(int)

    # ── Grid advantage: top 3 on grid have big advantage ──
    df["FrontRow"] = (df["GridPosition"] <= 3).astype(int)

    print(f"  Features ready. Target: Top {TOP_N} finish")
    return df, le_team, le_driver


def get_feature_columns():
    """Returns the list of features used for training."""
    return [
        "GridPosition",       # Starting position (most important!)
        "DriverForm_Last3",   # Driver's recent performance
        "TeamForm_Last3",     # Team's recent performance
        "AvgFinishAtTrack",   # Historical performance at this track
        "AirTemp",            # Weather
        "Rainfall",           # Wet race? (big upset potential)
        "FrontRow",           # Starting in top 3
        "TeamEncoded",        # Which team
        "DriverEncoded",      # Which driver
    ]

# ─── TRAIN MODEL ──────────────────────────────────────────────────────────────

def train_model(df):
    print("\n Training prediction model...")

    features = get_feature_columns()
    X = df[features]
    y = df["IsTopN"]

    # Split into training and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # ── Model 1: Random Forest ──
    print("\n  Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced"   # Handles imbalanced classes (few winners vs many losers)
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"   Random Forest Accuracy: {rf_acc:.1%}")

    # ── Model 2: Gradient Boosting (usually more accurate) ──
    print("\n  Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    print(f"   Gradient Boosting Accuracy: {gb_acc:.1%}")

    # Pick the best model
    best_model = rf_model if rf_acc >= gb_acc else gb_model
    best_name = "Random Forest" if rf_acc >= gb_acc else "Gradient Boosting"
    best_acc = max(rf_acc, gb_acc)

    print(f"\n  Best model: {best_name} ({best_acc:.1%} accuracy)")

    # Feature importance — what matters most?
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\n  Feature Importance (what the model relies on most):")
    for _, row in feature_importance.iterrows():
        bar = "█" * int(row["Importance"] * 50)
        print(f"   {row['Feature']:<25} {bar} {row['Importance']:.3f}")

    return best_model, best_name, best_acc

# ─── PREDICT RACE ─────────────────────────────────────────────────────────────

def predict_race(model, le_team, le_driver, race_entry):
    """
    Predicts win probability for a single race.

    race_entry: list of dicts, one per driver. Example:
    [
        {"driver": "VER", "team": "Red Bull Racing", "grid": 1, "form": 18.0, ...},
        {"driver": "HAM", "team": "Mercedes",         "grid": 2, "form": 15.0, ...},
        ...
    ]
    """
    features = get_feature_columns()
    results = []

    for driver in race_entry:
        # Encode team and driver
        try:
            team_enc = le_team.transform([driver["team"]])[0]
        except:
            team_enc = 0

        try:
            driver_enc = le_driver.transform([driver["driver"]])[0]
        except:
            driver_enc = 0

        row = {
            "GridPosition": driver.get("grid", 10),
            "DriverForm_Last3": driver.get("driver_form", 8.0),
            "TeamForm_Last3": driver.get("team_form", 15.0),
            "AvgFinishAtTrack": driver.get("track_avg", 8.0),
            "AirTemp": driver.get("air_temp", 25),
            "Rainfall": int(driver.get("rainfall", False)),
            "FrontRow": int(driver.get("grid", 10) <= 3),
            "TeamEncoded": team_enc,
            "DriverEncoded": driver_enc,
        }

        X = pd.DataFrame([row])[features]
        prob = model.predict_proba(X)[0][1]   # Probability of top N finish
        results.append({"Driver": driver["driver"], "Team": driver["team"],
                         "Grid": driver.get("grid"), "TopNProbability": prob})

    results_df = pd.DataFrame(results).sort_values("TopNProbability", ascending=False)
    results_df["Rank"] = range(1, len(results_df) + 1)
    return results_df

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load and prepare data
    df = load_data()
    df, le_team, le_driver = prepare_features(df)

    # Train the model
    model, model_name, accuracy = train_model(df)

    # Save model + encoders
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "le_team": le_team,
                     "le_driver": le_driver, "accuracy": accuracy,
                     "model_name": model_name}, f)

    print(f"\n Model saved to: {MODEL_PATH}")

    # ── Example: Predict a hypothetical race ──
    print("\n" + "="*60)
    print("🏁 EXAMPLE PREDICTION — Hypothetical 2026 Race")
    print("="*60)

    sample_race = [
        {"driver": "NOR", "team": "McLaren",  "grid": 1, "driver_form": 20.0, "team_form": 35.0, "track_avg": 4.0},
        {"driver": "VER", "team": "Red Bull Racing", "grid": 2, "driver_form": 22.0, "team_form": 38.0, "track_avg": 3.0},
        {"driver": "LEC", "team": "Ferrari",  "grid": 3, "driver_form": 16.0, "team_form": 28.0, "track_avg": 5.0},
        {"driver": "HAM", "team": "Mercedes", "grid": 4, "driver_form": 14.0, "team_form": 25.0, "track_avg": 6.0},
        {"driver": "RUS", "team": "Mercedes", "grid": 5, "driver_form": 12.0, "team_form": 25.0, "track_avg": 7.0},
        {"driver": "SAI", "team": "Ferrari",  "grid": 6, "driver_form": 13.0, "team_form": 28.0, "track_avg": 6.5},
        {"driver": "PIA", "team": "McLaren",  "grid": 7, "driver_form": 10.0, "team_form": 35.0, "track_avg": 8.0},
        {"driver": "ALO", "team": "Aston Martin", "grid": 8, "driver_form": 9.0, "team_form": 18.0, "track_avg": 7.0},
    ]

    predictions = predict_race(model, le_team, le_driver, sample_race)
    print(f"\n Predicted Top {TOP_N} probabilities:\n")
    print(predictions[["Rank", "Driver", "Team", "Grid", "TopNProbability"]].to_string(index=False))
    print(f"\n Predicted winner: {predictions.iloc[0]['Driver']} ({predictions.iloc[0]['TopNProbability']:.1%} chance of top {TOP_N})")

    print("\n🎉 Done! Next: cd ../visualizations && python visualize.py")