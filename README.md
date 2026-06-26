# FIFA World Cup 2026 Predictor 🏆

A machine learning model that predicts match outcomes and simulates the full 2026 World Cup tournament bracket.

## How It Works

- Trains a **logistic regression model from scratch** using gradient descent on historical World Cup match data (1930–2022)
- Uses **ELO ratings** as the primary feature to measure team strength
- Simulates the full 2026 tournament — group stage through the final

## Model Performance

- **Test Accuracy: 66.67%** on historical World Cup matches
- Features used: ELO difference, match stage (group stage vs knockout)

## Predicted 2026 Champion

🇪🇸 **Spain**

Predicted path: Round of 32 → Round of 16 → Quarter Finals → Semi Finals → **Final vs France**

## Files

| File | Description |
|------|-------------|
| `worldCupPredictor.py` | Main script — training, simulation, results |
| `wc_matches_historical.csv` | Historical WC match data with ELO ratings |
| `wc_2026_fixtures.csv` | 2026 fixture list |
| `elo_ratings_wc2026.csv` | Current ELO ratings for all 48 teams |


## Tech Stack

Python, NumPy, Pandas — no ML libraries used for the model itself.
