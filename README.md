<div align="center">

# 🏆 FIFA World Cup 2026 Predictor

### Logistic regression from scratch, 20 World Cups of data, one bold pick

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![No ML libraries](https://img.shields.io/badge/ML_libraries_used-0-E34F26?style=flat)
![Prediction](https://img.shields.io/badge/predicted_champion-🇪🇸_Spain-C60B1E?style=flat)
![Result](https://img.shields.io/badge/actual_champion-🇪🇸_Spain_✅-2EA44F?style=flat)

</div>

---

## 🎯 The prediction vs. what happened

Using only **pre-tournament Elo ratings** (from 27 May 2026), the model simulated every match from the group stage to the final and named **Spain** as champion. The prediction was committed here on 26 June 2026, three weeks before the final.

| | Model said | Reality |
|---|---|---|
| 🏆 Champion | 🇪🇸 **Spain** | 🇪🇸 **Spain** ✅ |
| Final opponent | 🇫🇷 France | 🇦🇷 Argentina (Spain won 1–0 after extra time) |
| France's exit | Final | Semi-final, knocked out by... Spain |

The model got the winner right. It also had Spain beating France, just one round later than it actually happened.

## How it works

1. **Data**: 117 key matches from all 20 World Cups (1930–2022), mostly knockout games, each with both teams' pre-match Elo ratings
2. **Features**: Elo difference between the two teams, and match stage (group vs knockout)
3. **Model**: logistic regression trained with **hand-written gradient descent** (sigmoid, loss and weight updates all in NumPy, no scikit-learn)
4. **Simulation**: plays the real 2026 fixture list using the latest Elo ratings (27 May 2026). It ranks all 12 groups, picks the 8 best third-placed teams, then runs the knockout rounds (Round of 32 → Round of 16 → QF → SF → Final)

**Test accuracy: 66.67%** on a held-out 20% of the historical matches (the model predicts *home-side win or not*). That's not bad for a model that mostly asks one question: "who has the higher Elo?"

## Run it

```bash
pip install pandas numpy
python worldCupDataAnalyzing.py
```

The script prints the training loss per epoch, the test accuracy, the group tables and the full knockout bracket, and then announces the champion.

## Files

| File | Description |
|------|-------------|
| `worldCupDataAnalyzing.py` | Training, evaluation and full tournament simulation |
| `wc_matches_historical.csv` | 117 World Cup matches (1930–2022) with pre-match Elo ratings |
| `wc_2026_fixtures.csv` | The 2026 fixture list (48 teams, 104 matches) |
| `elo_ratings_wc2026.csv` | Historical Elo rating snapshots for every national team |

## Limitations (being honest)

- Elo difference carries almost all the signal. Form, injuries and squad changes aren't modelled
- Group games become draws when the model is unsure (win probability between 40% and 60%). Knockout games simply go to whichever team has the higher probability. There are no simulated penalties
- 66.67% accuracy means the model is wrong about a third of the time. Getting the champion right also took some luck 🍀

---

<div align="center">

Made by [Syed Muhammad Ali](https://github.com/MuhammadAli128106)

</div>
