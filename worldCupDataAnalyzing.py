import pandas as pd
import numpy as np

#loading the dataa

df = pd.read_csv("wc_matches_historical.csv")

cols_to_keep = [
    "wc_year", "stage", "date", "home_team", "away_team",
    "home_goals", "away_goals", "home_pre_match_elo",
    "away_pre_match_elo", "result_type"
]
df = df[cols_to_keep]

stage_mapping = {
    'Group Stage': 0, 'Final Round': 0,
    'Round of 16': 1, 'Quarter-final': 2,
    'Semi-final': 3, 'Third-place': 3, 'Final': 4
}
df['stage_encoded']  = df['stage'].map(stage_mapping)
df['result']         = df['result_type'].map({"Home Win": 1, "Draw": 0, "Away Win": -1})
df['elo_difference'] = df['home_pre_match_elo'] - df['away_pre_match_elo']



df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
split       = int(0.8 * len(df_shuffled))
train       = df_shuffled[:split]
test        = df_shuffled[split:]

X_train = train[['elo_difference', 'stage_encoded']].to_numpy()
y_train = train['result'].to_numpy()
X_test  = test[['elo_difference', 'stage_encoded']].to_numpy()
y_test  = test['result'].to_numpy()

#normalizing elo and stages to fix scales bcs their vals were too diff

X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0)

X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled  = (X_test  - X_mean) / X_std

#converting obtained vals to binary
y_train_binary = (y_train == 1).astype(float)
y_test_binary  = (y_test  == 1).astype(float)


#training the model
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

weights = np.zeros(2)
bias    = 0
alpha   = 0.01
epochs  = 1000

print("=" * 45)
print("TRAINING")
print("=" * 45)

for epoch in range(epochs):
    logits     = np.dot(X_train_scaled, weights) + bias
    preds      = sigmoid(logits)
    error      = preds - y_train_binary

    dW = np.dot(X_train_scaled.T, error) / len(X_train_scaled)
    db = np.mean(error)

    weights -= alpha * dW
    bias    -= alpha * db

    if epoch % 100 == 0:
        loss = -np.mean(
            y_train_binary * np.log(preds + 1e-8) +
            (1 - y_train_binary) * np.log(1 - preds + 1e-8)
        )
        print(f"  Epoch {epoch:4d}  |  Loss: {loss:.4f}")

print(f"\n  Final weights : {weights}")
print(f"  Final bias    : {bias:.4f}")

#testing phase

test_preds = (sigmoid(np.dot(X_test_scaled, weights) + bias) >= 0.5).astype(float)
accuracy   = np.mean(test_preds == y_test_binary)

print("\n" + "=" * 45)
print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
print("=" * 45)

#now loading the current wc elo and fixtures

elo_df     = pd.read_csv("elo_ratings_wc2026.csv")
latest_elo = elo_df[elo_df['snapshot_date'] == '2026-05-27'][['country', 'rating']]

fixtures   = pd.read_csv("wc_2026_fixtures.csv")
group_stage = fixtures[fixtures['stage'] == 'group-stage'].copy()

#name mismatches were present which led to incomplete data, fixed them here
name_mapping = {
    'Korea Republic' : 'South Korea',
    'Curacao'        : 'Curaçao',
    'Congo DR'       : 'DR Congo',
    'Turkiye'        : 'Turkey',
    'IR Iran'        : 'Iran',
    "Cote d'Ivoire"  : 'Ivory Coast',
    'Cabo Verde'     : 'Cape Verde'
}
group_stage['home_team'] = group_stage['home_team'].replace(name_mapping)
group_stage['away_team'] = group_stage['away_team'].replace(name_mapping)

# elos merged
group_stage = group_stage.merge(
    latest_elo.rename(columns={'country': 'home_team', 'rating': 'home_elo'}),
    on='home_team', how='left'
)
group_stage = group_stage.merge(
    latest_elo.rename(columns={'country': 'away_team', 'rating': 'away_elo'}),
    on='away_team', how='left'
)

#DA SIMULATION

def simulate_group_stage(group_stage):
    points = {}
    groups = {}

    for _, row in group_stage.iterrows():
        home  = row['home_team']
        away  = row['away_team']
        group = row['group']

        if home not in points:
            points[home] = 0
            groups[home] = group
        if away not in points:
            points[away] = 0
            groups[away] = group

        elo_diff = row['home_elo'] - row['away_elo']
        features = np.array([[elo_diff, 0]])
        prob     = sigmoid(np.dot((features - X_mean) / X_std, weights) + bias)[0]

        if prob >= 0.6:
            points[home] += 3
        elif prob <= 0.4:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1

    return pd.DataFrame({
        'team'  : list(points.keys()),
        'points': list(points.values()),
        'group' : list(groups.values())
    })

points_df  = simulate_group_stage(group_stage)

# top 2 from each group advance
qualifiers = (points_df
              .sort_values('points', ascending=False)
              .groupby('group').head(2)
              .sort_values(['group', 'points'], ascending=[True, False]))

winners    = qualifiers.groupby('group').first().reset_index()
runners_up = qualifiers.groupby('group').last().reset_index()

# best 8 third-place teams also advance
third_place       = points_df[~points_df['team'].isin(qualifiers['team'])]
third_place_teams = third_place.sort_values('points', ascending=False).head(8)['team'].tolist()

print("\n" + "=" * 45)
print("GROUP STAGE RESULTS")
print("=" * 45)
print("\nGroup Winners:")
print(winners[['group', 'team', 'points']].to_string(index=False))
print("\nRunners Up:")
print(runners_up[['group', 'team', 'points']].to_string(index=False))
print("\nBest Third-Place Teams:")
print(points_df[points_df['team'].isin(third_place_teams)][['group', 'team', 'points']].to_string(index=False))

#knockout stage

def get_elo(team):
    match = latest_elo[latest_elo['country'] == team]['rating']
    return match.values[0] if len(match) > 0 else 1500

def simulate_knockout_round(teams, stage_enc, round_name):
    print(f"\n── {round_name} ──")
    round_winners = []

    for i in range(0, len(teams), 2):
        team1, team2 = teams[i], teams[i+1]
        elo_diff     = get_elo(team1) - get_elo(team2)
        features     = np.array([[elo_diff, stage_enc]])
        prob         = sigmoid(np.dot((features - X_mean) / X_std, weights) + bias)[0]
        winner       = team1 if prob >= 0.5 else team2

        print(f"  {team1:25s} vs {team2:25s}  →  {winner}")
        round_winners.append(winner)

    return round_winners

#r32 match order
round_of_32 = [
    winners['team'][0],  runners_up['team'][1],
    winners['team'][1],  runners_up['team'][0],
    winners['team'][2],  runners_up['team'][3],
    winners['team'][3],  runners_up['team'][2],
    winners['team'][4],  runners_up['team'][5],
    winners['team'][5],  runners_up['team'][4],
    winners['team'][6],  runners_up['team'][7],
    winners['team'][7],  runners_up['team'][6],
    winners['team'][8],  runners_up['team'][9],
    winners['team'][9],  runners_up['team'][8],
    winners['team'][10], runners_up['team'][11],
    winners['team'][11], runners_up['team'][10],
    third_place_teams[0], third_place_teams[1],
    third_place_teams[2], third_place_teams[3],
    third_place_teams[4], third_place_teams[5],
    third_place_teams[6], third_place_teams[7],
]

print("\n" + "=" * 45)
print("KNOCKOUT STAGE")
print("=" * 45)

r32_winners = simulate_knockout_round(round_of_32,   stage_enc=1, round_name="Round of 32")
r16_winners = simulate_knockout_round(r32_winners,   stage_enc=2, round_name="Round of 16")
qf_winners  = simulate_knockout_round(r16_winners,   stage_enc=3, round_name="Quarter Finals")
sf_winners  = simulate_knockout_round(qf_winners,    stage_enc=4, round_name="Semi Finals")
champion    = simulate_knockout_round(sf_winners,    stage_enc=4, round_name="Final")

print("\n" + "=" * 45)
print(f"🏆  2026 WORLD CUP CHAMPION: {champion[0].upper()}")
print("=" * 45)