import math
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pulp
import streamlit as st

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Showdown Game Simulator",
    layout="wide",
)


# ──────────────────────────────────────────────────────────────
# 0. CORE SIM MODEL: POLICY + GAME ENGINE
# ──────────────────────────────────────────────────────────────

PLAY_TYPES_SIMPLIFIED = [
    "run",
    "pass",
    "punt",
    "field_goal",
    "kickoff",
    "kneel",
    "spike",
    "two_pt",
    "other",
]

StateKey = Tuple[int, str, str, str]  # (down, togo_bucket, yard_bucket, quarter_bucket)


def simplify_play_type(row: pd.Series) -> str:
    pt = str(row.get("PlayType", "")).lower()

    if "kick" in pt and "off" in pt:
        return "kickoff"
    if "punt" in pt:
        return "punt"
    if "field goal" in pt or "field_goal" in pt:
        return "field_goal"
    if "kneel" in pt:
        return "kneel"
    if "spike" in pt:
        return "spike"
    if bool(row.get("IsTwoPointConversion", 0)):
        return "two_pt"

    is_rush = bool(row.get("IsRush", 0))
    is_pass = bool(row.get("IsPass", 0))

    if is_rush and not is_pass:
        return "run"
    if is_pass and not is_rush:
        return "pass"

    if "run" in pt:
        return "run"
    if "pass" in pt:
        return "pass"

    return "other"


def bucket_togo(togo: float) -> str:
    if pd.isna(togo):
        return "unknown"
    togo = float(togo)
    if togo <= 2:
        return "1-2"
    elif togo <= 5:
        return "3-5"
    elif togo <= 9:
        return "6-9"
    elif togo <= 15:
        return "10-15"
    else:
        return "16+"


def bucket_yardline(yardline_fixed: float) -> str:
    if pd.isna(yardline_fixed):
        return "unknown"
    y = float(yardline_fixed)
    if y <= 10:
        return "own_deep"
    elif y <= 25:
        return "own_red_zone"
    elif y <= 50:
        return "own_half"
    elif y <= 80:
        return "opp_half"
    elif y <= 90:
        return "red_zone"
    else:
        return "goal_to_go"


def bucket_quarter(q: int) -> str:
    if pd.isna(q):
        return "unknown"
    q = int(q)
    if q <= 4:
        return str(q)
    return "ot"


@dataclass
class PlayTypeDistribution:
    probs: Dict[str, float]


@dataclass
class OutcomeDistribution:
    yards_samples: np.ndarray
    td_prob: float
    int_prob: float
    fumble_prob: float


@dataclass
class PolicyModel:
    play_type_dists: Dict[StateKey, PlayTypeDistribution]
    outcome_dists_run: Dict[StateKey, OutcomeDistribution]
    outcome_dists_pass: Dict[StateKey, OutcomeDistribution]
    default_play_probs: Dict[str, float]
    rng_seed: int = 42

    def get_play_type_probs(self, state: StateKey) -> Dict[str, float]:
        if state in self.play_type_dists:
            return self.play_type_dists[state].probs

        down, togo_b, yard_b, q = state
        for fallback_state in [
            (down, togo_b, yard_b, "unknown"),
            (down, togo_b, "unknown", "unknown"),
        ]:
            if fallback_state in self.play_type_dists:
                return self.play_type_dists[fallback_state].probs

        return self.default_play_probs

    def get_outcome_dist(self, play_type: str, state: StateKey) -> OutcomeDistribution:
        table = self.outcome_dists_run if play_type == "run" else self.outcome_dists_pass

        if state in table:
            return table[state]

        down, togo_b, yard_b, q = state
        for key in [
            (down, togo_b, yard_b, "unknown"),
            (down, "unknown", "unknown", "unknown"),
        ]:
            if key in table:
                return table[key]

        return OutcomeDistribution(
            yards_samples=np.array([0.0]),
            td_prob=0.01,
            int_prob=0.02 if play_type == "pass" else 0.0,
            fumble_prob=0.01,
        )


def build_policy_from_pbp(df: pd.DataFrame, min_samples_per_bin: int = 30) -> PolicyModel:
    df = df.copy()

    df["play_type_sim"] = df.apply(simplify_play_type, axis=1)
    df = df.dropna(subset=["Down", "ToGo", "YardLineFixed", "Quarter"])

    df["Down"] = df["Down"].astype(int)
    df["togo_bucket"] = df["ToGo"].apply(bucket_togo)
    df["yard_bucket"] = df["YardLineFixed"].apply(bucket_yardline)
    df["quarter_bucket"] = df["Quarter"].apply(bucket_quarter)
    df["state_key"] = list(
        zip(df["Down"], df["togo_bucket"], df["yard_bucket"], df["quarter_bucket"])
    )

    play_type_dists: Dict[StateKey, PlayTypeDistribution] = {}
    for state_key, group in df.groupby("state_key"):
        counts = group["play_type_sim"].value_counts()
        total = counts.sum()
        if total < min_samples_per_bin:
            continue
        probs = {pt: counts.get(pt, 0) / total for pt in PLAY_TYPES_SIMPLIFIED}
        s = sum(probs.values())
        if s == 0:
            continue
        probs = {k: v / s for k, v in probs.items()}
        play_type_dists[state_key] = PlayTypeDistribution(probs=probs)

    global_counts = df["play_type_sim"].value_counts()
    global_total = global_counts.sum()
    default_play_probs = {
        pt: global_counts.get(pt, 0) / global_total for pt in PLAY_TYPES_SIMPLIFIED
    }

    def build_outcomes_for_type(filtered: pd.DataFrame) -> Dict[StateKey, OutcomeDistribution]:
        out: Dict[StateKey, OutcomeDistribution] = {}
        for state_key, group in filtered.groupby("state_key"):
            if len(group) < min_samples_per_bin:
                continue
            yards = group["Yards"].fillna(0).to_numpy(dtype=float)
            td_prob = group["IsTouchdown"].fillna(0).mean()
            int_prob = group["IsInterception"].fillna(0).mean()
            fumble_prob = group["IsFumble"].fillna(0).mean()
            out[state_key] = OutcomeDistribution(
                yards_samples=yards,
                td_prob=float(td_prob),
                int_prob=float(int_prob),
                fumble_prob=float(fumble_prob),
            )
        return out

    outcome_dists_run = build_outcomes_for_type(df[df["play_type_sim"] == "run"])
    outcome_dists_pass = build_outcomes_for_type(df[df["play_type_sim"] == "pass"])

    return PolicyModel(
        play_type_dists=play_type_dists,
        outcome_dists_run=outcome_dists_run,
        outcome_dists_pass=outcome_dists_pass,
        default_play_probs=default_play_probs,
        rng_seed=42,
    )


@dataclass
class GameState:
    offense: str
    defense: str
    quarter: int = 1
    seconds_in_quarter: int = 15 * 60
    ball_on_yard: int = 25
    down: int = 1
    togo: int = 10
    score: Dict[str, int] = field(default_factory=dict)
    num_plays: int = 0
    max_plays: int = 200

    def copy(self) -> "GameState":
        return GameState(
            offense=self.offense,
            defense=self.defense,
            quarter=self.quarter,
            seconds_in_quarter=self.seconds_in_quarter,
            ball_on_yard=self.ball_on_yard,
            down=self.down,
            togo=self.togo,
            score=dict(self.score),
            num_plays=self.num_plays,
            max_plays=self.max_plays,
        )


@dataclass
class PlayResult:
    play_type: str
    yards_gained: float
    td: bool
    turnover: bool
    turnover_type: str
    new_state: GameState


@dataclass
class GameResult:
    final_state: GameState
    play_by_play: List[PlayResult]
    player_stats: pd.DataFrame


def advance_clock(state: GameState, seconds: int) -> None:
    s = max(1, seconds)
    while s > 0:
        if s < state.seconds_in_quarter:
            state.seconds_in_quarter -= s
            s = 0
        else:
            s -= state.seconds_in_quarter
            state.quarter += 1
            if state.quarter > 4:
                state.seconds_in_quarter = 0
                break
            state.seconds_in_quarter = 15 * 60


def is_game_over(state: GameState) -> bool:
    if state.quarter > 4:
        return True
    if state.quarter == 4 and state.seconds_in_quarter <= 0:
        return True
    if state.num_plays >= state.max_plays:
        return True
    return False


def switch_possession(state: GameState, new_ball_on: int = 25) -> None:
    state.offense, state.defense = state.defense, state.offense
    state.ball_on_yard = new_ball_on
    state.down = 1
    state.togo = 10


def state_to_policy_key(state: GameState) -> StateKey:
    togo_bucket = bucket_togo(state.togo)
    yard_bucket = bucket_yardline(state.ball_on_yard)
    quarter_bucket = bucket_quarter(state.quarter)
    return (state.down, togo_bucket, yard_bucket, quarter_bucket)


def init_player_stats(players_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for _, row in players_df.iterrows():
        pid = str(row["player_id"])
        stats[pid] = {
            "player_id": pid,
            "name": row["name"],
            "team": row["team"],
            "position": row["position"],
            "rush_attempts": 0,
            "rush_yards": 0.0,
            "rush_tds": 0,
            "targets": 0,
            "receptions": 0,
            "rec_yards": 0.0,
            "rec_tds": 0,
            "pass_attempts": 0,
            "pass_yards": 0.0,
            "pass_tds": 0,
            "interceptions": 0,
        }
    return stats


def build_team_usage(players_df: pd.DataFrame) -> Dict[str, Dict[str, any]]:
    team_usage: Dict[str, Dict[str, any]] = {}

    for team, group in players_df.groupby("team"):
        g = group.copy()

        qbs = g[g["position"] == "QB"]
        qb_id = str(qbs.iloc[0]["player_id"]) if len(qbs) > 0 else None

        rush_df = g[g["rush_share"] > 0].copy()
        if len(rush_df) > 0:
            rush_probs = rush_df["rush_share"].values
            rush_probs = rush_probs / rush_probs.sum()
            rush_ids = rush_df["player_id"].astype(str).values
        else:
            rush_ids = np.array([], dtype=str)
            rush_probs = np.array([])

        tgt_df = g[g["target_share"] > 0].copy()
        if len(tgt_df) > 0:
            target_probs = tgt_df["target_share"].values
            target_probs = target_probs / target_probs.sum()
            target_ids = tgt_df["player_id"].astype(str).values
        else:
            target_ids = np.array([], dtype=str)
            target_probs = np.array([])

        team_usage[team] = {
            "qb_id": qb_id,
            "rush_ids": rush_ids,
            "rush_probs": rush_probs,
            "target_ids": target_ids,
            "target_probs": target_probs,
        }

    return team_usage


def simulate_single_play(
    policy: PolicyModel,
    state: GameState,
    rng: np.random.Generator,
    team_usage: Dict[str, Dict[str, any]],
    player_stats: Dict[str, Dict[str, float]],
) -> PlayResult:
    key = state_to_policy_key(state)
    play_probs = policy.get_play_type_probs(key)
    types = list(play_probs.keys())
    probs = np.array([play_probs[t] for t in types])
    probs = probs / probs.sum()
    play_type = rng.choice(types, p=probs)

    yards = 0.0
    td = False
    turnover = False
    turnover_type = "none"

    seconds_runoff = int(rng.integers(15, 40))
    advance_clock(state, seconds_runoff)

    if is_game_over(state):
        return PlayResult(play_type, yards, td, turnover, turnover_type, state.copy())

    offense_team = state.offense
    usage = team_usage.get(offense_team, {})
    qb_id = usage.get("qb_id")

    if play_type in ["punt", "field_goal", "kickoff", "kneel", "spike", "two_pt", "other"]:
        if play_type == "punt":
            new_yard = max(1, 100 - max(5, state.ball_on_yard + 40))
            switch_possession(state, new_ball_on=new_yard)
            turnover = True
            turnover_type = "downs"
        elif play_type == "field_goal":
            dist_to_posts = 100 - state.ball_on_yard + 17
            make_prob = 0.70 if dist_to_posts <= 57 else 0.25
            if rng.random() < make_prob:
                state.score[state.offense] = state.score.get(state.offense, 0) + 3
                switch_possession(state, new_ball_on=25)
            else:
                switch_possession(state, new_ball_on=max(1, 100 - state.ball_on_yard))
            turnover = True
            turnover_type = "downs"
        elif play_type == "kickoff":
            switch_possession(state, new_ball_on=25)
        elif play_type == "kneel":
            pass
        elif play_type == "spike":
            state.down += 1
            if state.down > 4:
                switch_possession(state, new_ball_on=max(1, 100 - state.ball_on_yard))
                turnover = True
                turnover_type = "downs"
        elif play_type == "two_pt":
            if rng.random() < 0.5:
                state.score[state.offense] = state.score.get(state.offense, 0) + 2
            switch_possession(state, new_ball_on=25)
            turnover = True
            turnover_type = "downs"

        state.num_plays += 1
        return PlayResult(play_type, yards, td, turnover, turnover_type, state.copy())

    # Run or pass
    outcome_dist = policy.get_outcome_dist(play_type, key)
    if len(outcome_dist.yards_samples) > 0:
        yards = float(rng.choice(outcome_dist.yards_samples))
    else:
        yards = 0.0

    td = rng.random() < outcome_dist.td_prob
    is_int = rng.random() < outcome_dist.int_prob if play_type == "pass" else False
    is_fumble = rng.random() < outcome_dist.fumble_prob

    state.ball_on_yard = int(np.clip(state.ball_on_yard + yards, 1, 99))

    # Assign stats to players
    pid_rusher = None
    pid_receiver = None

    if play_type == "run":
        rush_ids = usage.get("rush_ids", np.array([], dtype=str))
        rush_probs = usage.get("rush_probs", np.array([]))
        if len(rush_ids) > 0:
            pid_rusher = str(rng.choice(rush_ids, p=rush_probs))
        elif qb_id is not None:
            pid_rusher = qb_id
        if pid_rusher is not None and pid_rusher in player_stats:
            ps = player_stats[pid_rusher]
            ps["rush_attempts"] += 1
            ps["rush_yards"] += yards
            if td:
                ps["rush_tds"] += 1

    if play_type == "pass":
        if qb_id is not None and qb_id in player_stats:
            ps_qb = player_stats[qb_id]
            ps_qb["pass_attempts"] += 1
            ps_qb["pass_yards"] += yards
            if td:
                ps_qb["pass_tds"] += 1
            if is_int:
                ps_qb["interceptions"] += 1

        target_ids = usage.get("target_ids", np.array([], dtype=str))
        target_probs = usage.get("target_probs", np.array([]))
        if len(target_ids) > 0:
            pid_receiver = str(rng.choice(target_ids, p=target_probs))
            if pid_receiver in player_stats:
                ps_r = player_stats[pid_receiver]
                ps_r["targets"] += 1
                ps_r["receptions"] += 1
                ps_r["rec_yards"] += yards
                if td:
                    ps_r["rec_tds"] += 1

    # Update game state
    if td:
        state.score[state.offense] = state.score.get(state.offense, 0) + 7
        switch_possession(state, new_ball_on=25)
        turnover = True
        turnover_type = "none"
    elif is_int or is_fumble:
        turnover = True
        turnover_type = "int" if is_int else "fumble"
        switch_possession(state, new_ball_on=max(1, 100 - state.ball_on_yard))
    else:
        if yards >= state.togo:
            state.down = 1
            state.togo = 10
        else:
            state.togo = max(1, state.togo - int(yards))
            state.down += 1
            if state.down > 4:
                turnover = True
                turnover_type = "downs"
                switch_possession(state, new_ball_on=max(1, 100 - state.ball_on_yard))

    state.num_plays += 1

    return PlayResult(play_type, yards, td, turnover, turnover_type, state.copy())


def simulate_game(
    policy: PolicyModel,
    home_team: str,
    away_team: str,
    players_df: pd.DataFrame,
    seed: int = 123,
    max_plays: int = 200,
) -> GameResult:
    rng = np.random.default_rng(seed)

    if rng.random() < 0.5:
        offense, defense = home_team, away_team
    else:
        offense, defense = away_team, home_team

    state = GameState(
        offense=offense,
        defense=defense,
        quarter=1,
        seconds_in_quarter=15 * 60,
        ball_on_yard=25,
        down=1,
        togo=10,
        score={home_team: 0, away_team: 0},
        num_plays=0,
        max_plays=max_plays,
    )

    player_stats = init_player_stats(players_df)
    team_usage = build_team_usage(players_df)
    play_log: List[PlayResult] = []

    while not is_game_over(state):
        pr = simulate_single_play(policy, state, rng, team_usage, player_stats)
        play_log.append(pr)
        if is_game_over(state):
            break

    stats_df = pd.DataFrame(list(player_stats.values()))
    return GameResult(final_state=state.copy(), play_by_play=play_log, player_stats=stats_df)


# ──────────────────────────────────────────────────────────────
# 1. DK SHOWDOWN SCORING + OPTIMIZER
# ──────────────────────────────────────────────────────────────

def dk_scoring(row: pd.Series) -> float:
    pts = 0.0
    pts += row.get("pass_yards", 0.0) / 25.0
    pts += row.get("pass_tds", 0) * 4
    pts -= row.get("interceptions", 0) * 1

    pts += row.get("rush_yards", 0.0) / 10.0
    pts += row.get("rush_tds", 0) * 6

    pts += row.get("receptions", 0) * 1
    pts += row.get("rec_yards", 0.0) / 10.0
    pts += row.get("rec_tds", 0) * 6

    if row.get("rush_yards", 0.0) >= 100:
        pts += 3
    if row.get("rec_yards", 0.0) >= 100:
        pts += 3
    if row.get("pass_yards", 0.0) >= 300:
        pts += 3

    return float(pts)


def prepare_slate_with_fpts(
    slate_df: pd.DataFrame,
    player_stats: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    df_stats = player_stats.copy()
    df_stats["dk_fpts"] = df_stats.apply(dk_scoring, axis=1)

    merge_on_id = "ID" in slate_df.columns and "dk_id" in df_stats.columns
    if merge_on_id:
        df_stats = df_stats.rename(columns={"dk_id": "ID"})
        merged = slate_df.merge(df_stats, on="ID", how="left", suffixes=("", "_stat"))
    else:
        name_col_slate = "Name" if "Name" in slate_df.columns else slate_df.columns[0]
        name_col_stats = "name" if "name" in df_stats.columns else df_stats.columns[0]
        merged = slate_df.merge(
            df_stats, left_on=name_col_slate, right_on=name_col_stats, how="left", suffixes=("", "_stat")
        )

    if "Roster Position" in merged.columns:
        merged["is_cpt"] = merged["Roster Position"].astype(str).str.contains("CPT")
    else:
        merged["is_cpt"] = False

    merged["sim_fpts"] = merged["dk_fpts"].fillna(0.0)
    merged["effective_fpts"] = merged["sim_fpts"] * np.where(merged["is_cpt"], 1.5, 1.0)
    merged["TeamAbbrev"] = merged.get("TeamAbbrev", merged.get("team", ""))

    return merged


def solve_showdown_optimal(
    slate_with_fpts: pd.DataFrame,
    home_team: str,
    away_team: str,
    salary_cap: int = 50000,
    n_slots: int = 6,
) -> Optional[pd.DataFrame]:
    df = slate_with_fpts.copy()
    if "Salary" not in df.columns:
        st.error("Slate CSV must contain a 'Salary' column.")
        return None

    df["Salary"] = df["Salary"].astype(float)
    df["TeamAbbrev"] = df["TeamAbbrev"].astype(str)

    player_ids_raw = df.get("ID", df.get("player_id", df.index)).astype(str)
    df["base_id"] = player_ids_raw
    if "Roster Position" in df.columns:
        dup_key = df["Name"].astype(str) + "_" + df["TeamAbbrev"].astype(str)
        df["base_id"] = dup_key

    prob = pulp.LpProblem("DK_Showdown", pulp.LpMaximize)

    x = {
        i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in df.index
    }

    prob += pulp.lpSum(x[i] * df.loc[i, "effective_fpts"] for i in df.index)

    prob += pulp.lpSum(x[i] * df.loc[i, "Salary"] for i in df.index) <= salary_cap
    prob += pulp.lpSum(x[i] for i in df.index) == n_slots

    if "is_cpt" in df.columns:
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "is_cpt"]) == 1

    prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "TeamAbbrev"] == home_team) >= 1
    prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "TeamAbbrev"] == away_team) >= 1

    for base_id, group in df.groupby("base_id"):
        idxs = list(group.index)
        if len(idxs) > 1:
            prob += pulp.lpSum(x[i] for i in idxs) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    chosen_idxs = [i for i in df.index if x[i].value() == 1]
    return df.loc[chosen_idxs].copy()


# ──────────────────────────────────────────────────────────────
# 2. SESSION HELPERS
# ──────────────────────────────────────────────────────────────

if "policy_model" not in st.session_state:
    st.session_state["policy_model"] = None

if "pbp_df_head" not in st.session_state:
    st.session_state["pbp_df_head"] = None

if "players_df" not in st.session_state:
    st.session_state["players_df"] = None

if "slate_df" not in st.session_state:
    st.session_state["slate_df"] = None


# ──────────────────────────────────────────────────────────────
# 3. SIDEBAR NAV
# ──────────────────────────────────────────────────────────────

st.sidebar.title("🏈 NFL Showdown Simulator")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "1. Train Policy Model",
        "2. Upload Players & Slate",
        "3. Single Game Simulation",
        "4. Showdown Monte Carlo",
    ],
)


# ──────────────────────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────────────────────

if page == "Home":
    st.title("NFL Showdown Game Simulator")
    st.markdown(
        """
This app ties together:

1. **Play-by-play policy model** trained from historical PBP  
2. **Game engine** that simulates a full NFL game (possession, downs, scoring)  
3. **Player stat allocation** using usage shares (rush & target shares)  
4. **DK Showdown scoring + optimizer** to find the optimal lineup per sim  
5. **Monte Carlo** to estimate `% of sims each player is in the optimal lineup`  

### Recommended workflow

1. Go to **1. Train Policy Model** and load your PBP data  
2. Go to **2. Upload Players & Slate** and load:
   - A *players usage* CSV (player_id, name, team, position, rush_share, target_share, optional dk_id)
   - A *DraftKings Showdown slate* CSV (Name, Roster Position, Salary, TeamAbbrev, ID, etc.)  
3. Use **3. Single Game Simulation** to sanity-check game outputs + player box scores  
4. Use **4. Showdown Monte Carlo** to run many sims and see optimal lineup frequencies  
        """
    )


# ──────────────────────────────────────────────────────────────
# PAGE 1: TRAIN POLICY MODEL
# ──────────────────────────────────────────────────────────────

elif page == "1. Train Policy Model":
    st.title("1. Train Play-Calling & Outcome Policy Model")

    st.markdown(
        """
Upload a play-by-play CSV (like the `pbp-2024` sample you showed)  
The app will build an empirical model of:

- **Play type probabilities** given down / distance / field position / quarter  
- **Outcome distributions** (yards gained, TD/INT/fumble probabilities) for run & pass  
        """
    )

    pbp_file = st.file_uploader("Upload PBP CSV", type=["csv"])

    min_samples_per_bin = st.slider(
        "Minimum samples per state bin",
        min_value=10,
        max_value=200,
        value=40,
        step=10,
        help="State bins with fewer samples than this will fall back to broader defaults.",
    )

    if pbp_file is not None:
        df = pd.read_csv(pbp_file)
        st.session_state["pbp_df_head"] = df.head(50)
        st.write("Preview of uploaded PBP:")
        st.dataframe(df.head(20), use_container_width=True)

        required_cols = [
            "Down",
            "ToGo",
            "YardLineFixed",
            "Quarter",
            "PlayType",
            "IsRush",
            "IsPass",
            "Yards",
            "IsTouchdown",
            "IsInterception",
            "IsFumble",
            "IsTwoPointConversion",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("Train Policy Model"):
                with st.spinner("Training empirical policy model from PBP..."):
                    policy = build_policy_from_pbp(df, min_samples_per_bin=min_samples_per_bin)
                    st.session_state["policy_model"] = policy

                st.success("Policy model trained and stored in session!")
                st.write(
                    f"Number of state bins with play-type distributions: {len(policy.play_type_dists)}"
                )
                st.write(
                    f"Run outcome bins: {len(policy.outcome_dists_run)}, "
                    f"Pass outcome bins: {len(policy.outcome_dists_pass)}"
                )
    else:
        st.info("Upload a PBP CSV to begin.")


# ──────────────────────────────────────────────────────────────
# PAGE 2: UPLOAD PLAYERS & SLATE
# ──────────────────────────────────────────────────────────────

elif page == "2. Upload Players & Slate":
    st.title("2. Upload Players Usage & DK Showdown Slate")

    st.markdown(
        """
### Players Usage CSV (required)

Expected columns (you can adapt, but this is the default):

- `player_id` – unique id per player  
- `name` – player name  
- `team` – team abbreviation (e.g., KC, BAL)  
- `position` – QB / RB / WR / TE / etc.  
- `rush_share` – share of team rush attempts (0–1)  
- `target_share` – share of team targets (0–1)  
- Optional: `dk_id` – DraftKings ID that matches the Showdown slate `ID` column  

### Showdown Slate CSV (required)

Assumes a **DraftKings Showdown** export, with columns like:

- `Name`  
- `Roster Position` (e.g. CPT, FLEX)  
- `Salary`  
- `TeamAbbrev`  
- `ID` (numeric DK player id)  
        """
    )

    players_file = st.file_uploader("Upload Players Usage CSV", type=["csv"], key="players_csv")
    slate_file = st.file_uploader("Upload DK Showdown Slate CSV", type=["csv"], key="slate_csv")

    if players_file is not None:
        players_df = pd.read_csv(players_file)
        st.session_state["players_df"] = players_df
        st.subheader("Players Usage Preview")
        st.dataframe(players_df.head(20), use_container_width=True)

    if slate_file is not None:
        slate_df = pd.read_csv(slate_file)
        st.session_state["slate_df"] = slate_df
        st.subheader("Showdown Slate Preview")
        st.dataframe(slate_df.head(20), use_container_width=True)

    if st.session_state["players_df"] is not None:
        teams = sorted(st.session_state["players_df"]["team"].unique().tolist())
        st.markdown("### Detected teams from players CSV:")
        st.write(teams)


# ──────────────────────────────────────────────────────────────
# PAGE 3: SINGLE GAME SIMULATION
# ──────────────────────────────────────────────────────────────

elif page == "3. Single Game Simulation":
    st.title("3. Single Game Play-by-Play Simulation")

    policy = st.session_state.get("policy_model", None)
    players_df = st.session_state.get("players_df", None)

    if policy is None:
        st.error("Please train a policy model on page 1 first.")
    elif players_df is None:
        st.error("Please upload a players usage CSV on page 2.")
    else:
        teams = sorted(players_df["team"].unique().tolist())
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Home Team", teams, index=0)
        with col2:
            away_team = st.selectbox("Away Team", teams, index=min(1, len(teams) - 1))

        max_plays = st.slider("Max plays per game", 80, 250, 160, step=10)
        seed = st.number_input("Random seed", value=123, step=1)

        if st.button("Simulate Single Game"):
            with st.spinner("Simulating game..."):
                game_result = simulate_game(
                    policy=policy,
                    home_team=home_team,
                    away_team=away_team,
                    players_df=players_df,
                    seed=int(seed),
                    max_plays=int(max_plays),
                )

            st.subheader("Final Score")
            s = game_result.final_state.score
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric(home_team, s.get(home_team, 0))
            with col_b:
                st.metric(away_team, s.get(away_team, 0))
            with col_c:
                st.metric("Total Plays", game_result.final_state.num_plays)

            st.subheader("Top Player Stats by DK Points (single sim)")

            stats_df = game_result.player_stats.copy()
            stats_df["dk_fpts"] = stats_df.apply(dk_scoring, axis=1)
            stats_df = stats_df.sort_values("dk_fpts", ascending=False)

            st.dataframe(
                stats_df[
                    [
                        "player_id",
                        "name",
                        "team",
                        "position",
                        "rush_attempts",
                        "rush_yards",
                        "rush_tds",
                        "targets",
                        "receptions",
                        "rec_yards",
                        "rec_tds",
                        "pass_attempts",
                        "pass_yards",
                        "pass_tds",
                        "interceptions",
                        "dk_fpts",
                    ]
                ].head(30),
                use_container_width=True,
            )

            with st.expander("Raw Play-by-Play (sim)"):
                pbp_rows = []
                for i, pr in enumerate(game_result.play_by_play, start=1):
                    pbp_rows.append(
                        {
                            "play_index": i,
                            "play_type": pr.play_type,
                            "yards": pr.yards_gained,
                            "td": pr.td,
                            "turnover": pr.turnover,
                            "turnover_type": pr.turnover_type,
                            "offense_after": pr.new_state.offense,
                            "defense_after": pr.new_state.defense,
                            "quarter": pr.new_state.quarter,
                            "seconds_in_quarter": pr.new_state.seconds_in_quarter,
                            "ball_on_yard": pr.new_state.ball_on_yard,
                            "down": pr.new_state.down,
                            "togo": pr.new_state.togo,
                            "score_home": pr.new_state.score.get(home_team, 0),
                            "score_away": pr.new_state.score.get(away_team, 0),
                        }
                    )
                st.dataframe(pd.DataFrame(pbp_rows), use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PAGE 4: SHOWDOWN MONTE CARLO
# ──────────────────────────────────────────────────────────────

elif page == "4. Showdown Monte Carlo":
    st.title("4. Showdown Monte Carlo – Optimal Lineup Frequencies")

    policy = st.session_state.get("policy_model", None)
    players_df = st.session_state.get("players_df", None)
    slate_df = st.session_state.get("slate_df", None)

    if policy is None:
        st.error("Please train a policy model on page 1 first.")
    elif players_df is None:
        st.error("Please upload a players usage CSV on page 2.")
    elif slate_df is None:
        st.error("Please upload a DK showdown slate CSV on page 2.")
    else:
        teams = sorted(players_df["team"].unique().tolist())
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Home Team", teams, index=0, key="mc_home")
        with col2:
            away_team = st.selectbox("Away Team", teams, index=min(1, len(teams) - 1), key="mc_away")

        n_sims = st.number_input("Number of simulations", value=200, min_value=10, max_value=5000, step=50)
        max_plays = st.slider("Max plays per game", 80, 250, 160, step=10, key="mc_max_plays")
        base_seed = st.number_input("Base random seed", value=99, step=1, key="mc_seed")

        if st.button("Run Monte Carlo"):
            with st.spinner("Running simulations and optimizing showdowns..."):
                rng = np.random.default_rng(int(base_seed))

                player_ids = slate_df.get("ID", slate_df.get("Name", slate_df.index)).astype(str).values
                base_ids = slate_df.get("Name", slate_df.index).astype(str).values

                optimal_counts: Dict[str, int] = {str(pid): 0 for pid in player_ids}
                optimal_cpt_counts: Dict[str, int] = {str(pid): 0 for pid in player_ids}

                progress_bar = st.progress(0)
                for i in range(int(n_sims)):
                    seed = int(rng.integers(0, 1_000_000_000))
                    game_result = simulate_game(
                        policy=policy,
                        home_team=home_team,
                        away_team=away_team,
                        players_df=players_df,
                        seed=seed,
                        max_plays=int(max_plays),
                    )

                    slate_with_fpts = prepare_slate_with_fpts(
                        slate_df, game_result.player_stats, home_team, away_team
                    )

                    optimal_lineup = solve_showdown_optimal(
                        slate_with_fpts, home_team=home_team, away_team=away_team
                    )
                    if optimal_lineup is None:
                        continue

                    for _, row in optimal_lineup.iterrows():
                        pid = str(row.get("ID", row.get("Name", "")))
                        if pid not in optimal_counts:
                            optimal_counts[pid] = 0
                            optimal_cpt_counts[pid] = 0
                        optimal_counts[pid] += 1
                        if row.get("is_cpt", False):
                            optimal_cpt_counts[pid] += 1

                    if (i + 1) % max(1, int(n_sims) // 100) == 0:
                        progress_bar.progress((i + 1) / n_sims)

                progress_bar.progress(1.0)

            n = float(n_sims)
            out_rows = []
            for idx, row in slate_df.iterrows():
                pid = str(row.get("ID", row.get("Name", idx)))
                name = row.get("Name", str(pid))
                team = row.get("TeamAbbrev", "")
                roster_pos = row.get("Roster Position", "")
                count = optimal_counts.get(pid, 0)
                cpt_count = optimal_cpt_counts.get(pid, 0)
                out_rows.append(
                    {
                        "ID": pid,
                        "Name": name,
                        "TeamAbbrev": team,
                        "Roster Position": roster_pos,
                        "Optimal Lineup %": count / n,
                        "Optimal CPT %": cpt_count / n,
                        "Optimal Count": count,
                        "Optimal CPT Count": cpt_count,
                    }
                )

            result_df = pd.DataFrame(out_rows).sort_values(
                "Optimal Lineup %", ascending=False
            )

            st.subheader("Optimal Lineup Frequencies")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "Download results as CSV",
                data=result_df.to_csv(index=False),
                file_name="showdown_monte_carlo_results.csv",
                mime="text/csv",
            )
