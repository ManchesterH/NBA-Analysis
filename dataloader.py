import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.player import Player

class DataLoader:
    def __init__(self, csv_path: str, historical_csv_path: str = None):
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        if 'Hollinger' in first_line or 'Qualified' in first_line:
            self.df = pd.read_csv(csv_path, skiprows=1)
            self._is_mcm_format = True
        else:
            self.df = pd.read_csv(csv_path)
            self._is_mcm_format = False
        self.historical_ewa = {}
        if historical_csv_path:
            self._load_historical_ewa(historical_csv_path)
        else:
            self._build_ewa_lookup()
        self._prepare_data()
    
    def _load_historical_ewa(self, historical_csv_path: str):
        try:
            with open(historical_csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            if 'Hollinger' in first_line or 'Qualified' in first_line:
                hist_df = pd.read_csv(historical_csv_path, skiprows=1)
            else:
                hist_df = pd.read_csv(historical_csv_path)
            hist_df.columns = [c.strip() for c in hist_df.columns]
            player_col = 'PLAYER' if 'PLAYER' in hist_df.columns else 'Player'
            hist_df['PlayerName'] = hist_df[player_col].apply(
                lambda x: str(x).rsplit(',', 1)[0].strip() if pd.notna(x) else "Unknown"
            )
            ewa_col = 'EWA' if 'EWA' in hist_df.columns else 'Estimated Wins Added'
            if ewa_col in hist_df.columns:
                hist_df[ewa_col] = pd.to_numeric(hist_df[ewa_col], errors='coerce')
                for name in hist_df['PlayerName'].unique():
                    player_data = hist_df[hist_df['PlayerName'] == name][ewa_col].dropna()
                    if len(player_data) > 0:
                        best_ewa = player_data.max()
                        avg_ewa = player_data[player_data > 0].mean() if (player_data > 0).any() else 0
                        self.historical_ewa[name] = {
                            'best': best_ewa,
                            'avg': avg_ewa if pd.notna(avg_ewa) else 0
                        }
            print(f"Loaded historical EWA for {len(self.historical_ewa)} players")
        except Exception as e:
            print(f"Warning: Could not load historical EWA: {e}")
    
    def _build_ewa_lookup(self):
        ewa_col = 'EWA' if 'EWA' in self.df.columns else 'Estimated Wins Added'
        player_col = 'PLAYER' if 'PLAYER' in self.df.columns else 'Player'
        if ewa_col not in self.df.columns:
            return
        temp_df = self.df.copy()
        temp_df[ewa_col] = pd.to_numeric(temp_df[ewa_col], errors='coerce')
        temp_df['PlayerName'] = temp_df[player_col].apply(
            lambda x: str(x).rsplit(',', 1)[0].strip() if pd.notna(x) else "Unknown"
        )
        for name in temp_df['PlayerName'].unique():
            player_data = temp_df[temp_df['PlayerName'] == name][ewa_col].dropna()
            if len(player_data) > 0:
                best_ewa = player_data.max()
                avg_ewa = player_data[player_data > 0].mean() if (player_data > 0).any() else 0
                self.historical_ewa[name] = {
                    'best': best_ewa,
                    'avg': avg_ewa if pd.notna(avg_ewa) else 0
                }

    def _prepare_data(self):
        self.df.columns = [c.strip() for c in self.df.columns]
        if 'RK' in self.df.columns:
            self.df = self.df[~self.df['RK'].astype(str).str.startswith('RK')]
            self.df['RK'] = pd.to_numeric(self.df['RK'], errors='coerce')
        player_col = 'PLAYER' if 'PLAYER' in self.df.columns else 'Player'
        def split_player_team(s):
            if pd.isna(s):
                return "Unknown", "UNK"
            parts = str(s).rsplit(',', 1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
            return str(s).strip(), "UNK"
        self.df[['PlayerName', 'TeamAbbr']] = self.df[player_col].apply(lambda x: pd.Series(split_player_team(x)))
        for col in ['PER', 'EWA', 'VA', 'GP', 'MPG']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        rank_col = 'RK' if 'RK' in self.df.columns else 'Rank'
        if rank_col in self.df.columns:
            self.df['Rank'] = pd.to_numeric(self.df[rank_col], errors='coerce')
        else:
            self.df['Rank'] = range(1, len(self.df) + 1)
        top_30_df = self.df[self.df['Rank'] <= 30]
        self.years_top_30_map = top_30_df.groupby('PlayerName').size().to_dict()
        if '赛季年份' in self.df.columns:
            self.career_start = self.df.groupby('PlayerName')['赛季年份'].min().to_dict()
        else:
            self.career_start = self.df.groupby('PlayerName').size().to_dict()

    def get_roster_for_team(self, target_team: str, year: int = None) -> List[Player]:
        if '赛季年份' in self.df.columns and year is not None:
            season_df = self.df[self.df['赛季年份'] == year].copy()
        else:
            season_df = self.df.copy()
        team_rows = []
        for _, row in season_df.iterrows():
            raw_team = row['TeamAbbr']
            if pd.isna(raw_team):
                continue
            current_team = str(raw_team).split('/')[-1]
            if current_team == target_team:
                team_rows.append(row)
        if not team_rows:
            return []
        team_df = pd.DataFrame(team_rows)
        per_col = 'PER' if 'PER' in team_df.columns else 'Player Efficiency Rating'
        team_df[per_col] = pd.to_numeric(team_df[per_col], errors='coerce')
        idx = team_df.groupby('PlayerName')[per_col].idxmax()
        team_df = team_df.loc[idx.dropna()]
        if self._is_mcm_format:
            team_df = team_df.nlargest(15, per_col)
        players = []
        for _, row in team_df.iterrows():
            name = row['PlayerName']
            per = row.get('PER', row.get('Player Efficiency Rating', 15.0))
            if pd.isna(per): per = 15.0
            per = float(per)
            ewa = row.get('EWA', row.get('Estimated Wins Added', 0.0))
            if pd.isna(ewa): ewa = 0.0
            ewa = float(ewa)
            if ewa == 0.0 and name in self.historical_ewa:
                hist = self.historical_ewa[name]
                ewa = hist.get('avg', 0.0)
                if ewa == 0.0:
                    ewa = hist.get('best', 0.0)
            rank = int(row.get('Rank', 999)) if not pd.isna(row.get('Rank', 999)) else 999
            seniority = self.career_start.get(name, 1)
            age = 22 + min(seniority, 15)
            years_top_30 = self.years_top_30_map.get(name, 0)
            if ewa == 0.0 and per > 15.0:
                 ewa_est = (per - 15.0) * 1.5 
            else:
                 ewa_est = ewa
            salary_est = 2.0 + (ewa_est / 20.0) * 50.0
            if per > 25.0: salary_est = max(salary_est, 45.0)
            elif per > 20.0: salary_est = max(salary_est, 25.0)
            elif per > 15.0: salary_est = max(salary_est, 5.0)
            salary_est = max(1.5, min(65.0, salary_est)) 
            p = Player(
                name=name,
                age=age,
                per=per,
                ewa=ewa,
                rank_per=rank,
                years_top_30=years_top_30,
                salary=round(salary_est, 2)
            )
            players.append(p)
        players.sort(key=lambda x: x.per, reverse=True)
        return players

    def get_calibration_stats(self, year: int = None):
        if '赛季年份' in self.df.columns and year is not None:
            season_df = self.df[self.df['赛季年份'] == year]
        else:
            season_df = self.df
        if season_df.empty:
            return None
        per_col = 'PER' if 'PER' in season_df.columns else 'Player Efficiency Rating'
        pers = season_df[per_col].dropna()
        ewa_col = 'EWA' if 'EWA' in season_df.columns else 'Estimated Wins Added'
        ewas = season_df[ewa_col].dropna() if ewa_col in season_df.columns else pd.Series([0])
        return {
            "avg_per": pers.mean(),
            "std_per": pers.std(),
            "top_10_avg_per": pers.nlargest(10).mean(),
            "top_50_avg_per": pers.nlargest(50).mean(),
            "max_per": pers.max(),
            "avg_ewa": ewas.mean(),
            "max_ewa": ewas.max()
        }
