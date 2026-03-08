import sqlite3
import pandas as pd
import numpy as np
import glicko2
import re
from collections import deque
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

KIMARITE_CATEGORY = {
    '押し出し': '突き押し', '突き出し': '突き押し', '押し倒し': '突き押し', '突き倒し': '突き押し',
    '寄り切り': '四つ', '寄り倒し': '四つ', '上手投げ': '四つ', '下手投げ': '四つ', '上手出し投げ': '四つ', '下手出し投げ': '四つ', '掬い投げ': '四つ', '外掛け': '四つ', '内掛け': '四つ',
    '叩き込み': '引き', '引き落とし': '引き', '突き落とし': '引き', '肩透かし': '引き', '引き技': '引き'
}

def parse_rank(rank_str, kakuzuke):
    if not isinstance(rank_str, str): return 400.0
    ew_offset = 0.0
    if '西' in rank_str: ew_offset = 0.5
    if '横綱' in rank_str: return 1.0 + ew_offset
    if '大関' in rank_str: return 2.0 + ew_offset
    if '関脇' in rank_str: return 3.0 + ew_offset
    if '小結' in rank_str: return 4.0 + ew_offset
    
    num = 10.0
    if '筆頭' in rank_str:
        num = 1.0
    else:
        m_arab = re.search(r'[0-9０-９]+', rank_str)
        if m_arab:
            num = float(m_arab.group().translate(str.maketrans('０１２３４５６７８９', '0123456789')))
        else:
            m_kanji = re.search(r'([一二三四五六七八九十百]+)枚目', rank_str)
            if m_kanji:
                kanji_nums = {'一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9}
                kanji_str = m_kanji.group(1)
                n = 0
                if '百' in kanji_str:
                    parts = kanji_str.split('百')
                    n += kanji_nums.get(parts[0], 1) * 100
                    kanji_str = parts[1]
                if '十' in kanji_str:
                    parts = kanji_str.split('十')
                    n += kanji_nums.get(parts[0], 1) * 10
                    kanji_str = parts[1]
                n += kanji_nums.get(kanji_str, 0)
                if n > 0: num = float(n)
                    
    if '前頭' in rank_str or '幕内' in rank_str: return 4.0 + num + ew_offset
    if '十両' in rank_str: return 25.0 + num + ew_offset
    if '幕下' in rank_str: return 40.0 + num + ew_offset
    if '三段目' in rank_str: return 100.0 + num + ew_offset
    if '序二段' in rank_str: return 200.0 + num + ew_offset
    if '序ノ口' in rank_str: return 300.0 + num + ew_offset
    
    if kakuzuke == 1: return 4.0 + num + ew_offset
    if kakuzuke == 2: return 25.0 + num + ew_offset
    if kakuzuke == 3: return 40.0 + num + ew_offset
    if kakuzuke == 4: return 100.0 + num + ew_offset
    if kakuzuke == 5: return 200.0 + num + ew_offset
    if kakuzuke == 6: return 300.0 + num + ew_offset
    return 400.0 + ew_offset

def get_initial_rating(kakuzuke):
    mapping = {1: 2000, 2: 1800, 3: 1600, 4: 1400, 5: 1200, 6: 1000}
    return mapping.get(kakuzuke, 1500)

def get_profile(history_queue):
    prof = {'win': {'突き押し':0, '四つ':0, '引き':0, 'その他':0},
            'loss': {'突き押し':0, '四つ':0, '引き':0, 'その他':0},
            'total_win':0, 'total_loss':0}
    for cat, outcome in history_queue:
        prof[outcome][cat] += 1
        prof['total_' + outcome] += 1
    return prof

def build_and_train_model():
    conn = sqlite3.connect('sumo_data.db')
    query = '''
        SELECT year, month, basho_id, day, kakuzuke, east_id, east_rank, east_result, kimarite, west_result, west_id, west_rank, east_name, west_name 
        FROM bouts 
        ORDER BY year ASC, month ASC, day ASC
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()

    rikishi_ratings = {}
    recent_bouts = {}
    direct_h2h = {}
    X = []
    y = []

    basho_groups = df.groupby(['year', 'month', 'basho_id'], sort=False)

    for (year, month, basho_id), basho_df in basho_groups:
        basho_active_rikishi = set()
        daily_groups = basho_df.groupby('day', sort=False)
        
        for day, daily_df in daily_groups:
            match_data = {}
            daily_active = set()

            for _, row in daily_df.iterrows():
                e_id = row['east_id'] if row['east_id'] != 0 else row['east_name']
                w_id = row['west_id'] if row['west_id'] != 0 else row['west_name']
                kakuzuke = row['kakuzuke']
                kimarite = row['kimarite']
                category = KIMARITE_CATEGORY.get(kimarite, 'その他')
                
                is_played = row['east_result'] in ['〇', '●']
                east_win = 1 if row['east_result'] == '〇' else 0

                basho_active_rikishi.add(e_id)
                basho_active_rikishi.add(w_id)

                if e_id not in rikishi_ratings: rikishi_ratings[e_id] = glicko2.Player(rating=get_initial_rating(kakuzuke))
                if w_id not in rikishi_ratings: rikishi_ratings[w_id] = glicko2.Player(rating=get_initial_rating(kakuzuke))
                if e_id not in recent_bouts: recent_bouts[e_id] = deque(maxlen=90)
                if w_id not in recent_bouts: recent_bouts[w_id] = deque(maxlen=90)

                pair_key = tuple(sorted([str(e_id), str(w_id)]))
                if pair_key not in direct_h2h: direct_h2h[pair_key] = {str(e_id): 0, str(w_id): 0}

                e_rating = rikishi_ratings[e_id].getRating()
                w_rating = rikishi_ratings[w_id].getRating()
                rating_diff = e_rating - w_rating

                e_rank_num = parse_rank(row['east_rank'], kakuzuke)
                w_rank_num = parse_rank(row['west_rank'], kakuzuke)
                rank_num_diff = w_rank_num - e_rank_num

                prof_A = get_profile(recent_bouts[e_id])
                prof_B = get_profile(recent_bouts[w_id])

                aff_A, aff_B = 0, 0
                for cat in ['突き押し', '四つ', '引き', 'その他']:
                    e_tot_win = prof_A['total_win']
                    e_win_rate = prof_A['win'][cat] / e_tot_win if e_tot_win > 0 else 0
                    w_tot_loss = prof_B['total_loss']
                    w_loss_rate = prof_B['loss'][cat] / w_tot_loss if w_tot_loss > 0 else 0
                    w_tot_win = prof_B['total_win']
                    w_win_rate = prof_B['win'][cat] / w_tot_win if w_tot_win > 0 else 0
                    e_tot_loss = prof_A['total_loss']
                    e_loss_rate = prof_A['loss'][cat] / e_tot_loss if e_tot_loss > 0 else 0
                    aff_A += e_win_rate * w_loss_rate
                    aff_B += w_win_rate * e_loss_rate
                aff_diff = aff_A - aff_B

                h2h_e = direct_h2h[pair_key][str(e_id)]
                h2h_w = direct_h2h[pair_key][str(w_id)]
                total_h2h = h2h_e + h2h_w
                past_win_rate = h2h_e / total_h2h if total_h2h > 0 else 0.5

                if is_played:
                    X.append([rating_diff, aff_diff, past_win_rate, rank_num_diff])
                    y.append(east_win)

                if is_played:
                    daily_active.add(e_id)
                    daily_active.add(w_id)
                    if e_id not in match_data: match_data[e_id] = []
                    if w_id not in match_data: match_data[w_id] = []
                    match_data[e_id].append((w_id, east_win))
                    match_data[w_id].append((e_id, 1 - east_win))

                    if east_win == 1:
                        recent_bouts[e_id].append((category, 'win'))
                        recent_bouts[w_id].append((category, 'loss'))
                        direct_h2h[pair_key][str(e_id)] += 1
                    else:
                        recent_bouts[w_id].append((category, 'win'))
                        recent_bouts[e_id].append((category, 'loss'))
                        direct_h2h[pair_key][str(w_id)] += 1

            update_params = {}
            for r_id in daily_active:
                rating_list = []
                rd_list = []
                outcome_list = []
                for opponent_id, outcome in match_data[r_id]:
                    opp = rikishi_ratings[opponent_id]
                    rating_list.append(opp.getRating())
                    rd_list.append(opp.getRd())
                    outcome_list.append(outcome)
                update_params[r_id] = (rating_list, rd_list, outcome_list)

            for r_id in daily_active:
                player = rikishi_ratings[r_id]
                player.update_player(update_params[r_id][0], update_params[r_id][1], update_params[r_id][2])

        for r_id, player in rikishi_ratings.items():
            if r_id not in basho_active_rikishi:
                player.did_not_compete()

    X = np.array(X)
    y = np.array(y)
    
    # 総合モデル（ロジスティック回帰）
    model_lr = LogisticRegression()
    model_lr.fit(X, y)
    joblib.dump(model_lr, 'sumo_model_lr.pkl')

    # 正答率特化モデル（ランダムフォレスト）
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_rf.fit(X, y)
    joblib.dump(model_rf, 'sumo_model_rf.pkl')

    print("モデルの学習と保存が完了しました。")

if __name__ == "__main__":
    build_and_train_model()