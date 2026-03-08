import sqlite3
import pandas as pd
import numpy as np
import glicko2
import re
from collections import deque

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

def get_rating_data():
    conn = sqlite3.connect('sumo_data.db')
    query = '''
        SELECT year, month, basho_id, day, kakuzuke, east_id, east_rank, east_name, east_result, kimarite, west_result, west_name, west_rank, west_id 
        FROM bouts 
        ORDER BY year ASC, month ASC, day ASC
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, {}, {}, pd.DataFrame(), {}

    rikishi_ratings = {}
    history_records = []
    bouts_history = []
    recent_bouts = {}
    direct_h2h = {}
    latest_name_map = {}
    latest_status = {}
    kakuzuke_map_dict = {1: '幕内', 2: '十両', 3: '幕下', 4: '三段目', 5: '序二段', 6: '序ノ口'}
    
    latest_basho_id = df['basho_id'].max()
    active_rikishi = set() 

    basho_groups = df.groupby(['year', 'month', 'basho_id'], sort=False)

    for (year, month, basho_id), basho_df in basho_groups:
        basho_active_rikishi = set()
        daily_groups = basho_df.groupby('day', sort=False)
        basho_ranks = {}
        basho_rank_nums = {}
        
        for day, daily_df in daily_groups:
            match_data = {}
            daily_active = set()

            for _, row in daily_df.iterrows():
                e_id = row['east_id'] if row['east_id'] != 0 else row['east_name']
                w_id = row['west_id'] if row['west_id'] != 0 else row['west_name']
                
                latest_name_map[e_id] = row['east_name']
                latest_name_map[w_id] = row['west_name']
                
                basho_ranks[e_id] = row['east_rank']
                basho_ranks[w_id] = row['west_rank']

                kakuzuke = row['kakuzuke']
                kimarite = row['kimarite']
                category = KIMARITE_CATEGORY.get(kimarite, 'その他')
                
                is_played = row['east_result'] in ['〇', '●']
                east_win = 1 if row['east_result'] == '〇' else 0

                e_rank_num = parse_rank(row['east_rank'], kakuzuke)
                w_rank_num = parse_rank(row['west_rank'], kakuzuke)
                basho_rank_nums[e_id] = e_rank_num
                basho_rank_nums[w_id] = w_rank_num

                latest_status[row['east_name']] = {'rank_num': e_rank_num, 'kakuzuke': kakuzuke_map_dict.get(kakuzuke, "不明")}
                latest_status[row['west_name']] = {'rank_num': w_rank_num, 'kakuzuke': kakuzuke_map_dict.get(kakuzuke, "不明")}

                if basho_id == latest_basho_id:
                    active_rikishi.add(e_id)
                    active_rikishi.add(w_id)

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

                bouts_history.append({
                    'basho': f"{year}年{month:02d}月場所",
                    'day': f"{day}日目",
                    'day_num': day,
                    'kakuzuke': kakuzuke,
                    'east_id': e_id,
                    'east_rank': row['east_rank'],
                    'east_name': row['east_name'],
                    'east_result': row['east_result'],
                    'kimarite': kimarite,
                    'west_result': row['west_result'],
                    'west_name': row['west_name'],
                    'west_rank': row['west_rank'],
                    'west_id': w_id,
                    'rating_diff': rating_diff,
                    'affinity_diff': aff_diff,
                    'past_win_rate': past_win_rate,
                    'rank_num_diff': rank_num_diff
                })

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
            
            active_ratings = [rikishi_ratings[n].getRating() for n in basho_active_rikishi]
            mean_rating = np.mean(active_ratings) if active_ratings else 1500
            std_rating = np.std(active_ratings) if active_ratings else 1
            if std_rating == 0: std_rating = 1
            
            for r_id in basho_active_rikishi:
                rating = rikishi_ratings[r_id].getRating()
                deviation = ((rating - mean_rating) / std_rating) * 10 + 50
                history_records.append({
                    'basho': f"{year}年{month:02d}月場所",
                    'day': day,
                    'id': r_id,
                    'name': latest_name_map[r_id],
                    'rank': basho_ranks.get(r_id, ""),
                    'rank_num': basho_rank_nums.get(r_id, 400.0),
                    'rating': rating,
                    'deviation': deviation
                })

        for r_id, player in rikishi_ratings.items():
            if r_id not in basho_active_rikishi:
                player.did_not_compete()

    history_df = pd.DataFrame(history_records)
    bouts_history_df = pd.DataFrame(bouts_history)
    
    bouts_history_df['latest_east_name'] = bouts_history_df['east_id'].map(latest_name_map)
    bouts_history_df['latest_west_name'] = bouts_history_df['west_id'].map(latest_name_map)

    ranking = []
    for r_id, player in rikishi_ratings.items():
        if r_id in active_rikishi and player.getRd() < 200: # 幕下以下も含むため制限緩和
            ranking.append({
                'id': r_id,
                'name': latest_name_map[r_id],
                'rating': round(player.getRating(), 1),
                'rd': round(player.getRd(), 1)
            })
        
    ranking_df = pd.DataFrame(ranking).sort_values(by='rating', ascending=False).reset_index(drop=True)
    ranking_df.index += 1
    
    final_profiles = {}
    for r_id, queue in recent_bouts.items():
        prof = get_profile(queue)
        for cat in ['突き押し', '四つ', '引き', 'その他']:
            prof['win_rate_' + cat] = prof['win'][cat] / prof['total_win'] if prof['total_win'] > 0 else 0.0
            prof['loss_rate_' + cat] = prof['loss'][cat] / prof['total_loss'] if prof['total_loss'] > 0 else 0.0
        final_profiles[latest_name_map[r_id]] = prof
        
    final_rikishi_ratings = {latest_name_map[r_id]: player for r_id, player in rikishi_ratings.items()}
    
    final_h2h = {}
    for (id1, id2), scores in direct_h2h.items():
        name1 = latest_name_map.get(int(id1), id1) if id1.isdigit() else latest_name_map.get(id1, id1)
        name2 = latest_name_map.get(int(id2), id2) if id2.isdigit() else latest_name_map.get(id2, id2)
            
        pair_key = tuple(sorted([name1, name2]))
        if pair_key not in final_h2h:
            final_h2h[pair_key] = {name1: 0, name2: 0}
        
        final_h2h[pair_key][name1] += scores[id1]
        final_h2h[pair_key][name2] += scores[id2]

    return ranking_df, history_df, final_rikishi_ratings, final_profiles, final_h2h, bouts_history_df, latest_status