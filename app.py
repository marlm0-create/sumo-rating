import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import time
import requests
import sqlite3
from bs4 import BeautifulSoup
from rating_calc import get_rating_data, parse_rank

# スマホ向けにサイドバーを初期状態で閉じ、横幅を広く使う設定
st.set_page_config(page_title="大相撲 AIレーティング", layout="wide", initial_sidebar_state="collapsed")

# 初回ロード時のプレビュー表示処理
if 'init_done' not in st.session_state:
    st.title("大相撲 AIレーティングシステム")
    st.info("システムを初期化しています。AIモデルと全期間のデータを計算中です。")
    
    try:
        conn = sqlite3.connect('sumo_data.db')
        preview_query = "SELECT year, month, day, east_name, east_result, west_name, west_result FROM bouts ORDER BY year DESC, month DESC, day DESC LIMIT 10"
        preview_df = pd.read_sql_query(preview_query, conn)
        conn.close()
        
        preview_df.columns = ['年', '月', '日目', '東方', '結果', '西方', '結果']
        st.write("プレビュー: 直近の取組データ")
        st.dataframe(preview_df, width="stretch")
    except Exception:
        pass

    with st.spinner("AIエンジンのロード中..."):
        # バックグラウンドで本処理を実行
        ranking_df, history_df, rikishi_ratings, profiles, direct_h2h, bouts_history_df, latest_status = get_rating_data()
        try:
            model_lr = joblib.load('sumo_model_lr.pkl')
            model_rf = joblib.load('sumo_model_rf.pkl')
        except FileNotFoundError:
            model_lr, model_rf = None, None

        # キャッシュ化の代わりにセッションへ保持
        st.session_state.app_data = (ranking_df, history_df, rikishi_ratings, profiles, direct_h2h, bouts_history_df, latest_status)
        st.session_state.models = (model_lr, model_rf)
        st.session_state.init_done = True
        st.rerun()

# データ展開
ranking_df, history_df, rikishi_ratings, profiles, direct_h2h, bouts_history_df, latest_status = st.session_state.app_data
model_lr, model_rf = st.session_state.models

if ranking_df.empty or model_lr is None or model_rf is None:
    st.warning("データまたはモデルが存在しません。スクレイピングとモデル学習を実行してください。")
    st.stop()

# モデル選択UI
st.sidebar.header("予測モデル設定")
selected_model_name = st.sidebar.radio(
    "使用するAIモデルを選択",
    ["総合AIモデル (番付・レーティング基準)", "特化AIモデル (正答率・直近傾向基準)"]
)
active_model = model_lr if "総合" in selected_model_name else model_rf

KIMARITE_CATEGORY = {
    '押し出し': '突き押し', '突き出し': '突き押し', '押し倒し': '突き押し', '突き倒し': '突き押し',
    '寄り切り': '四つ', '寄り倒し': '四つ', '上手投げ': '四つ', '下手投げ': '四つ', '上手出し投げ': '四つ', '下手出し投げ': '四つ', '掬い投げ': '四つ', '外掛け': '四つ', '内掛け': '四つ',
    '叩き込み': '引き', '引き落とし': '引き', '突き落とし': '引き', '肩透かし': '引き', '引き技': '引き'
}

def make_sortable_str(val_for_sort, display_str, reverse=False):
    if pd.isna(val_for_sort):
        v = 0 if not reverse else 999999
    else:
        v = int(val_for_sort * 100) + 100000 
    if reverse:
        v = 200000 - v
    s = f"{v:06d}"
    mapping = {'0': '\u200b', '1': '\u200c', '2': '\u200d', '3': '\u200e',
               '4': '\u200f', '5': '\u202a', '6': '\u202b', '7': '\u202c',
               '8': '\u202d', '9': '\u202e'}
    prefix = "".join(mapping.get(c, '\u200b') for c in s)
    return prefix + str(display_str)

def calculate_affinity_diff(prof_A, prof_B):
    affinity_A = 0
    affinity_B = 0
    categories = ['突き押し', '四つ', '引き', 'その他']
    for cat in categories:
        affinity_A += prof_A['win_rate_' + cat] * prof_B['loss_rate_' + cat]
        affinity_B += prof_B['win_rate_' + cat] * prof_A['loss_rate_' + cat]
    return affinity_A - affinity_B

def highlight_result(val):
    if val == '〇': return 'color: red; font-weight: bold;'
    elif val == '●': return 'color: blue; font-weight: bold;'
    return ''

def highlight_upset(val):
    if val == '番狂わせ': return 'background-color: yellow; color: black; font-weight: bold;'
    return ''

def format_trend_val(diff):
    if pd.isna(diff): return "-"
    diff = int(diff)
    if diff > 0: return f"↑ {abs(diff)}"
    if diff < 0: return f"↓ {abs(diff)}"
    return "→ 0"

def color_trend(val):
    if isinstance(val, str):
        if '↑' in val: return 'color: red; font-weight: bold;'
        if '↓' in val: return 'color: blue; font-weight: bold;'
        if '→' in val: return 'color: gray; font-weight: bold;'
    return ''

def plot_kimarite_chart(counts_series, title, global_ratio_dict):
    if counts_series.empty:
        st.write("データがありません。")
        return
    total = counts_series.sum()
    data = []
    for k, v in counts_series.items():
        cat = KIMARITE_CATEGORY.get(k, 'その他')
        ratio = v / total
        global_ratio = global_ratio_dict.get(k, 0.0)
        data.append({
            '決まり手': k, '回数': v, '割合': ratio, '割合表示': f"{ratio*100:.1f}%", 
            '分類': cat, '全体割合': global_ratio
        })
    df = pd.DataFrame(data).sort_values(by='割合', ascending=False).head(10)
    kimarite_order = df['決まり手'].tolist()
    
    base = alt.Chart(df).encode(x=alt.X('決まり手:N', sort=kimarite_order, title=None, axis=alt.Axis(labelAngle=-45, labelOverlap=False)))
    bars = base.mark_bar().encode(
        y=alt.Y('割合:Q', title='割合', axis=alt.Axis(format='%')),
        color=alt.Color('分類:N', scale=alt.Scale(domain=['突き押し', '四つ', '引き', 'その他'], range=['#d62728', '#1f77b4', '#2ca02c', '#7f7f7f']), legend=alt.Legend(title="分類")),
        tooltip=['決まり手', '分類', '回数', alt.Tooltip('割合:Q', format='.1%'), alt.Tooltip('全体割合:Q', format='.1%')]
    )
    ticks_bg = base.mark_tick(color='white', thickness=3, size=35).encode(y=alt.Y('全体割合:Q'))
    ticks_fg = base.mark_tick(color='black', thickness=3, size=35, strokeDash=[4, 4]).encode(y=alt.Y('全体割合:Q'))
    text = base.mark_text(align='center', baseline='bottom', dy=-5).encode(y=alt.Y('割合:Q'), text='割合表示:N')
    st.altair_chart((bars + ticks_bg + ticks_fg + text).properties(title=title, height=350), width="stretch")

def apply_row_styles(row):
    styles = [''] * len(row)
    prob = row['勝率予想_raw']
    res = row['結果']
    prob_color = 'red' if prob > 0.5 else 'blue'
    res_color = 'red' if res == '〇' else 'blue'
    alpha = min(abs(prob - 0.5) * 1.5, 0.8)
    
    bg_prob = f'background-color: rgba(255, 0, 0, {alpha});' if prob > 0.5 else f'background-color: rgba(0, 0, 255, {alpha});'
    bg_res = f'background-color: rgba(255, 0, 0, {alpha});' if res == '〇' else f'background-color: rgba(0, 0, 255, {alpha});'
    
    idx_prob = row.index.get_loc('勝率予想')
    idx_res = row.index.get_loc('結果')
    
    if row['特記事項'] == '番狂わせ':
        bg_upset = 'background-color: yellow; color: black;'
        for i in range(len(styles)): styles[i] = bg_upset
        styles[idx_prob] += f' color: {prob_color}; font-weight: bold;'
        styles[idx_res] += f' color: {res_color}; font-weight: bold;'
    else:
        styles[idx_prob] = f'color: {prob_color}; font-weight: bold; {bg_prob}'
        styles[idx_res] = f'color: {res_color}; font-weight: bold; {bg_res}'
    return styles

@st.cache_data(ttl=3600)
def fetch_makuuchi_banzuke(bouts_df):
    rikishi_list = []
    try:
        url = "https://www.sumo.or.jp/ResultBanzuke/table/"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, 'html.parser')
            table = soup.find('table', class_='mdTable1')
            if table:
                for row in table.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) == 3:
                        e_name_tag = cells[0].find('a')
                        rank = cells[1].text.strip()
                        w_name_tag = cells[2].find('a')
                        
                        e_room, w_room = "unknown", "unknown"
                        e_text = cells[0].text
                        w_text = cells[2].text
                        if e_name_tag:
                            e_room = e_text.replace(e_name_tag.text, '').strip()
                        if w_name_tag:
                            w_room = w_text.replace(w_name_tag.text, '').strip()

                        if e_name_tag: 
                            rikishi_list.append({'name': e_name_tag.text.strip(), 'rank': f"東{rank}", 'room': e_room})
                        if w_name_tag: 
                            rikishi_list.append({'name': w_name_tag.text.strip(), 'rank': f"西{rank}", 'room': w_room})
                if rikishi_list:
                    return rikishi_list
    except Exception:
        pass
    
    latest_basho = bouts_df['basho'].iloc[-1]
    makuuchi_bouts = bouts_df[(bouts_df['basho'] == latest_basho) & (bouts_df['kakuzuke'] == 1)]
    rikishi_dict = {}
    for _, row in makuuchi_bouts.iterrows():
        if row['latest_east_name']: rikishi_dict[row['latest_east_name']] = row['east_rank']
        if row['latest_west_name']: rikishi_dict[row['latest_west_name']] = row['west_rank']
    return [{'name': k, 'rank': v, 'room': 'unknown'} for k, v in rikishi_dict.items()]

def create_win_prob_matrix(rikishi_list, rikishi_ratings, profiles, direct_h2h, model):
    matrix = {}
    makuuchi_names = [r['name'] for r in rikishi_list]
    rank_nums = {r['name']: parse_rank(r['rank'], 1) for r in rikishi_list}
    rooms = {r['name']: r['room'] for r in rikishi_list}
    
    import glicko2
    for i in range(len(makuuchi_names)):
        for j in range(i+1, len(makuuchi_names)):
            rikishi_A = makuuchi_names[i]
            rikishi_B = makuuchi_names[j]
            
            player_A = rikishi_ratings.get(rikishi_A, glicko2.Player(rating=1500))
            player_B = rikishi_ratings.get(rikishi_B, glicko2.Player(rating=1500))
            prof_A = profiles.get(rikishi_A, {'win_rate_突き押し':0, 'win_rate_四つ':0, 'win_rate_引き':0, 'win_rate_その他':0, 'loss_rate_突き押し':0, 'loss_rate_四つ':0, 'loss_rate_引き':0, 'loss_rate_その他':0})
            prof_B = profiles.get(rikishi_B, {'win_rate_突き押し':0, 'win_rate_四つ':0, 'win_rate_引き':0, 'win_rate_その他':0, 'loss_rate_突き押し':0, 'loss_rate_四つ':0, 'loss_rate_引き':0, 'loss_rate_その他':0})
            
            pair_key = tuple(sorted([rikishi_A, rikishi_B]))
            h2h_A_wins = direct_h2h.get(pair_key, {}).get(rikishi_A, 0)
            h2h_B_wins = direct_h2h.get(pair_key, {}).get(rikishi_B, 0)
            total_matches = h2h_A_wins + h2h_B_wins
            past_win_rate = h2h_A_wins / total_matches if total_matches > 0 else 0.5
            
            rating_diff = player_A.getRating() - player_B.getRating()
            affinity_diff = calculate_affinity_diff(prof_A, prof_B)
            rank_num_diff = rank_nums[rikishi_B] - rank_nums[rikishi_A]
            
            X_input = np.array([[rating_diff, affinity_diff, past_win_rate, rank_num_diff]])
            prob_A = model.predict_proba(X_input)[0][1]
            
            matrix[(rikishi_A, rikishi_B)] = prob_A
            matrix[(rikishi_B, rikishi_A)] = 1.0 - prob_A
            
    return matrix, rank_nums, rooms

def run_simulation(makuuchi_names, win_prob_matrix, rank_nums, rooms, num_simulations):
    results = {name: {'yusho': 0, 'wins': []} for name in makuuchi_names}
    top16_names = sorted(makuuchi_names, key=lambda x: rank_nums[x])[:16]
    
    for sim in range(num_simulations):
        wins = {name: 0 for name in makuuchi_names}
        match_history = {name: set() for name in makuuchi_names}
        
        for day in range(15):
            matched = set()
            
            if day < 5:
                sorted_rikishi = sorted(makuuchi_names, key=lambda x: (rank_nums[x], np.random.rand()))
            else:
                sorted_rikishi = sorted(makuuchi_names, key=lambda x: (-wins[x], rank_nums[x], np.random.rand()))

            if day < 10:
                for i, rikishi_A in enumerate(sorted_rikishi):
                    if rikishi_A not in top16_names or rikishi_A in matched: continue
                    
                    for j in range(i+1, len(sorted_rikishi)):
                        rikishi_B = sorted_rikishi[j]
                        if rikishi_B in top16_names and rikishi_B not in matched and rikishi_B not in match_history[rikishi_A]:
                            if rooms[rikishi_A] != "unknown" and rooms[rikishi_A] == rooms[rikishi_B]: 
                                continue
                            
                            matched.add(rikishi_A)
                            matched.add(rikishi_B)
                            match_history[rikishi_A].add(rikishi_B)
                            match_history[rikishi_B].add(rikishi_A)
                            
                            prob_A = win_prob_matrix.get((rikishi_A, rikishi_B), 0.5)
                            if np.random.rand() < prob_A:
                                wins[rikishi_A] += 1
                            else:
                                wins[rikishi_B] += 1
                            break

            for i, rikishi_A in enumerate(sorted_rikishi):
                if rikishi_A in matched: continue
                
                for j in range(i+1, len(sorted_rikishi)):
                    rikishi_B = sorted_rikishi[j]
                    if rikishi_B not in matched and rikishi_B not in match_history[rikishi_A]:
                        if rooms[rikishi_A] != "unknown" and rooms[rikishi_A] == rooms[rikishi_B]: 
                            continue 
                        
                        matched.add(rikishi_A)
                        matched.add(rikishi_B)
                        match_history[rikishi_A].add(rikishi_B)
                        match_history[rikishi_B].add(rikishi_A)
                        
                        prob_A = win_prob_matrix.get((rikishi_A, rikishi_B), 0.5)
                        if np.random.rand() < prob_A:
                            wins[rikishi_A] += 1
                        else:
                            wins[rikishi_B] += 1
                        break
        
        max_win = max(wins.values())
        yusho_candidates = [name for name, w in wins.items() if w == max_win]
        winner = np.random.choice(yusho_candidates)
        results[winner]['yusho'] += 1
        
        for name in makuuchi_names:
            results[name]['wins'].append(wins[name])
            
    return results

# ==========================================
# メイン画面構築
# ==========================================

st.title("大相撲 AIレーティングシステム")

global_kimarite_counts = bouts_history_df['kimarite'].value_counts()
global_kimarite_ratio = (global_kimarite_counts / global_kimarite_counts.sum()).to_dict()

kakuzuke_map = {1: '幕内', 2: '十両', 3: '幕下', 4: '三段目', 5: '序二段', 6: '序ノ口'}
kakuzuke_map_rev = {v: k for k, v in kakuzuke_map.items()}
bouts_history_df['階級'] = bouts_history_df['kakuzuke'].map(kakuzuke_map)

# 2つのモデルの予測値をセット
X_hist = bouts_history_df[['rating_diff', 'affinity_diff', 'past_win_rate', 'rank_num_diff']].values
probs_lr = model_lr.predict_proba(X_hist)
probs_rf = model_rf.predict_proba(X_hist)

bouts_history_df['東予測_lr'] = probs_lr[:, 1]
bouts_history_df['西予測_lr'] = probs_lr[:, 0]
bouts_history_df['東予測_rf'] = probs_rf[:, 1]
bouts_history_df['西予測_rf'] = probs_rf[:, 0]

# 選択されたモデルを適用
if "総合" in selected_model_name:
    bouts_history_df['東予測_raw'] = bouts_history_df['東予測_lr']
    bouts_history_df['西予測_raw'] = bouts_history_df['西予測_lr']
else:
    bouts_history_df['東予測_raw'] = bouts_history_df['東予測_rf']
    bouts_history_df['西予測_raw'] = bouts_history_df['西予測_rf']

# ★ 不足していた判定用の関数を追加
def check_upset(row):
    if row['east_result'] == '〇' and row['東予測_raw'] <= 0.4: return "番狂わせ"
    if row['east_result'] == '●' and row['東予測_raw'] >= 0.6: return "番狂わせ"
    return ""

bouts_history_df['特記事項'] = bouts_history_df.apply(check_upset, axis=1)
all_rikishi_sorted = sorted(list(rikishi_ratings.keys()), key=lambda x: rikishi_ratings[x].getRating(), reverse=True)

if 'target_rikishi' not in st.session_state:
    st.session_state.target_rikishi = all_rikishi_sorted[0]
if 'compare_rikishi' not in st.session_state:
    st.session_state.compare_rikishi = "すべて"

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👑 順位推移", "📅 星取表", "⚔️ 対戦分析", "🔮 未来予測", "🚀 急上昇", "🎲 シミュ"
])

# ==========================================
# タブ1：ランキング推移
# ==========================================
with tab1:
    st.header("年代別ランキング・レーティング推移")
    basho_list = history_df['basho'].unique().tolist()
    
    with st.expander("検索条件の設定", expanded=True):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            selected_target_basho = st.select_slider("時代（場所）を選択", options=basho_list, value=basho_list[-1])
        with col_t2:
            days_in_basho = sorted(history_df[history_df['basho'] == selected_target_basho]['day'].unique().tolist())
            if not days_in_basho: days_in_basho = [15]
            selected_target_day = st.select_slider("日目を選択", options=days_in_basho, value=days_in_basho[-1])

    with st.expander("アニメーション再生（推移を自動で動かす）", expanded=False):
        col_t3, col_t4, col_t5 = st.columns([2, 1, 1])
        with col_t3:
            default_start = basho_list[-16] if len(basho_list) >= 16 else basho_list[0]
            anim_start_basho, anim_end_basho = st.select_slider("アニメーション期間", options=basho_list, value=(default_start, basho_list[-1]))
        with col_t4:
            display_metric_anim = st.radio("グラフ指標", ["レーティング", "偏差値"], horizontal=True)
        with col_t5:
            st.write("")
            play_anim = st.button("▶ 再生する")
    
    col_r1, col_r2 = st.columns([1, 2])
    
    with col_r1:
        ranking_placeholder = st.empty()
        
        def display_ranking(basho_name, day_num, is_animating=False):
            current_idx = basho_list.index(basho_name)
            target_basho_df = history_df[(history_df['basho'] == basho_name) & (history_df['day'] == day_num)].copy()
            if target_basho_df.empty: return
            
            target_basho_df['current_rank'] = target_basho_df['deviation'].rank(ascending=False, method='min')
            day15_history = history_df[history_df['day'] == 15].copy()
            day15_history['rank_in_basho'] = day15_history.groupby('basho')['deviation'].rank(ascending=False, method='min')

            def get_prev_rank(row):
                wrestler = row['name']
                past_records = day15_history[(day15_history['name'] == wrestler) & (day15_history['basho'].isin(basho_list[:current_idx]))]
                if not past_records.empty: return past_records.iloc[-1]['rank_in_basho']
                return np.nan

            target_basho_df['prev_rank'] = target_basho_df.apply(get_prev_rank, axis=1)
            target_basho_df['rank_diff'] = target_basho_df['prev_rank'] - target_basho_df['current_rank']
            target_basho_df['変動'] = target_basho_df.apply(lambda r: make_sortable_str(r['rank_diff'], format_trend_val(r['rank_diff']), reverse=True), axis=1)
            target_basho_df['番付'] = target_basho_df.apply(lambda r: make_sortable_str(r['rank_num'], r['rank']), axis=1)
            
            sort_option = "順位" if is_animating else st.session_state.get('sort_option_tab1', '順位')
            
            if sort_option == "番付":
                target_basho_df = target_basho_df.sort_values(by=['rank_num', 'current_rank'], ascending=[True, True])
            elif sort_option == "変動幅":
                target_basho_df = target_basho_df.sort_values(by=['rank_diff', 'current_rank'], ascending=[False, True])
            else:
                target_basho_df = target_basho_df.sort_values(by='current_rank')
            
            display_df = target_basho_df[['current_rank', '変動', 'name', '番付', 'rating', 'deviation']].copy()
            display_df.columns = ['順位', '変動', '四股名', '番付', 'レーティング', '偏差値']
            
            with ranking_placeholder.container():
                st.write(f"**{basho_name} {day_num}日目のランキング**")
                
                event = st.dataframe(
                    display_df.style.format({'レーティング': '{:.1f}', '偏差値': '{:.1f}', '順位': '{:.0f}'}).map(color_trend, subset=['変動']), 
                    height=600, width="stretch", hide_index=True,
                    on_select="rerun", selection_mode="single-row"
                )
                if event and len(event.selection.rows) > 0:
                    selected_name = display_df.iloc[event.selection.rows[0]]['四股名']
                    st.session_state.target_rikishi = selected_name

        if not play_anim:
            st.session_state.sort_option_tab1 = st.radio("並び替えの基準:", ["順位", "番付", "変動幅"], horizontal=True)
            display_ranking(selected_target_basho, selected_target_day, is_animating=False)

    with col_r2:
        graph_placeholder = st.empty()
        
        if play_anim:
            start_idx = basho_list.index(anim_start_basho)
            end_idx = basho_list.index(anim_end_basho)
            anim_bashos = basho_list[start_idx:end_idx+1]
            
            final_df = history_df[(history_df['basho'] == anim_end_basho) & (history_df['day'] == 15)].copy()
            top6_rikishi = final_df.sort_values(by='deviation', ascending=False).head(6)['name'].tolist()
            
            metric_col = 'rating' if display_metric_anim == "レーティング" else 'deviation'
            top6_data = history_df[(history_df['name'].isin(top6_rikishi)) & (history_df['day'] == 15) & (history_df['basho'].isin(anim_bashos))].copy()
            
            if not top6_data.empty:
                y_scale = alt.Scale(domain=[top6_data[metric_col].min() * 0.98, top6_data[metric_col].max() * 1.02])
            else:
                y_scale = alt.Scale(zero=False)

            animated_data = pd.DataFrame()
            
            for b in anim_bashos:
                display_ranking(b, 15, is_animating=True)
                new_step_data = top6_data[top6_data['basho'] == b]
                animated_data = pd.concat([animated_data, new_step_data], ignore_index=True)
                
                if not animated_data.empty:
                    chart = alt.Chart(animated_data).encode(
                        x=alt.X('basho:O', sort=anim_bashos, scale=alt.Scale(domain=anim_bashos), axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
                        y=alt.Y(f'{metric_col}:Q', scale=y_scale),
                        color=alt.Color('name:N', legend=alt.Legend(orient='bottom', columns=3)),
                        tooltip=['basho', 'name', 'rank', alt.Tooltip(f'{metric_col}:Q', format='.1f')]
                    ).mark_line(point=True).properties(height=600)
                    graph_placeholder.altair_chart(chart, width="stretch")
                
                time.sleep(1.0) 
            
        else:
            display_rikishi_static = st.multiselect("グラフに表示する力士", options=all_rikishi_sorted, default=ranking_df['name'].tolist()[:3] if not ranking_df.empty else [])
            metric_static = st.radio("グラフ表示指標", ["レーティング", "偏差値"], horizontal=True)
            
            if display_rikishi_static:
                metric_col = 'rating' if metric_static == "レーティング" else 'deviation'
                filtered_data = history_df[(history_df['name'].isin(display_rikishi_static)) & (history_df['day'] == 15)]
                
                line_chart = alt.Chart(filtered_data).mark_line(point=True).encode(
                    x=alt.X('basho:O', sort=basho_list, axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
                    y=alt.Y(f'{metric_col}:Q', scale=alt.Scale(zero=False)),
                    color=alt.Color('name:N', legend=alt.Legend(orient='bottom')),
                    tooltip=['basho', 'name', alt.Tooltip(f'{metric_col}:Q', format='.1f')]
                ).properties(height=600)
                
                rule = alt.Chart(pd.DataFrame({'basho': [selected_target_basho]})).mark_rule(color='red', strokeDash=[5,5]).encode(x=alt.X('basho:O', sort=basho_list))
                graph_placeholder.altair_chart(line_chart + rule, width="stretch")

# ==========================================
# タブ2：星取表・場所検索
# ==========================================
with tab2:
    with st.expander("検索条件の設定", expanded=True):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            selected_basho = st.selectbox("場所を選択", options=basho_list, index=len(basho_list)-1)
        with col_s2:
            selected_kakuzuke = st.selectbox("階級を選択（序ノ口まで対応）", options=["すべて", "幕内", "十両", "幕下", "三段目", "序二段", "序ノ口"])
    
    st.subheader(f"{selected_basho} 星取表")
    bouts_basho = bouts_history_df[bouts_history_df['basho'] == selected_basho]
    if selected_kakuzuke != "すべて":
        bouts_basho = bouts_basho[bouts_basho['階級'] == selected_kakuzuke]
        
    east_df = bouts_basho[['day_num', '階級', 'east_rank', 'latest_east_name', 'east_result']].copy()
    east_df.columns = ['day', '階級', '番付', 'name', '結果']
    west_df = bouts_basho[['day_num', '階級', 'west_rank', 'latest_west_name', 'west_result']].copy()
    west_df.columns = ['day', '階級', '番付', 'name', '結果']
    
    hoshitori_base = pd.concat([east_df, west_df])
    
    if not hoshitori_base.empty:
        hoshitori_base['rank_num'] = hoshitori_base.apply(lambda row: parse_rank(row['番付'], kakuzuke_map_rev.get(row['階級'], 1)), axis=1)
        hoshitori_base = hoshitori_base.sort_values(by='rank_num') 
        hoshitori_base['番付'] = hoshitori_base.apply(lambda row: make_sortable_str(row['rank_num'], row['番付']), axis=1)

        hoshitori_pivot = hoshitori_base.pivot_table(index=['階級', '番付', 'name'], columns='day', values='結果', aggfunc='first', sort=False).reset_index()
        day_columns = sorted([col for col in hoshitori_pivot.columns if isinstance(col, int)])
        
        display_columns = ['階級', '番付', 'name'] + day_columns
        hoshitori_display = hoshitori_pivot[display_columns].copy()
        
        rename_dict = {'name': '四股名'}
        for d in day_columns: rename_dict[d] = f"{d}日目"
        hoshitori_display = hoshitori_display.rename(columns=rename_dict)
        
        st.write("※表の行をタップすると対象力士が切り替わります。")
        event2 = st.dataframe(
            hoshitori_display.style.map(highlight_result, subset=[f"{d}日目" for d in day_columns]), 
            hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row"
        )
        if event2 and len(event2.selection.rows) > 0:
            selected_name = hoshitori_display.iloc[event2.selection.rows[0]]['四股名']
            st.session_state.target_rikishi = selected_name
    else:
        st.info("データがありません。")

    st.subheader("取組ごとの詳細結果・予測の答え合わせ")
    day_list = bouts_history_df[bouts_history_df['basho'] == selected_basho]['day'].unique()
    selected_day = st.selectbox("日目を選択", options=day_list, key="selectbox_tab2_day")

    filtered_bouts = bouts_history_df[(bouts_history_df['basho'] == selected_basho) & (bouts_history_df['day'] == selected_day)].copy()
    if selected_kakuzuke != "すべて":
        filtered_bouts = filtered_bouts[filtered_bouts['階級'] == selected_kakuzuke]

    filtered_bouts['東予測'] = filtered_bouts['東予測_raw'].apply(lambda x: f"{x * 100:.1f}%")
    filtered_bouts['西予測'] = filtered_bouts['西予測_raw'].apply(lambda x: f"{x * 100:.1f}%")
    
    display_cols = ['basho', 'day', '階級', 'east_rank', 'latest_east_name', '東予測', 'east_result', 'kimarite', 'west_result', '西予測', 'latest_west_name', 'west_rank', '特記事項']
    display_df = filtered_bouts[display_cols].copy()
    display_df.columns = ['場所', '日目', '階級', '東番付', '東方', '東予測', '東結果', '決まり手', '西結果', '西予測', '西方', '西番付', '特記事項']
    
    st.dataframe(display_df.style.map(highlight_result, subset=['東結果', '西結果']).map(highlight_upset, subset=['特記事項']), hide_index=True, width="stretch")


# ==========================================
# タブ3：力士別・対戦成績
# ==========================================
with tab3:
    with st.expander("対象力士の選択", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            idx_A = all_rikishi_sorted.index(st.session_state.target_rikishi) if st.session_state.target_rikishi in all_rikishi_sorted else 0
            rikishi_A = st.selectbox("対象力士", options=all_rikishi_sorted, index=idx_A)
            st.session_state.target_rikishi = rikishi_A
        with col_b:
            options_B = ["すべて"] + all_rikishi_sorted
            idx_B = options_B.index(st.session_state.compare_rikishi) if st.session_state.compare_rikishi in options_B else 0
            rikishi_B = st.selectbox("比較相手（任意）", options=options_B, index=idx_B)
            st.session_state.compare_rikishi = rikishi_B

    if rikishi_A and rikishi_B != "すべて":
        st.markdown("---")
        player_A = rikishi_ratings[rikishi_A]
        player_B = rikishi_ratings[rikishi_B]
        prof_A = profiles[rikishi_A]
        prof_B = profiles[rikishi_B]
        
        pair_key = tuple(sorted([rikishi_A, rikishi_B]))
        h2h_A_wins = direct_h2h.get(pair_key, {}).get(rikishi_A, 0)
        h2h_B_wins = direct_h2h.get(pair_key, {}).get(rikishi_B, 0)
        total_matches = h2h_A_wins + h2h_B_wins
        past_win_rate = h2h_A_wins / total_matches if total_matches > 0 else 0.5
        
        rating_diff = player_A.getRating() - player_B.getRating()
        affinity_diff = calculate_affinity_diff(prof_A, prof_B)
        
        e_rank_num = latest_status.get(rikishi_A, {}).get('rank_num', 400)
        w_rank_num = latest_status.get(rikishi_B, {}).get('rank_num', 400)
        rank_num_diff = w_rank_num - e_rank_num
        
        X_input = np.array([[rating_diff, affinity_diff, past_win_rate, rank_num_diff]])
        prob_A = active_model.predict_proba(X_input)[0][1]
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric(f"AI勝率予測: {rikishi_A}", f"{prob_A * 100:.1f}%")
        col_m2.metric(f"過去の直接対決", f"{rikishi_A} {h2h_A_wins}勝 - {h2h_B_wins}勝 {rikishi_B}")

    if rikishi_A:
        st.markdown("---")
        st.write(f"**{rikishi_A} の決まり手傾向（直近）**")
        target_bouts = bouts_history_df[(bouts_history_df['latest_east_name'] == rikishi_A) | (bouts_history_df['latest_west_name'] == rikishi_A)]
        
        wins = []
        losses = []
        for _, row in target_bouts.iterrows():
            is_east = (row['latest_east_name'] == rikishi_A)
            res = row['east_result'] if is_east else row['west_result']
            if res == '〇':
                wins.append(row['kimarite'])
            elif res == '●':
                losses.append(row['kimarite'])
                
        win_counts = pd.Series(wins).value_counts()
        loss_counts = pd.Series(losses).value_counts()
        
        col_w, col_l = st.columns(2)
        with col_w: plot_kimarite_chart(win_counts, "勝利時の決まり手", global_kimarite_ratio)
        with col_l: plot_kimarite_chart(loss_counts, "敗北時の決まり手", global_kimarite_ratio)

    st.markdown("---")
    st.write(f"**{rikishi_A} の過去の対戦履歴**")
    if rikishi_B == "すべて":
        rikishi_bouts = bouts_history_df[(bouts_history_df['latest_east_name'] == rikishi_A) | (bouts_history_df['latest_west_name'] == rikishi_A)].copy()
    else:
        rikishi_bouts = bouts_history_df[((bouts_history_df['latest_east_name'] == rikishi_A) & (bouts_history_df['latest_west_name'] == rikishi_B)) | ((bouts_history_df['latest_east_name'] == rikishi_B) & (bouts_history_df['latest_west_name'] == rikishi_A))].copy()

    formatted_bouts = []
    for _, row in rikishi_bouts.iterrows():
        is_east = (row['latest_east_name'] == rikishi_A)
        target_result = row['east_result'] if is_east else row['west_result']
        opponent = row['latest_west_name'] if is_east else row['latest_east_name']
        target_prob = row['東予測_raw'] if is_east else row['西予測_raw']
        target_rank = row['east_rank'] if is_east else row['west_rank']
        opponent_rank = row['west_rank'] if is_east else row['east_rank']
        
        formatted_bouts.append({
            '場所': row['basho'], '日目': row['day'], '番付': target_rank, '対戦相手': opponent, '相手番付': opponent_rank,
            '勝率予想': f"{target_prob * 100:.1f}%", '結果': target_result, '決まり手': row['kimarite'],
            '特記事項': row['特記事項'], '勝率予想_raw': target_prob
        })
        
    target_df = pd.DataFrame(formatted_bouts).iloc[::-1]

    if not target_df.empty:
        st.write("※表の行をタップすると、対象力士がその対戦相手に切り替わります。")
        event = st.dataframe(
            target_df.style.apply(apply_row_styles, axis=1), 
            hide_index=True, 
            width="stretch", 
            column_config={"勝率予想_raw": None},
            on_select="rerun",
            selection_mode="single-row"
        )
        
        if event and len(event.selection.rows) > 0:
            selected_idx = event.selection.rows[0]
            selected_opponent = target_df.iloc[selected_idx]['対戦相手']
            if selected_opponent in all_rikishi_sorted:
                st.session_state.target_rikishi = selected_opponent
                st.rerun()
    else:
        st.write("対戦履歴がありません。")

# ==========================================
# タブ4：未来の取組と勝敗予測
# ==========================================
with tab4:
    st.header("未来の取組と勝敗予測")
    unplayed_bouts = bouts_history_df[bouts_history_df['east_result'] == '-'].copy()
    
    if not unplayed_bouts.empty:
        latest_unplayed_basho = unplayed_bouts['basho'].iloc[-1]
        st.write(f"**対象場所:** {latest_unplayed_basho}")
        
        with st.expander("表示条件の絞り込み", expanded=True):
            col_f1, col_f2 = st.columns(2)
            available_days = unplayed_bouts[unplayed_bouts['basho'] == latest_unplayed_basho]['day'].unique()
            with col_f1:
                selected_future_day = st.selectbox("日目", options=available_days, key="selectbox_tab4_future_day")
            with col_f2:
                selected_future_kakuzuke = st.selectbox("階級（序ノ口まで予測対応）", ["すべて", "幕内", "十両", "幕下", "三段目", "序二段", "序ノ口"], key="selectbox_tab4_kakuzuke")
        
        display_future = unplayed_bouts[(unplayed_bouts['basho'] == latest_unplayed_basho) & (unplayed_bouts['day'] == selected_future_day)].copy()
        if selected_future_kakuzuke != "すべて":
            display_future = display_future[display_future['階級'] == selected_future_kakuzuke]
        
        display_future['東予測(総合)'] = display_future['東予測_lr'].apply(lambda x: f"{x * 100:.1f}%")
        display_future['西予測(総合)'] = display_future['西予測_lr'].apply(lambda x: f"{x * 100:.1f}%")
        display_future['東予測(特化)'] = display_future['東予測_rf'].apply(lambda x: f"{x * 100:.1f}%")
        display_future['西予測(特化)'] = display_future['西予測_rf'].apply(lambda x: f"{x * 100:.1f}%")
        
        display_cols = ['階級', 'east_rank', 'latest_east_name', '東予測(総合)', '東予測(特化)', '西予測(総合)', '西予測(特化)', 'latest_west_name', 'west_rank']
        disp_df = display_future[display_cols].copy()
        disp_df.columns = ['階級', '東番付', '東方', '東予測(総合)', '東予測(特化)', '西予測(総合)', '西予測(特化)', '西方', '西番付']
        
        st.dataframe(disp_df, hide_index=True, width="stretch")
    else:
        st.info("現在、未発表または未取組のデータはありません。（場所前または全日程終了）")

# ==========================================
# タブ5：急上昇力士と予測精度分析
# ==========================================
with tab5:
    st.header("🚀 レーティング急上昇力士")
    st.write("最新場所とその前の場所を比較し、レーティングが大きく伸びている力士を抽出します。")
    
    if len(basho_list) >= 2:
        latest_basho = basho_list[-1]
        prev_basho = basho_list[-2]
        
        latest_df = history_df[history_df['basho'] == latest_basho].groupby('name').last().reset_index()
        prev_df = history_df[history_df['basho'] == prev_basho].groupby('name').last().reset_index()
        
        merged_trend = pd.merge(latest_df[['name', 'rating']], prev_df[['name', 'rating']], on='name', suffixes=('_latest', '_prev'))
        merged_trend['diff'] = merged_trend['rating_latest'] - merged_trend['rating_prev']
        merged_trend['kakuzuke'] = merged_trend['name'].apply(lambda x: latest_status.get(x, {}).get('kakuzuke', '不明'))
        
        trend_kakuzuke = st.selectbox("階級で絞り込み", ["幕内", "十両", "幕下", "三段目", "序二段", "序ノ口"], key="trend_kakuzuke_select")
        filtered_trend = merged_trend[merged_trend['kakuzuke'] == trend_kakuzuke].sort_values('diff', ascending=False).head(10)
        
        filtered_trend['rating_latest'] = filtered_trend['rating_latest'].round(1)
        filtered_trend['diff'] = filtered_trend['diff'].round(1).apply(lambda x: f"+{x}" if x > 0 else str(x))
        
        disp_trend = filtered_trend[['name', 'rating_latest', 'diff']].rename(columns={'name':'四股名', 'rating_latest':'最新レーティング', 'diff':'上昇幅'})
        st.dataframe(disp_trend, hide_index=True, width="stretch")
    
    st.markdown("---")
    st.header("🎯 AIモデルの予測精度分析")
    st.write("過去の実際の取組結果と、事前の勝敗予測がどれくらい一致していたかを検証します。")
    
    valid_bouts = bouts_history_df[bouts_history_df['east_result'].isin(['〇', '●'])].copy()
    
    if not valid_bouts.empty:
        valid_bouts['actual'] = (valid_bouts['east_result'] == '〇')
        valid_bouts['pred_lr'] = (valid_bouts['東予測_lr'] >= 0.5)
        valid_bouts['pred_rf'] = (valid_bouts['東予測_rf'] >= 0.5)
        valid_bouts['correct_lr'] = (valid_bouts['actual'] == valid_bouts['pred_lr'])
        valid_bouts['correct_rf'] = (valid_bouts['actual'] == valid_bouts['pred_rf'])

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("総合AI 予測的中率", f"{valid_bouts['correct_lr'].mean()*100:.1f} %", f"全 {len(valid_bouts):,} 取組対象")
        col_m2.metric("特化AI 予測的中率", f"{valid_bouts['correct_rf'].mean()*100:.1f} %", f"全 {len(valid_bouts):,} 取組対象")
        
        st.write("**場所ごとの的中率推移**")
        acc_by_basho = valid_bouts.groupby('basho')[['correct_lr', 'correct_rf']].mean().reset_index()
        acc_melted = acc_by_basho.melt(id_vars='basho', value_vars=['correct_lr', 'correct_rf'], var_name='model', value_name='accuracy')
        acc_melted['model'] = acc_melted['model'].map({'correct_lr': '総合AI', 'correct_rf': '特化AI'})
        
        st.altair_chart(alt.Chart(acc_melted).mark_line(point=True).encode(
            x=alt.X('basho:O', sort=basho_list, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('accuracy:Q', scale=alt.Scale(domain=[0.4, 0.8])),
            color='model:N',
            tooltip=['basho', 'model', alt.Tooltip('accuracy:Q', format='.1%')]
        ).properties(height=350), width="stretch")

# ==========================================
# タブ6：場所全体シミュレーション
# ==========================================
with tab6:
    st.header("場所全体シミュレーション（幕内）")
    st.write("最新の幕内番付に基づき、15日間の全取組を仮想的に編成・対戦させ、モンテカルロシミュレーションによって結果を予測します。")
    
    sim_count = st.slider("シミュレーション試行回数", min_value=100, max_value=5000, value=1000, step=100)
    
    if st.button("シミュレーションを実行する"):
        with st.spinner(f"最新の番付情報を取得し、{sim_count}回の場所シミュレーションを実行しています..."):
            makuuchi_list = fetch_makuuchi_banzuke(bouts_history_df)
            
            if not makuuchi_list:
                st.error("幕内力士情報の取得に失敗しました。")
            else:
                matrix, rank_nums, rooms = create_win_prob_matrix(makuuchi_list, rikishi_ratings, profiles, direct_h2h, active_model)
                makuuchi_names = [r['name'] for r in makuuchi_list]
                
                results = run_simulation(makuuchi_names, matrix, rank_nums, rooms, sim_count)
                
                sim_data = []
                for r in makuuchi_list:
                    name = r['name']
                    data = results[name]
                    yusho_prob = data['yusho'] / sim_count
                    avg_wins = np.mean(data['wins'])
                    kachikoshi_prob = sum(1 for w in data['wins'] if w >= 8) / sim_count
                    max_w = max(data['wins'])
                    min_w = min(data['wins'])
                    
                    sim_data.append({
                        '四股名': name,
                        '番付': r['rank'],
                        '優勝確率': yusho_prob,
                        '平均勝利数': avg_wins,
                        '最高': max_w,
                        '最低': min_w,
                        '勝ち越し確率': kachikoshi_prob,
                        'rank_num': rank_nums[name]
                    })
                
                sim_df = pd.DataFrame(sim_data)
                sim_df['番付'] = sim_df.apply(lambda row: make_sortable_str(row['rank_num'], row['番付']), axis=1)
                sim_df = sim_df.sort_values('優勝確率', ascending=False)
                
                st.session_state.sim_df = sim_df
                st.session_state.sim_raw_results = results
                st.session_state.sim_count = sim_count

    if 'sim_df' in st.session_state:
        st.write("※表の行をタップすると、その力士のシミュレーション詳細（15日間の勝利数分布）が下に表示されます。")
        sim_df = st.session_state.sim_df
        display_cols = ['四股名', '番付', '優勝確率', '平均勝利数', '最高', '最低', '勝ち越し確率']
        
        event_sim = st.dataframe(
            sim_df[display_cols].style.format({
                '優勝確率': '{:.1%}',
                '平均勝利数': '{:.1f}勝',
                '最高': '{}勝',
                '最低': '{}勝',
                '勝ち越し確率': '{:.1%}'
            }),
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row"
        )

        if event_sim and len(event_sim.selection.rows) > 0:
            selected_idx = event_sim.selection.rows[0]
            selected_name = sim_df.iloc[selected_idx]['四股名']
            st.subheader(f"{selected_name} のシミュレーション詳細（勝利数分布）")
            
            wins_data = st.session_state.sim_raw_results[selected_name]['wins']
            win_counts = pd.Series(wins_data).value_counts().sort_index()
            
            full_index = pd.Index(range(16))
            win_counts = win_counts.reindex(full_index, fill_value=0)
            
            dist_df = pd.DataFrame({
                '勝利数': win_counts.index,
                '回数': win_counts.values,
                '確率': win_counts.values / st.session_state.sim_count
            })
            dist_df['確率表示'] = dist_df['確率'].apply(lambda x: f"{x*100:.1f}%")
            
            base = alt.Chart(dist_df).encode(
                x=alt.X('勝利数:O', title='勝利数', axis=alt.Axis(labelAngle=0))
            )
            bars = base.mark_bar(color='#1f77b4').encode(
                y=alt.Y('確率:Q', title='確率', axis=alt.Axis(format='%')),
                tooltip=['勝利数', '回数', alt.Tooltip('確率:Q', format='.1%')]
            )
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5
            ).encode(
                y=alt.Y('確率:Q'),
                text='確率表示:N'
            )
            st.altair_chart((bars + text).properties(height=350), width="stretch")