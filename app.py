import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import time
from rating_calc import get_rating_data, parse_rank

st.set_page_config(page_title="大相撲レーティング", layout="wide")

KIMARITE_CATEGORY = {
    '押し出し': '突き押し', '突き出し': '突き押し', '押し倒し': '突き押し', '突き倒し': '突き押し',
    '寄り切り': '四つ', '寄り倒し': '四つ', '上手投げ': '四つ', '下手投げ': '四つ', '上手出し投げ': '四つ', '下手出し投げ': '四つ', '掬い投げ': '四つ', '外掛け': '四つ', '内掛け': '四つ',
    '叩き込み': '引き', '引き落とし': '引き', '突き落とし': '引き', '肩透かし': '引き', '引き技': '引き'
}

@st.cache_data
def load_data():
    return get_rating_data()

@st.cache_resource
def load_model():
    try:
        return joblib.load('sumo_model.pkl')
    except FileNotFoundError:
        return None

@st.cache_data
def calculate_yusho(bouts_df):
    yusho_dict = {}
    for (basho, kakuzuke), group in bouts_df.groupby(['basho', 'kakuzuke']):
        win_counts = {}
        for _, row in group.iterrows():
            if row['east_result'] == '〇': win_counts[row['latest_east_name']] = win_counts.get(row['latest_east_name'], 0) + 1
            if row['west_result'] == '〇': win_counts[row['latest_west_name']] = win_counts.get(row['latest_west_name'], 0) + 1
        if win_counts:
            max_wins = max(win_counts.values())
            winners = [k for k, v in win_counts.items() if v == max_wins]
            yusho_dict[(basho, kakuzuke)] = (max_wins, winners)
    return yusho_dict

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

def format_dev_diff(val):
    if pd.isna(val): return "-"
    if val > 0: return f"↑ {val:.1f}"
    if val < 0: return f"↓ {abs(val):.1f}"
    return "→ 0.0"

def color_diff(val):
    if pd.isna(val): return ""
    if val > 0: return 'color: red; font-weight: bold;'
    if val < 0: return 'color: blue; font-weight: bold;'
    return 'color: gray; font-weight: bold;'

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
    st.altair_chart((bars + ticks_bg + ticks_fg + text).properties(title=title, height=350), use_container_width=True)

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

st.title("大相撲 レーティング・対戦分析システム")

ranking_df, history_df, rikishi_ratings, profiles, direct_h2h, bouts_history_df, latest_status = load_data()
model = load_model()

if ranking_df.empty or model is None:
    st.warning("データまたはモデルが存在しません。スクレイピングとモデル学習を実行してください。")
    st.stop()

yusho_data = calculate_yusho(bouts_history_df)
global_kimarite_counts = bouts_history_df['kimarite'].value_counts()
global_kimarite_ratio = (global_kimarite_counts / global_kimarite_counts.sum()).to_dict()

kakuzuke_map = {1: '幕内', 2: '十両', 3: '幕下', 4: '三段目', 5: '序二段', 6: '序ノ口'}
kakuzuke_map_rev = {v: k for k, v in kakuzuke_map.items()}
bouts_history_df['階級'] = bouts_history_df['kakuzuke'].map(kakuzuke_map)

X_hist = bouts_history_df[['rating_diff', 'affinity_diff', 'past_win_rate', 'rank_num_diff']].values
probs = model.predict_proba(X_hist)
bouts_history_df['東予測_raw'] = probs[:, 1]
bouts_history_df['西予測_raw'] = probs[:, 0]

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

tab1, tab2, tab3, tab4 = st.tabs(["年代別ランキング・推移", "場所・日程から検索", "力士・対戦成績から検索", "最新の取組・勝敗予測"])

with tab1:
    st.header("年代別ランキング・レーティング推移")
    basho_list = history_df['basho'].unique().tolist()
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        selected_target_basho = st.select_slider("時代（場所）を選択してください", options=basho_list, value=basho_list[-1])
    with col_t2:
        days_in_basho = sorted(history_df[history_df['basho'] == selected_target_basho]['day'].unique().tolist())
        if not days_in_basho: days_in_basho = [15]
        selected_target_day = st.select_slider("日目を選択してください", options=days_in_basho, value=days_in_basho[-1])

    st.markdown("---")
    st.subheader("アニメーション設定と推移グラフ")
    col_t3, col_t4, col_t5 = st.columns([2, 1, 1])
    with col_t3:
        default_start = basho_list[-16] if len(basho_list) >= 16 else basho_list[0]
        default_end = basho_list[-1]
        anim_start_basho, anim_end_basho = st.select_slider("アニメーション期間", options=basho_list, value=(default_start, default_end))
    with col_t4:
        display_metric_anim = st.radio("グラフ表示指標", ["レーティング", "偏差値"], horizontal=True, key="anim_metric")
    with col_t5:
        st.write("")
        st.write("")
        play_anim = st.button("アニメーション再生（表とグラフ連動）")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
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
            
            if is_animating:
                sort_option = "順位"
            else:
                sort_option = st.session_state.get('sort_option_tab1', '順位')

            if sort_option == "番付": target_basho_df = target_basho_df.sort_values(by=['rank_num', 'current_rank'], ascending=[True, True])
            elif sort_option == "変動幅": target_basho_df = target_basho_df.sort_values(by=['rank_diff', 'current_rank'], ascending=[False, True])
            else: target_basho_df = target_basho_df.sort_values(by='current_rank')
            
            display_ranking_df = target_basho_df[['current_rank', '変動', 'name', '番付', 'rating', 'deviation']].copy()
            display_ranking_df.columns = ['順位', '変動', '四股名', '番付', 'レーティング', '偏差値']
            
            with ranking_placeholder.container():
                st.subheader(f"{basho_name} {day_num}日目 のランキング")
                
                event = st.dataframe(
                    display_ranking_df.style.format({'レーティング': '{:.1f}', '偏差値': '{:.1f}', '順位': '{:.0f}'}).map(color_trend, subset=['変動']), 
                    height=800, use_container_width=True, hide_index=True,
                    on_select="rerun", selection_mode="single-row"
                )
                if event and len(event.selection.rows) > 0:
                    selected_name = display_ranking_df.iloc[event.selection.rows[0]]['四股名']
                    st.session_state.target_rikishi = selected_name
                    st.write(f"「{selected_name}」を選択しました。力士別・対戦成績と予測分析タブで詳細を確認できます。")

        if not play_anim:
            st.session_state.sort_option_tab1 = st.radio("表の並び替え基準:", ["順位", "番付", "変動幅"], horizontal=True, key="sort_radio_tab1")
            display_ranking(selected_target_basho, selected_target_day, is_animating=False)

    with col2:
        graph_placeholder = st.empty()
        
        if play_anim:
            start_idx = basho_list.index(anim_start_basho)
            end_idx = basho_list.index(anim_end_basho)
            anim_bashos = basho_list[start_idx:end_idx+1]
            
            final_basho = anim_end_basho
            final_day15_df = history_df[(history_df['basho'] == final_basho) & (history_df['day'] == 15)].copy()
            final_day15_df['rank_in_basho'] = final_day15_df['deviation'].rank(ascending=False, method='min')
            top6_rikishi = final_day15_df.sort_values(by='rank_in_basho').head(6)['name'].tolist()
            
            metric_col = 'rating' if display_metric_anim == "レーティング" else 'deviation'
            
            top6_data_all = history_df[
                (history_df['name'].isin(top6_rikishi)) & 
                (history_df['day'] == 15) & 
                (history_df['basho'].isin(anim_bashos))
            ].copy()
            
            if not top6_data_all.empty:
                y_min = top6_data_all[metric_col].min()
                y_max = top6_data_all[metric_col].max()
                y_scale = alt.Scale(domain=[y_min * 0.98 if y_min > 0 else y_min * 1.02, 
                                             y_max * 1.02 if y_max > 0 else y_max * 0.98])
            else:
                y_scale = alt.Scale(zero=False)

            animated_data = pd.DataFrame()
            
            def create_anim_chart(data, x_sort_order, y_metric, y_axis_scale):
                base_chart = alt.Chart(data).encode(
                    x=alt.X('basho:O', title='場所', sort=x_sort_order, scale=alt.Scale(domain=x_sort_order), axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
                    y=alt.Y(f'{y_metric}:Q', title=display_metric_anim, scale=y_axis_scale),
                    color=alt.Color('name:N', title='力士', legend=alt.Legend(orient='bottom', columns=3)),
                    tooltip=['basho', 'name', 'rank', alt.Tooltip(f'{y_metric}:Q', format='.1f')]
                )
                lines = base_chart.mark_line().encode()
                points = base_chart.mark_point(size=50, filled=True).encode()
                return (lines + points).properties(title=f"終了時点上位6人の{display_metric_anim}推移 ({anim_start_basho} ～ {final_basho})", height=1000)

            for b in anim_bashos:
                basho_name = b 
                display_ranking(b, 15, is_animating=True)
                
                new_step_data = top6_data_all[top6_data_all['basho'] == b]
                animated_data = pd.concat([animated_data, new_step_data], ignore_index=True)
                
                if not animated_data.empty:
                    chart = create_anim_chart(animated_data, anim_bashos, metric_col, y_scale)
                    graph_placeholder.altair_chart(chart, use_container_width=True)
                
                time.sleep(1.0) 
            
            st.write("アニメーション再生が完了しました。")

        else:
            st.write("直近のレーティング推移（15日目）")
            display_rikishi_static = st.multiselect("力士を選択", options=all_rikishi_sorted, default=ranking_df['name'].tolist()[:3] if not ranking_df.empty else [], key="multiselect_static_tab1")
            if display_rikishi_static:
                filtered_history_static = history_df[(history_df['name'].isin(display_rikishi_static)) & (history_df['day'] == 15)]
                metric_col_static = 'rating' if st.session_state.get('static_metric_tab1', '偏差値') == "レーティング" else 'deviation'
                y_title_static = "レーティング" if metric_col_static == 'rating' else "偏差値"
                
                line_chart_static = alt.Chart(filtered_history_static).mark_line(point=True).encode(
                    x=alt.X('basho:O', title='場所', sort=basho_list, axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
                    y=alt.Y(f'{metric_col_static}:Q', scale=alt.Scale(zero=False), title=y_title_static),
                    color=alt.Color('name:N', title='力士', legend=alt.Legend(orient='bottom')),
                    tooltip=['basho', 'name', 'rank', alt.Tooltip(f'{metric_col_static}:Q', format='.1f')]
                ).properties(height=1000)
                rule_static = alt.Chart(pd.DataFrame({'basho': [selected_target_basho]})).mark_rule(color='red', strokeDash=[5, 5]).encode(x=alt.X('basho:O', sort=basho_list))
                graph_placeholder.altair_chart(line_chart_static + rule_static, use_container_width=True)
                
                st.session_state.static_metric_tab1 = st.radio("表示指標", ["レーティング", "偏差値"], horizontal=True, key="metric_radio_static_tab1")

with tab2:
    st.header("場所・日程による検索と星取表")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        selected_basho = st.selectbox("場所を選択", options=basho_list, index=len(basho_list)-1)
    with col_s2:
        kakuzuke_options = ["すべて", "幕内", "十両", "幕下", "三段目", "序二段", "序ノ口"]
        selected_kakuzuke = st.selectbox("階級で絞り込み", options=kakuzuke_options)

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
        day_columns = [col for col in hoshitori_pivot.columns if isinstance(col, int)]
        day_columns_sorted = sorted(day_columns)
        
        display_columns = ['階級', '番付', 'name'] + day_columns_sorted
        hoshitori_display = hoshitori_pivot[display_columns].copy()
        
        rename_dict = {'name': '四股名'}
        for d in day_columns_sorted: rename_dict[d] = f"{d}日目"
        hoshitori_display = hoshitori_display.rename(columns=rename_dict)
        
        event2 = st.dataframe(
            hoshitori_display.style.map(highlight_result, subset=[f"{d}日目" for d in day_columns_sorted]), 
            hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row"
        )
        if event2 and len(event2.selection.rows) > 0:
            selected_name = hoshitori_display.iloc[event2.selection.rows[0]]['四股名']
            st.session_state.target_rikishi = selected_name
            st.write(f"「{selected_name}」を選択しました。力士別・対戦成績と予測分析タブで詳細を確認できます。")
    else:
        st.write("データがありません。")

    st.subheader("取組ごとの詳細検索")
    day_list = bouts_history_df[bouts_history_df['basho'] == selected_basho]['day'].unique()
    selected_day = st.selectbox("日目を選択", options=day_list)

    filtered_bouts = bouts_history_df[(bouts_history_df['basho'] == selected_basho) & (bouts_history_df['day'] == selected_day)].copy()
    if selected_kakuzuke != "すべて":
        filtered_bouts = filtered_bouts[filtered_bouts['階級'] == selected_kakuzuke]

    filtered_bouts['東予測'] = filtered_bouts['東予測_raw'].apply(lambda x: f"{x * 100:.1f}%")
    filtered_bouts['西予測'] = filtered_bouts['西予測_raw'].apply(lambda x: f"{x * 100:.1f}%")
    
    display_cols = ['basho', 'day', '階級', 'east_rank', 'latest_east_name', '東予測', 'east_result', 'kimarite', 'west_result', '西予測', 'latest_west_name', 'west_rank', '特記事項']
    display_df = filtered_bouts[display_cols].copy()
    display_df.columns = ['場所', '日目', '階級', '東番付', '東方', '東予測', '東結果', '決まり手', '西結果', '西予測', '西方', '西番付', '特記事項']
    
    st.dataframe(display_df.style.map(highlight_result, subset=['東結果', '西結果']).map(highlight_upset, subset=['特記事項']), hide_index=True, use_container_width=True)

with tab3:
    st.header("力士別・対戦成績と予測分析")
    col_a, col_b = st.columns(2)
    with col_a:
        idx_A = all_rikishi_sorted.index(st.session_state.target_rikishi) if st.session_state.target_rikishi in all_rikishi_sorted else 0
        rikishi_A = st.selectbox("対象力士を選択", options=all_rikishi_sorted, index=idx_A)
        st.session_state.target_rikishi = rikishi_A
    with col_b:
        options_B = ["すべて"] + all_rikishi_sorted
        idx_B = options_B.index(st.session_state.compare_rikishi) if st.session_state.compare_rikishi in options_B else 0
        rikishi_B = st.selectbox("比較対象（任意）", options=options_B, index=idx_B)
        st.session_state.compare_rikishi = rikishi_B

    if rikishi_A:
        st.subheader(f"{rikishi_A} の決まり手分布")
        target_bouts = bouts_history_df[(bouts_history_df['latest_east_name'] == rikishi_A) | (bouts_history_df['latest_west_name'] == rikishi_A)]
        wins, losses = [], []
        for _, row in target_bouts.iterrows():
            is_east = (row['latest_east_name'] == rikishi_A)
            res = row['east_result'] if is_east else row['west_result']
            if res == '〇': wins.append(row['kimarite'])
            else: losses.append(row['kimarite'])
                
        win_counts = pd.Series(wins).value_counts()
        loss_counts = pd.Series(losses).value_counts()
        
        col_w, col_l = st.columns(2)
        with col_w: plot_kimarite_chart(win_counts, "勝利時の決まり手", global_kimarite_ratio)
        with col_l: plot_kimarite_chart(loss_counts, "敗北時の決まり手", global_kimarite_ratio)

    if rikishi_A and rikishi_B != "すべて":
        player_A = rikishi_ratings[rikishi_A]
        player_B = rikishi_ratings[rikishi_B]
        prof_A, prof_B = profiles[rikishi_A], profiles[rikishi_B]
        
        pair_key = tuple(sorted([rikishi_A, rikishi_B]))
        h2h_A_wins = direct_h2h.get(pair_key, {}).get(rikishi_A, 0)
        h2h_B_wins = direct_h2h.get(pair_key, {}).get(rikishi_B, 0)
        total_matches = h2h_A_wins + h2h_B_wins
        past_win_rate = h2h_A_wins / total_matches if total_matches > 0 else 0.5
        
        rating_diff = player_A.getRating() - player_B.getRating()
        affinity_diff = calculate_affinity_diff(prof_A, prof_B)
        
        e_rank_num = latest_status[rikishi_A]['rank_num'] if rikishi_A in latest_status else 400
        w_rank_num = latest_status[rikishi_B]['rank_num'] if rikishi_B in latest_status else 400
        rank_num_diff = w_rank_num - e_rank_num
        
        X_input = np.array([[rating_diff, affinity_diff, past_win_rate, rank_num_diff]])
        prob_A = model.predict_proba(X_input)[0][1]
        
        st.subheader("現在の勝率予測")
        st.markdown(f"**{rikishi_A}**: {prob_A * 100:.1f} %  /  **{rikishi_B}**: {(1 - prob_A) * 100:.1f} %")
        st.markdown(f"**通算成績**: {rikishi_A} {h2h_A_wins}勝 - {h2h_B_wins}勝 {rikishi_B}")

    st.subheader(f"{rikishi_A} の過去の対戦履歴")
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
        styled_target_df = target_df.style.apply(apply_row_styles, axis=1)
        st.write("表の行を選択すると対象力士がその対戦相手に切り替わります。")
        
        event = st.dataframe(
            styled_target_df, 
            hide_index=True, 
            use_container_width=True, 
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

with tab4:
    st.header("未来の取組と勝敗予測")
    unplayed_bouts = bouts_history_df[bouts_history_df['east_result'] == '-'].copy()
    
    if not unplayed_bouts.empty:
        latest_unplayed_basho = unplayed_bouts['basho'].iloc[-1]
        st.subheader(f"対象場所: {latest_unplayed_basho}")
        
        available_days = unplayed_bouts[unplayed_bouts['basho'] == latest_unplayed_basho]['day'].unique()
        selected_future_day = st.selectbox("日目を選択", options=available_days)
        
        display_future = unplayed_bouts[(unplayed_bouts['basho'] == latest_unplayed_basho) & (unplayed_bouts['day'] == selected_future_day)].copy()
        
        display_future['東予測'] = display_future['東予測_raw'].apply(lambda x: f"{x * 100:.1f}%")
        display_future['西予測'] = display_future['西予測_raw'].apply(lambda x: f"{x * 100:.1f}%")
        
        display_cols = ['階級', 'east_rank', 'latest_east_name', '東予測', '西予測', 'latest_west_name', 'west_rank']
        disp_df = display_future[display_cols].copy()
        disp_df.columns = ['階級', '東番付', '東方', '東予測', '西予測', '西方', '西番付']
        
        st.dataframe(disp_df, hide_index=True, use_container_width=True)
    else:
        st.write("現在、未取組のデータはありません。（場所前または全日程終了）")