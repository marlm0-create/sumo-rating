import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
import concurrent.futures

def create_database():
    conn = sqlite3.connect('sumo_data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS bouts (
            year INTEGER,
            month INTEGER,
            basho_id INTEGER,
            day INTEGER,
            kakuzuke INTEGER,
            east_id INTEGER,
            east_rank TEXT,
            east_name TEXT,
            east_result TEXT,
            kimarite TEXT,
            west_result TEXT,
            west_name TEXT,
            west_rank TEXT,
            west_id INTEGER
        )
    ''')
    conn.commit()
    return conn

def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return session

def make_request_with_retry(session, url, payload, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = session.post(url, data=payload, timeout=20)
            if response.status_code == 200:
                return response
        except requests.exceptions.RequestException:
            pass
        time.sleep(3 + attempt * 2) 
    return None

def get_basho_id(session, year, month):
    url = "https://www.sumo.or.jp/ResultRikishiDataDaicho/torikumi/"
    payload = {"year": str(year), "basho_month": f"{month:02d}", "contents": "torikumi"}
    response = make_request_with_retry(session, url, payload)
    if not response: return None
    match = re.search(r'torikumi\(\d+,\s*\d+,\s*(\d+)\)', response.text)
    if match: return int(match.group(1))
    return None

def fetch_day_data(args):
    year, month, basho_id, day, kakuzuke = args
    session = get_session()
    url = "https://www.sumo.or.jp/ResultRikishiDataDaicho/torikumi"
    payload = {"basho_id": basho_id, "day": day, "kakuzuke": kakuzuke}
    
    response = make_request_with_retry(session, url, payload)
    if not response: return []

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', class_='mdTable1')
    if not table: return []

    bouts = []
    rows = table.find_all('tr')
    for row in rows:
        if row.find('th'): continue
        cells = row.find_all('td', recursive=False)
        if len(cells) >= 5:
            try:
                east_td = cells[0]
                east_rank = east_td.find('span', class_='rank').text.strip()
                east_name = east_td.find('span', class_='name').text.strip()
                
                east_a = east_td.find('a')
                east_id = 0
                if east_a and 'href' in east_a.attrs:
                    m = re.search(r'/profile/(\d+)', east_a['href'])
                    if m: east_id = int(m.group(1))
                
                kimarite = cells[2].text.strip()
                
                west_td = cells[4]
                west_rank = west_td.find('span', class_='rank').text.strip()
                west_name = west_td.find('span', class_='name').text.strip()

                west_a = west_td.find('a')
                west_id = 0
                if west_a and 'href' in west_a.attrs:
                    m = re.search(r'/profile/(\d+)', west_a['href'])
                    if m: west_id = int(m.group(1))
                
                if kimarite == "":
                    east_result = "-"
                    west_result = "-"
                else:
                    east_win = 'win' in east_td.get('class', [])
                    east_result = "〇" if east_win else "●"
                    west_win = 'win' in west_td.get('class', [])
                    west_result = "〇" if west_win else "●"
                
                bouts.append((year, month, basho_id, day, kakuzuke, east_id, east_rank, east_name, east_result, kimarite, west_result, west_name, west_rank, west_id))
            except AttributeError:
                continue
    return bouts

def main():
    conn = create_database()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM bouts WHERE east_result = '-' OR east_result = ''")
    conn.commit()
    
    years = range(2013, 2027)
    months = [1, 3, 5, 7, 9, 11]
    main_session = get_session()
    tasks = []
    
    print("データベースをスキャンし、不足または未取組の日程を特定します...")
    for year in years:
        for month in months:
            if year == 2020 and month == 5: continue
            
            cursor.execute('SELECT kakuzuke, day FROM bouts WHERE year = ? AND month = ?', (year, month))
            records = cursor.fetchall()
            
            existing_data = {k: set() for k in range(1, 7)}
            for kakuzuke, day in records:
                existing_data[kakuzuke].add(day)
            
            is_complete = all(len(existing_data[k]) >= 15 for k in range(1, 7))
            if is_complete:
                continue
                
            basho_id = get_basho_id(main_session, year, month)
            if not basho_id: continue
            
            missing_info = []
            for kakuzuke in range(1, 7):
                missing_days = set(range(1, 16)) - existing_data[kakuzuke]
                if missing_days:
                    missing_info.append(f"階級{kakuzuke}({len(missing_days)}日)")
                    for day in missing_days:
                        tasks.append((year, month, basho_id, day, kakuzuke))
            
            if missing_info:
                print(f"  {year}年{month}月 (ID:{basho_id}): 取得対象 -> {', '.join(missing_info)}")

    if not tasks:
        print("すべてのデータが揃っています。更新対象はありません。")
        conn.close()
        return
        
    print(f"\n合計 {len(tasks)} 件のタスクを実行します。")
    MAX_WORKERS = 5
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_day_data, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                bouts = future.result()
                if bouts:
                    cursor.executemany('''INSERT INTO bouts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', bouts)
                    conn.commit()
                    print(f"    - 取得完了: {task[0]}年{task[1]}月 階級{task[4]} {task[3]}日目")
                else:
                    print(f"    - データ空: {task[0]}年{task[1]}月 階級{task[4]} {task[3]}日目")
            except Exception as e:
                print(f"    - エラー発生: {task} -> {e}")
                
    print("データ取得が完了しました。")
    conn.close()

if __name__ == "__main__":
    main()