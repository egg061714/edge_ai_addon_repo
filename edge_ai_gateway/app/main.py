import asyncio
import json
import time
import threading
import traceback
from typing import Optional
from collections import deque
import psutil
import os
import pandas as pd
import numpy as np
import joblib
import paho.mqtt.client as mqtt
from aioesphomeapi import APIClient

# =========================================
# 1. 檔案路徑與全域設定
# =========================================
MODEL_IFOREST_PATH = "/share/edge_ai_gateway/pi_model_second_iforest.joblib"
MODEL_ZSCORE_PATH  = "/share/edge_ai_gateway/pi_model_second_zscore_params.joblib"
CONF_PATH          = "/share/edge_ai_gateway/runtime_config.json"
MODEL_IF_THRESHOLD_PATH = "/share/edge_ai_gateway/pi_model_second_if_threshold.joblib"
FEATURE_COLS = []
TRAINED_COLS = []
LAST_CONF_TIME = 0.0  # 用於偵測檔案變動
WINDOW_SIZE = 10
LATEST_SENSOR_DATA = {} # 改為動態初始化
LATEST_CONTEXT_DATA = {
    "motion": 0.0
}

STATE = {
    "buffer": deque(maxlen=WINDOW_SIZE),
    "alarm": False,
    "last_change": 0.0,
    "switch_key": None,
    "total_count": 0
}

models = {"iforest": None, "zscore": None,"if_threshold": None}

# =========================================
# 2. 工具函式
# =========================================
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_and_apply_config():
    global FEATURE_COLS, TRAINED_COLS, LAST_CONF_TIME, LATEST_SENSOR_DATA
    try:
        current_time = os.path.getmtime(CONF_PATH)
        if current_time > LAST_CONF_TIME:
            print(f"[CONFIG] 偵測到配置檔更新，載入中...", flush=True)
            conf = load_json(CONF_PATH)
            
            # 從 ai_model 區塊讀取映射 (假設你 JSON 已補上 ai_model 區段)
            mapping = conf.get("ai_model", {}).get("feature_mapping", {})
            if mapping:
                # 按照 Index 排序，確保輸入模型的維度正確
                sorted_mapping = sorted(mapping.items(), key=lambda x: x[1])
                FEATURE_COLS = [item[0] for item in sorted_mapping]
                # 這裡的 TRAINED_COLS 通常是模型固定的，可以視需求對應
                TRAINED_COLS = ["temperature", "humidity", "light", "voltage"] 
                
                # 初始化感測器快取
                for col in FEATURE_COLS:
                    if col not in LATEST_SENSOR_DATA:
                        LATEST_SENSOR_DATA[col] = 0.0
                
                print(f"[CONFIG] 映射更新完成: {FEATURE_COLS}", flush=True)
                LAST_CONF_TIME = current_time
            return conf
    except Exception as e:
        print(f"[CONFIG ERROR] 載入失敗: {e}", flush=True)
    return None



def load_ai_models():
    print(f"[BOOT] 載入 AI 模型中...", flush=True)
    models["iforest"] = joblib.load(MODEL_IFOREST_PATH)
    models["zscore"] = joblib.load(MODEL_ZSCORE_PATH)
    models["if_threshold"] = joblib.load(MODEL_IF_THRESHOLD_PATH)
    print(f"[BOOT] 模型載入完成！Z-score 欄位: {list(models['zscore'].keys())}", flush=True)

async def get_switch_key(client: APIClient, name_contains: str) -> int:
    entities, _ = await client.list_entities_services()
    for e in entities:
        name = (getattr(e, "name", "") or "").lower()
        obj  = (getattr(e, "object_id", "") or "").lower()
        if name_contains.lower() in name or name_contains.lower() in obj:
            key = getattr(e, "key", None)
            if key is not None: return key
    raise RuntimeError(f"Switch not found: '{name_contains}'")

async def esphome_set_switch(esph_conf: dict, name_contains: str, turn_on: bool):
    """執行 ESPHome 繼電器控制"""
    client = APIClient(
        address=esph_conf["host"], 
        port=int(esph_conf.get("port", 6053)), 
        password=None, 
        noise_psk=esph_conf.get("encryption_key")
    )
    try:
        await client.connect(login=True)
        if STATE["switch_key"] is None:
            STATE["switch_key"] = await get_switch_key(client, name_contains)
        
        await client.switch_command(STATE["switch_key"], turn_on)
        print(f"[ESPHome] 成功切換 {name_contains} -> {turn_on}", flush=True)
    finally:
        await client.disconnect()

def get_system_usage():
    # 獲取當前進程 (Gateway AI) 的資源佔用
    process = psutil.Process(os.getpid())
    cpu_usage = psutil.cpu_percent(interval=None) # 全域 CPU 使用率
    ram_usage = process.memory_info().rss / (1024 * 1024) # 本程式佔用 RAM (MB)
    system_ram = psutil.virtual_memory().percent # 系統總記憶體使用率
    
    return cpu_usage, ram_usage, system_ram
# =========================================
# 3. 邊緣推論核心
# =========================================
# def extract_robust_features(window_data):
#     """將視窗資料轉換為 16 維特徵 (4感測器 * 4統計量)"""
#     window = np.array(window_data, dtype=np.float64)
#     curr_val = window[-1]
#     win_mean = np.mean(window, axis=0)
#     win_std = np.std(window, axis=0)
#     win_trend = window[-1] - window[0]
#     return np.concatenate([curr_val, win_mean, win_std, win_trend]).reshape(1, -1)



def extract_robust_features(window_data):
    """
    與訓練端 create_robust_window_features() 對齊
    4 感測器時輸出 30 維：
    curr_val, win_mean, win_std, win_trend,
    diff_mean, slopes, cum_dev, time_sin, time_cos
    """

    window = np.array(window_data, dtype=np.float64)

    curr_val = window[-1]
    win_mean = np.mean(window, axis=0)
    win_std = np.std(window, axis=0)
    win_trend = window[-1] - window[0]

    diff = np.diff(window, axis=0)
    diff_mean = np.mean(diff, axis=0)

    x = np.arange(window.shape[0])
    slopes = []

    for j in range(window.shape[1]):
        y = window[:, j]
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)

    slopes = np.array(slopes)

    deviation = window - np.mean(window, axis=0)
    cum_dev = np.sum(deviation, axis=0)

    now = pd.Timestamp.now()
    seconds_in_day = now.hour * 3600 + now.minute * 60 + now.second

    time_sin = np.sin(2 * np.pi * seconds_in_day / 86400)
    time_cos = np.cos(2 * np.pi * seconds_in_day / 86400)

    combined = np.concatenate([
        curr_val,
        win_mean,
        win_std,
        win_trend,
        diff_mean,
        slopes,
        cum_dev,
        [time_sin, time_cos]
    ])

    return combined.reshape(1, -1)

def infer_hybrid_model_with_root_cause(current_vals, window_data):

    # =====================================
    # 1. Z-score 計算
    # =====================================
    z_scores = []

    for i, trained_key in enumerate(TRAINED_COLS):

        if trained_key not in models["zscore"]:
            z_scores.append(0.0)
            continue

        params = models["zscore"][trained_key]

        z = abs(
            (current_vals[i] - params["mean"]) /
            (params["std"] + 1e-6)
        )

        z_scores.append(z)

    z_max = max(z_scores)

    # =====================================
    # 2. 強突波異常
    # =====================================
    if z_max > 4.5:
        return True, FEATURE_COLS[np.argmax(z_scores)]

    # =====================================
    # 3. Isolation Forest score
    # =====================================
    X_features = extract_robust_features(window_data)

    if_score = -models["iforest"].decision_function(X_features)[0]

    if_threshold = models["if_threshold"]["if_threshold"]

    # =====================================
    # 4. Drift anomaly
    # =====================================
    if (if_score > if_threshold) and (z_max > 2.6):

        return True, FEATURE_COLS[np.argmax(z_scores)]

    return False, None

async def handle_sensor_data(current_vals: list, conf: dict):
    try:
        if len(STATE["buffer"]) < WINDOW_SIZE:
            return

        print(f"[推論啟動] 數據: {current_vals}", flush=True)
        is_anomaly, reason_sensor = infer_hybrid_model_with_root_cause(current_vals, STATE["buffer"])
        print(f"[AI 結果] 異常: {is_anomaly}, 根因: {reason_sensor}", flush=True)

        # ==============================
        # 第二層：Context-aware Decision
        # motion 不進模型，只作為情境輔助
        # ==============================
        motion = float(LATEST_CONTEXT_DATA.get("motion", 0.0))

        if is_anomaly:
            if motion >= 1.0:
                context_reason = "有人活動下的環境異常"
            else:
                context_reason = "無人狀態下的環境異常"
        else:
            context_reason = "正常"

        print(f"[Context] motion={motion} | 情境判斷: {context_reason}", flush=True)

        hold = float(conf.get("hold_seconds", 5))
        now = time.time()
        in_hold = (now - STATE["last_change"]) < hold

        should_alarm = is_anomaly if not STATE["alarm"] else not (not is_anomaly and not in_hold)

        if should_alarm != STATE["alarm"]:
            target = "General Alarm"

            if reason_sensor == "mq5":
                target = "Gas Valve"
            elif reason_sensor in ["temperature", "humidity"]:
                target = "Fan Relay"
            elif reason_sensor == "pm25":
                target = "Air Purifier"

            STATE["alarm"] = should_alarm
            STATE["last_change"] = now

            print(
                f"[ACTION] 狀態改變為: {should_alarm} | 原因: {reason_sensor} | Context: {context_reason}",
                flush=True
            )

    except Exception as e:
        print(f"[CRITICAL ERROR] handle_sensor_data 崩潰: {repr(e)}", flush=True)
        traceback.print_exc()

async def periodic_inference_loop(conf: dict):

    interval = float(conf.get("inference_interval_seconds", 5.0))
    print(f"[SYSTEM] 啟動定頻推論，週期: {interval}s", flush=True)
    while True:
        await asyncio.sleep(interval)
        new_conf = load_and_apply_config()
        if new_conf: conf = new_conf
        cpu_usage, ram_usage, system_ram = get_system_usage()
        print(f"[PERF] CPU: {cpu_usage}% | AI RAM: {ram_usage:.2f}MB | Sys RAM: {system_ram}%", flush=True)
        if len(STATE["buffer"]) >= WINDOW_SIZE:
            # 拍照當下快照
            current_vals = [float(LATEST_SENSOR_DATA[col]) for col in FEATURE_COLS]
            await handle_sensor_data(current_vals, conf)
        else:
            print(f"[SYSTEM] 等待資料存滿... ({len(STATE['buffer'])}/10)", flush=True)

# =========================================
# 4. MQTT 與啟動
# =========================================
def main():
    print("[BOOT] Edge AI Gateway starting...", flush=True)
    conf = load_json(CONF_PATH)
    load_ai_models()

    # 啟動非同步背景執行緒
    loop = asyncio.new_event_loop()
    def runner():
        asyncio.set_event_loop(loop)
        loop.run_forever()
    threading.Thread(target=runner, daemon=True).start()
    asyncio.run_coroutine_threadsafe(periodic_inference_loop(conf), loop)

    mqtt_conf = conf["mqtt"]
    topic = str(mqtt_conf["topic"]).strip()

    def on_connect(client, userdata, flags, rc):
        print(f"[MQTT] 已連線, 訂閱: {topic}", flush=True)
        client.subscribe(topic)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", errors="ignore"))
            if not isinstance(payload, dict): return

            # 更新 LATEST_SENSOR_DATA 快取
            updated = []
            for k, v in payload.items():

                # ==============================
                # motion：只作為情境資料，不進模型 buffer
                # ==============================
                if k == "motion":
                    if isinstance(v, bool):
                        LATEST_CONTEXT_DATA["motion"] = 1.0 if v else 0.0
                    elif isinstance(v, str):
                        LATEST_CONTEXT_DATA["motion"] = 1.0 if v.upper() in ["ON", "TRUE", "1"] else 0.0
                    else:
                        LATEST_CONTEXT_DATA["motion"] = 1.0 if float(v) > 0 else 0.0

                    updated.append(k)
                    continue

                # ==============================
                # 模型感測資料：temperature / humidity / mq5 / pm25
                # ==============================
                if k in LATEST_SENSOR_DATA:
                    LATEST_SENSOR_DATA[k] = float(v)
                    updated.append(k)
            
            if updated:
                # 每當資料更新，就塞入滑動視窗
                snap = [float(LATEST_SENSOR_DATA[c]) for c in FEATURE_COLS]
                STATE["buffer"].append(snap)
                STATE["total_count"] += 1
                
                print(f"[數據流入] 序號:#{STATE['total_count']} | 來源:{updated} | Buffer:{len(STATE['buffer'])}/10", flush=True)
        except Exception as e:
            print(f"[MQTT ERROR] {e}", flush=True)

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    if mqtt_conf.get("username"):
        c.username_pw_set(mqtt_conf["username"], mqtt_conf.get("password"))
    
    c.on_connect = on_connect
    c.on_message = on_message
    c.connect(mqtt_conf["broker"], int(mqtt_conf.get("port", 1883)))
    c.loop_forever()

if __name__ == "__main__":
    main()