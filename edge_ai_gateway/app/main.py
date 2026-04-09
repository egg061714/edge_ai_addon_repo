import asyncio
import json
import time
import threading
from typing import Optional
from collections import deque

import numpy as np
import joblib
import paho.mqtt.client as mqtt
from aioesphomeapi import APIClient

# =========================================
# 1. 檔案路徑與全域設定 (配合昨天匯出的 Hybrid 模型)
# =========================================
MODEL_IFOREST_PATH = "/share/edge_ai_gateway/pi_model_iforest.joblib"
MODEL_ZSCORE_PATH  = "/share/edge_ai_gateway/pi_model_zscore_params.joblib"
CONF_PATH          = "/share/edge_ai_gateway/runtime_config.json"

FEATURE_COLS = ["temperature", "humidity", "mq5", "pm25"]
WINDOW_SIZE = 10

# --- 新增：用來儲存各個 ESP32 傳來的最新數值 ---
LATEST_SENSOR_DATA = {
    "temperature": 25.0,  # 給定合理的初始安全值
    "humidity": 50.0,
    "mq5": 0.0,
    "pm25": 0.0
}

STATE = {
    "buffer": deque(maxlen=WINDOW_SIZE),
    "alarm": False,
    "last_change": 0.0,
    "switch_key": None,
}

# 全域模型變數
models = {
    "iforest": None,
    "zscore": None
}

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ai_models():
    """載入 Scikit-learn 模型與 Z-score 參數"""
    print(f"[BOOT] 載入 AI 模型中...", flush=True)
    models["iforest"] = joblib.load(MODEL_IFOREST_PATH)
    models["zscore"] = joblib.load(MODEL_ZSCORE_PATH)
    print(f"[BOOT] 模型載入完成！", flush=True)

# =========================================
# 2. ESPHome 控制邏輯 (保留你原本的優秀設計)
# =========================================
async def get_switch_key(client: APIClient, name_contains: str) -> int:
    entities, _services = await client.list_entities_services()
    for e in entities:
        name = (getattr(e, "name", "") or "").lower()
        obj  = (getattr(e, "object_id", "") or "").lower()
        if name_contains.lower() in name or name_contains.lower() in obj:
            key = getattr(e, "key", None)
            if key is not None: return key
    raise RuntimeError(f"Switch not found: '{name_contains}'")

async def esphome_set_switch(esph: dict, name_contains: str, turn_on: bool):
    noise_psk: Optional[str] = esph.get("encryption_key")
    client = APIClient(address=esph["host"], port=int(esph.get("port", 6053)), password=None, noise_psk=noise_psk)
    
    res = client.connect(login=True)
    if asyncio.iscoroutine(res): await res

    if STATE["switch_key"] is None:
        STATE["switch_key"] = await get_switch_key(client, name_contains)

    try:
        res = client.switch_command(STATE["switch_key"], turn_on)
        if asyncio.iscoroutine(res): await res
    except TypeError:
        res = client.switch_command(key=STATE["switch_key"], state=turn_on)
        if asyncio.iscoroutine(res): await res

    res = client.disconnect()
    if asyncio.iscoroutine(res): await res

# =========================================
# 3. 邊緣推論核心 (特徵提取 + Hybrid 融合)
# =========================================
def extract_robust_features(window_data):
    """將 10 筆資料轉換為模型需要的 16 維特徵 (當下, 平均, 波動, 趨勢)"""
    window = np.array(window_data) # shape: (10, 4)
    curr_val = window[-1]
    win_mean = np.mean(window, axis=0)
    win_std = np.std(window, axis=0)
    win_trend = window[-1] - window[0]
    
    # 回傳 shape: (1, 16)
    return np.concatenate([curr_val, win_mean, win_std, win_trend]).reshape(1, -1)

def infer_hybrid_model(current_vals, window_data):
    """
    執行 Hybrid 推論：
    1. Z-score 抓瞬間突波 (單點)
    2. iForest 抓趨勢脈絡 (區塊)
    """
    # --- 1. Z-score 檢測 ---
    z_alarm = False
    for i, col in enumerate(FEATURE_COLS):
        val = current_vals[i]
        params = models["zscore"][col]
        z_score = abs((val - params["mean"]) / params["std"])
        if z_score > params["threshold"]:
            z_alarm = True
            print(f"[Z-SCORE 觸發] {col} 數值異常! (z={z_score:.2f})")
            break # 只要有一個維度突破天際就報警
            
    # --- 2. Isolation Forest 檢測 ---
    if_alarm = False
    X_features = extract_robust_features(window_data)
    pred = models["iforest"].predict(X_features)
    if pred[0] == -1:
        if_alarm = True
        print(f"[IFOREST 觸發] 發現區塊/趨勢異常!")
        
    # --- 3. 邏輯融合 (Logical OR) ---
    return z_alarm or if_alarm

async def handle_sensor_data(current_vals: list, conf: dict):
    # 將新資料推入滑動視窗
    STATE["buffer"].append(current_vals)
    
    # 如果資料還沒收滿 Window Size (10筆)，先不推論
    if len(STATE["buffer"]) < WINDOW_SIZE:
        print(f"[BUFFER] 收集資料中... ({len(STATE['buffer'])}/{WINDOW_SIZE})")
        return

    # 進行推論
    is_anomaly = infer_hybrid_model(current_vals, STATE["buffer"])
    
    # 加入原有的防抖/遲滯邏輯 (Hold Seconds)
    hold = float(conf.get("hold_seconds", 5))
    now = time.time()
    in_hold = (now - STATE["last_change"]) < hold
    
    # 決定是否真正要觸發警報
    if STATE["alarm"]:
        # 如果正在警報中，必須等異常消失「且」過了 hold 時間才能解除
        should_alarm = not (not is_anomaly and not in_hold)
    else:
        # 如果沒在警報，一發現異常立刻觸發
        should_alarm = is_anomaly

    print(f"[DECISION] AI判定異常={is_anomaly} | 最終繼電器狀態={should_alarm} | in_hold={in_hold}", flush=True)

    # 狀態改變才發送控制指令
    if should_alarm == STATE["alarm"]: return

    esph = conf["esphome"]
    name_contains = esph.get("switch_name_contains", "Fan Relay")

    try:
        await esphome_set_switch(esph, name_contains, should_alarm)
        STATE["alarm"] = should_alarm
        STATE["last_change"] = time.time()
        print(f"[ACTION] 成功切換警報狀態: {should_alarm}", flush=True)
    except Exception as e:
        print("[ERROR] ESPHome 控制失敗:", repr(e), flush=True)
        STATE["switch_key"] = None

async def periodic_inference_loop(conf: dict):
    """定頻採樣迴圈：每隔 N 秒將最新的感測器數值截圖，送入 AI 推論"""
    interval = float(conf.get("inference_interval_seconds", 5.0)) # 預設每 5 秒推論一次
    print(f"[SYSTEM] 啟動定頻推論迴圈，每 {interval} 秒執行一次...", flush=True)
    
    while True:
        await asyncio.sleep(interval)
        
        # 依照 FEATURE_COLS 順序，從快取中拿出最新數值
        current_vals = [float(LATEST_SENSOR_DATA[col]) for col in FEATURE_COLS]
        
        # 丟進我們寫好的處理邏輯
        await handle_sensor_data(current_vals, conf)


# =========================================
# 4. MQTT 接收與啟動程序
# =========================================
def start_async_loop():
    loop = asyncio.new_event_loop()
    def runner():
        asyncio.set_event_loop(loop)
        loop.run_forever()
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return loop

def main():
    print("[BOOT] Edge AI Gateway starting...", flush=True)
    
    # 載入設定與模型
    conf = load_json(CONF_PATH)
    load_ai_models()

    mqtt_conf = conf["mqtt"]
    broker = mqtt_conf["broker"]
    port   = int(mqtt_conf.get("port", 1883))
    topic  = str(mqtt_conf["topic"]).strip()

    print(f"[MQTT] broker={broker} topic={topic}", flush=True)
    loop = start_async_loop()

    def on_connect(client, userdata, flags, rc, *args, **kwargs):
        print("[MQTT] 已連線, 開始訂閱:", topic, flush=True)
        client.subscribe(topic, qos=0)

    def on_message(client, userdata, msg):
        try:
            # 假設 ESP32 傳來的格式可能是不完整的 JSON
            # 例如 ESP32-A 傳: {"temperature": 26.5, "humidity": 65}
            # 例如 ESP32-B 傳: {"mq5": 150}
            payload_str = msg.payload.decode("utf-8", errors="ignore").strip()
            data_dict = json.loads(payload_str)
            
            # --- 關鍵修改：只更新有收到的欄位 ---
            updated_keys = []
            for key in data_dict.keys():
                if key in LATEST_SENSOR_DATA:
                    LATEST_SENSOR_DATA[key] = float(data_dict[key])
                    updated_keys.append(key)
            
            if updated_keys:
                print(f"[MQTT 更新快取] {updated_keys} -> {payload_str}")

        except json.JSONDecodeError:
            print("[MQTT] Payload 不是有效的 JSON 格式", flush=True)
        except Exception as e:
            print("[MQTT] 處理訊息時發生錯誤:", repr(e), flush=True)

    c = mqtt.Client()
    c.on_connect = on_connect
    c.on_message = on_message
    
    if mqtt_conf.get("username"):
        c.username_pw_set(mqtt_conf["username"], mqtt_conf.get("password"))

    c.connect(broker, port, keepalive=60)
    c.loop_forever()

if __name__ == "__main__":
    main()