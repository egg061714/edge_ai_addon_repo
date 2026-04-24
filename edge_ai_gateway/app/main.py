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

FEATURE_COLS = ["temperature", "humidity", "mq5", "dust_ratio"]
WINDOW_SIZE = 10

# --- 新增：用來儲存各個 ESP32 傳來的最新數值 ---
LATEST_SENSOR_DATA = {
    "temperature": 25.0,  # 給定合理的初始安全值
    "humidity": 50.0,
    "mq5": 0.0,
    "dust_ratio": 0.0
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

    print(f"Z-score 欄位列表: {models['zscore'].keys()}")
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
    try:
        window = np.array(window_data, dtype=np.float64) 
        # 如果這裡噴錯，代表 window_data 內容物有問題（例如有 None 或長度不一）
        
        curr_val = window[-1]
        win_mean = np.mean(window, axis=0)
        win_std = np.std(window, axis=0)
        win_trend = window[-1] - window[0]
        
        res = np.concatenate([curr_val, win_mean, win_std, win_trend]).reshape(1, -1)
        return res
    except Exception as e:
        print(f"[FEATURE ERROR] 矩陣運算失敗: {repr(e)}", flush=True)
        return None

def infer_hybrid_model_with_root_cause(current_vals, window_data):
    # --- 1. Z-score 診斷 (最直接的根因) ---
    for i, col in enumerate(FEATURE_COLS):
        if col not in models["zscore"]:
            print(f"ERROR: Z-score 參數缺少欄位 {col}", flush=True)
            return False, None
        
        params = models["zscore"][col]
        z = abs((current_vals[i] - params["mean"]) / params["std"])
        if z > params["threshold"]:
            return True, col # 直接抓到現行犯
            
    # --- 2. IForest 診斷 (脈絡異常) ---
    print("DEBUG: 開始 IForest 特徵提取...", flush=True)
    try:
        X_features = extract_robust_features(window_data)
        print(f"DEBUG: X_features shape = {X_features.shape}", flush=True)
        
        pred = models["iforest"].predict(X_features)[0]
        print(f"DEBUG: IForest 預測結果 = {pred}", flush=True)
        
        if pred == -1:
            # 原本的 root cause 邏輯...
            return True, "Context Anomaly"
    except Exception as e:
        print(f"ERROR: IForest 推論失敗: {repr(e)}", flush=True)

    return False, None

async def handle_sensor_data(current_vals: list, conf: dict):
    # 將新資料推入滑動視窗
    # STATE["buffer"].append(current_vals)
    print(f"[推論啟動] 準備餵入模型: {current_vals} | 視窗長度: {len(STATE['buffer'])}", flush=True)
    # 如果資料還沒收滿 Window Size (10筆)，先不推論
    if len(STATE["buffer"]) < WINDOW_SIZE:
        print(f"[BUFFER] 收集資料中... ({len(STATE['buffer'])}/{WINDOW_SIZE})")
        return

    # 進行推論
    is_anomaly, reason_sensor = infer_hybrid_model_with_root_cause(current_vals, STATE["buffer"])
    
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
    if should_alarm and is_anomaly:
        print(f"[判定] 異常感測器: {reason_sensor} | 數值: {current_vals}")
        
        # 這裡就是你的決策映射 (Decision Mapping)
        target_switch = "General Alarm" # 預設
        if reason_sensor == "mq5":
            target_switch = "Gas Valve"
        elif reason_sensor in ["temperature", "humidity"]:
            target_switch = "Fan Relay"
        elif reason_sensor == "dust_ratio":
            target_switch = "Air Purifier"

        # 如果狀態改變，才去控制對應的裝置
        if should_alarm != STATE["alarm"]:
            await esphome_set_switch(conf["esphome"], target_switch, True)

    try:
        await esphome_set_switch(conf["esphome"], target_switch, should_alarm)
        STATE["alarm"] = should_alarm
        STATE["last_change"] = time.time()
        print(f"[ACTION] 成功切換警報狀態: {should_alarm}", flush=True)
    except Exception as e:
        print("[ERROR] ESPHome 控制失敗:", repr(e), flush=True)
        STATE["switch_key"] = None

async def periodic_inference_loop(conf: dict):
    interval = float(conf.get("inference_interval_seconds", 5.0))
    while True:
        await asyncio.sleep(interval)
        
        # 檢查視窗夠不夠長
        if len(STATE["buffer"]) >= WINDOW_SIZE:
            # 取得當前最新的視窗快照進行 AI 推論
            current_vals = list(STATE["buffer"])[-1] 
            await handle_sensor_data(current_vals, conf)
        else:
            print(f"[SYSTEM] 等待資料存滿中... ({len(STATE['buffer'])}/10)", flush=True)


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
    asyncio.run_coroutine_threadsafe(periodic_inference_loop(conf), loop)

    def on_connect(client, userdata, flags, rc, *args, **kwargs):
        print("[MQTT] 已連線, 開始訂閱:", topic, flush=True)
        client.subscribe(topic, qos=0)

    def on_message(client, userdata, msg):
        # print(f"DEBUG: 收到訊息了! Topic={msg.topic} Payload={msg.payload}", flush=True)
        try:
            payload_str = msg.payload.decode("utf-8", errors="ignore").strip()
            
            # 嘗試解析 JSON
            try:
                data_dict = json.loads(payload_str)
                # 如果解析出來不是 dict (例如是純數字 0.52)
                if not isinstance(data_dict, dict):
                    # 假設這個純數字屬於某個預設感測器 (例如 mq5)
                    # 你可以根據 topic 名稱來判斷，或者預設給一個欄位
                    data_dict = {"mq5": float(payload_str)} 
            except json.JSONDecodeError:
                # 如果連 JSON 都不是，嘗試直接轉 float
                data_dict = {"mq5": float(payload_str)}

            # --- 更新快取邏輯 ---
            # --- 更新快取邏輯 (建議修改版) ---
            updated_keys = []
            for key, value in data_dict.items():
                try:
                    # 直接寫入或更新，不檢查 key 是否已存在
                    LATEST_SENSOR_DATA[key] = float(value)
                    updated_keys.append(key)
                except (ValueError, TypeError):
                    continue 
            
            if updated_keys:
                print(f"[MQTT 更新快取] {updated_keys} -> {data_dict}")
                
                # 這裡補上：只要有任何資料更新，就把當前的四個數值狀態存入 buffer
                current_snap = [float(LATEST_SENSOR_DATA.get(col, 0.0)) for col in FEATURE_COLS]
                STATE["buffer"].append(current_snap)
                
                # 這樣你就會看到 buffer 快速跳動了 (1/10, 2/10...)
                if len(STATE["buffer"]) < WINDOW_SIZE:
                    print(f"[BUFFER] 視窗累積中: {len(STATE['buffer'])}/10", flush=True)

        except Exception as e:
            print(f"[MQTT 錯誤] 無法處理此 Payload: {payload_str} | 錯誤: {repr(e)}", flush=True)

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    c.on_connect = on_connect
    c.on_message = on_message
    
    if mqtt_conf.get("username"):
        c.username_pw_set(mqtt_conf["username"], mqtt_conf.get("password"))

    c.connect(broker, port, keepalive=60)
    c.loop_forever()

if __name__ == "__main__":
    main()