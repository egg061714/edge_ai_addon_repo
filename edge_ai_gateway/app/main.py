import asyncio
import json
import time
import threading
import traceback
from typing import Optional
from collections import deque

import numpy as np
import joblib
import paho.mqtt.client as mqtt
from aioesphomeapi import APIClient

# =========================================
# 1. 檔案路徑與全域設定
# =========================================
MODEL_IFOREST_PATH = "/share/edge_ai_gateway/pi_model_iforest.joblib"
MODEL_ZSCORE_PATH  = "/share/edge_ai_gateway/pi_model_zscore_params.joblib"
CONF_PATH          = "/share/edge_ai_gateway/runtime_config.json"

# 模型訓練時的原始欄位 (對應 Z-score 字典的 Key)
TRAINED_COLS = ["temperature", "humidity", "light", "voltage"]
# 實際部署時的感測器映射順序
# [temp->temp, humi->humi, light->mq5, voltage->dust_ratio]
FEATURE_COLS = ["temperature", "humidity", "mq5", "dust_ratio"]
WINDOW_SIZE = 10

LATEST_SENSOR_DATA = {
    "temperature": 25.0,
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

models = {"iforest": None, "zscore": None}

# =========================================
# 2. 工具函式
# =========================================
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ai_models():
    print(f"[BOOT] 載入 AI 模型中...", flush=True)
    models["iforest"] = joblib.load(MODEL_IFOREST_PATH)
    models["zscore"] = joblib.load(MODEL_ZSCORE_PATH)
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

# =========================================
# 3. 邊緣推論核心
# =========================================
def extract_robust_features(window_data):
    """將視窗資料轉換為 16 維特徵 (4感測器 * 4統計量)"""
    window = np.array(window_data, dtype=np.float64)
    curr_val = window[-1]
    win_mean = np.mean(window, axis=0)
    win_std = np.std(window, axis=0)
    win_trend = window[-1] - window[0]
    return np.concatenate([curr_val, win_mean, win_std, win_trend]).reshape(1, -1)

def infer_hybrid_model_with_root_cause(current_vals, window_data):
    # --- 1. Z-score 診斷 (映射至訓練時的 Key) ---
    for i, trained_key in enumerate(TRAINED_COLS):
        if trained_key in models["zscore"]:
            params = models["zscore"][trained_key]
            z = abs((current_vals[i] - params["mean"]) / (params["std"] + 1e-6))
            if z > params["threshold"]:
                return True, FEATURE_COLS[i] # 回傳現實中的感測器名稱

    # --- 2. IForest 診斷 ---
    X_features = extract_robust_features(window_data)
    if models["iforest"].predict(X_features)[0] == -1:
        # 找出偏離平均最嚴重的作為根因
        window_np = np.array(window_data)
        deviation = np.abs(current_vals - np.mean(window_np, axis=0)) / (np.std(window_np, axis=0) + 1e-6)
        return True, FEATURE_COLS[np.argmax(deviation)]
        
    return False, None

async def handle_sensor_data(current_vals: list, conf: dict):
    try:
        # 如果視窗還沒滿，不執行推論
        if len(STATE["buffer"]) < WINDOW_SIZE:
            return

        print(f"[推論啟動] 數據: {current_vals}", flush=True)
        is_anomaly, reason_sensor = infer_hybrid_model_with_root_cause(current_vals, STATE["buffer"])
        print(f"[AI 結果] 異常: {is_anomaly}, 根因: {reason_sensor}", flush=True)

        # 防抖邏輯
        hold = float(conf.get("hold_seconds", 5))
        now = time.time()
        in_hold = (now - STATE["last_change"]) < hold
        
        # 狀態決策
        should_alarm = is_anomaly if not STATE["alarm"] else not (not is_anomaly and not in_hold)

        if should_alarm != STATE["alarm"]:
            # 根據原因選擇控制裝置
            target = "General Alarm"
            if reason_sensor == "mq5": target = "Gas Valve"
            elif reason_sensor in ["temperature", "humidity"]: target = "Fan Relay"

            # await esphome_set_switch(conf["esphome"], target, should_alarm)
            STATE["alarm"] = should_alarm
            STATE["last_change"] = now
            print(f"[ACTION] 狀態改變為: {should_alarm} (原因: {reason_sensor})", flush=True)

    except Exception as e:
        print(f"[CRITICAL ERROR] handle_sensor_data 崩潰: {repr(e)}", flush=True)
        traceback.print_exc()

async def periodic_inference_loop(conf: dict):
    interval = float(conf.get("inference_interval_seconds", 5.0))
    print(f"[SYSTEM] 啟動定頻推論，週期: {interval}s", flush=True)
    while True:
        await asyncio.sleep(interval)
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
                if k in LATEST_SENSOR_DATA:
                    LATEST_SENSOR_DATA[k] = float(v)
                    updated.append(k)
            
            if updated:
                # 每當資料更新，就塞入滑動視窗
                snap = [float(LATEST_SENSOR_DATA[c]) for c in FEATURE_COLS]
                STATE["buffer"].append(snap)
                print(f"[MQTT 更新] {updated} | Buffer: {len(STATE['buffer'])}/10", flush=True)

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