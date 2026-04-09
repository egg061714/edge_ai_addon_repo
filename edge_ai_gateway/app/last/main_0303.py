import asyncio
import json
import time
import threading
from typing import Optional

import paho.mqtt.client as mqtt
from aioesphomeapi import APIClient

MODEL_PATH = "/share/edge_ai_gateway/model.json"
CONF_PATH  = "/share/edge_ai_gateway/runtime_config.json"

STATE = {
    "ewma": None,
    "alarm": False,
    "last_change": 0.0,
    "switch_key": None,
}

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

async def get_switch_key(client: APIClient, name_contains: str) -> int:
    entities, _services = await client.list_entities_services()
    for e in entities:
        name = (getattr(e, "name", "") or "").lower()
        obj  = (getattr(e, "object_id", "") or "").lower()
        if name_contains.lower() in name or name_contains.lower() in obj:
            key = getattr(e, "key", None)
            if key is not None:
                return key
    raise RuntimeError(f"Switch not found (name contains '{name_contains}')")

async def esphome_set_switch(esph: dict, name_contains: str, turn_on: bool):
    """
   兼容不同 aioesphomeapi 版本：
    - switch_command 可能是 coroutine，也可能是同步回 None
    - disconnect 也可能是 coroutine，也可能是同步
    """
    noise_psk: Optional[str] = esph.get("encryption_key")  # None => no encryption

    client = APIClient(
        address=esph["host"],
        port=int(esph.get("port", 6053)),
        password=None,
        noise_psk=noise_psk,
    )

    # connect
    res = client.connect(login=True)
    if asyncio.iscoroutine(res):
        await res

    # cache switch key
    if STATE["switch_key"] is None:
        STATE["switch_key"] = await get_switch_key(client, name_contains)

    # switch command (兼容 await / non-await)
    try:
        res = client.switch_command(STATE["switch_key"], turn_on)
        if asyncio.iscoroutine(res):
            await res
    except TypeError:
        # 有些版本參數名可能是 state
        res = client.switch_command(key=STATE["switch_key"], state=turn_on)
        if asyncio.iscoroutine(res):
            await res

    # disconnect (兼容 await / non-await)
    res = client.disconnect()
    if asyncio.iscoroutine(res):
        await res

def infer_should_alarm(v: float, model: dict):
    alpha = float(model.get("alpha", 0.3))
    mu = float(model["mean"])
    std = float(model["std"])
    z_on = float(model.get("z_on", 1.3))
    z_off = float(model.get("z_off", 0.8))
    hold = float(model.get("hold_seconds", 5))

    now = time.time()
    in_hold = (now - STATE["last_change"]) < hold

    if STATE["ewma"] is None:
        STATE["ewma"] = v
    else:
        STATE["ewma"] = alpha * v + (1 - alpha) * STATE["ewma"]

    z = abs((STATE["ewma"] - mu) / (std if std != 0 else 1e-6))

    if STATE["alarm"]:
        should_alarm = not ((z <= z_off) and (not in_hold))
    else:
        should_alarm = (z >= z_on)

    return should_alarm, z, STATE["ewma"]

async def handle_value(v: float, model: dict, conf: dict):
    should_alarm, z, ewma = infer_should_alarm(v, model)

    print(
        f"[DECISION] ppm={v:.3f} ewma={ewma:.3f} z={z:.2f} alarm={STATE['alarm']} -> {should_alarm}",
        flush=True
    )

    if should_alarm == STATE["alarm"]:
        return

    esph = conf["esphome"]
    name_contains = esph.get("switch_name_contains", "Fan Relay")

    try:
        await esphome_set_switch(esph, name_contains, should_alarm)

        # ✅ 只有控制成功才更新狀態
        STATE["alarm"] = should_alarm
        STATE["last_change"] = time.time()

        print(f"[ACTION] alarm={should_alarm} (switch='{name_contains}')", flush=True)

    except Exception as e:
        print("[ERROR] ESPHome control failed:", repr(e), flush=True)
        STATE["switch_key"] = None

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
    print(f"[BOOT] reading {MODEL_PATH} and {CONF_PATH}", flush=True)

    model = load_json(MODEL_PATH)
    conf  = load_json(CONF_PATH)

    print(
        f"[MODEL] mean={model.get('mean')} std={model.get('std')} alpha={model.get('alpha')} "
        f"z_on={model.get('z_on')} z_off={model.get('z_off')} hold={model.get('hold_seconds')}",
        flush=True
    )

    mqtt_conf = conf["mqtt"]
    broker = mqtt_conf["broker"]
    port   = int(mqtt_conf.get("port", 1883))
    topic  = str(mqtt_conf["topic"]).strip()
    username = mqtt_conf.get("username")
    password = mqtt_conf.get("password")

    print(f"[MQTT] broker={broker} port={port} topic={topic}", flush=True)

    loop = start_async_loop()

    def on_connect(client, userdata, flags, rc, *args, **kwargs):
        print("[MQTT] connected rc=", rc, "sub=", topic, flush=True)
        client.subscribe(topic, qos=0)

    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", errors="ignore").strip()
            v = float(payload)
            print(f"[MQTT] ppm={v}", flush=True)

            asyncio.run_coroutine_threadsafe(handle_value(v, model, conf), loop)

        except Exception as e:
            print("[MQTT] bad payload:", repr(e), "raw=", msg.payload[:80], flush=True)

    c = mqtt.Client()  # DeprecationWarning 可先忽略
    c.on_connect = on_connect
    c.on_message = on_message

    if username and password:
        c.username_pw_set(username, password)

    c.connect(broker, port, keepalive=60)
    print("[MQTT] connect() called", flush=True)
    c.loop_forever()

if __name__ == "__main__":
    main()