import asyncio
import json
import time
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
    noise_psk: Optional[str] = esph.get("encryption_key")  # None => no encryption

    client = APIClient(
        address=esph["host"],
        port=int(esph.get("port", 6053)),
        password=None,
        noise_psk=noise_psk,
    )
    await client.connect(login=True)

    if STATE["switch_key"] is None:
        STATE["switch_key"] = await get_switch_key(client, name_contains)

    await client.switch_command(STATE["switch_key"], turn_on)
    await client.disconnect()

def infer_should_alarm(v: float, model: dict):
    """
    Returns: (should_alarm, z, ewma)
    EWMA + z-score + hysteresis + hold time.
    """
    alpha = float(model.get("alpha", 0.2))
    mu = float(model["mean"])
    std = float(model["std"])
    z_on = float(model.get("z_on", 3.0))
    z_off = float(model.get("z_off", 2.0))
    hold = float(model.get("hold_seconds", 30))

    now = time.time()
    in_hold = (now - STATE["last_change"]) < hold

    # EWMA update
    if STATE["ewma"] is None:
        STATE["ewma"] = v
    else:
        STATE["ewma"] = alpha * v + (1 - alpha) * STATE["ewma"]

    z = abs((STATE["ewma"] - mu) / (std if std != 0 else 1e-6))

    # hysteresis decision
    if STATE["alarm"]:
        should_alarm = not ((z <= z_off) and (not in_hold))
    else:
        should_alarm = (z >= z_on)

    return should_alarm, z, STATE["ewma"]

async def handle_value(v: float, model: dict, conf: dict):
    should_alarm, z, ewma = infer_should_alarm(v, model)

    # ✅ 每筆都印決策
    print(
        f"[DECISION] ppm={v:.3f} ewma={ewma:.3f} z={z:.2f} alarm={STATE['alarm']} -> {should_alarm}",
        flush=True
    )

    # 只有狀態改變才做控制
    if should_alarm != STATE["alarm"]:
        STATE["alarm"] = should_alarm
        STATE["last_change"] = time.time()

        esph = conf["esphome"]
        name_contains = esph.get("switch_name_contains", "Fan Relay")

        try:
            await esphome_set_switch(esph, name_contains, should_alarm)
            print(f"[ACTION] alarm={should_alarm} (switch='{name_contains}')", flush=True)
        except Exception as e:
            print("[ERROR] ESPHome control failed:", repr(e), flush=True)
            # 讓下次重新搜尋 switch key
            STATE["switch_key"] = None

def main():
    print("[BOOT] Edge AI Gateway starting...", flush=True)
    print(f"[BOOT] reading {MODEL_PATH} and {CONF_PATH}", flush=True)

    model = load_json(MODEL_PATH)
    conf  = load_json(CONF_PATH)
    print(f"[MODEL] mean={model.get('mean')} std={model.get('std')} alpha={model.get('alpha')} "
      f"z_on={model.get('z_on')} z_off={model.get('z_off')} hold={model.get('hold_seconds')}", flush=True)
    mqtt_conf = conf["mqtt"]
    broker = mqtt_conf["broker"]
    port   = int(mqtt_conf.get("port", 1883))
    topic  = str(mqtt_conf["topic"]).strip()
    username = mqtt_conf.get("username")
    password = mqtt_conf.get("password")

    print(f"[MQTT] broker={broker} port={port} topic={topic}", flush=True)

    loop = asyncio.get_event_loop()

    def on_connect(client, userdata, flags, rc, *args, **kwargs):
        print("[MQTT] connected rc=", rc, "sub=", topic, flush=True)
        client.subscribe(topic, qos=0)

    def on_disconnect(client, userdata, rc, *args, **kwargs):
        print("[MQTT] disconnected rc=", rc, flush=True)

    def on_log(client, userdata, level, buf):
        print("[MQTT-LOG]", buf, flush=True)

    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", errors="ignore").strip()
            v = float(payload)
            print(f"[MQTT] ppm={v}", flush=True)
            loop.create_task(handle_value(v, model, conf))
        except Exception as e:
            print("[MQTT] bad payload:", repr(e), "raw=", msg.payload[:80], flush=True)

    c = mqtt.Client()  # DeprecationWarning 可先忽略，功能正常
    c.on_connect = on_connect
    c.on_disconnect = on_disconnect
    # c.on_log = on_log
    c.on_message = on_message

    if username and password:
        c.username_pw_set(username, password)

    try:
        c.connect(broker, port, keepalive=60)
        print("[MQTT] connect() called", flush=True)
    except Exception as e:
        print("[MQTT] connect failed:", repr(e), flush=True)
        raise

    c.loop_forever()

if __name__ == "__main__":
    main()