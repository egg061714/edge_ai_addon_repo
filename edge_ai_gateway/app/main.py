import asyncio
import json
import time
from typing import Optional
import os
import paho.mqtt.client as mqtt
from aioesphomeapi import APIClient

broker1= "192.168.1.115"
topic1= " home/env1/sensor/mq5_voltage/state"
username1= "egg_home"
password1= "0908885645"

MODEL_PATH = "/share/edge_ai_gateway/model.json"
CONF_PATH  = "/share/edge_ai_gateway/runtime_config.json"
STATE = {
    "ewma": None,          # EWMA value
    "alarm": False,        # current alarm state
    "last_change": 0.0,    # last toggle time
    "switch_key": None,    # cached switch key
}

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

async def get_switch_key(client: APIClient, name_contains: str) -> int:
    """Find switch entity key by name/object_id substring match."""
    entities, _services = await client.list_entities_services()

    # Switch entities are in entities list; we match by name / object_id
    for e in entities:
        name = (getattr(e, "name", "") or "").lower()
        obj = (getattr(e, "object_id", "") or "").lower()
        if name_contains.lower() in name or name_contains.lower() in obj:
            key = getattr(e, "key", None)
            if key is not None:
                return key
    raise RuntimeError(f"Switch not found (name contains '{name_contains}')")

async def esphome_set_switch(esph: dict, name_contains: str, turn_on: bool):
    """Connect ESPHome device via native API and toggle a switch."""
    noise_psk: Optional[str] = esph.get("encryption_key")  # None => no encryption

    client = APIClient(
        address=esph["host"],
        port=int(esph.get("port", 6053)),
        password=None,
        noise_psk=noise_psk,
    )

    await client.connect(login=True)

    # cache switch key to avoid listing entities every time
    if STATE["switch_key"] is None:
        STATE["switch_key"] = await get_switch_key(client, name_contains)

    await client.switch_command(STATE["switch_key"], turn_on)
    await client.disconnect()

def infer_should_alarm(v: float, model: dict) -> bool:
    """
    EWMA + z-score + hysteresis + hold time.
    - Trigger when z >= z_on
    - Release when z <= z_off AND hold time passed
    """
    alpha = float(model.get("alpha", 0.2))
    mu = float(model["mean"])
    std = float(model["std"])
    z_on = float(model.get("z_on", 3.0))
    z_off = float(model.get("z_off", 2.0))
    hold = float(model.get("hold_seconds", 30))

    now = time.time()
    in_hold = (now - STATE["last_change"]) < hold

    # update ewma
    if STATE["ewma"] is None:
        STATE["ewma"] = v
    else:
        STATE["ewma"] = alpha * v + (1 - alpha) * STATE["ewma"]

    z = abs((STATE["ewma"] - mu) / (std if std != 0 else 1e-6))

    if STATE["alarm"]:
        # release condition
        if (z <= z_off) and (not in_hold):
            return False
        return True
    else:
        # trigger condition
        if z >= z_on:
            return True
        return False

async def handle_value(v: float, model: dict, conf: dict):
    should_alarm = infer_should_alarm(v, model)

    if should_alarm != STATE["alarm"]:
        STATE["alarm"] = should_alarm
        STATE["last_change"] = time.time()

        esph = conf["esphome"]
        name_contains = esph.get("switch_name_contains", "Fan Relay")

        try:
            await esphome_set_switch(esph, name_contains, should_alarm)
            print(f"[ACTION] alarm={should_alarm} ppm={v:.3f} ewma={STATE['ewma']:.3f}")
        except Exception as e:
            # If key cached but device rebooted / entity list changed, clear cache and retry once
            print("[ERROR] ESPHome control failed:", repr(e))
            STATE["switch_key"] = None

def main():

    model = load_json(MODEL_PATH)
    conf = load_json(CONF_PATH)

    mqtt_conf = conf["mqtt"]
    broker = mqtt_conf[broker1]
    port = int(mqtt_conf.get("port", 1883))
    topic = mqtt_conf[topic1]
    username = mqtt_conf.get(username1)
    password = mqtt_conf.get(password1)

    loop = asyncio.get_event_loop()

    def on_connect(client, userdata, flags, rc):
        print("[MQTT] connected rc=", rc, "sub=", topic)
        client.subscribe(topic, qos=0)

    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", errors="ignore").strip()
            v = float(payload)  # your payload is plain number like "5.08"
        except Exception as e:
            print("[MQTT] bad payload:", repr(e), "raw=", msg.payload[:50])
            return

        loop.create_task(handle_value(v, model, conf))

    c = mqtt.Client()
    if username is not None and password is not None:
        c.username_pw_set(username, password)

    c.on_connect = on_connect
    c.on_message = on_message

    c.connect(broker, port, keepalive=60)
    c.loop_forever()

if __name__ == "__main__":
    print("[BOOT] Edge AI Gateway starting...")
    print(f"[BOOT] reading {MODEL_PATH} and {CONF_PATH}")
    main()