import threading
import time

import paho.mqtt.client as mqtt

from testaifix57 import app, engine


MQTT_BROKER = ""
MQTT_PORT = 1883

TOPIC_INSPECTION_REQUEST = "factory/main/inspection/request"
TOPIC_INSPECTION_RESULT = "factory/pi/inspection/result"

RESULT_READY_VERDICTS = {"OK", "NG"}
INSPECTION_TIMEOUT_SECONDS = 8.0
STABLE_HITS_REQUIRED = 5
MONITOR_INTERVAL_SECONDS = 0.2
MIN_INSPECTION_DWELL_SECONDS = 1.2


class InspectionCoordinator:
    def __init__(self):
        self.client = mqtt.Client(client_id="pi_aoi_controller", clean_session=True)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self.lock = threading.Lock()
        self.pending_cycle_id = None
        self.request_started_at = 0.0
        self.last_signature = None
        self.stable_hits = 0

    def start(self):
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        threading.Thread(target=self.client.loop_forever, daemon=True).start()
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def _on_connect(self, client, userdata, flags, rc):
        client.subscribe(TOPIC_INSPECTION_REQUEST)

    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode(errors="ignore").strip()
        if msg.topic != TOPIC_INSPECTION_REQUEST or not payload:
            return

        try:
            cycle_id = int(payload)
        except ValueError:
            return

        with self.lock:
            self.pending_cycle_id = cycle_id
            self.request_started_at = time.time()
            self.last_signature = None
            self.stable_hits = 0

    def _monitor_loop(self):
        while True:
            time.sleep(MONITOR_INTERVAL_SECONDS)

            with self.lock:
                cycle_id = self.pending_cycle_id
                started_at = self.request_started_at

            if cycle_id is None:
                continue

            snapshot = engine.get_status()["current"]
            board_type = snapshot.get("board_type") or snapshot.get("board_candidate") or "UNKNOWN"
            verdict = snapshot.get("verdict") or "WAIT"
            score = snapshot.get("classification_confidence", 0)
            signature = (board_type, verdict, score)
            dwell_elapsed = time.time() - started_at

            if dwell_elapsed >= MIN_INSPECTION_DWELL_SECONDS and board_type != "UNKNOWN" and verdict in RESULT_READY_VERDICTS:
                with self.lock:
                    if signature == self.last_signature:
                        self.stable_hits += 1
                    else:
                        self.last_signature = signature
                        self.stable_hits = 1

                    if self.stable_hits >= STABLE_HITS_REQUIRED:
                        self._publish_result_locked(cycle_id, snapshot)
                        self._clear_pending_locked()
                continue

            if time.time() - started_at >= INSPECTION_TIMEOUT_SECONDS:
                timeout_snapshot = dict(snapshot)
                timeout_snapshot["board_type"] = board_type
                timeout_snapshot["verdict"] = "TIMEOUT" if verdict.startswith("WAIT") else verdict
                with self.lock:
                    self._publish_result_locked(cycle_id, timeout_snapshot)
                    self._clear_pending_locked()

    def _publish_result_locked(self, cycle_id, snapshot):
        board_type = snapshot.get("board_type") or snapshot.get("board_candidate") or "UNKNOWN"
        verdict = snapshot.get("verdict") or "UNKNOWN"
        payload = ",".join(
            [
                str(cycle_id),
                str(board_type),
                str(verdict),
                str(snapshot.get("classification_confidence", 0)),
            ]
        )
        self.client.publish(TOPIC_INSPECTION_RESULT, payload, qos=1, retain=False)

    def _clear_pending_locked(self):
        self.pending_cycle_id = None
        self.request_started_at = 0.0
        self.last_signature = None
        self.stable_hits = 0


coordinator = InspectionCoordinator()
coordinator.start()


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        engine.stop()
