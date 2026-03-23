import time
import uuid
from datetime import datetime
from typing import Callable

from model import Alert

# Last N seconds to count same-fault events
WINDOW_SEC = 5.0
# Min same-fault count in window to establish continuous fault
K = 3
# Seconds without receiving current fault to consider it ended
L = 10.0



class StateManager:
    def __init__(self, k: int = K, l_sec: float = L, window_sec: float = WINDOW_SEC):
        self.k = k
        self.l_sec = l_sec
        self.window_sec = window_sec

        # Current active fault
        # {
        #   "id": str,
        #   "asset_id": str,
        #   "fault_type": str,
        #   "start_ts": datetime,
        #   "last_received_ts": float
        # }
        self._current: dict | None = None

        # Recent events used for establishing continuous faults
        # [{"asset_id": str, "fault_type": str, "ts": float}]
        self._recent: list[dict] = []

    def _now(self) -> float:
        return time.time()

    def _utcnow(self) -> datetime:
        return datetime.utcnow()

    def _prune_recent(self, now_ts: float) -> None:
        cutoff = now_ts - self.window_sec
        self._recent = [e for e in self._recent if e["ts"] >= cutoff]

    def _clear_recent(self, reason: str = "") -> None:
        print("\n========== CLEAR RECENT WINDOW ==========")
        if reason:
            print(f"reason: {reason}")
        print(f"clearing {len(self._recent)} recent event(s)")
        print("========================================\n")
        self._recent = []

    def _latest_recent_ts(self) -> float | None:
        if not self._recent:
            return None
        return max(e["ts"] for e in self._recent)

    def _count_in_window(self, asset_id: str, fault_type: str) -> int:
        return sum(
            1
            for e in self._recent
            if e["asset_id"] == asset_id and e["fault_type"] == fault_type
        )

    def _established_faults_in_window(self) -> list[tuple[str, str, float, float]]:
        """
        Returns:
            [(asset_id, fault_type, earliest_ts_float, latest_ts_float), ...]
        for faults with count >= K in the recent window.
        """
        seen: set[tuple[str, str]] = set()
        result: list[tuple[str, str, float, float]] = []

        for e in self._recent:
            key = (e["asset_id"], e["fault_type"])
            if key in seen:
                continue
            seen.add(key)

            matching = [
                x for x in self._recent
                if x["asset_id"] == e["asset_id"] and x["fault_type"] == e["fault_type"]
            ]

            if len(matching) >= self.k:
                earliest = min(x["ts"] for x in matching)
                latest = max(x["ts"] for x in matching)
                result.append((e["asset_id"], e["fault_type"], earliest, latest))

        # Prefer most recently active established fault
        result.sort(key=lambda item: item[3], reverse=True)
        return result

    def _end_current_fault(
        self,
        end_ts: datetime,
        on_fault_period_end: Callable[[str, datetime], None],
        reason: str = "",
    ) -> None:
        if self._current is None:
            return

        print("\n========== STATE CHANGE: END FAULT ==========")
        print(f"reason          : {reason}")
        print(f"fault_id        : {self._current['id']}")
        print(f"asset_id        : {self._current['asset_id']}")
        print(f"fault_type      : {self._current['fault_type']}")
        print(f"start_ts        : {self._current['start_ts']}")
        print(f"last_received_ts: {self._current['last_received_ts']}")
        print(f"end_ts          : {end_ts}")
        print("=============================================\n")

        fault_id = self._current["id"]
        on_fault_period_end(fault_id, end_ts)
        self._current = None

    def process_alert(
        self,
        payload: Alert,
        on_fault_period_end: Callable[[str, datetime], None],
        now_ts: float | None = None,
    ) -> dict | None:
        """
        Process one alert and update fault state.

        Uses server receive time for all timing logic.

        Returns a dict when a NEW fault is established:
        {
            "id": str,
            "asset_id": str,
            "fault_type": str,
            "start_ts": datetime
        }
        otherwise None.
        """
        now_ts = now_ts if now_ts is not None else self._now()

        asset_id = payload.asset_id
        fault_type = payload.condition_name or payload.message or ""

        print("\n--------------- ALERT RECEIVED ---------------")
        print(f"asset_id   : {asset_id}")
        print(f"fault_type : {fault_type}")
        print(f"server_ts  : {now_ts}")
        print(f"current    : {self._current}")
        print("---------------------------------------------\n")

        # If there is a big gap from the latest recent event, clear the window first.
        latest_recent_ts = self._latest_recent_ts()
        if latest_recent_ts is not None and (now_ts - latest_recent_ts >= self.l_sec):
            self._clear_recent(
                reason=f"gap between latest recent event and new event >= {self.l_sec} sec"
            )

        # If current fault exists and timed out, end it and clear the window.
        if self._current is not None:
            current_last_ts = self._current["last_received_ts"]

            if now_ts - current_last_ts >= self.l_sec:
                self._end_current_fault(
                    end_ts=self._utcnow(),
                    on_fault_period_end=on_fault_period_end,
                    reason=f"timeout: no matching alert for >= {self.l_sec} sec",
                )
                self._clear_recent(reason="current fault timed out")

        # Add current alert after timeout/gap handling
        self._recent.append({
            "asset_id": asset_id,
            "fault_type": fault_type,
            "ts": now_ts,
        })
        self._prune_recent(now_ts)

        print("Recent events in window:")
        for e in self._recent:
            print(e)
        print()

        # If a current fault still exists, handle same/different fault
        if self._current is not None:
            current_asset_id = self._current["asset_id"]
            current_fault_type = self._current["fault_type"]

            # Same fault -> only refresh last_received_ts
            if current_asset_id == asset_id and current_fault_type == fault_type:
                self._current["last_received_ts"] = now_ts

                print("\n========== STATE UNCHANGED: REFRESH ==========")
                print(f"fault_id        : {self._current['id']}")
                print(f"asset_id        : {self._current['asset_id']}")
                print(f"fault_type      : {self._current['fault_type']}")
                print(f"last_received_ts: {self._current['last_received_ts']}")
                print("==============================================\n")
                return None

            # Different fault may replace current if established
            established = self._established_faults_in_window()
            other = [
                (aid, ftype, start_float, latest_float)
                for (aid, ftype, start_float, latest_float) in established
                if (aid, ftype) != (current_asset_id, current_fault_type)
            ]

            if other:
                aid, ftype, start_float, latest_float = other[0]

                self._end_current_fault(
                    end_ts=self._utcnow(),
                    on_fault_period_end=on_fault_period_end,
                    reason="different established fault detected",
                )

                new_id = str(uuid.uuid4())
                start_dt = self._utcnow()

                self._current = {
                    "id": new_id,
                    "asset_id": aid,
                    "fault_type": ftype,
                    "start_ts": start_dt,
                    "last_received_ts": latest_float,
                }

                print("\n========== STATE CHANGE: START FAULT ==========")
                print(f"fault_id        : {new_id}")
                print(f"asset_id        : {aid}")
                print(f"fault_type      : {ftype}")
                print(f"start_ts        : {start_dt}")
                print(f"last_received_ts: {latest_float}")
                print("===============================================\n")

                return {
                    "id": new_id,
                    "asset_id": aid,
                    "fault_type": ftype,
                    "start_ts": start_dt,
                }

            return None

        # No current fault -> check whether one is established
        established = self._established_faults_in_window()
        if established:
            aid, ftype, start_float, latest_float = established[0]

            new_id = str(uuid.uuid4())
            start_dt = self._utcnow()

            self._current = {
                "id": new_id,
                "asset_id": aid,
                "fault_type": ftype,
                "start_ts": start_dt,
                "last_received_ts": latest_float,
            }

            print("\n========== STATE CHANGE: START FAULT ==========")
            print(f"fault_id        : {new_id}")
            print(f"asset_id        : {aid}")
            print(f"fault_type      : {ftype}")
            print(f"start_ts        : {start_dt}")
            print(f"last_received_ts: {latest_float}")
            print("===============================================\n")

            return {
                "id": new_id,
                "asset_id": aid,
                "fault_type": ftype,
                "start_ts": start_dt,
            }

        print("No established fault yet.\n")
        return None

    def expire_current_fault(
        self,
        on_fault_period_end: Callable[[str, datetime], None],
        now_ts: float | None = None,
    ) -> bool:
        """
        Optional manual timeout check if no alerts are arriving.
        """
        now_ts = now_ts if now_ts is not None else self._now()

        if self._current is None:
            return False

        if now_ts - self._current["last_received_ts"] >= self.l_sec:
            self._end_current_fault(
                end_ts=self._utcnow(),
                on_fault_period_end=on_fault_period_end,
                reason="manual expire_current_fault() timeout check",
            )
            self._clear_recent(reason="manual timeout expiration")
            return True

        return False

    def get_state(self) -> dict | None:
        return dict(self._current) if self._current else None