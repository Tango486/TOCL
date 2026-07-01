import math
import os
import pickle
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _stable_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    y = np.exp(x)
    total = np.sum(y)
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(y, dtype=np.float64) / max(len(y), 1)
    return y / total


@dataclass
class LevelRecord:
    level_id: int
    params: np.ndarray
    created_step: int
    source: str = "new"
    score: float = 0.0
    raw_score: float = 0.0
    n_seen: int = 0
    n_sampled: int = 0
    staleness: float = 0.0
    last_reward: Optional[float] = None
    last_td_error: Optional[float] = None


class TOCLSampler:
    """Traffic operation aware replay sampler over normalized source contexts."""

    def __init__(self, param_env_bounds, seed, args, run_dir):
        self.param_env_bounds = OrderedDict(param_env_bounds)
        self.param_keys = list(self.param_env_bounds.keys())
        self.dim = len(self.param_keys)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.args = args
        self.run_dir = run_dir

        self.buffer_size = int(getattr(args, "tocl_buffer_size", 4000))
        self.replay_prob = float(getattr(args, "tocl_replay_prob", 0.65))
        self.min_seen_before_replay = int(getattr(args, "tocl_min_seen_before_replay", 10))
        self.score_transform = getattr(args, "tocl_score_transform", "rank")
        self.temperature = float(getattr(args, "tocl_temperature", 0.5))
        self.staleness_coef = float(getattr(args, "tocl_staleness_coef", 0.25))
        self.staleness_temperature = float(getattr(args, "tocl_staleness_temperature", 1.0))
        self.ema_alpha = float(getattr(args, "tocl_ema_alpha", 0.5))

        self.score_clip = float(getattr(args, "tocl_score_clip", 5.0))
        self.new_candidate_count = int(getattr(args, "tocl_new_candidate_count", 8))
        self.static_select_prob = float(getattr(args, "tocl_static_select_prob", 0.25))
        self.static_temperature = float(getattr(args, "tocl_static_temperature", 0.35))
        self.scale_window = int(getattr(args, "tocl_scale_window", 128))

        static_weights = {
            "flow": float(getattr(args, "tocl_static_flow_weight", 0.30)),
            "spatial": float(getattr(args, "tocl_static_spatial_weight", 0.20)),
            "temporal": float(getattr(args, "tocl_static_temporal_weight", 0.20)),
            "left": float(getattr(args, "tocl_static_left_weight", 0.15)),
            "del": float(getattr(args, "tocl_static_del_weight", 0.10)),
            "size": float(getattr(args, "tocl_static_size_weight", 0.05)),
        }
        static_weight_sum = sum(static_weights.values())
        if static_weight_sum <= 0:
            raise ValueError("At least one static score weight must be positive.")
        self.static_weights = {
            key: float(value / static_weight_sum)
            for key, value in static_weights.items()
        }

        self.delay_weight = float(getattr(args, "tocl_delay_weight", 0.26))
        self.wait_weight = float(getattr(args, "tocl_wait_weight", 0.22))
        self.queue_weight = float(getattr(args, "tocl_queue_weight", 0.22))
        self.pressure_weight = float(getattr(args, "tocl_pressure_weight", 0.16))
        self.td_weight = float(getattr(args, "tocl_td_weight", 0.04))
        self.static_weight = float(getattr(args, "tocl_static_weight", 0.10))

        self.delay_history = deque(maxlen=self.scale_window)
        self.wait_history = deque(maxlen=self.scale_window)
        self.queue_history = deque(maxlen=self.scale_window)
        self.pressure_history = deque(maxlen=self.scale_window)
        self.td_history = deque(maxlen=self.scale_window)

        self.next_level_id = 1
        self.levels = OrderedDict()

    @property
    def eligible_records(self):
        return [record for record in self.levels.values() if record.n_seen > 0]

    def _uniform_params(self):
        return self.rng.uniform(0.0, 1.0, size=self.dim).astype(np.float32)

    def _insert_level(self, params, total_num_steps, source="new", initial_score=0.0):
        record = LevelRecord(
            level_id=self.next_level_id,
            params=np.asarray(params, dtype=np.float32),
            created_step=int(total_num_steps),
            source=source,
            score=float(initial_score),
            raw_score=float(initial_score),
        )
        self.next_level_id += 1
        self.levels[record.level_id] = record
        self._enforce_capacity(exclude_ids={record.level_id})
        return record

    def _enforce_capacity(self, exclude_ids=None):
        exclude_ids = exclude_ids or set()
        while len(self.levels) > self.buffer_size:
            candidates = [record for record in self.levels.values() if record.level_id not in exclude_ids]
            if not candidates:
                return
            victim = min(candidates, key=lambda record: (record.score, record.n_seen, -record.staleness))
            self.levels.pop(victim.level_id, None)

    def _idx(self, key):
        try:
            return self.param_keys.index(key)
        except ValueError:
            return None

    def _norm_value(self, params, key, default=0.0):
        idx = self._idx(key)
        if idx is None or idx >= len(params):
            return float(default)
        return float(np.clip(params[idx], 0.0, 1.0))

    def static_score(self, params):
        flow = self._norm_value(params, "flow_scale", 0.0)
        turn = self._norm_value(params, "turn_left_ratio", 0.0)
        spatial = self._norm_value(params, "spatial_imbalance", 0.0)
        temporal = self._norm_value(params, "temporal_variation", 0.0)
        deletion = self._norm_value(params, "del_factor", 0.0)
        x_nodes = self._norm_value(params, "x_nodes", 0.5)
        y_nodes = self._norm_value(params, "y_nodes", 0.5)
        size = 0.5 * (x_nodes + y_nodes)
        w = self.static_weights
        return float(np.clip(
            w["flow"] * flow
            + w["spatial"] * spatial
            + w["temporal"] * temporal
            + w["left"] * turn
            + w["del"] * deletion
            + w["size"] * size,
            0.0,
            1.0,
        ))

    def _scale(self, value, history):
        values = list(history) + [float(value)]
        scale = float(np.quantile(values, 0.80)) if values else 1.0
        return max(scale, 1.0)

    def _positive_risk(self, row, key, history):
        raw = 0.0 if row.get(key) is None else max(float(row.get(key)), 0.0)
        scale = self._scale(raw, history)
        history.append(raw)
        return float(np.clip(raw / scale, 0.0, self.score_clip)), float(scale)

    def _td_risk(self, row):
        raw = 0.0 if row.get("td_error") is None else max(float(row.get("td_error")), 0.0)
        scale = self._scale(raw, self.td_history)
        self.td_history.append(raw)
        return float(np.clip(raw / scale, 0.0, self.score_clip)), float(scale)

    def _score_weights(self, records, scores, transform, temperature):
        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) == 0:
            return scores
        if transform == "constant":
            weights = np.ones_like(scores)
        elif transform == "max":
            weights = np.zeros_like(scores)
            weights[np.argmax(scores)] = 1.0
        elif transform == "rank":
            order = np.flip(np.argsort(scores))
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(order)) + 1
            weights = 1.0 / np.power(ranks.astype(np.float64), 1.0 / max(temperature, 1e-6))
        elif transform == "softmax":
            weights = _stable_softmax(scores / max(temperature, 1e-6))
        else:
            weights = np.power(np.clip(scores, 0.0, None) + 1e-3, 1.0 / max(temperature, 1e-6))

        weights = np.asarray(weights, dtype=np.float64)
        total = np.sum(weights)
        if total <= 0 or not np.isfinite(total):
            return np.ones(len(records), dtype=np.float64) / len(records)
        return weights / total

    def _sample_replay_record(self):
        records = self.eligible_records
        if not records:
            return None
        scores = [record.score for record in records]
        weights = self._score_weights(records, scores, self.score_transform, self.temperature)
        if self.staleness_coef > 0:
            stale_scores = [record.staleness for record in records]
            stale_weights = self._score_weights(records, stale_scores, "power", self.staleness_temperature)
            weights = (1.0 - self.staleness_coef) * weights + self.staleness_coef * stale_weights
            weights = weights / np.sum(weights)
        index = int(self.rng.choice(np.arange(len(records)), p=weights))
        for record in records:
            record.staleness += 1.0
        records[index].staleness = 0.0
        return records[index]

    def _is_warm(self):
        if self.min_seen_before_replay > 0:
            return len(self.eligible_records) >= self.min_seen_before_replay
        return len(self.eligible_records) >= max(1, int(math.ceil(0.2 * max(self.buffer_size, 1))))

    def _extra(self, record, candidate_count=1):
        return {
            "tocl_operational_score": getattr(record, "tocl_operational_score", None),
            "tocl_priority": getattr(record, "tocl_priority", record.score),
            "tocl_static_score": float(self.static_score(record.params)),
            "tocl_delay_risk": getattr(record, "tocl_delay_risk", None),
            "tocl_wait_risk": getattr(record, "tocl_wait_risk", None),
            "tocl_queue_risk": getattr(record, "tocl_queue_risk", None),
            "tocl_pressure_risk": getattr(record, "tocl_pressure_risk", None),
            "tocl_td_risk": getattr(record, "tocl_td_risk", None),
            "candidate_count": int(candidate_count),
        }

    def _record_to_info(self, record, total_num_steps, source=None, extra=None):
        record.n_sampled += 1
        info = {
            "family": "tocl",
            "strategy": "tocl",
            "source": source or record.source,
            "level_id": record.level_id,
            "score": float(record.score),
            "raw_score": float(record.raw_score),
            "n_seen": int(record.n_seen),
            "n_sampled": int(record.n_sampled),
            "staleness": float(record.staleness),
            "created_step": int(record.created_step),
            "generation_seed": int(self.seed * 100000 + record.level_id),
            "total_num_steps": int(total_num_steps),
        }
        if extra:
            info.update(extra)
        return record.params.copy(), info

    def _sample_static_new_record(self, total_num_steps):
        candidate_count = max(1, self.new_candidate_count)
        candidates = []
        for _ in range(candidate_count):
            params = self._uniform_params()
            candidates.append((self.static_score(params), params))
        risks = np.asarray([item[0] for item in candidates], dtype=np.float64)
        weights = _stable_softmax(risks / max(self.static_temperature, 1e-6))
        index = int(self.rng.choice(np.arange(len(candidates)), p=weights))
        static_risk, params = candidates[index]
        record = self._insert_level(
            params,
            total_num_steps,
            source="static_new",
            initial_score=self.static_weight * static_risk,
        )
        record.tocl_priority = float(self.static_weight * static_risk)
        return self._record_to_info(
            record,
            total_num_steps,
            source="static_new",
            extra=self._extra(record, candidate_count=candidate_count),
        )

    def sample(self, total_num_steps):
        if self._is_warm() and self.rng.rand() < self.replay_prob:
            record = self._sample_replay_record()
            if record is not None:
                return self._record_to_info(record, total_num_steps, source="replay", extra=self._extra(record))

        if self.rng.rand() < self.static_select_prob:
            return self._sample_static_new_record(total_num_steps)

        record = self._insert_level(self._uniform_params(), total_num_steps, source="new")
        return self._record_to_info(record, total_num_steps, source="new", extra=self._extra(record))

    def update(self, feedback_rows, total_num_steps):
        updates = []
        for row in feedback_rows:
            info = row.get("info") or {}
            record = self.levels.get(info.get("level_id"))
            if record is None:
                updates.append(None)
                continue

            delay_risk, delay_scale = self._positive_risk(row, "delay_time", self.delay_history)
            wait_risk, wait_scale = self._positive_risk(row, "wait_time", self.wait_history)
            queue_risk, queue_scale = self._positive_risk(row, "queue_len", self.queue_history)
            pressure_risk, pressure_scale = self._positive_risk(row, "pressure", self.pressure_history)
            td_risk, td_scale = self._td_risk(row)
            static_score = self.static_score(record.params)
            operational_score = (
                self.delay_weight * delay_risk
                + self.wait_weight * wait_risk
                + self.queue_weight * queue_risk
                + self.pressure_weight * pressure_risk
                + self.td_weight * td_risk
            )
            raw_score = float(operational_score + self.static_weight * static_score)
            if record.n_seen == 0:
                record.score = raw_score
            else:
                record.score = float((1.0 - self.ema_alpha) * record.score + self.ema_alpha * raw_score)
            record.raw_score = raw_score
            record.n_seen += 1
            record.last_reward = row.get("reward")
            record.last_td_error = row.get("td_error")
            record.tocl_priority = record.score
            record.tocl_operational_score = float(operational_score)
            record.tocl_delay_risk = delay_risk
            record.tocl_wait_risk = wait_risk
            record.tocl_queue_risk = queue_risk
            record.tocl_pressure_risk = pressure_risk
            record.tocl_td_risk = td_risk
            updates.append({
                "level_id": record.level_id,
                "updated_score": float(record.score),
                "raw_score": float(record.raw_score),
                "n_seen": int(record.n_seen),
                "tocl_priority": float(record.score),
                "tocl_operational_score": float(operational_score),
                "tocl_static_score": float(static_score),
                "tocl_delay_risk": float(delay_risk),
                "tocl_wait_risk": float(wait_risk),
                "tocl_queue_risk": float(queue_risk),
                "tocl_pressure_risk": float(pressure_risk),
                "tocl_td_risk": float(td_risk),
                "tocl_delay_scale": delay_scale,
                "tocl_wait_scale": wait_scale,
                "tocl_queue_scale": queue_scale,
                "tocl_pressure_scale": pressure_scale,
                "tocl_td_scale": td_scale,
            })
        return updates

    def dump(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump({
                "strategy": "tocl",
                "seed": self.seed,
                "next_level_id": self.next_level_id,
                "levels": self.levels,
                "delay_history": list(self.delay_history),
                "wait_history": list(self.wait_history),
                "queue_history": list(self.queue_history),
                "pressure_history": list(self.pressure_history),
                "td_history": list(self.td_history),
                "static_weights": dict(self.static_weights),
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
