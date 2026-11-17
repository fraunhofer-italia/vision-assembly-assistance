import numpy as np
import copy
from scipy.signal import savgol_filter

class TrajectoryPreprocessor:

    @staticmethod
    def preprocess(traj, oob_policy=None):
        """
        Full preprocessing pipeline
        """
        if not traj:
            return traj

        if oob_policy is None:
            oob_policy = {"mode": "none"}
        bounds = oob_policy.get("bounds", [0, 1])
        mode = oob_policy.get("mode", "none")

        traj_copy = copy.deepcopy(traj)

        cleaned = []

        for p in traj_copy:
            if (
                p["x"] is None or
                p["y"] is None or
                np.isnan(p["x"]) or
                np.isnan(p["y"])
            ):
                continue

            cleaned.append(p)

        if not cleaned:
            return []
        
        cleaned.sort(key=lambda p : p["t"])

        t0 = cleaned[0]["t"]

        for p in cleaned:
            p["t"]= p["t"] - t0
            if p["t"] < 0:
                p["t"] = 0.0

        cleaned = TrajectoryPreprocessor._apply_oob_policy(
            cleaned,
            oob_policy=oob_policy
        )

        if not cleaned:
            return []

        # Suppress isolated one-frame spikes (common from brief hand-toggle jitter).
        cleaned = TrajectoryPreprocessor._suppress_single_frame_spikes(
            cleaned,
            spike_thresh=0.06,
        )

        if len(cleaned) >= 5:
            xs = np.array([p["x"] for p in cleaned])
            ys = np.array([p["y"] for p in cleaned])

            xs = savgol_filter(xs, window_length=5, polyorder=2)
            ys = savgol_filter(ys, window_length=5, polyorder=2)

            smoothed = []
            for i, p in enumerate(cleaned):
                smoothed.append({
                    **p,
                    "x": float(xs[i]),
                    "y": float(ys[i])
                })

            # Smoothing can slightly push edge samples outside bounds.
            if mode not in ("none", None):
                smoothed = TrajectoryPreprocessor._apply_oob_policy(
                    smoothed,
                    {"mode": "clamp", "bounds": bounds}
                )
            return smoothed

        if mode not in ("none", None):
            cleaned = TrajectoryPreprocessor._apply_oob_policy(
                cleaned,
                {"mode": "clamp", "bounds": bounds}
            )
        return cleaned

    @staticmethod
    def _suppress_single_frame_spikes(traj, spike_thresh=0.06):
        if len(traj) < 3:
            return traj
        out = [dict(p) for p in traj]
        th = float(max(0.0, spike_thresh))
        for i in range(1, len(out) - 1):
            p0 = out[i - 1]
            p1 = out[i]
            p2 = out[i + 1]
            d01 = ((float(p1["x"]) - float(p0["x"])) ** 2 + (float(p1["y"]) - float(p0["y"])) ** 2) ** 0.5
            d12 = ((float(p1["x"]) - float(p2["x"])) ** 2 + (float(p1["y"]) - float(p2["y"])) ** 2) ** 0.5
            d02 = ((float(p0["x"]) - float(p2["x"])) ** 2 + (float(p0["y"]) - float(p2["y"])) ** 2) ** 0.5
            if d01 > th and d12 > th and d02 < (0.5 * th):
                out[i]["x"] = float(0.5 * (float(p0["x"]) + float(p2["x"])))
                out[i]["y"] = float(0.5 * (float(p0["y"]) + float(p2["y"])))
        return out

    @staticmethod
    def _is_in_bounds(p, low, high):
        return low <= p["x"] <= high and low <= p["y"] <= high

    @staticmethod
    def _apply_oob_policy(traj, oob_policy):
        if not traj:
            return traj

        mode = (oob_policy or {}).get("mode", "none")
        bounds = (oob_policy or {}).get("bounds", [0, 1])
        max_gap_threshold = int((oob_policy or {}).get("max_gap_threshold", 20))
        low, high = float(bounds[0]), float(bounds[1])

        if mode in ("none", None):
            return traj

        if mode == "clamp":
            clamped = []
            for p in traj:
                new_p = dict(p)
                new_p["x"] = min(max(float(new_p["x"]), low), high)
                new_p["y"] = min(max(float(new_p["y"]), low), high)
                clamped.append(new_p)
            return clamped

        if mode == "drop":
            return [p for p in traj if TrajectoryPreprocessor._is_in_bounds(p, low, high)]

        if mode == "drop_interpolate":
            n = len(traj)
            valid = np.array(
                [TrajectoryPreprocessor._is_in_bounds(p, low, high) for p in traj],
                dtype=bool
            )

            if valid.all():
                return traj

            valid_idx = np.where(valid)[0]
            if len(valid_idx) == 0:
                return []

            out = [dict(p) for p in traj]

            i = 0
            while i < n:
                if valid[i]:
                    i += 1
                    continue

                start = i
                while i < n and not valid[i]:
                    i += 1
                end = i - 1
                gap_len = end - start + 1

                prev_candidates = valid_idx[valid_idx < start]
                next_candidates = valid_idx[valid_idx > end]
                prev_i = int(prev_candidates[-1]) if len(prev_candidates) > 0 else None
                next_i = int(next_candidates[0]) if len(next_candidates) > 0 else None

                for j in range(start, end + 1):
                    if prev_i is None and next_i is None:
                        continue
                    if prev_i is None:
                        out[j]["x"] = float(out[next_i]["x"])
                        out[j]["y"] = float(out[next_i]["y"])
                        continue
                    if next_i is None:
                        out[j]["x"] = float(out[prev_i]["x"])
                        out[j]["y"] = float(out[prev_i]["y"])
                        continue

                    # Interpolate only short gaps; long gaps use nearest valid sample.
                    if gap_len <= max_gap_threshold:
                        t0 = float(out[prev_i]["t"])
                        t1 = float(out[next_i]["t"])
                        if abs(t1 - t0) < 1e-9:
                            alpha = 0.5
                        else:
                            alpha = (float(out[j]["t"]) - t0) / (t1 - t0)
                        out[j]["x"] = float(out[prev_i]["x"] + alpha * (out[next_i]["x"] - out[prev_i]["x"]))
                        out[j]["y"] = float(out[prev_i]["y"] + alpha * (out[next_i]["y"] - out[prev_i]["y"]))
                    else:
                        if (j - prev_i) <= (next_i - j):
                            out[j]["x"] = float(out[prev_i]["x"])
                            out[j]["y"] = float(out[prev_i]["y"])
                        else:
                            out[j]["x"] = float(out[next_i]["x"])
                            out[j]["y"] = float(out[next_i]["y"])

            # guard against interpolation overshoot due to float noise
            return TrajectoryPreprocessor._apply_oob_policy(
                out,
                {"mode": "clamp", "bounds": [low, high]}
            )

        raise ValueError(f"Unsupported oob_policy mode: {mode}")
        

        



        '''
        traj = TrajectoryPreprocessor._sort_by_time(traj)
        traj = TrajectoryPreprocessor._normalize_time(traj)
        traj = TrajectoryPreprocessor._remove_invalid(traj)
        traj = TrajectoryPreprocessor._smooth_xy(traj)

        return traj
    
    #-----------------------------------------
    #       STEP 1 : SORT TIMESTAMPS
    #-----------------------------------------

    @staticmethod
    def _sort_by_time(traj):
        return sorted(traj, key=lambda p: p["t"])
    
    #-----------------------------------------
    #       STEP 2 : Normalize time
    #   Converts monotonic time --> relative time
    #-----------------------------------------

    @staticmethod
    def _normalize_time(traj):
        t0 = traj[0]["t"]

        return [
            {**p, "t": p["t"] - t0}
            for p in traj
        ]
    
    #-----------------------------------------
    #       STEP 3 : Remove invalid points
    #-----------------------------------------

    @staticmethod
    def _remove_invalid(traj):

        cleaned = []

        for p in traj:
            if (
                p["x"] is None or
                p["y"] is None or
                np.isnan(p["x"]) or
                np.isnan(p["y"])
            ):
                continue

            cleaned.append(p)

        return cleaned
    
    #-----------------------------------------
    #       STEP 4 : Smooth spatial jitter
    #-----------------------------------------

    @staticmethod
    def _smooth_xy(traj):
        if len(traj) < 5:
            return traj
        
        xs = np.array([p["x"] for p in traj])
        ys = np.array([p["y"] for p in traj])

        xs = savgol_filter(xs, window_length=5, polyorder=2)
        ys = savgol_filter(ys, window_length=5, polyorder=2)

        smoothed = []
        for i, p in enumerate(traj):
            smoothed.append({
                **p,
                "x": float(xs[i]),
                "y": float(ys[i])
            })

        return smoothed
        '''
