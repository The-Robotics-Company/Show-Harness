"""Parallel-env variant of :class:`core.sim.mvtoken_robolab_runner.MvTokenRobolabRunner`.

Same MVTOKEN loop -- two views -> one token -> a few relative-IK control steps -- run for
``num_envs`` RoboLab envs of one Isaac Lab instance in LOCKSTEP:

* every macro-step, each live env gets its own VLM call (a thread pool; the endpoint has
  to accept concurrent requests -- ``scripts/codex/codex_proxy.py`` does), its own
  controller (gripper state, orientation reference) and its own :class:`EpisodeLogger`
  (steps / images / video per env);
* every macro-step is ``M = max(sim_steps_per_decision + settle, gripper_hold_steps)``
  control steps for all envs; a shorter token chunk (a move is 8 steps, a GRASP 10) is
  padded with zero-delta holds. Under relative IK a zero delta re-targets the current pose,
  so the padding is "stay put" and not extra travel (see ``settle_steps_per_decision`` in
  ``configs/robot_robolab.yaml``);
* the auto-release reflex (a closed gripper holding nothing is reopened before the next
  decision) is executed as that env's NEXT macro-step -- a forced RELEASE that skips the
  VLM -- so it costs the env the same ``gripper_hold_steps`` it costs the sequential runner
  and never stalls the other envs;
* an env that says DONE, succeeds (RoboLab freezes it) or is truncated (the task's
  ``episode_length_s``, the same for every env) just holds until the others finish.

Because every env sees the same 180 s of sim time and RoboLab's layout is fixed at
construction, one 5-env run is the same experiment as five sequential episodes, at
one-fifth of the wall time.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np

from core.action_units import MOVE_ATOMS
from core.record.episode_logger import EpisodeLogger
from core.record.images import prepare_view
from core.sim.mvtoken_robolab_runner import (
    DONE_TOKEN,
    FALLBACK_TOKEN,
    GRASP_TOKEN,
    RECENT_MOVES_MAX,
    RELEASE_TOKEN,
    video_fps,
)
from core.sim.robolab_task import (
    reset_robolab,
    rl_ee_quat,
    rl_gripper_width,
    rl_rgb,
    rl_success,
    rl_tcp,
    step_robolab_batched,
)
from core.v0_types import EpisodeResult, V0Config


class MvTokenRobolabParallelRunner:
    """Closed-loop MVTOKEN rollout over all envs of one RoboLab env in lockstep."""

    def __init__(
        self,
        *,
        env: Any,
        task_description: str,
        controllers: list,
        agents: list,
        loggers: list[EpisodeLogger],
        config: V0Config,
        max_steps: int,
        num_steps_wait: int,
        sim_steps_per_decision: int,
        settle_steps_per_decision: int,
        agentview_camera: str,
        wrist_camera: str,
        agentview_rotation_degrees: int,
        wrist_rotation_degrees: int,
        agentview_flip: str,
        wrist_flip: str,
        use_wrist_image: bool,
        auto_release: Any = None,
        debug: bool = False,
        prompt_log_every: int = 20,
        agentview_square_size: Optional[int] = None,
        agentview_crop_aspect: Optional[float] = None,
        wrist_crop_aspect: Optional[float] = None,
        wrist_square_size: Optional[int] = None,
        gripper_hold_steps: int = 0,
    ) -> None:
        self.env = env
        self.n = int(env.num_envs)
        if not (len(controllers) == len(agents) == len(loggers) == self.n):
            raise ValueError(
                f"need one controller/agent/logger per env: {len(controllers)}/{len(agents)}/"
                f"{len(loggers)} for {self.n} envs"
            )
        self.task_description = str(task_description)
        self.controllers = list(controllers)
        self.agents = list(agents)
        self.loggers = list(loggers)
        self.config = config
        self.max_steps = int(max_steps)
        self.num_steps_wait = int(num_steps_wait)
        self.sim_steps_per_decision = max(1, int(sim_steps_per_decision))
        self.settle_steps_per_decision = max(0, int(settle_steps_per_decision))
        self.gripper_hold_steps = int(gripper_hold_steps) or self.sim_steps_per_decision
        # Lockstep macro-step length: the longest chunk any token produces.
        self.macro_steps = max(
            self.sim_steps_per_decision + self.settle_steps_per_decision, self.gripper_hold_steps
        )
        self.agentview_camera = str(agentview_camera)
        self.wrist_camera = str(wrist_camera)
        self.agentview_rotation_degrees = int(agentview_rotation_degrees)
        self.wrist_rotation_degrees = int(wrist_rotation_degrees)
        self.agentview_flip = str(agentview_flip or "none")
        self.wrist_flip = str(wrist_flip or "none")
        self.use_wrist_image = bool(use_wrist_image)
        self.auto_release = auto_release
        self.debug = bool(debug)
        self.prompt_log_every = int(prompt_log_every)
        self.agentview_square_size = int(agentview_square_size) if agentview_square_size else None
        self.agentview_crop_aspect = float(agentview_crop_aspect) if agentview_crop_aspect else None
        self.wrist_crop_aspect = float(wrist_crop_aspect) if wrist_crop_aspect else None
        self.wrist_square_size = int(wrist_square_size) if wrist_square_size else None

    # ------------------------------------------------------------------ main loop
    def run(self) -> list[EpisodeResult]:
        n = self.n
        for agent in self.agents:
            reset_history = getattr(agent, "reset", None)
            if callable(reset_history):
                reset_history()
        for c in self.controllers:
            c.set_orientation_reference(None)
            c.open_gripper()
        obs, _term0, _trunc0 = reset_robolab(
            self.env, hold_action=self.controllers[0].open_gripper(), settle_steps=self.num_steps_wait
        )
        for i, c in enumerate(self.controllers):
            c.set_orientation_reference(rl_ee_quat(self.env, i))

        success = [rl_success(self.env, i) for i in range(n)]
        done = [bool(s) for s in success]
        end_reason = ["success" if s else "max_steps_exceeded" for s in success]
        steps = [0] * n
        recent_moves: list[list[str]] = [[] for _ in range(n)]
        pending_release = [False] * n
        video_paths: list[Any] = [lg.run_dir / "rollout_failure.mp4" for lg in self.loggers]
        pool = ThreadPoolExecutor(max_workers=n)

        try:
            for step_idx in range(self.max_steps):
                active = [i for i in range(n) if not done[i]]
                if not active:
                    break
                t_decide = time.monotonic()
                views = {i: self._images(obs, i) for i in active}

                # ---- decisions: forced RELEASE for the auto-release reflex, else the VLM
                futures = {}
                tokens: dict[int, str] = {}
                responses: dict[int, Any] = {}
                auto_flag: dict[int, bool] = {}
                for i in active:
                    if pending_release[i]:
                        tokens[i], responses[i], auto_flag[i] = RELEASE_TOKEN, None, True
                        pending_release[i] = False
                        continue
                    auto_flag[i] = False
                    futures[i] = pool.submit(self._decide, i, views[i][0], views[i][1], recent_moves[i])
                for i, fut in futures.items():
                    tokens[i], responses[i] = fut.result()
                decide_s = time.monotonic() - t_decide

                for i in active:
                    steps[i] = step_idx + 1
                    if not auto_flag[i] and self.prompt_log_every > 0 and step_idx % self.prompt_log_every == 0:
                        prompt_text = getattr(self.agents[i], "last_prompt", "")
                        if prompt_text:
                            self.loggers[i].save_controller_prompt(
                                step_idx, prompt_text, media=getattr(self.agents[i], "last_media", None)
                            )
                    if tokens[i] == DONE_TOKEN:
                        print(f"[mvtoken-robolab] env {i} step {step_idx}: DONE emitted -- holding.")
                        done[i] = True
                        end_reason[i] = "done"

                # ---- execute all chunks in lockstep
                chunks = [self._chunk(i, tokens.get(i)) if (i in active and not done[i]) else None
                          for i in range(n)]
                term_acc = np.zeros(n, dtype=bool)
                trunc_acc = np.zeros(n, dtype=bool)
                for k in range(self.macro_steps):
                    actions = np.zeros((n, 7), dtype=np.float32)
                    for i in range(n):
                        c = self.controllers[i]
                        base = chunks[i][k] if chunks[i] is not None and k < len(chunks[i]) else c.hold_action()
                        actions[i] = c.with_orientation_hold(base, rl_ee_quat(self.env, i))
                    obs, term, trunc, _info = step_robolab_batched(self.env, actions)
                    term_acc |= term
                    trunc_acc |= trunc

                # ---- bookkeeping per env that acted this macro-step
                for i in active:
                    if chunks[i] is None:      # DONE this step: nothing executed for it
                        continue
                    token = tokens[i]
                    success[i] = bool(term_acc[i]) or rl_success(self.env, i)
                    released_pending = False
                    if self._empty_grasp(i):
                        pending_release[i] = True
                        released_pending = True
                        print(f"[mvtoken-robolab] env {i} step {step_idx}: auto-release -- gripper width "
                              f"{rl_gripper_width(self.env, i):.4f}m < {self.auto_release.empty_width_m:.4f}m; "
                              "opening gripper next step")
                    if token in MOVE_ATOMS:
                        recent_moves[i].insert(0, token)
                        del recent_moves[i][RECENT_MOVES_MAX:]
                    record = self._record(i, step_idx, token, responses[i], success[i],
                                          bool(term_acc[i] or trunc_acc[i]), auto_flag[i], released_pending)
                    record["decide_ms"] = int(decide_s * 1000)
                    agentview, wrist = views[i]
                    self.loggers[i].log_step(step_idx=step_idx, agentview=agentview, wrist=wrist, record=record)
                    if success[i]:
                        done[i] = True
                        end_reason[i] = "success"
                    elif trunc_acc[i]:
                        print(f"[mvtoken-robolab] env {i} step {step_idx}: env truncated (episode time-out).")
                        done[i] = True
                        end_reason[i] = "env_truncated"
                live = sum(1 for i in range(n) if not done[i])
                print(f"[mvtoken-robolab] macro-step {step_idx}: {live}/{n} envs live, decisions {decide_s:.1f} s, "
                      f"tokens {[tokens.get(i, '-') for i in range(n)]}", flush=True)
        finally:
            pool.shutdown(wait=False)
            for i in range(n):
                video_paths[i] = self.loggers[i].close(success=success[i], fps=video_fps(self.config.video_fps))
                self.loggers[i].write_summary({
                    "success": success[i],
                    "steps": steps[i],
                    "max_steps": self.max_steps,
                    "end_reason": end_reason[i],
                    "video_path": str(video_paths[i]),
                    "run_dir": str(self.loggers[i].run_dir),
                    "control_mode": "robolab_mvtoken_parallel",
                    "env_id": i,
                    "num_envs": n,
                    "macro_steps": self.macro_steps,
                    "task": self.task_description,
                })

        return [
            EpisodeResult(success=success[i], steps=steps[i], end_reason=end_reason[i],
                          video_path=str(video_paths[i]), run_dir=str(self.loggers[i].run_dir))
            for i in range(n)
        ]

    # ------------------------------------------------------------------ helpers
    def _decide(self, i: int, agentview, wrist, recent_moves: list[str]):
        gripper_state = "closed" if self.controllers[i].state.gripper_name == "CLOSE" else "open"
        try:
            response = self.agents[i].decide(
                task=self.task_description,
                gripper_state=gripper_state,
                recent_moves=", ".join(recent_moves) if recent_moves else "none",
                agentview_image=agentview,
                wrist_image=wrist,
                debug=self.debug,
            )
            return response.token, response
        except RuntimeError as exc:
            print(f"[mvtoken-robolab] env {i}: VLM token parse failed ({exc}); falling back to {FALLBACK_TOKEN}")
            return FALLBACK_TOKEN, None

    def _chunk(self, i: int, token: Optional[str]) -> list[np.ndarray]:
        """Per-control-step base actions for one token (rotation slots filled later)."""
        c = self.controllers[i]
        if token in MOVE_ATOMS:
            action = c.action_for_atomic(token)
            chunk = [action] * self.sim_steps_per_decision
            if self.settle_steps_per_decision:
                chunk += [c.hold_action()] * self.settle_steps_per_decision
            return chunk
        if token == GRASP_TOKEN:
            return [c.close_gripper()] * self.gripper_hold_steps
        if token == RELEASE_TOKEN:
            return [c.open_gripper()] * self.gripper_hold_steps
        return [c.hold_action()]

    def _empty_grasp(self, i: int) -> bool:
        if self.auto_release is None or not self.auto_release.enabled:
            return False
        if self.controllers[i].state.gripper_name != "CLOSE":
            return False
        return bool(self.auto_release.should_release(rl_gripper_width(self.env, i), True))

    def _images(self, obs: dict[str, Any], i: int):
        agentview = prepare_view(
            rl_rgb(obs, self.agentview_camera, i),
            rotation_degrees=self.agentview_rotation_degrees,
            flip=self.agentview_flip,
            crop_aspect=self.agentview_crop_aspect,
            square_size=self.agentview_square_size,
        )
        wrist = (
            prepare_view(
                rl_rgb(obs, self.wrist_camera, i),
                rotation_degrees=self.wrist_rotation_degrees,
                flip=self.wrist_flip,
                crop_aspect=self.wrist_crop_aspect,
                square_size=self.wrist_square_size,
            )
            if self.use_wrist_image
            else None
        )
        return agentview, wrist

    def _record(self, i: int, step_idx: int, token: str, response: Any, success: bool, env_done: bool,
                auto_released: bool, release_pending: bool) -> dict[str, Any]:
        record: dict[str, Any] = {
            "i": int(step_idx),
            "env": int(i),
            "stage": "-",
            "act": token,
            "eef": [round(float(x), 3) for x in rl_tcp(self.env, i)],
            "w": round(rl_gripper_width(self.env, i), 5),
            "grip": self.controllers[i].state.gripper_name,
        }
        if auto_released:
            record["auto_release"] = True       # this RELEASE was the reflex, not the policy
        if release_pending:
            record["auto_release_pending"] = True
        latency_s = (getattr(response, "payload", None) or {}).get("latency_s")
        if latency_s is not None:
            record["vlm_ms"] = int(round(float(latency_s) * 1000.0))
        if success:
            record["ok"] = True
        if env_done:
            record["env_done"] = True
        return record
