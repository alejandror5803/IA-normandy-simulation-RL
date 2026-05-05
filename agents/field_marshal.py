# Field Marshal - strategic command layer for the Normandy simulation
#
# AI technique from the course (Topic 3 - smolagents):
#
#   SmolAgentsFieldMarshal uses a ToolCallingAgent backed by a HuggingFace LLM
#   to act as a strategic advisor for BOTH blue and red commanders.
#   It is called every FM_STRATEGY_INTERVAL episodes to avoid burning API quota.
#
#   What it actually does (this is the interesting part):
#
#   Option A - Dynamic reward multipliers:
#     The LLM outputs scaling factors for each action type (capture / kill / defend).
#     These get applied to the reward signal in the training loop, so the LLM is
#     literally reshaping the reward function every 500 episodes based on performance.
#     Example: if blue keeps fighting but ignoring objectives, the LLM cranks up
#     CAPTURE_MULT and dials down KILL_MULT so the Q-agents learn to prioritise caps.
#
#   Option B - Per-commander epsilon control:
#     The LLM can set individual exploration rates for each of the 4 blue commanders
#     and a global rate for the 12 red commanders.
#     Stagnating commanders get pushed toward exploration; converged ones stay put.
#
#   Tools given to the LLM:
#     get_battle_performance   - recent win rate, rewards, captures, blue survivors
#     get_commander_stats      - per-commander epsilon and avg Q-value
#     get_luftwaffe_report     - aircraft status and missions done
#
#   Expected LLM output (one line, parsed with regex):
#     CAPTURE_MULT=1.5  KILL_MULT=0.8  DEFEND_MULT=1.0
#     BLUE_EPS=0.15,0.20,0.15,0.25  RED_EPS=0.35
#     RATIONALE=<one sentence>
#
#   Multiplier range: [0.5, 2.5]  |  Epsilon range: [0.05, 0.70]
#
# If smolagents is not installed or the API call fails, the fallback rule-based
# implementation computes reasonable values from the same stats.

import re
import io
import contextlib
import numpy as np
import env.env_config as cfg


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _build_llm_model():
    # returns the right smolagents model based on FM_PROVIDER in env_config
    # "groq" uses OpenAIServerModel pointing at Groq's OpenAI-compatible API
    # "hf"   uses InferenceClientModel (HuggingFace serverless inference)
    if cfg.FM_PROVIDER == "groq":
        from smolagents import OpenAIServerModel
        return OpenAIServerModel(
            model_id=cfg.GROQ_MODEL_ID,
            api_base="https://api.groq.com/openai/v1",
            api_key=cfg.GROQ_API_KEY or None,
            # fail fast on 429/timeouts, then we fallback to rule-based without blocking training
            retry=False,
            client_kwargs={"max_retries": 0, "timeout": 20},
        )
    else:
        from smolagents import InferenceClientModel
        return InferenceClientModel(
            cfg.FM_MODEL_ID,
            token=cfg.HF_TOKEN or None,
            retry=False,
        )


class RuleBasedFieldMarshal:
    # original heuristic field marshal, kept as fallback when LLM is unavailable
    # now also outputs reward multipliers and epsilon suggestions

    def update_strategy(self, stats: dict) -> dict:
        win_rate     = stats.get("win_rate",     0.5)
        avg_captures = stats.get("avg_captures", 0.0)
        blue_eps     = stats.get("blue_epsilons", [0.3, 0.3, 0.3, 0.3])
        red_eps_cur  = stats.get("red_epsilon",   0.3)

        # reward multipliers: steer toward what is currently lacking
        if avg_captures < 1.0:
            # blue is not capping at all - massively incentivise capture
            capture_mult = 2.0
            kill_mult    = 0.7
            defend_mult  = 0.8
        elif win_rate < 0.3:
            # losing a lot - push aggression
            capture_mult = 1.1
            kill_mult    = 1.8
            defend_mult  = 1.4
        elif win_rate > 0.65:
            # doing well - slight push toward caps to close out games
            capture_mult = 1.3
            kill_mult    = 1.0
            defend_mult  = 0.9
        else:
            capture_mult = 1.2
            kill_mult    = 1.1
            defend_mult  = 1.0

        # epsilon nudges: good performers exploit more, bad ones explore more
        new_blue_eps = []
        for eps in blue_eps:
            if win_rate > 0.6:
                new_blue_eps.append(round(_clamp(eps * 0.88, 0.05, 0.70), 3))
            elif win_rate < 0.3:
                new_blue_eps.append(round(_clamp(eps * 1.15, 0.05, 0.70), 3))
            else:
                new_blue_eps.append(round(_clamp(eps, 0.05, 0.70), 3))

        if win_rate > 0.6:
            new_red_eps = round(_clamp(red_eps_cur * 0.90, 0.05, 0.70), 3)
        elif win_rate < 0.3:
            new_red_eps = round(_clamp(red_eps_cur * 1.10, 0.05, 0.70), 3)
        else:
            new_red_eps = round(_clamp(red_eps_cur, 0.05, 0.70), 3)

        return {
            "capture_mult": capture_mult,
            "kill_mult":    kill_mult,
            "defend_mult":  defend_mult,
            "blue_eps":     new_blue_eps,
            "red_eps":      new_red_eps,
            "rationale":    f"Rule-based fallback (win={win_rate:.0%}, caps={avg_captures:.1f})",
            "source":       "rule_based",
        }


class SmolAgentsFieldMarshal:
    # LLM-powered strategic advisor (Topic 3 - smolagents ToolCallingAgent)
    #
    # Every FM_STRATEGY_INTERVAL episodes it:
    #   1. Receives battle stats via tools
    #   2. Outputs reward multipliers (Option A) + per-commander epsilons (Option B)
    #   3. Falls back to RuleBasedFieldMarshal if the API call fails

    _SYSTEM_PROMPT = (
        "You are Generalfeldmarschall von Rundstedt. You command 4 German Tiger "
        "platoons (blue) against 12 Allied Sherman platoons (red) at Normandy 1944. "
        "You have 3 objectives: A, B (most valuable), C. You also control Luftwaffe air support.\n\n"
        "Use the three tools to gather battle data, then output EXACTLY this format on one line:\n\n"
        "CAPTURE_MULT=<0.5-2.5>  KILL_MULT=<0.5-2.5>  DEFEND_MULT=<0.5-2.5>  "
        "BLUE_EPS=<e0,e1,e2,e3>  RED_EPS=<r>  RATIONALE=<one sentence>\n\n"
        "CAPTURE_MULT, KILL_MULT, DEFEND_MULT: reward multipliers that reshape how much "
        "each action type is worth. Values above 1.0 push agents toward that behaviour.\n"
        "BLUE_EPS: comma-separated exploration rates for blue commanders 0-3 (range 0.05-0.70). "
        "Lower = more exploitation of learned policy. Higher = more random exploration.\n"
        "RED_EPS: single exploration rate for all red commanders (range 0.05-0.70).\n\n"
        "Example of good reasoning: if win rate is low but captures are decent, "
        "push KILL_MULT up and nudge low-performing blue commanders toward exploration."
    )

    def __init__(self, model_id: str = cfg.FM_MODEL_ID):
        self._fallback  = RuleBasedFieldMarshal()
        self._available = False
        self._agent     = None
        self._last_strategy: dict = {
            "capture_mult": 1.0,
            "kill_mult":    1.0,
            "defend_mult":  1.0,
            "blue_eps":     [0.3, 0.3, 0.3, 0.3],
            "red_eps":      0.3,
            "rationale":    "Initial defaults - waiting for first LLM call.",
            "source":       "default",
        }

        try:
            from smolagents import ToolCallingAgent, tool

            @tool
            def get_battle_performance(
                avg_reward: float,
                win_rate: float,
                avg_captures: float,
                blue_alive_avg: float,
            ) -> str:
                """
                Returns a formatted battle performance report for the last training window.

                Args:
                    avg_reward:     Average total episode reward over the last window.
                    win_rate:       Fraction of episodes won by the blue (German) team.
                    avg_captures:   Average objectives captured per episode (max 3).
                    blue_alive_avg: Average blue platoons still alive at episode end.
                """
                if win_rate >= 0.6:
                    trend = "performing well"
                elif win_rate >= 0.35:
                    trend = "struggling"
                else:
                    trend = "losing badly"
                return (
                    f"BATTLE PERFORMANCE\n"
                    f"  avg reward      : {avg_reward:.1f}\n"
                    f"  win rate        : {win_rate:.1%}\n"
                    f"  avg caps / ep   : {avg_captures:.2f} / 3\n"
                    f"  avg blue alive  : {blue_alive_avg:.1f} / 4\n"
                    f"  overall trend   : {trend}\n"
                )

            @tool
            def get_commander_stats(
                blue_epsilons_csv: str,
                blue_q_means_csv: str,
                red_epsilon: float,
            ) -> str:
                """
                Returns per-commander exploration rates and average Q-values.
                Use this to identify which commanders need more exploration vs exploitation.

                Args:
                    blue_epsilons_csv: Comma-separated epsilons for blue commanders 0-3.
                    blue_q_means_csv:  Comma-separated mean Q-table values for blue commanders 0-3.
                    red_epsilon:       Current exploration rate shared by all red commanders.
                """
                lines = ["COMMANDER STATS\n"]
                try:
                    eps_vals = [float(x) for x in blue_epsilons_csv.split(",")]
                    q_vals   = [float(x) for x in blue_q_means_csv.split(",")]
                    for i, (eps, q) in enumerate(zip(eps_vals, q_vals)):
                        confidence = "confident" if q > 0.01 else ("learning" if q > -0.01 else "confused")
                        lines.append(f"  blue cmd {i}: eps={eps:.3f}  q_mean={q:.4f}  status={confidence}\n")
                except Exception:
                    lines.append("  (parse error - raw values not available)\n")
                lines.append(f"  red (all 12): eps={red_epsilon:.3f}\n")
                return "".join(lines)

            @tool
            def get_luftwaffe_report(missions_done: int, lw_active: bool) -> str:
                """
                Returns Luftwaffe air support status.

                Args:
                    missions_done: Total bombing missions completed this training run.
                    lw_active:     Whether the aircraft is still operational (not shot down).
                """
                status = "operational" if lw_active else "SHOT DOWN"
                return (
                    f"LUFTWAFFE REPORT\n"
                    f"  status        : {status}\n"
                    f"  missions done : {missions_done}\n"
                )

            self._agent = ToolCallingAgent(
                tools=[get_battle_performance, get_commander_stats, get_luftwaffe_report],
                model=_build_llm_model(),
                max_steps=4,
            )
            self._available = True
            print(f"[FieldMarshal] smolagents ready -- model: {model_id}")

        except ImportError as exc:
            print(f"[FieldMarshal] smolagents import error ({exc}) -- using rule-based fallback.")
        except Exception as exc:
            print(f"[FieldMarshal] smolagents init failed ({exc}) -- using rule-based fallback.")

    def update_strategy(self, stats: dict) -> dict:
        # query the LLM and return a strategy dict with reward multipliers + epsilons
        # silently falls back to rule-based if anything goes wrong
        if not self._available:
            self._last_strategy = self._fallback.update_strategy(stats)
            return self._last_strategy

        try:
            blue_eps_csv = ",".join(f"{e:.3f}" for e in stats.get("blue_epsilons", [0.3, 0.3, 0.3, 0.3]))
            q_means_csv  = ",".join(f"{q:.4f}" for q in stats.get("blue_q_means",  [0.0,  0.0,  0.0,  0.0]))

            task = (
                f"{self._SYSTEM_PROMPT}\n\n"
                f"Analyse the recent battle data and output your strategic directive.\n"
                f"Call get_battle_performance with: "
                f"avg_reward={stats.get('avg_reward', 0):.1f}, "
                f"win_rate={stats.get('win_rate', 0):.3f}, "
                f"avg_captures={stats.get('avg_captures', 0):.2f}, "
                f"blue_alive_avg={stats.get('blue_alive_avg', 0):.1f}.\n"
                f"Call get_commander_stats with: "
                f"blue_epsilons_csv='{blue_eps_csv}', "
                f"blue_q_means_csv='{q_means_csv}', "
                f"red_epsilon={stats.get('red_epsilon', 0.3):.3f}.\n"
                f"Call get_luftwaffe_report with: "
                f"missions_done={stats.get('lw_missions', 0)}, "
                f"lw_active={stats.get('lw_active', True)}.\n"
                f"Then output your directive in the required format."
            )
            last_exc = None
            for attempt in range(3):
                try:
                    _sink = io.StringIO()
                    with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
                        raw = self._agent.run(task)
                    break
                except Exception as e:
                    last_exc = e
                    if attempt < 2:
                        continue
            else:
                raise last_exc

            strategy = self._parse_directive(str(raw), stats)
            self._last_strategy = strategy
            print(
                f"[FieldMarshal] LLM directive -> "
                f"cap*{strategy['capture_mult']}  kil*{strategy['kill_mult']}  def*{strategy['defend_mult']}  "
                f"blue_eps={strategy['blue_eps']}  red_eps={strategy['red_eps']}\n"
                f"  Rationale: {strategy['rationale']}"
            )
            return strategy

        except Exception as exc:
            err = str(exc).lower()
            if "rate limit" in err or "429" in err:
                self._available = False
                self._last_strategy = self._fallback.update_strategy(stats)
                self._last_strategy["source"] = "rate_limit_fallback"
                print("[FieldMarshal] rate limit reached -- switching to rule-based fallback.")
                return self._last_strategy
            print("[FieldMarshal] LLM call failed -- keeping last strategy.")
            return self._last_strategy

    @property
    def current_strategy(self) -> dict:
        return self._last_strategy

    def _parse_directive(self, raw: str, stats: dict) -> dict:
        # extract all fields from the LLM output string using regex
        # falls back to rule-based if parsing fails

        def parse_float(pattern, default):
            m = re.search(pattern, raw, re.I)
            return _clamp(float(m.group(1)), 0.5, 2.5) if m else default

        capture_mult = parse_float(r"CAPTURE_MULT\s*=\s*([0-9.]+)", 1.0)
        kill_mult    = parse_float(r"KILL_MULT\s*=\s*([0-9.]+)",    1.0)
        defend_mult  = parse_float(r"DEFEND_MULT\s*=\s*([0-9.]+)",  1.0)

        # per-commander epsilons (4 comma-separated floats)
        blue_eps = [0.3, 0.3, 0.3, 0.3]
        m = re.search(r"BLUE_EPS\s*=\s*([\d.,\s]+)", raw, re.I)
        if m:
            try:
                parsed = [_clamp(float(x.strip()), 0.05, 0.70) for x in m.group(1).split(",")]
                if len(parsed) == 4:
                    blue_eps = [round(v, 3) for v in parsed]
            except (ValueError, IndexError):
                pass

        red_eps = stats.get("red_epsilon", 0.3)
        m = re.search(r"RED_EPS\s*=\s*([0-9.]+)", raw, re.I)
        if m:
            red_eps = round(_clamp(float(m.group(1)), 0.05, 0.70), 3)

        m = re.search(r"RATIONALE\s*=\s*(.+)", raw, re.I)
        rationale = m.group(1).strip() if m else raw[:150]

        # if we couldn't parse at least one multiplier, fall back entirely
        if not re.search(r"CAPTURE_MULT|KILL_MULT|DEFEND_MULT", raw, re.I):
            fallback = self._fallback.update_strategy(stats)
            fallback["source"] = "parse_fallback"
            return fallback

        return {
            "capture_mult": capture_mult,
            "kill_mult":    kill_mult,
            "defend_mult":  defend_mult,
            "blue_eps":     blue_eps,
            "red_eps":      red_eps,
            "rationale":    rationale,
            "source":       "llm",
        }


# backward-compatible alias so existing imports still work
field_marshal = RuleBasedFieldMarshal


# =============================================================================
#  Allied Field Marshal (red team advisor)
#
#  Mirror of SmolAgentsFieldMarshal but for the Allied side.
#  Controls exploration rates for the 12 red commanders, grouped into 4 squads
#  of 3 (squad 0 = cmds 0-2, squad 1 = cmds 3-5, squad 2 = cmds 6-8, squad 3 = cmds 9-11).
#
#  Only epsilon is adjusted for red - reward multipliers belong to the blue FM.
#  Output format:
#    RED_EPS=e0,e1,e2,e3  RATIONALE=<one sentence>
# =============================================================================

class RuleBasedAlliedMarshal:
    # rule-based fallback for the Allied (red) side

    def update_strategy(self, stats: dict) -> dict:
        red_win_rate  = stats.get("red_win_rate",  0.5)
        avg_red_alive = stats.get("avg_red_alive", 6.0)
        group_eps     = stats.get("red_group_eps", [0.3, 0.3, 0.3, 0.3])

        new_eps = []
        for eps in group_eps:
            if red_win_rate > 0.6:
                # winning - exploit more
                new_eps.append(round(_clamp(eps * 0.88, 0.05, 0.70), 3))
            elif red_win_rate < 0.3:
                # losing - explore more, maybe find better tactics
                new_eps.append(round(_clamp(eps * 1.15, 0.05, 0.70), 3))
            else:
                new_eps.append(round(_clamp(eps, 0.05, 0.70), 3))

        return {
            "red_group_eps": new_eps,
            "rationale":     f"Allied fallback (red_win={red_win_rate:.0%}, avg_alive={avg_red_alive:.1f})",
            "source":        "rule_based",
        }


class AlliedFieldMarshal:
    # LLM-powered advisor for the Allied (red) side
    # Controls per-squad epsilon for the 12 red Sherman commanders
    # Completely separate from SmolAgentsFieldMarshal (different prompt, different tools, different role)

    _SYSTEM_PROMPT = (
        "You are General Omar Bradley commanding 12 Allied Sherman platoons at Normandy, 1944. "
        "Your forces are split into 4 squads of 3 Shermans each (Squad 0-3). "
        "You face 4 German Tiger platoons - your advantage is numbers, not armour. "
        "Your three objectives are A, B (most valuable), and C.\n\n"
        "Use the tools to analyse your squads performance, then output EXACTLY this format:\n\n"
        "RED_EPS=<e0,e1,e2,e3>  RATIONALE=<one sentence>\n\n"
        "RED_EPS: comma-separated exploration rates for squads 0-3 (range 0.05-0.70). "
        "Lower means the squad sticks to its learned tactics. "
        "Higher means the squad tries new approaches.\n\n"
        "Squads that are getting wiped out should explore more (higher epsilon). "
        "Squads that are surviving and contributing should exploit their strategy (lower epsilon). "
        "If all squads are losing, raise all epsilons to force new behaviours."
    )

    def __init__(self, model_id: str = cfg.FM_MODEL_ID):
        self._fallback  = RuleBasedAlliedMarshal()
        self._available = False
        self._agent     = None
        self._last_strategy: dict = {
            "red_group_eps": [0.3, 0.3, 0.3, 0.3],
            "rationale":     "Initial defaults - waiting for first LLM call.",
            "source":        "default",
        }

        try:
            from smolagents import ToolCallingAgent, tool

            @tool
            def get_allied_battle_performance(
                red_win_rate: float,
                avg_red_alive: float,
                avg_blue_alive: float,
            ) -> str:
                """
                Returns a battle performance report from the Allied perspective.

                Args:
                    red_win_rate:   Fraction of episodes where red team won (all blue eliminated).
                    avg_red_alive:  Average red Shermans still alive at episode end (max 12).
                    avg_blue_alive: Average German Tigers still alive at episode end (max 4).
                """
                if red_win_rate >= 0.6:
                    trend = "dominating"
                elif red_win_rate >= 0.35:
                    trend = "contested"
                else:
                    trend = "being overwhelmed"
                return (
                    f"ALLIED BATTLE REPORT\n"
                    f"  red win rate    : {red_win_rate:.1%}\n"
                    f"  avg red alive   : {avg_red_alive:.1f} / 12\n"
                    f"  avg blue alive  : {avg_blue_alive:.1f} / 4\n"
                    f"  overall trend   : {trend}\n"
                )

            @tool
            def get_squad_stats(
                squad_epsilons_csv: str,
                squad_q_means_csv: str,
            ) -> str:
                """
                Returns per-squad exploration rates and average Q-values for the 4 Allied squads.
                Each squad contains 3 Sherman commanders.

                Args:
                    squad_epsilons_csv: Comma-separated current epsilon for squads 0-3.
                    squad_q_means_csv:  Comma-separated mean Q-table value for squads 0-3.
                """
                lines = ["SQUAD STATS (3 Shermans each)\n"]
                try:
                    eps_vals = [float(x) for x in squad_epsilons_csv.split(",")]
                    q_vals   = [float(x) for x in squad_q_means_csv.split(",")]
                    for i, (eps, q) in enumerate(zip(eps_vals, q_vals)):
                        status = "effective" if q > 0.01 else ("adapting" if q > -0.01 else "struggling")
                        lines.append(f"  squad {i}: eps={eps:.3f}  q_mean={q:.4f}  status={status}\n")
                except Exception:
                    lines.append("  (parse error)\n")
                return "".join(lines)

            self._agent = ToolCallingAgent(
                tools=[get_allied_battle_performance, get_squad_stats],
                model=_build_llm_model(),
                max_steps=3,
            )
            self._available = True
            print(f"[AlliedMarshal] smolagents ready -- model: {model_id}")

        except ImportError as exc:
            print(f"[AlliedMarshal] smolagents import error ({exc}) -- using rule-based fallback.")
        except Exception as exc:
            print(f"[AlliedMarshal] smolagents init failed ({exc}) -- using rule-based fallback.")

    def update_strategy(self, stats: dict) -> dict:
        # query LLM with red team stats and return individual squad epsilons
        if not self._available:
            self._last_strategy = self._fallback.update_strategy(stats)
            return self._last_strategy

        try:
            sq_eps_csv = ",".join(f"{e:.3f}" for e in stats.get("red_group_eps",  [0.3, 0.3, 0.3, 0.3]))
            sq_q_csv   = ",".join(f"{q:.4f}" for q in stats.get("red_group_qmeans", [0.0, 0.0, 0.0, 0.0]))

            task = (
                f"{self._SYSTEM_PROMPT}\n\n"
                f"Analyse your squads and decide on exploration rates.\n"
                f"Call get_allied_battle_performance with: "
                f"red_win_rate={stats.get('red_win_rate', 0):.3f}, "
                f"avg_red_alive={stats.get('avg_red_alive', 6):.1f}, "
                f"avg_blue_alive={stats.get('avg_blue_alive', 2):.1f}.\n"
                f"Call get_squad_stats with: "
                f"squad_epsilons_csv='{sq_eps_csv}', "
                f"squad_q_means_csv='{sq_q_csv}'.\n"
                f"Then output your directive in the required format."
            )
            last_exc = None
            for attempt in range(3):
                try:
                    _sink = io.StringIO()
                    with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
                        raw = self._agent.run(task)
                    break
                except Exception as e:
                    last_exc = e
                    if attempt < 2:
                        continue
            else:
                raise last_exc

            strategy = self._parse_directive(str(raw), stats)
            self._last_strategy = strategy
            print(
                f"[AlliedMarshal] LLM directive -> "
                f"red_squad_eps={strategy['red_group_eps']}\n"
                f"  Rationale: {strategy['rationale']}"
            )
            return strategy

        except Exception as exc:
            err = str(exc).lower()
            if "rate limit" in err or "429" in err:
                self._available = False
                self._last_strategy = self._fallback.update_strategy(stats)
                self._last_strategy["source"] = "rate_limit_fallback"
                print("[AlliedMarshal] rate limit reached -- switching to rule-based fallback.")
                return self._last_strategy
            print("[AlliedMarshal] LLM call failed -- keeping last strategy.")
            return self._last_strategy

    @property
    def current_strategy(self) -> dict:
        return self._last_strategy

    def _parse_directive(self, raw: str, stats: dict) -> dict:
        red_eps = [0.3, 0.3, 0.3, 0.3]
        m = re.search(r"RED_EPS\s*=\s*([\d.,\s]+)", raw, re.I)
        if m:
            try:
                parsed = [_clamp(float(x.strip()), 0.05, 0.70) for x in m.group(1).split(",")]
                if len(parsed) == 4:
                    red_eps = [round(v, 3) for v in parsed]
            except (ValueError, IndexError):
                pass

        m = re.search(r"RATIONALE\s*=\s*(.+)", raw, re.I)
        rationale = m.group(1).strip() if m else raw[:150]

        if not re.search(r"RED_EPS", raw, re.I):
            fallback = self._fallback.update_strategy(stats)
            fallback["source"] = "parse_fallback"
            return fallback

        return {
            "red_group_eps": red_eps,
            "rationale":     rationale,
            "source":        "llm",
        }
