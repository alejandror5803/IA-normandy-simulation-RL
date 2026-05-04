# Field Marshal - strategic command layer for the blue (German) team
#
# AI technique from the course used here:
#
#   smolagents (Topic 3)
#     LLM-based strategic advisor implemented with a smolagents ToolCallingAgent.
#     The idea is the same as wrapping an environment as tools that the agent can call,
#     which is what we saw in class with CartPole-v1.
#     To avoid burning HuggingFace API quota the LLM is only called every
#     FM_STRATEGY_INTERVAL episodes (default 500). Between calls the last
#     strategy directive stays active.
#
#     Tools given to the agent:
#       get_battle_performance  - formatted report of recent training stats
#       get_luftwaffe_report    - Luftwaffe mission count and aircraft status
#
#     The agent outputs a structured directive:
#       PRIORITY=<CAPTURE|ATTACK|DEFEND>  AGGRESSION=<LOW|MEDIUM|HIGH>
#     which the training loop uses to add a small shaped reward bonus when
#     the commander follows the recommended priority action.
#
# If smolagents is not installed or the API call fails we fall back to the
# original rule-based implementation so the simulation still runs.

import re
import env.env_config as cfg

# meta-action codes (shared with normandy_env)
CAPTURE_A = 0
CAPTURE_B = 1
CAPTURE_C = 2
ATTACK    = 3
DEFEND    = 4

# maps LLM directive strings to META-action indices used by command_agent
_PRIORITY_MAP = {
    "CAPTURE": cfg.META_CAPTURE,
    "ATTACK":  cfg.META_ATTACK,
    "DEFEND":  cfg.META_DEFENSE,
}


class RuleBasedFieldMarshal:
    # original heuristic field marshal, kept as fallback when LLM is unavailable

    def count_alive(self, platoons):
        return sum(1 for p in platoons if p.is_alive())

    def count_enemies_in_range(self, observations):
        return sum(len(obs["enemies_in_range"]) for obs in observations)

    def choose_action(self, blue_platoons, red_platoons, objectives, observations):
        blue_alive           = self.count_alive(blue_platoons)
        red_alive            = self.count_alive(red_platoons)
        total_enemies_nearby = self.count_enemies_in_range(observations)

        if blue_alive < red_alive / 2:
            return DEFEND
        if total_enemies_nearby >= 3:
            return DEFEND
        if not objectives["B"]["captured"]:
            return CAPTURE_B
        if not objectives["A"]["captured"]:
            return CAPTURE_A
        if not objectives["C"]["captured"]:
            return CAPTURE_C
        return ATTACK

    def update_strategy(self, stats: dict) -> dict:
        # compatibility shim - same interface as SmolAgentsFieldMarshal.update_strategy
        win_rate     = stats.get("win_rate", 0.5)
        avg_captures = stats.get("avg_captures", 0.0)

        if win_rate < 0.25:
            priority = cfg.META_DEFENSE
            label    = "DEFEND"
        elif avg_captures < 1.5:
            priority = cfg.META_CAPTURE
            label    = "CAPTURE"
        else:
            priority = cfg.META_ATTACK
            label    = "ATTACK"

        return {
            "priority_action": priority,
            "priority_label":  label,
            "aggression":      "MEDIUM",
            "rationale":       f"Rule-based fallback (win_rate={win_rate:.0%})",
            "source":          "rule_based",
        }


class SmolAgentsFieldMarshal:
    # LLM-powered strategic advisor using the smolagents ToolCallingAgent (Topic 3)
    #
    # The LLM is queried every FM_STRATEGY_INTERVAL episodes. If smolagents is not
    # installed or the API call fails, it transparently delegates to RuleBasedFieldMarshal.

    _SYSTEM_PROMPT = (
        "You are Generalfeldmarschall von Rundstedt commanding German Tiger tank "
        "platoons at Normandy, 1944. You have 4 Tiger platoons (B0-B3) against 12 "
        "Allied Sherman platoons. Your three objectives are A, B (most valuable), "
        "and C. You also have Luftwaffe air support.\n\n"
        "Analyse the battle performance data provided by your tools and output "
        "exactly one line with this format:\n"
        "PRIORITY=<CAPTURE|ATTACK|DEFEND>  AGGRESSION=<LOW|MEDIUM|HIGH>  "
        "RATIONALE=<one sentence>\n\n"
        "Use CAPTURE when the main bottleneck is not holding objectives. "
        "Use ATTACK when losses to enemy fire are the core problem. "
        "Use DEFEND when survival rate is critically low."
    )

    def __init__(self, model_id: str = cfg.FM_MODEL_ID):
        self._fallback  = RuleBasedFieldMarshal()
        self._available = False
        self._agent     = None
        self._last_strategy: dict = {
            "priority_action": cfg.META_CAPTURE,
            "priority_label":  "CAPTURE",
            "aggression":      "MEDIUM",
            "rationale":       "Initial default — waiting for first LLM call.",
            "source":          "default",
        }

        try:
            from smolagents import ToolCallingAgent, HfApiModel, tool

            @tool
            def get_battle_performance(
                avg_reward: float,
                win_rate: float,
                avg_captures: float,
                blue_alive_avg: float,
            ) -> str:
                """
                Returns a formatted battle performance report.

                Args:
                    avg_reward:     Average total episode reward over the last window.
                    win_rate:       Fraction of episodes won by the blue (German) team.
                    avg_captures:   Average number of objectives captured per episode.
                    blue_alive_avg: Average number of blue platoons alive at episode end.
                """
                if win_rate >= 0.6:
                    trend = "performing well"
                elif win_rate >= 0.35:
                    trend = "struggling"
                else:
                    trend = "losing badly"
                return (
                    f"BATTLE PERFORMANCE REPORT\n"
                    f"  Average reward   : {avg_reward:.1f}\n"
                    f"  Win rate         : {win_rate:.1%}\n"
                    f"  Avg obj captured : {avg_captures:.2f} / 3\n"
                    f"  Avg platoons left: {blue_alive_avg:.1f} / 4\n"
                    f"  Overall trend    : {trend}\n"
                )

            @tool
            def get_luftwaffe_report(missions_done: int, lw_active: bool) -> str:
                """
                Returns a Luftwaffe air support status report.

                Args:
                    missions_done: Total bombing missions completed this training run.
                    lw_active:     Whether the aircraft is still operational.
                """
                status = "operational" if lw_active else "SHOT DOWN"
                return (
                    f"LUFTWAFFE REPORT\n"
                    f"  Status        : {status}\n"
                    f"  Missions done : {missions_done}\n"
                )

            self._agent = ToolCallingAgent(
                tools=[get_battle_performance, get_luftwaffe_report],
                model=HfApiModel(model_id),
                system_prompt=self._SYSTEM_PROMPT,
                max_steps=4,
            )
            self._available = True
            print(f"[FieldMarshal] smolagents ready — model: {model_id}")

        except ImportError:
            print("[FieldMarshal] smolagents not installed — using rule-based fallback.")
        except Exception as exc:
            print(f"[FieldMarshal] smolagents init failed ({exc}) — using rule-based fallback.")

    def update_strategy(self, stats: dict) -> dict:
        # query the LLM with current battle stats and return a strategy dict
        # falls back to rule-based if smolagents is unavailable or the call fails
        if not self._available:
            self._last_strategy = self._fallback.update_strategy(stats)
            return self._last_strategy

        try:
            task = (
                f"Analyse the following recent battle data and recommend a strategy.\n"
                f"Call get_battle_performance with: "
                f"avg_reward={stats.get('avg_reward', 0):.1f}, "
                f"win_rate={stats.get('win_rate', 0):.3f}, "
                f"avg_captures={stats.get('avg_captures', 0):.2f}, "
                f"blue_alive_avg={stats.get('blue_alive_avg', 0):.1f}.\n"
                f"Call get_luftwaffe_report with: "
                f"missions_done={stats.get('lw_missions', 0)}, "
                f"lw_active={stats.get('lw_active', True)}.\n"
                f"Then output your strategic directive."
            )
            raw      = self._agent.run(task)
            strategy = self._parse_directive(str(raw), stats)
            self._last_strategy = strategy
            print(
                f"[FieldMarshal] LLM directive -> "
                f"PRIORITY={strategy['priority_label']}  "
                f"AGGRESSION={strategy['aggression']}\n"
                f"  Rationale: {strategy['rationale']}"
            )
            return strategy

        except Exception as exc:
            print(f"[FieldMarshal] LLM call failed ({exc}) — keeping last strategy.")
            return self._last_strategy

    @property
    def current_strategy(self) -> dict:
        return self._last_strategy

    def _parse_directive(self, raw: str, stats: dict) -> dict:
        # extract PRIORITY / AGGRESSION / RATIONALE from the LLM output string
        # falls back to rule-based if the LLM didn't follow the expected format
        priority_match   = re.search(r"PRIORITY\s*=\s*(CAPTURE|ATTACK|DEFEND)", raw, re.I)
        aggression_match = re.search(r"AGGRESSION\s*=\s*(LOW|MEDIUM|HIGH)", raw, re.I)
        rationale_match  = re.search(r"RATIONALE\s*=\s*(.+)", raw, re.I)

        if not priority_match:
            fallback = self._fallback.update_strategy(stats)
            fallback["source"] = "parse_fallback"
            return fallback

        priority_label = priority_match.group(1).upper()
        aggression     = aggression_match.group(1).upper() if aggression_match else "MEDIUM"
        rationale      = rationale_match.group(1).strip()  if rationale_match  else raw[:120]

        return {
            "priority_action": _PRIORITY_MAP.get(priority_label, cfg.META_CAPTURE),
            "priority_label":  priority_label,
            "aggression":      aggression,
            "rationale":       rationale,
            "source":          "llm",
        }


# backward-compatible alias so existing imports still work
field_marshal = RuleBasedFieldMarshal
