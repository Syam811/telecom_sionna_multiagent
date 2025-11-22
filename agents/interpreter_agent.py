import re
import logging

class InterpreterAgent:
    def __init__(self):
        self.logger = logging.getLogger("InterpreterAgent")

    def run(self, prompt: str):
        prompt_l = prompt.lower()
        self.logger.info(f"Input prompt: {prompt}")

        # ---- MIMO DETECTION ----
        mimo_keywords = ["mimo", "1x1", "2x2", "4x4", "8x8", "antenna", "antennas"]
        if any(k in prompt_l for k in mimo_keywords):
            task_type = "mimo_comparison"
            self.logger.info(f"Classified task_type: {task_type}")
            return TaskSpec(task_type)

        # ---- MULTI RADIO MAP ----
        if "multi" in prompt_l and "radio" in prompt_l:
            task_type = "multi_radio_map"
            self.logger.info(f"Classified task_type: {task_type}")
            return TaskSpec(task_type)

        if "multi-transmitter" in prompt_l or "multiple tx" in prompt_l:
            task_type = "multi_radio_map"
            self.logger.info(f"Classified task_type: {task_type}")
            return TaskSpec(task_type)

        # ---- SINGLE RADIO MAP ----
        if "radio map" in prompt_l or "heatmap" in prompt_l or "coverage" in prompt_l:
            task_type = "radiomap"
            self.logger.info(f"Classified task_type: {task_type}")
            return TaskSpec(task_type)

        # ---- BER ----
        ber_keywords = ["ber", "bit error", "error rate"]
        if any(k in prompt_l for k in ber_keywords):
            task_type = "ber"
            self.logger.info(f"Classified task_type: {task_type}")
            return TaskSpec(task_type)

        # ---- CONSTELLATION ----
        const_keywords = ["constellation", "scatter", "symbol plot"]
        if any(k in prompt_l for k in const_keywords):
            task_type = "constellation"
            self.logger.info(f"Classified task_type: {task_type}")
            return TaskSpec(task_type)

        # Default fallback
        self.logger.info("Could not classify → defaulting to constellation")
        return TaskSpec("constellation")


class TaskSpec:
    def __init__(self, task_type):
        self.task_type = task_type
        self.parameters = {}
