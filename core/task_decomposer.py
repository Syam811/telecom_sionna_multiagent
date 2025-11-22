"""
Drop your existing TaskDecomposer here.
Must implement:
  classify(prompt) -> str task_type
  extract_params(prompt, task_type) -> dict
"""

class TaskDecomposer:
    def classify(self, prompt: str) -> str:
        # TODO: replace with your draft decomposer
        prompt_l = prompt.lower()
        if "constellation" in prompt_l or "qam" in prompt_l:
            return "constellation"
        if "ber" in prompt_l or "bit error" in prompt_l:
            return "ber"
        if "mimo" in prompt_l or "antenna" in prompt_l:
            return "mimo_comparison"
        if "radio map" in prompt_l or "coverage" in prompt_l:
            return "radiomap"
        if "multi" in prompt_l and "transmitter" in prompt_l:
            return "multi_radio_map"
        return "ber"

    def extract_params(self, prompt: str, task_type: str) -> dict:
        # TODO: replace with your draft extraction logic
        # return a dict: modulation, snr_db list/range, tx_pos, rx_pos, n_tx, n_rx, channel, etc.
        return {}
