import json
from main import TelecomMultiAgentAssistant

def run_eval(path="eval/sample_tasks.json"):
    assistant = TelecomMultiAgentAssistant()
    data = json.load(open(path))

    correct_task = 0
    correct_tool = 0

    for t in data:
        prompt = t["prompt"]
        task = assistant.interpreter.run(prompt)
        task = assistant.extractor.run(task)

        exp_type = t["expected_task_type"]
        exp_tool = t["expected_tool"]

        if task.task_type == exp_type:
            correct_task += 1

        tool_name = {
            "constellation": "simulate_constellation",
            "ber": "simulate_ber",
            "mimo_comparison": "simulate_ber_mimo",
            "radiomap": "simulate_radio_map",
            "multi_radio_map": "simulate_multi_radio_map",
        }.get(task.task_type)

        if tool_name == exp_tool:
            correct_tool += 1

    print(f"Task-type accuracy: {correct_task}/{len(data)}")
    print(f"Tool-choice accuracy: {correct_tool}/{len(data)}")

if __name__ == "__main__":
    run_eval()
