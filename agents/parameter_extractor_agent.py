from core.logger import setup_logger

class ParameterExtractorAgent:
    def __init__(self, decomposer):
        self.decomposer = decomposer
        self.logger = setup_logger("ParameterExtractorAgent")

    def run(self, task_spec):
        params = self.decomposer.extract_params(task_spec.raw_prompt, task_spec.task_type)
        task_spec.parameters = params
        self.logger.info(f"Extracted params: {params}")
        return task_spec
