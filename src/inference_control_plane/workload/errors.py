"""Domain errors shared by Workload Optimizer services and routes."""


class WorkloadOptimizerError(RuntimeError):
    def __init__(
        self, message: str, *, status_code: int = 400, code: str = "workload_optimizer_error"
    ):
        self.status_code = status_code
        self.code = code
        super().__init__(message)
