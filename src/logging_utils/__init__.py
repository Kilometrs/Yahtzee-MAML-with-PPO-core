import os


def create_logger(project_name: str, task_name: str, config: dict):
    """Return ClearMLLogger or TensorBoardLogger based on CLEARML_OFF env var."""
    if os.environ.get("CLEARML_OFF", "").strip() in ("1", "true", "yes"):
        from logging_utils.tensorboard_logger import TensorBoardLogger
        return TensorBoardLogger(project_name, task_name, config)
    from logging_utils.clearml_logger import ClearMLLogger
    return ClearMLLogger(project_name, task_name, config)
