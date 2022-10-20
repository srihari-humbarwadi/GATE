from gate.configs.task.image_classification import (HundredClassClassificationTask,
                                                    TenClassClassificationTask,
                                                    ThousandClassClassificationTask,
                                                    VariableClassClassificationTask)
from hydra.core.config_store import ConfigStore


def add_task_configs(config_store: ConfigStore):
    config_store.store(
        group="task",
        name="Classification10",
        node=TenClassClassificationTask,
    )

    config_store.store(
        group="task",
        name="Classification100",
        node=HundredClassClassificationTask,
    )

    config_store.store(
        group="task",
        name="Classification1000",
        node=ThousandClassClassificationTask,
    )

    config_store.store(
        group="task",
        name="VariableClassClassification",
        node=VariableClassClassificationTask,
    )

    return config_store
