import datetime

from setuptools import setup

base_version = "0.3"
version_str = f"{base_version}.{datetime.date.today().month}.{datetime.date.today().day}.{datetime.datetime.now().hour}"

setup(
    name="gate",
    version=version_str,
    packages=[
        "gate.learners",
        "gate.model_blocks",
        "gate.model_blocks.auto_builder_modules",
        "gate.datasets",
        "gate.datamodules",
        "gate.datamodules.tf_hub",
        "gate.models",
        "gate.tasks",
        "gate.base",
        "gate.base.callbacks",
        "gate.base.utils",
        "gate.base.vendor",
        "gate.configs",
        "gate",
    ],
    url="https://github.com/AntreasAntoniou/GATE",
    license="GNU-v-3.0",
    author="Antreas Antoniou",
    author_email="a.antoniou@ed.ac.uk",
    description="Generalization After Transfer Evaluation on Modality, "
    "Task and Data Domain - A codebase enabling evaluation "
    "of architectures after transfer to a number of novel "
    "data domains, tasks and modalities",
)
