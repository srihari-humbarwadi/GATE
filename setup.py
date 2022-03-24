from setuptools import setup

setup(
    name="gate",
    version="0.3.1",
    packages=[
        "gate.learners",
        "gate.model_blocks",
        "gate.datasets",
        "gate.datamodules",
        "gate.models",
        "gate.tasks",
        "gate.base",
        "gate",
    ],
    url="",
    license="",
    author="Antreas Antoniou",
    author_email="a.antoniou@ed.ac.uk",
    description="Generalization After Transfer Evaluation - A codebase enabling"
    " evaluation of architectures after transfer to a number of novel "
    "data domains, tasks and modalities",
)
