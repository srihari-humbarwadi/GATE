from setuptools import setup

setup(
    name="gate",
    version="0.2.0",
    packages=[
        "gate.adaptation_schemes",
        "gate.model_blocks",
        "gate.model_blocks.auto_builder_modules",
        "gate.model_blocks.tokenizers",
        "gate.datasets",
        "gate.models",
        "gate.tasks",
        "gate.utils",
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
