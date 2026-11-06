from setuptools import setup

setup(
    name="mlp",
    version="1.0.8",
    author="tmouche",
    description="Multylayer perceptron framework",
    package=[
        "mlp",
        "mlp/core",
        "mlp/ml_tools",
        "mlp/utils"
    ],
    install_requires=[
        "numpy>=2.5.4",
        "pandas>=3.1.0",
        "plotly>=7.1.0",
    ]
)
