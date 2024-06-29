"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_car_sale


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_car_sale,
            inputs="car_sale_dataset",
            outputs="preprocess_car_sale_data",
            name="preprocess_car_sale_node",
        ),
    ])
