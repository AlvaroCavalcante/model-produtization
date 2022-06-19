import datetime
import os
from typing import List

import tensorflow_model_analysis as tfma
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx import v1 as tfx
from absl import logging
logging.set_verbosity(logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_pipeline_name = 'cartola_pro_clients'

# CHANGE WITH YOUR PROJECT ROOT
_project_root = '/home/alvaro/Desktop/model_produtization'
_data_root = os.path.join(_project_root, 'data')

_module_file = os.path.join(_project_root, 'src/tfx_utils.py')

_serving_model_dir = os.path.join(
    _project_root, 'serving_model', _pipeline_name)


_tfx_root = os.path.join(_project_root, 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    '--direct_num_workers=0',
]

_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}


def _get_eval_config():
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='pro_target')],
        slicing_specs=[
            # Use the whole dataset.
            tfma.SlicingSpec(),
            # Calculate metrics for each class.
            tfma.SlicingSpec(feature_keys=['pro_target']),
        ],
        metrics_specs=[
            tfma.MetricsSpec(per_slice_thresholds={
                'binary_accuracy':
                    tfma.PerSliceMetricThresholds(thresholds=[
                        tfma.PerSliceMetricThreshold(
                            slicing_specs=[tfma.SlicingSpec()],
                            threshold=tfma.MetricThreshold(
                                value_threshold=tfma.GenericValueThreshold(
                                    lower_bound={'value': 0.4}),
                                change_threshold=tfma.GenericChangeThreshold(
                                    direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                    absolute={'value': -1e-10}))
                        )]),
            })],
    )
    return eval_config


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     metadata_path: str,
                     beam_pipeline_args: List[str]) -> pipeline.Pipeline:

    data_root_runtime = data_types.RuntimeParameter(
        'data_root', ptype=str, default=data_root)

    # Brings data into the pipeline.
    example_gen = tfx.components.CsvExampleGen(input_base=data_root_runtime)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])

    # Generates schema based on statistics files.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    # Performs transformations and feature engineering in training and serving.
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        materialize=False,
        module_file=module_file)

    # Implements keras model.
    trainer = tfx.components.Trainer(
        module_file=module_file,
        examples=example_gen.outputs['examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=tfx.proto.TrainArgs(num_steps=2000),
        eval_args=tfx.proto.EvalArgs(num_steps=400))

    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
                'latest_blessed_model_resolver')

    eval_config = _get_eval_config()
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, statistics_gen, schema_gen,
            example_validator, transform, trainer,
            model_resolver, evaluator, pusher
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path),
        beam_pipeline_args=beam_pipeline_args)


DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
    _create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        data_root=_data_root,
        module_file=_module_file,
        serving_model_dir=_serving_model_dir,
        metadata_path=_metadata_path,
        beam_pipeline_args=_beam_pipeline_args))
