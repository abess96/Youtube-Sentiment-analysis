"""
Tests for params.yaml and dvc.yaml structural changes introduced in this PR.

params.yaml changes:
- mlflow.experiments dict with 7 structured experiment names
- Legacy mlflow.experiment_name kept for backward compatibility

dvc.yaml changes:
- Stage commands updated to new module paths
- Dependencies updated to new file locations
"""

import os
import sys
import unittest

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
PARAMS_PATH = os.path.join(REPO_ROOT, 'params.yaml')
DVC_PATH = os.path.join(REPO_ROOT, 'dvc.yaml')


def load_params():
    with open(PARAMS_PATH, 'r') as f:
        return yaml.safe_load(f)


def load_dvc():
    with open(DVC_PATH, 'r') as f:
        return yaml.safe_load(f)


class TestParamsYamlMlflowExperiments(unittest.TestCase):
    """Tests for the new mlflow.experiments block in params.yaml."""

    def setUp(self):
        self.params = load_params()

    def test_mlflow_key_exists(self):
        self.assertIn('mlflow', self.params)

    def test_mlflow_experiments_key_exists(self):
        self.assertIn('experiments', self.params['mlflow'])

    def test_experiments_has_data_preprocessing(self):
        experiments = self.params['mlflow']['experiments']
        self.assertIn('data_preprocessing', experiments)

    def test_experiments_has_feature_engineering(self):
        experiments = self.params['mlflow']['experiments']
        self.assertIn('feature_engineering', experiments)

    def test_experiments_has_model_training(self):
        experiments = self.params['mlflow']['experiments']
        self.assertIn('model_training', experiments)

    def test_experiments_has_model_evaluation(self):
        experiments = self.params['mlflow']['experiments']
        self.assertIn('model_evaluation', experiments)

    def test_experiments_has_model_registry(self):
        experiments = self.params['mlflow']['experiments']
        self.assertIn('model_registry', experiments)

    def test_experiments_has_continuous_learning(self):
        experiments = self.params['mlflow']['experiments']
        self.assertIn('continuous_learning', experiments)

    def test_experiments_has_experiment_tracking(self):
        experiments = self.params['mlflow']['experiments']
        self.assertIn('experiment_tracking', experiments)

    def test_experiment_names_are_numbered(self):
        """Each experiment name should start with a two-digit number prefix."""
        experiments = self.params['mlflow']['experiments']
        for key, name in experiments.items():
            self.assertRegex(
                name,
                r'^\d{2}_',
                msg=f"Experiment '{key}' name '{name}' does not start with a numeric prefix"
            )

    def test_experiment_names_are_unique(self):
        """All experiment names should be unique."""
        experiments = self.params['mlflow']['experiments']
        names = list(experiments.values())
        self.assertEqual(len(names), len(set(names)))

    def test_data_preprocessing_experiment_name(self):
        name = self.params['mlflow']['experiments']['data_preprocessing']
        self.assertEqual(name, '01_Data_Preprocessing')

    def test_feature_engineering_experiment_name(self):
        name = self.params['mlflow']['experiments']['feature_engineering']
        self.assertEqual(name, '02_Feature_Engineering')

    def test_model_training_experiment_name(self):
        name = self.params['mlflow']['experiments']['model_training']
        self.assertEqual(name, '03_Model_Training')

    def test_model_evaluation_experiment_name(self):
        name = self.params['mlflow']['experiments']['model_evaluation']
        self.assertEqual(name, '04_Model_Evaluation')

    def test_model_registry_experiment_name(self):
        name = self.params['mlflow']['experiments']['model_registry']
        self.assertEqual(name, '05_Model_Registry')

    def test_continuous_learning_experiment_name(self):
        name = self.params['mlflow']['experiments']['continuous_learning']
        self.assertEqual(name, '06_Continuous_Learning')

    def test_experiment_tracking_experiment_name(self):
        name = self.params['mlflow']['experiments']['experiment_tracking']
        self.assertEqual(name, '07_Experiment_Tracking')

    def test_legacy_experiment_name_preserved(self):
        """Legacy experiment_name key must exist for backward compatibility."""
        self.assertIn('experiment_name', self.params['mlflow'])
        self.assertIsNotNone(self.params['mlflow']['experiment_name'])

    def test_legacy_experiment_name_matches_model_training(self):
        """Legacy experiment_name should correspond to model training stage."""
        legacy = self.params['mlflow']['experiment_name']
        structured = self.params['mlflow']['experiments']['model_training']
        self.assertEqual(legacy, structured)

    def test_all_seven_experiments_defined(self):
        experiments = self.params['mlflow']['experiments']
        self.assertEqual(len(experiments), 7)

    def test_experiment_names_all_strings(self):
        experiments = self.params['mlflow']['experiments']
        for key, value in experiments.items():
            self.assertIsInstance(value, str, msg=f"Experiment '{key}' value is not a string")


class TestDvcYamlStagePaths(unittest.TestCase):
    """Tests that dvc.yaml stage commands reference the new module paths."""

    def setUp(self):
        self.dvc = load_dvc()
        self.stages = self.dvc.get('stages', {})

    def test_model_building_uses_new_path(self):
        cmd = self.stages['model_building']['cmd']
        self.assertIn('src/models/base/model_factory.py', cmd)
        self.assertNotIn('src/model/model_building.py', cmd)

    def test_advanced_model_training_uses_trainer(self):
        cmd = self.stages['advanced_model_training']['cmd']
        self.assertIn('src/training/trainer.py', cmd)
        self.assertNotIn('src/model/advanced_model_training.py', cmd)

    def test_model_evaluation_uses_new_path(self):
        cmd = self.stages['model_evaluation']['cmd']
        self.assertIn('src/evaluation/metrics.py', cmd)
        self.assertNotIn('src/model/model_evaluation.py', cmd)

    def test_model_registration_uses_new_path(self):
        cmd = self.stages['model_registration']['cmd']
        self.assertIn('src/mlops/model_registry.py', cmd)
        self.assertNotIn('src/model/register_model.py', cmd)

    def test_comprehensive_evaluation_uses_new_path(self):
        cmd = self.stages['comprehensive_evaluation']['cmd']
        self.assertIn('src/evaluation/evaluation_pipeline.py', cmd)
        self.assertNotIn('src/model/evaluation_pipeline.py', cmd)

    def test_enhanced_tracking_uses_new_path(self):
        cmd = self.stages['enhanced_tracking']['cmd']
        self.assertIn('src/mlops/experiment_tracking.py', cmd)
        self.assertNotIn('src/model/train_with_enhanced_tracking.py', cmd)

    def test_continuous_learning_uses_new_path(self):
        cmd = self.stages['continuous_learning']['cmd']
        self.assertIn('src/examples/continuous_learning_example.py', cmd)
        self.assertNotIn('src/model/continuous_learning_example.py', cmd)

    def test_advanced_model_training_deps_use_new_paths(self):
        deps = self.stages['advanced_model_training']['deps']
        dep_paths = [str(d) for d in deps]
        self.assertTrue(any('src/training/trainer.py' in d for d in dep_paths))
        self.assertTrue(any('src/models/base/model_factory.py' in d for d in dep_paths))
        self.assertTrue(any('src/models/base/base_model.py' in d for d in dep_paths))

    def test_comprehensive_evaluation_deps_use_new_paths(self):
        deps = self.stages['comprehensive_evaluation']['deps']
        dep_paths = [str(d) for d in deps]
        self.assertTrue(any('src/evaluation/evaluator.py' in d for d in dep_paths))
        self.assertTrue(any('src/evaluation/bias_detection.py' in d for d in dep_paths))
        self.assertTrue(any('src/evaluation/drift_detection.py' in d for d in dep_paths))

    def test_continuous_learning_deps_use_new_training_paths(self):
        deps = self.stages['continuous_learning']['deps']
        dep_paths = [str(d) for d in deps]
        self.assertTrue(any('src/training/incremental_learner.py' in d for d in dep_paths))
        self.assertTrue(any('src/training/active_learner.py' in d for d in dep_paths))
        self.assertTrue(any('src/training/auto_retrainer.py' in d for d in dep_paths))

    def test_continuous_learning_deps_use_evaluation_drift_detection(self):
        deps = self.stages['continuous_learning']['deps']
        dep_paths = [str(d) for d in deps]
        self.assertTrue(any('src/evaluation/drift_detection.py' in d for d in dep_paths))
        # Ensure old path not present
        self.assertFalse(any('src/model/drift_detector.py' in d for d in dep_paths))

    def test_enhanced_tracking_deps_no_old_paths(self):
        deps = self.stages['enhanced_tracking']['deps']
        dep_paths = [str(d) for d in deps]
        self.assertFalse(any('enhanced_mlflow_tracker.py' in d for d in dep_paths))
        self.assertFalse(any('model_lifecycle_manager.py' in d for d in dep_paths))

    def test_no_stage_references_old_model_dir(self):
        """No stage command should reference the old src/model/ directory."""
        for stage_name, stage_config in self.stages.items():
            cmd = stage_config.get('cmd', '')
            self.assertNotIn(
                'src/model/',
                cmd,
                msg=f"Stage '{stage_name}' command still references old src/model/ path: {cmd}"
            )

    def test_model_building_dep_uses_new_path(self):
        deps = self.stages['model_building']['deps']
        dep_paths = [str(d) for d in deps]
        self.assertTrue(any('src/models/base/model_factory.py' in d for d in dep_paths))


if __name__ == '__main__':
    unittest.main()