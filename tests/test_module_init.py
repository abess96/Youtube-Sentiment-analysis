"""
Tests for all new __init__.py modules introduced in this PR:
- src/config/__init__.py
- src/evaluation/__init__.py
- src/examples/__init__.py
- src/explainability/__init__.py
- src/models/__init__.py
- src/models/base/__init__.py
- src/models/deep_learning/__init__.py
- src/models/ensemble/__init__.py
- src/models/traditional/__init__.py
- src/models/transformers/__init__.py
- src/training/__init__.py
- src/mlops/__init__.py
- src/serving/__init__.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestConfigInitModule(unittest.TestCase):
    """Tests for src/config/__init__.py."""

    def test_importable(self):
        import src.config
        self.assertIsNotNone(src.config)

    def test_exports_setup_mlflow(self):
        from src.config import setup_mlflow
        self.assertTrue(callable(setup_mlflow))

    def test_exports_get_or_create_experiment(self):
        from src.config import get_or_create_experiment
        self.assertTrue(callable(get_or_create_experiment))

    def test_all_defined(self):
        import src.config
        self.assertTrue(hasattr(src.config, '__all__'))

    def test_all_contains_expected_names(self):
        import src.config
        self.assertIn('setup_mlflow', src.config.__all__)
        self.assertIn('get_or_create_experiment', src.config.__all__)


class TestEvaluationInitModule(unittest.TestCase):
    """Tests for src/evaluation/__init__.py."""

    def test_importable(self):
        import src.evaluation
        self.assertIsNotNone(src.evaluation)

    def test_has_docstring(self):
        import src.evaluation
        self.assertIsNotNone(src.evaluation.__doc__)

    def test_docstring_mentions_evaluation(self):
        import src.evaluation
        self.assertIn('evaluation', src.evaluation.__doc__.lower())


class TestExamplesInitModule(unittest.TestCase):
    """Tests for src/examples/__init__.py."""

    def test_importable(self):
        import src.examples
        self.assertIsNotNone(src.examples)

    def test_has_docstring(self):
        import src.examples
        self.assertIsNotNone(src.examples.__doc__)


class TestExplainabilityInitModule(unittest.TestCase):
    """Tests for src/explainability/__init__.py."""

    def test_importable(self):
        import src.explainability
        self.assertIsNotNone(src.explainability)

    def test_has_docstring(self):
        import src.explainability
        self.assertIsNotNone(src.explainability.__doc__)

    def test_docstring_mentions_explainability(self):
        import src.explainability
        doc_lower = src.explainability.__doc__.lower()
        self.assertTrue(
            'explainab' in doc_lower or 'interpret' in doc_lower
        )


class TestModelsInitModule(unittest.TestCase):
    """Tests for src/models/__init__.py."""

    def test_importable(self):
        import src.models
        self.assertIsNotNone(src.models)

    def test_has_docstring(self):
        import src.models
        self.assertIsNotNone(src.models.__doc__)

    def test_exports_base_model(self):
        from src.models import BaseModel
        self.assertIsNotNone(BaseModel)

    def test_exports_model_config(self):
        from src.models import ModelConfig
        self.assertIsNotNone(ModelConfig)

    def test_exports_prediction_result(self):
        from src.models import PredictionResult
        self.assertIsNotNone(PredictionResult)

    def test_exports_model_factory(self):
        from src.models import ModelFactory
        self.assertIsNotNone(ModelFactory)

    def test_all_defined(self):
        import src.models
        self.assertTrue(hasattr(src.models, '__all__'))

    def test_all_contains_expected_names(self):
        import src.models
        for name in ['BaseModel', 'ModelConfig', 'PredictionResult', 'ModelFactory']:
            self.assertIn(name, src.models.__all__)


class TestModelsBaseInitModule(unittest.TestCase):
    """Tests for src/models/base/__init__.py."""

    def test_importable(self):
        import src.models.base
        self.assertIsNotNone(src.models.base)

    def test_exports_base_model(self):
        from src.models.base import BaseModel
        self.assertIsNotNone(BaseModel)

    def test_exports_model_factory(self):
        from src.models.base import ModelFactory
        self.assertIsNotNone(ModelFactory)

    def test_exports_model_config(self):
        from src.models.base import ModelConfig
        self.assertIsNotNone(ModelConfig)

    def test_exports_prediction_result(self):
        from src.models.base import PredictionResult
        self.assertIsNotNone(PredictionResult)

    def test_all_defined(self):
        import src.models.base
        self.assertTrue(hasattr(src.models.base, '__all__'))

    def test_all_contains_expected_base_classes(self):
        import src.models.base
        for name in ['BaseModel', 'ModelConfig', 'PredictionResult', 'ModelFactory']:
            self.assertIn(name, src.models.base.__all__)


class TestModelsDeepLearningInitModule(unittest.TestCase):
    """Tests for src/models/deep_learning/__init__.py."""

    def test_importable(self):
        import src.models.deep_learning
        self.assertIsNotNone(src.models.deep_learning)

    def test_has_docstring(self):
        import src.models.deep_learning
        self.assertIsNotNone(src.models.deep_learning.__doc__)


class TestModelsEnsembleInitModule(unittest.TestCase):
    """Tests for src/models/ensemble/__init__.py."""

    def test_importable(self):
        import src.models.ensemble
        self.assertIsNotNone(src.models.ensemble)

    def test_has_docstring(self):
        import src.models.ensemble
        self.assertIsNotNone(src.models.ensemble.__doc__)


class TestModelsTraditionalInitModule(unittest.TestCase):
    """Tests for src/models/traditional/__init__.py."""

    def test_importable(self):
        import src.models.traditional
        self.assertIsNotNone(src.models.traditional)

    def test_has_docstring(self):
        import src.models.traditional
        self.assertIsNotNone(src.models.traditional.__doc__)


class TestModelsTransformersInitModule(unittest.TestCase):
    """Tests for src/models/transformers/__init__.py."""

    def test_importable(self):
        import src.models.transformers
        self.assertIsNotNone(src.models.transformers)

    def test_has_docstring(self):
        import src.models.transformers
        self.assertIsNotNone(src.models.transformers.__doc__)


class TestTrainingInitModule(unittest.TestCase):
    """Tests for src/training/__init__.py."""

    def test_importable(self):
        import src.training
        self.assertIsNotNone(src.training)

    def test_has_docstring(self):
        import src.training
        self.assertIsNotNone(src.training.__doc__)

    def test_docstring_mentions_training(self):
        import src.training
        self.assertIn('training', src.training.__doc__.lower())


class TestMlopsInitModule(unittest.TestCase):
    """Tests for src/mlops/__init__.py."""

    def test_importable(self):
        import src.mlops
        self.assertIsNotNone(src.mlops)

    def test_has_docstring(self):
        import src.mlops
        self.assertIsNotNone(src.mlops.__doc__)

    def test_docstring_mentions_mlops(self):
        import src.mlops
        doc_lower = src.mlops.__doc__.lower()
        self.assertTrue(
            'mlops' in doc_lower or 'experiment' in doc_lower or 'model' in doc_lower
        )


class TestServingInitModule(unittest.TestCase):
    """Tests for src/serving/__init__.py."""

    def test_importable(self):
        import src.serving
        self.assertIsNotNone(src.serving)

    def test_has_docstring(self):
        import src.serving
        self.assertIsNotNone(src.serving.__doc__)


if __name__ == '__main__':
    unittest.main()