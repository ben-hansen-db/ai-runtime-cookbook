"""Exercise the workload entry point without starting Ray or accessing a workspace."""
import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import MagicMock, patch

from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.sdk.service.catalog import CatalogInfo, CatalogType, SchemaInfo


class DriverTests(unittest.TestCase):
    def run_driver(self, client, run_id, trial_error=None):
        modules = {name: MagicMock() for name in (
            "numpy", "ray", "ray.tune", "ray.tune.schedulers", "sklearn",
            "sklearn.datasets", "sklearn.metrics", "sklearn.model_selection", "xgboost",
        )}
        reached_ray = AssertionError("Ray started")
        if trial_error is None:
            modules["ray"].init.side_effect = reached_ray
        else:
            modules["ray"].cluster_resources.return_value = {"GPU": 1}
            modules["ray"].tune = modules["ray.tune"]
            modules["ray.tune"].Tuner.return_value.fit.side_effect = trial_error
            modules["sklearn.model_selection"].train_test_split.return_value = (MagicMock(),) * 4
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"MLFLOW_RUN_ID": run_id, "HYPERPARAMETERS_PATH": ""}),
            patch("databricks.sdk.WorkspaceClient", return_value=client),
            self.assertRaises(Exception) as caught,
        ):
            runpy.run_path(str(Path(__file__).with_name("tune.py")), run_name="__main__")
        return caught.exception, modules["ray"].init

    def test_shared_catalog_stops_before_ray_and_any_schema_write(self):
        workspace = MagicMock()
        workspace.catalogs.get.return_value = CatalogInfo(catalog_type=CatalogType.DELTASHARING_CATALOG)
        workspace.schemas.create.side_effect = PermissionDenied("Policy 42")
        error, start_ray = self.run_driver(workspace, "tracking-run")
        self.assertIn("uc_catalog", str(error))
        self.assertIn("main.default.xgboost_covertype", str(error))
        start_ray.assert_not_called()
        workspace.catalogs.get.assert_called_once_with("main", include_browse=True)
        workspace.schemas.create.assert_not_called()

    def test_browse_only_parent_stops_registration(self):
        for catalog, schema in [(True, False), (False, True)]:
            with self.subTest(catalog=catalog):
                workspace = MagicMock()
                workspace.catalogs.get.return_value = CatalogInfo(browse_only=catalog)
                workspace.schemas.get.return_value = SchemaInfo(browse_only=schema)
                error, start_ray = self.run_driver(workspace, "tracking-run")
                self.assertIn("USE CATALOG", str(error))
                if schema:
                    self.assertIn("USE SCHEMA", str(error))
                    workspace.schemas.get.assert_called_once_with("main.default", include_browse=True)
                start_ray.assert_not_called()
                workspace.schemas.create.assert_not_called()

    def test_failed_search_leaves_missing_schema_uncreated(self):
        workspace = MagicMock()
        workspace.catalogs.get.return_value = CatalogInfo()
        workspace.schemas.get.side_effect = NotFound("absent", error_code="SCHEMA_DOES_NOT_EXIST")
        workspace.current_user.me.return_value.user_name = None
        workspace.current_user.me.return_value.application_id = None
        failed_trial = ValueError("trial failed")
        error, _ = self.run_driver(workspace, "tracking-run", trial_error=failed_trial)
        self.assertIs(error, failed_trial)
        workspace.schemas.create.assert_not_called()
        workspace.registered_models.get.assert_not_called()

    def test_off_platform_skips_registration_preflight(self):
        workspace = MagicMock()
        _, start_ray = self.run_driver(workspace, "")
        start_ray.assert_called_once()
        self.assertEqual(workspace.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
