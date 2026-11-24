"""Use SDK response types to exercise registration checks without workspace or GPU access."""
from contextlib import redirect_stdout
from io import StringIO
from unittest import TestCase, main
from unittest.mock import Mock

from databricks.sdk.errors import (
    AlreadyExists, BadRequest, NotFound, PermissionDenied, ResourceAlreadyExists,
    ResourceConflict, ResourceDoesNotExist, TemporarilyUnavailable, Unauthenticated,
)
from databricks.sdk.service import catalog, iam
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED

from registry import ensure_uc_schema, model_logging_errors, preflight_registration


CATALOG = "training"
SCHEMA = "training.models"
MODEL = "training.models.forest"
CALLER = "trainer@example.org"


class PreflightTests(TestCase):
    def setUp(self):
        self.workspace = Mock()
        self.workspace.catalogs.get.return_value = catalog.CatalogInfo()
        self.workspace.schemas.get.return_value = catalog.SchemaInfo()
        self.workspace.registered_models.get.return_value = catalog.RegisteredModelInfo()
        self.workspace.current_user.me.return_value = iam.User(user_name=CALLER)
        self.workspace.grants.get_effective.return_value = catalog.EffectivePermissionsList()

    def preflight(self):
        with redirect_stdout(StringIO()) as log:
            preflight_registration(self.workspace, CATALOG, "models", MODEL)
        self.workspace.schemas.create.assert_not_called()
        return log.getvalue()

    def test_parent_read_outage_still_checks_other_parent(self):
        self.workspace.catalogs.get.side_effect = TemporarilyUnavailable("temporarily unavailable")
        self.workspace.schemas.get.return_value = catalog.SchemaInfo(browse_only=True)
        with self.assertRaisesRegex(RuntimeError, "USE CATALOG.*USE SCHEMA"):
            self.preflight()

    def test_metadata_timeout_is_unknown(self):
        for api, method in [("catalogs", "get"), ("schemas", "get"), ("registered_models", "get"),
                            ("current_user", "me"), ("grants", "get_effective")]:
            with self.subTest(api=api):
                read = getattr(getattr(self.workspace, api), method)
                read.side_effect = TimeoutError("metadata retries exhausted")
                log = self.preflight()
                self.assertIn("metadata retries exhausted", log)
                self.assertIn("confirm", log)
                read.side_effect = None

    def test_incomplete_visibility_reports_both_model_operations(self):
        self.workspace.registered_models.get.side_effect = PermissionDenied("limited visibility")
        log = self.preflight()
        for hint in ["USE CATALOG", "USE SCHEMA", "CREATE MODEL or CREATE FUNCTION", "CREATE MODEL VERSION"]:
            self.assertIn(hint, log)
        self.assertIn("visibility unknown", log)
        self.assertNotIn("CREATE SCHEMA", log)

    def test_model_not_found_can_be_a_hidden_model(self):
        self.workspace.registered_models.get.side_effect = ResourceDoesNotExist(
            "unavailable", error_code="RESOURCE_DOES_NOT_EXIST",
        )
        log = self.preflight()
        self.assertIn("not found", log)
        self.assertIn("CREATE MODEL or CREATE FUNCTION", log)
        self.assertIn("If it exists", log)

    def test_missing_schema_only_checks_catalog(self):
        self.workspace.schemas.get.side_effect = NotFound("absent", error_code="SCHEMA_DOES_NOT_EXIST")
        log = self.preflight()
        self.assertIn("CREATE SCHEMA on training", log)
        self.workspace.grants.get_effective.assert_called_once_with(
            "catalog", CATALOG, principal=CALLER, max_results=0, page_token=None,
        )
        self.workspace.registered_models.get.assert_not_called()

    def test_generic_schema_404_remains_unknown(self):
        self.workspace.schemas.get.side_effect = NotFound("unsupported endpoint", error_code="NOT_FOUND")
        log = self.preflight()
        self.assertIn("if the schema needs creation", log)
        self.workspace.registered_models.get.assert_called_once_with(MODEL, include_browse=True)

    def test_later_page_group_grants_and_group_ownership(self):
        self.workspace.registered_models.get.return_value = catalog.RegisteredModelInfo(owner="trainers")
        self.workspace.current_user.me.return_value.groups = [iam.ComplexValue(display="trainers")]

        def grants(kind, name, *, principal, max_results, page_token):
            self.assertEqual(principal, CALLER)
            self.assertEqual(max_results, 0)
            if page_token is None:
                return catalog.EffectivePermissionsList(next_page_token="next")
            self.assertEqual(page_token, "next")
            privileges = {("catalog", CATALOG): ["USE_CATALOG"],
                          ("schema", SCHEMA): ["USE_SCHEMA", "CREATE_FUNCTION"]}[kind, name]
            return catalog.EffectivePermissionsList.from_dict({"privilege_assignments": [{
                "principal": "inherited-training-group",
                "privileges": [{"privilege": value, "inherited_from_name": CATALOG}
                               for value in privileges],
            }]})

        self.workspace.grants.get_effective.side_effect = grants
        log = self.preflight()
        self.assertNotIn("confirm", log)
        self.assertEqual(self.workspace.grants.get_effective.call_count, 4)

    def test_version_authority_does_not_replace_mlflow_model_create(self):
        self.workspace.registered_models.get.return_value = catalog.RegisteredModelInfo(owner=CALLER)
        log = self.preflight()
        self.assertIn("CREATE MODEL or CREATE FUNCTION", log)
        self.assertNotIn("CREATE MODEL VERSION", log)

    def test_model_execute_does_not_confirm_version_write(self):
        self.workspace.registered_models.get.return_value = catalog.RegisteredModelInfo(browse_only=True)
        self.workspace.grants.get_effective.return_value = catalog.EffectivePermissionsList.from_dict({
            "privilege_assignments": [{"principal": CALLER, "privileges": [{"privilege": "EXECUTE"}]}],
        })
        self.assertIn("ownership or CREATE MODEL VERSION", self.preflight())
        self.workspace.grants.get_effective.assert_called_with(
            "function", MODEL, principal=CALLER, max_results=0, page_token=None,
        )

    def test_partial_grant_pages_do_not_turn_into_denials(self):
        page = catalog.EffectivePermissionsList(next_page_token="next")

        def grants(kind, name, *, principal, max_results, page_token):
            if not page_token:
                return page
            raise TemporarilyUnavailable("incomplete")

        self.workspace.grants.get_effective.side_effect = grants
        log = self.preflight()
        self.assertIn("incomplete", log)
        self.assertIn("CREATE MODEL or CREATE FUNCTION", log)
        self.assertIn("continuing", log)

    def test_version_privilege_depends_on_what_the_sdk_can_decode(self):
        response = catalog.EffectivePermissionsList.from_dict({"privilege_assignments": [{
            "principal": "trainers", "privileges": [{"privilege": "CREATE_MODEL_VERSION"}],
        }]})
        self.workspace.grants.get_effective.return_value = response
        log = self.preflight()
        represented = response.privilege_assignments[0].privileges[0].privilege is not None
        self.assertEqual("ownership or CREATE MODEL VERSION" in log, not represented)
        self.assertIn("CREATE MODEL or CREATE FUNCTION", log)

    def test_all_privileges_on_parents_and_model_ownership(self):
        self.workspace.registered_models.get.return_value = catalog.RegisteredModelInfo(owner=CALLER)
        self.workspace.grants.get_effective.return_value = catalog.EffectivePermissionsList.from_dict({
            "privilege_assignments": [{"principal": CALLER, "privileges": [{"privilege": "ALL_PRIVILEGES"}]}],
        })
        self.assertNotIn("confirm", self.preflight())

    def test_all_privileges_on_model_covers_only_version_authority(self):
        def grants(kind, name, **kwargs):
            if (kind, name) == ("function", MODEL):
                return catalog.EffectivePermissionsList.from_dict({"privilege_assignments": [{
                    "principal": CALLER, "privileges": [{"privilege": "ALL_PRIVILEGES"}],
                }]})
            return catalog.EffectivePermissionsList()

        self.workspace.grants.get_effective.side_effect = grants
        log = self.preflight()
        self.assertNotIn("ownership or CREATE MODEL VERSION", log)
        self.assertIn("CREATE MODEL or CREATE FUNCTION", log)

    def test_unknown_caller_never_requests_unfiltered_grants(self):
        self.workspace.current_user.me.side_effect = PermissionDenied("unavailable")
        log = self.preflight()
        self.assertIn("CREATE MODEL", log)
        self.workspace.grants.get_effective.assert_not_called()

    def test_unavailable_grants_are_advisory(self):
        self.workspace.grants.get_effective.side_effect = PermissionDenied("limited visibility")
        self.assertIn("CREATE MODEL VERSION", self.preflight())


class WriteBoundaryTests(TestCase):
    def test_existing_schema_skips_create(self):
        workspace = Mock()
        workspace.schemas.get.return_value = catalog.SchemaInfo()
        ensure_uc_schema(workspace, CATALOG, "models")
        workspace.schemas.create.assert_not_called()

    def test_concurrent_creation_accepts_only_structured_duplicates(self):
        workspace = Mock()
        workspace.schemas.get.side_effect = NotFound("gone", error_code="SCHEMA_DOES_NOT_EXIST")
        for duplicate in [AlreadyExists("opaque"), ResourceAlreadyExists("opaque"),
                          BadRequest("opaque", error_code="SCHEMA_ALREADY_EXISTS")]:
            with self.subTest(duplicate=type(duplicate)):
                workspace.schemas.create.side_effect = duplicate
                ensure_uc_schema(workspace, CATALOG, "models")
                workspace.schemas.create.assert_called_with(name="models", catalog_name=CATALOG)

    def test_schema_errors_preserve_type_and_only_annotate_denial(self):
        workspace = Mock()
        workspace.schemas.get.side_effect = PermissionDenied("hidden")
        for error in [PermissionDenied("policy 42"), ResourceConflict("already exists"),
                      Unauthenticated("expired"), TemporarilyUnavailable("permission unavailable")]:
            with self.subTest(error=type(error)):
                workspace.schemas.create.side_effect = error
                with self.assertRaises(type(error)) as caught:
                    ensure_uc_schema(workspace, CATALOG, "models")
                self.assertIs(caught.exception, error)
                notes = " ".join(getattr(error, "__notes__", []))
                self.assertEqual("CREATE SCHEMA" in notes, isinstance(error, PermissionDenied))

    def test_logging_preserves_sdk_mlflow_and_wrapped_denials(self):
        wrapped = RuntimeError("model logging failed")
        wrapped.__cause__ = PermissionDenied("policy 42")
        for error in [PermissionDenied("policy 42"), MlflowException("opaque", PERMISSION_DENIED),
                      RestException({"error_code": "PERMISSION_DENIED", "message": "artifact access"}),
                      wrapped]:
            with self.subTest(error=type(error)), self.assertRaises(type(error)) as caught:
                with model_logging_errors(MODEL):
                    raise error
            self.assertIs(caught.exception, error)
            self.assertIn("logging/registering " + MODEL, error.__notes__[0])
            self.assertIn("CREATE MODEL VERSION", error.__notes__[0])

    def test_mlflow_implicit_denial_context_respects_explicit_chaining(self):
        original = RestException({"error_code": "PERMISSION_DENIED", "message": "artifact credentials"})
        for chain in ["implicit", "suppressed", "unrelated cause"]:
            with self.subTest(chain=chain), self.assertRaises(MlflowException) as caught:
                with model_logging_errors(MODEL):
                    try:
                        raise original
                    except RestException:
                        error = MlflowException("could not upload artifact")
                        if chain == "suppressed":
                            raise error from None
                        if chain == "unrelated cause":
                            raise error from TimeoutError("expired")
                        raise error
            self.assertIs(caught.exception.__context__, original)
            self.assertEqual(bool(getattr(caught.exception, "__notes__", None)), chain == "implicit")

    def test_unrelated_logging_errors_keep_original_diagnostic(self):
        for error in [MlflowException("permission text only"), Unauthenticated("expired"),
                      ConnectionError("write interrupted")]:
            with self.subTest(error=type(error)), self.assertRaises(type(error)) as caught:
                with model_logging_errors(MODEL):
                    raise error
            self.assertIs(caught.exception, error)
            self.assertFalse(getattr(error, "__notes__", None))


if __name__ == "__main__":
    main()
