from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationResult:
    allowed: bool
    reason_code: str
    reason: str


class StageRegionContractValidator:
    """
    Enforces stage-specific region contracts for start/completion events.
    Deterministic rejects with explicit reason codes for auditability.
    """

    def validate_start(self, *, mode: str, ctx, stage_descriptor: dict | None) -> ValidationResult:
        desc = stage_descriptor or {}
        mode = str(mode).lower()
        if mode == "assembly":
            return self._validate_assembly_start(ctx, desc)
        return self._validate_dismantle_start(ctx, desc)

    def validate_completion(self, *, mode: str, ctx, stage_descriptor: dict | None) -> ValidationResult:
        desc = stage_descriptor or {}
        mode = str(mode).lower()
        if mode == "assembly":
            return self._validate_assembly_completion(ctx, desc)
        return self._validate_dismantle_completion(ctx, desc)

    @staticmethod
    def _resolve_scope(desc: dict, phase: str) -> str:
        scoped_key = f"{phase}_hand_scope"
        scope = str(desc.get(scoped_key, desc.get("hand_scope", "any"))).strip().lower()
        if scope not in {"active", "any"}:
            scope = "any"
        return scope

    @classmethod
    def _ctx_roles_regions(cls, ctx, desc: dict, phase: str):
        scope = cls._resolve_scope(desc, phase)
        if scope == "active":
            return set(ctx.active_roles), set(ctx.active_regions)
        return set(ctx.any_roles), set(ctx.any_regions)

    @staticmethod
    def _validate_dismantle_start(ctx, desc: dict) -> ValidationResult:
        roles, regions = StageRegionContractValidator._ctx_roles_regions(ctx, desc, "start")
        manual = bool(desc.get("manual_start_allowed", True))
        tool_regions = set(desc.get("start_tool_regions", []) or [])
        if not tool_regions:
            if manual:
                return ValidationResult(True, "OK_MANUAL_START", "Manual start allowed.")
            return ValidationResult(False, "MANUAL_START_DISABLED", "Manual start is disabled for this stage.")

        has_tool_pick = "tool_pick" in roles
        in_allowed_tool_region = bool(tool_regions & regions)
        if has_tool_pick and in_allowed_tool_region:
            return ValidationResult(True, "OK_TOOL_START", "Tool pick in allowed tool region.")
        if has_tool_pick and not in_allowed_tool_region:
            return ValidationResult(
                False,
                "START_TOOL_WRONG_REGION",
                "Tool pick detected but active region is not an allowed start tool region.",
            )
        if manual:
            return ValidationResult(True, "OK_MANUAL_FALLBACK", "Manual fallback start allowed.")
        return ValidationResult(False, "START_TOOL_MISSING", "Expected tool pick start was not detected.")

    @staticmethod
    def _validate_dismantle_completion(ctx, desc: dict) -> ValidationResult:
        roles_ctx, regions_ctx = StageRegionContractValidator._ctx_roles_regions(ctx, desc, "completion")
        roles = set(desc.get("completion_roles", []) or ["part_drop", "tool_drop"])
        regions = set(desc.get("completion_regions", []) or [])
        has_role = bool(roles & roles_ctx)
        if regions:
            in_region = bool(regions & regions_ctx)
            if has_role and in_region:
                return ValidationResult(True, "OK_COMPLETION", "Completion role in allowed completion region.")
            if has_role and not in_region:
                return ValidationResult(
                    False,
                    "COMPLETION_WRONG_REGION",
                    "Completion role detected but active region is not an allowed completion region.",
                )
            return ValidationResult(False, "COMPLETION_ROLE_MISSING", "Expected completion role not detected.")
        if has_role:
            return ValidationResult(True, "OK_COMPLETION_ROLE_ONLY", "Completion role detected (no region constraint).")
        return ValidationResult(False, "COMPLETION_ROLE_MISSING", "Expected completion role not detected.")

    @staticmethod
    def _validate_assembly_start(ctx, desc: dict) -> ValidationResult:
        roles_ctx, regions_ctx = StageRegionContractValidator._ctx_roles_regions(ctx, desc, "start")
        start_roles = set(desc.get("start_roles", []) or ["part_pick"])
        source_regions = set(desc.get("start_source_regions", []) or [])
        manual = bool(desc.get("manual_start_allowed", False))
        has_role = bool(start_roles & roles_ctx)
        if source_regions:
            in_region = bool(source_regions & regions_ctx)
            if has_role and in_region:
                return ValidationResult(True, "OK_START_PICK", "Start pick in allowed source region.")
            if has_role and not in_region:
                return ValidationResult(
                    False,
                    "START_WRONG_SOURCE_REGION",
                    "Start role detected but active region is not an allowed source region.",
                )
            if manual:
                return ValidationResult(True, "OK_MANUAL_START", "Manual start allowed.")
            return ValidationResult(False, "START_ROLE_MISSING", "Expected start role was not detected.")
        if has_role:
            return ValidationResult(True, "OK_START_ROLE_ONLY", "Start role detected (no source constraint).")
        if manual:
            return ValidationResult(True, "OK_MANUAL_START", "Manual start allowed.")
        return ValidationResult(False, "START_ROLE_MISSING", "Expected start role was not detected.")

    @staticmethod
    def _validate_assembly_completion(ctx, desc: dict) -> ValidationResult:
        roles_ctx, regions_ctx = StageRegionContractValidator._ctx_roles_regions(ctx, desc, "completion")
        mode = str(desc.get("completion_mode", "next_stage_pick")).lower()
        if mode == "next_stage_pick":
            roles = set(desc.get("next_stage_trigger_roles", []) or ["part_pick"])
            regions = set(desc.get("next_stage_source_regions", []) or [])
            has_role = bool(roles & roles_ctx)
            if regions:
                in_region = bool(regions & regions_ctx)
                if has_role and in_region:
                    return ValidationResult(True, "OK_NEXT_STAGE_PICK", "Next-stage pick intent detected.")
                if has_role and not in_region:
                    return ValidationResult(
                        False,
                        "NEXT_STAGE_PICK_WRONG_REGION",
                        "Next-stage pick role detected in wrong region.",
                    )
                return ValidationResult(False, "NEXT_STAGE_PICK_MISSING", "Next-stage pick intent missing.")
            if has_role:
                return ValidationResult(True, "OK_NEXT_STAGE_PICK_ROLE_ONLY", "Next-stage pick role detected.")
            return ValidationResult(False, "NEXT_STAGE_PICK_MISSING", "Next-stage pick intent missing.")

        # completion_proxy mode
        roles = set(desc.get("completion_roles", []) or ["assembly_target"])
        regions = set(desc.get("completion_regions", []) or [])
        has_role = bool(roles & roles_ctx)
        if regions:
            in_region = bool(regions & regions_ctx)
            if has_role and in_region:
                return ValidationResult(True, "OK_COMPLETION_PROXY", "Completion proxy detected in allowed region.")
            if has_role and not in_region:
                return ValidationResult(False, "COMPLETION_PROXY_WRONG_REGION", "Completion proxy detected in wrong region.")
            return ValidationResult(False, "COMPLETION_PROXY_MISSING", "Completion proxy missing.")
        if has_role:
            return ValidationResult(True, "OK_COMPLETION_PROXY_ROLE_ONLY", "Completion proxy role detected.")
        return ValidationResult(False, "COMPLETION_PROXY_MISSING", "Completion proxy missing.")
