from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Policy Engine")
printer = PrettyPrinter

Validator = Callable[[Dict[str, Any]], Tuple[bool, Optional[str]]]


@dataclass
class ContractValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class TaskContract:
    task_type: str
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, tuple[type, ...]] = field(default_factory=dict)
    allow_unknown_fields: bool = True
    validators: List[Validator] = field(default_factory=list)

    def validate(self, payload: Dict[str, Any]) -> ContractValidationResult:
        errors: List[str] = []

        for field_name in self.required_fields:
            if field_name not in payload:
                errors.append(f"missing required field '{field_name}'")

        known_fields = set(self.required_fields) | set(self.optional_fields) | set(self.field_types.keys())
        if not self.allow_unknown_fields and known_fields:
            for field_name in payload.keys():
                if field_name not in known_fields:
                    errors.append(f"unknown field '{field_name}' is not allowed")

        for field_name, expected_types in self.field_types.items():
            if field_name not in payload:
                continue
            if not isinstance(payload[field_name], expected_types):
                expected = ", ".join(t.__name__ for t in expected_types)
                actual = type(payload[field_name]).__name__
                errors.append(f"field '{field_name}' expected [{expected}] but got [{actual}]")

        for validator in self.validators:
            is_valid, message = validator(payload)
            if not is_valid:
                errors.append(message or f"custom validator failed for task '{self.task_type}'")

        return ContractValidationResult(valid=not errors, errors=errors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "required_fields": list(self.required_fields),
            "optional_fields": list(self.optional_fields),
            "field_types": {
                key: [value.__name__ for value in values]
                for key, values in self.field_types.items()
            },
            "allow_unknown_fields": self.allow_unknown_fields,
            "validators_count": len(self.validators),
        }


class TaskContractRegistry:
    def __init__(self):
        self._contracts: Dict[str, TaskContract] = {}

        logger.info("Task Contract Registry initialized")

    def register(self, contract: TaskContract) -> None:
        self._contracts[contract.task_type] = contract

    def register_contract(
        self,
        task_type: str,
        required_fields: Optional[List[str]] = None,
        optional_fields: Optional[List[str]] = None,
        field_types: Optional[Dict[str, tuple[type, ...]]] = None,
        allow_unknown_fields: bool = True,
        validators: Optional[List[Validator]] = None,
    ) -> TaskContract:
        contract = TaskContract(
            task_type=task_type,
            required_fields=required_fields or [],
            optional_fields=optional_fields or [],
            field_types=field_types or {},
            allow_unknown_fields=allow_unknown_fields,
            validators=validators or [],
        )
        self.register(contract)
        return contract

    def get(self, task_type: str) -> Optional[TaskContract]:
        return self._contracts.get(task_type)

    def validate(self, task_type: str, payload: Dict[str, Any]) -> ContractValidationResult:
        contract = self.get(task_type)
        if contract is None:
            return ContractValidationResult(valid=True, errors=[])
        return contract.validate(payload)

    def list_contracts(self) -> Dict[str, Dict[str, Any]]:
        return {task_type: contract.to_dict() for task_type, contract in self._contracts.items()}
