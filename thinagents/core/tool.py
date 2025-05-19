from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Union,
    get_args,
    get_origin,
    Annotated,
    Protocol,  # Already imported
    Literal,
    Final,
    ClassVar,
    TypeVar,
    List,
    Tuple,
    Set,
    FrozenSet,
    ParamSpec,
    runtime_checkable,
)

try:
    from typing import get_type_hints
except ImportError:
    from typing_extensions import get_type_hints  # type: ignore

import inspect
from collections.abc import Sequence, Mapping
import enum
from dataclasses import is_dataclass, fields
import functools

P = ParamSpec("P")
R = TypeVar("R", covariant=True)

JSONSchemaType = Dict[str, Any]

_PYDANTIC_V1 = False
_PYDANTIC_V2 = False
_BaseModel = object
IS_PYDANTIC_AVAILABLE = False

try:
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import __version__ as pydantic_version

    _BaseModel = PydanticBaseModel  # type: ignore
    if pydantic_version.startswith("1."):
        _PYDANTIC_V1 = True
    elif pydantic_version.startswith("2."):
        _PYDANTIC_V2 = True

    if _PYDANTIC_V1 or _PYDANTIC_V2:
        IS_PYDANTIC_AVAILABLE = True
except ImportError:
    pass


@runtime_checkable
class ThinAgentsTool(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def tool_schema(self) -> Dict[str, Any]: ...

    __name__: str


def map_type_to_schema(py_type: Any) -> JSONSchemaType:
    if py_type is type(None):
        return {"type": "null"}
    if py_type is str:
        return {"type": "string"}
    if py_type is int:
        return {"type": "integer"}
    if py_type is float:
        return {"type": "number"}
    if py_type is bool:
        return {"type": "boolean"}

    if isinstance(py_type, type) and issubclass(py_type, enum.Enum):
        values = [e.value for e in py_type]
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        if all(isinstance(v, int) for v in values):
            return {"type": "integer", "enum": values}
        if all(isinstance(v, (int, float)) for v in values):
            return {"type": "number", "enum": values}
        return {"enum": values}

    if (
        IS_PYDANTIC_AVAILABLE
        and isinstance(py_type, type)
        and issubclass(py_type, _BaseModel)
    ):
        schema: Optional[JSONSchemaType] = None
        if _PYDANTIC_V2 and hasattr(py_type, "model_json_schema"):
            schema = py_type.model_json_schema()  # type: ignore
        elif _PYDANTIC_V1 and hasattr(py_type, "schema"):
            schema = py_type.schema()  # type: ignore

        if schema is not None:
            return schema

    origin = get_origin(py_type)

    if origin is Literal:
        literal_values = get_args(py_type)
        if all(isinstance(v, str) for v in literal_values):
            return {"type": "string", "enum": list(literal_values)}
        if all(isinstance(v, int) for v in literal_values):
            return {"type": "integer", "enum": list(literal_values)}
        if all(isinstance(v, (int, float)) for v in literal_values):
            return {"type": "number", "enum": list(literal_values)}
        return {"enum": list(literal_values)}

    if origin in (list, List) or (
        isinstance(py_type, type)
        and issubclass(py_type, Sequence)
        and not issubclass(py_type, (str, bytes, bytearray))
    ):
        args = get_args(py_type)
        item_type = args[0] if args else Any
        return {"type": "array", "items": map_type_to_schema(item_type)}

    if origin in (tuple, Tuple):
        args = get_args(py_type)
        if not args:
            return {"type": "array"}
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": map_type_to_schema(args[0])}
        return {
            "type": "array",
            "prefixItems": [map_type_to_schema(arg) for arg in args],
            "minItems": len(args),
            "maxItems": len(args),
        }

    if origin in (set, Set, frozenset, FrozenSet):
        args = get_args(py_type)
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": map_type_to_schema(item_type),
            "uniqueItems": True,
        }

    if is_dataclass(py_type):
        props = {}
        required = []
        dc_fields = fields(py_type)
        type_hints_for_dc = get_type_hints(py_type, include_extras=True)

        for field in dc_fields:
            field_type = type_hints_for_dc.get(field.name, field.type)
            props[field.name] = map_type_to_schema(field_type)

            is_optional_type = False
            field_origin = get_origin(field_type)
            field_args = get_args(field_type)

            if field_origin is Union and type(None) in field_args:
                is_optional_type = True
            elif (
                field_origin is Optional
            ):  # Optional is syntactic sugar for Union[T, None]
                is_optional_type = True

            if (
                getattr(field, "default", inspect.Parameter.empty)
                is inspect.Parameter.empty  # More robust check for default
                and getattr(field, "default_factory", inspect.Parameter.empty)
                is inspect.Parameter.empty  # More robust check for default_factory
                and not is_optional_type
            ):
                required.append(field.name)
        return {
            "type": "object",
            "properties": props,
            "required": sorted(list(set(required))),  # Ensure uniqueness and order
            "additionalProperties": False,
        }

    if origin in (dict, Dict) or (
        isinstance(py_type, type) and issubclass(py_type, Mapping)
    ):
        args = get_args(py_type)
        if args and len(args) == 2:
            key_type, value_type = args
            schema_for_mapping = {
                "type": "object",
                "additionalProperties": map_type_to_schema(value_type),
            }
            if key_type is not str:  # JSON object keys must be strings
                schema_for_mapping["x-key-type"] = str(
                    key_type
                )  # Non-standard, but informative
            return schema_for_mapping
        else:  # Dict without type args, or Mapping without type args
            return {"type": "object", "additionalProperties": map_type_to_schema(Any)}

    if origin is Union:
        args = get_args(py_type)
        non_none_args = [a for a in args if a is not type(None)]

        if not non_none_args:  # Union of only NoneType e.g. Union[None]
            return {"type": "null"}

        # Handle Optional[T] as Union[T, NoneType]
        if type(None) in args:
            if len(non_none_args) == 1:  # This is effectively Optional[T]
                # OpenAPI 3.0+ suggests nullable: true, or combining with null type in anyOf/oneOf
                # For broader compatibility, using anyOf is safer.
                return {
                    "anyOf": [map_type_to_schema(non_none_args[0]), {"type": "null"}]
                }
            else:  # Union[A, B, NoneType]
                return {
                    "anyOf": [map_type_to_schema(a) for a in non_none_args]
                    + [{"type": "null"}]
                }
        else:  # Union[A, B] (no NoneType)
            return {"anyOf": [map_type_to_schema(a) for a in args]}

    # Optional[T] is handled by the Union case above because get_origin(Optional[T]) is Union
    # and get_args(Optional[T]) is (T, type(None)).
    # This specific 'origin is Optional' block might be redundant if Union handles it comprehensively.
    # However, sometimes type hints resolve Optional directly.
    if origin is Optional:  # Explicit check for Optional, though Union often catches it
        args = get_args(py_type)
        # args for Optional[T] will be (T, type(None))
        # If it's just Optional without a type (should not happen with valid type hints for Optional[X])
        # or if the first arg is None (e.g. Optional[None]), treat as null.
        if args and args[0] is not type(None):
            # This effectively becomes map_type_to_schema(Union[args[0], type(None)])
            return {"anyOf": [map_type_to_schema(args[0]), {"type": "null"}]}
        return {"type": "null"}

    if origin is Final or origin is ClassVar:
        args = get_args(py_type)
        if args:
            return map_type_to_schema(args[0])
        return {}  # Should not happen if used correctly, e.g., Final[int]

    if py_type is Any:
        return {}  # Represents any type, often no specific schema constraints

    if isinstance(py_type, TypeVar):
        constraints = getattr(py_type, "__constraints__", None)
        if constraints:  # For TypeVar('T', int, str)
            return {"anyOf": [map_type_to_schema(c) for c in constraints]}
        bound = getattr(py_type, "__bound__", None)
        if bound and bound is not object:  # For TypeVar('T', bound=Sequence)
            return map_type_to_schema(bound)
        return {}  # Unconstrained TypeVar, like Any

    # Default for unrecognized types or classes
    # Consider logging a warning here if a more specific schema is expected
    return {"type": "object"}  # Or simply {} if no assumptions can be made


def tool(fn_for_tool: Callable[P, R]) -> ThinAgentsTool[P, R]:
    annotated_desc = ""
    actual_func = fn_for_tool
    if get_origin(fn_for_tool) is Annotated:
        unwrapped_func, *meta = get_args(fn_for_tool)
        actual_func = unwrapped_func
        annotated_desc = next((m for m in meta if isinstance(m, str)), "")

    @functools.wraps(actual_func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return actual_func(*args, **kwargs)

    def tool_schema() -> Dict[str, Any]:
        sig = inspect.signature(actual_func)
        # include_extras=True is important for Annotated
        type_hints = get_type_hints(actual_func, include_extras=True)

        params_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,  # Usually good for tools to be strict
        }

        for name, param in sig.parameters.items():
            annotation = type_hints.get(name, param.annotation)
            if annotation is inspect.Parameter.empty:
                annotation = Any  # Default to Any if no type hint

            param_def = _generate_param_schema(name, param, annotation)
            params_schema["properties"][name] = param_def  # type: ignore
            if _is_required_parameter(param, annotation):
                params_schema["required"].append(name)

        params_schema["required"] = sorted(list(set(params_schema["required"])))

        func_doc = inspect.getdoc(actual_func)
        description = annotated_desc or func_doc or ""

        return {
            "type": "function",  # As per OpenAI spec
            "function": {
                "name": actual_func.__name__,
                "description": description,
                "parameters": params_schema,
            },
        }

    setattr(wrapper, "tool_schema", tool_schema)
    wrapper.__name__ = actual_func.__name__

    # The type: ignore below is because `wrapper` (a function) doesn't statically
    # appear as a ThinAgentsTool to the type checker just by setting an attribute.
    # Using `cast` can make this more explicit if preferred.
    # from typing import cast
    # return cast(ThinAgentsTool[P, R], wrapper)
    return wrapper  # type: ignore


def _generate_param_schema(
    name: str, param: inspect.Parameter, annotation: Any
) -> JSONSchemaType:
    base_type = annotation
    param_description_from_annotated: Optional[str] = None

    if get_origin(annotation) is Annotated:
        actual_type, *metadata = get_args(annotation)
        base_type = actual_type
        param_description_from_annotated = next(
            # Ensure description is not mistaken for a Pydantic Field description format
            (m for m in metadata if isinstance(m, str) and not m.startswith(":")),
            None,
        )

    core_type_schema = map_type_to_schema(base_type)
    param_final_schema = core_type_schema.copy()  # Start with base schema

    # Override title with capitalized parameter name
    param_final_schema["title"] = name.replace("_", " ").capitalize()

    # Override description if provided via Annotated
    if param_description_from_annotated:
        param_final_schema["description"] = param_description_from_annotated
    # If no Annotated description, and core_type_schema had one (e.g., from Pydantic model), it's preserved.

    if param.default is not inspect.Parameter.empty:
        param_final_schema["default"] = param.default

    return param_final_schema


def _is_required_parameter(param: inspect.Parameter, annotation: Any) -> bool:
    if param.default is not inspect.Parameter.empty:
        return False  # Has a default value, so not required

    # Unwrap Annotated to check the underlying type for optionality
    current_type_to_check = annotation
    if get_origin(current_type_to_check) is Annotated:
        args = get_args(current_type_to_check)
        if args:  # Annotated[type, ...]
            current_type_to_check = args[0]  # The actual type

    # Check for Optional[T] or Union[T, None]
    origin = get_origin(current_type_to_check)
    args = get_args(current_type_to_check)

    if origin is Union:
        if type(None) in args:
            return False  # It's a Union including None, so effectively optional
    # Note: `Optional[T]` is just `Union[T, NoneType]`, so the above `Union` check covers it.
    # An explicit `origin is Optional` check might be redundant but harmless.
    elif origin is Optional:
        return False  # It's Optional, so not required

    # If annotation is Any or empty, and no default, assume required unless specified otherwise.
    # This behavior might need adjustment based on desired strictness.
    # If it's Any or Parameter.empty and has no default, it's considered required.
    if annotation is Any or annotation is inspect.Parameter.empty:
        return True

    return True  # No default, not Optional/Union[..., None], so it's required
