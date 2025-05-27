import contextlib
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Union,
    get_args,
    get_origin,
    Annotated,
    Protocol,
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

with contextlib.suppress(ImportError):
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import __version__ as pydantic_version

    _BaseModel = PydanticBaseModel  # type: ignore
    if pydantic_version.startswith("1."):
        _PYDANTIC_V1 = True
    elif pydantic_version.startswith("2."):
        _PYDANTIC_V2 = True

    if _PYDANTIC_V1 or _PYDANTIC_V2:
        IS_PYDANTIC_AVAILABLE = True


@runtime_checkable
class ThinAgentsTool(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def tool_schema(self) -> Dict[str, Any]: ...

    __name__: str


_PRIMITIVE_TYPE_MAP = {
    type(None): {"type": "null"},
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}

def _handle_enum(py_type: Any) -> JSONSchemaType:
    """
    Convert a Python Enum type to a JSON schema representation.
    Handles string, integer, and number enums.
    """
    values = [e.value for e in py_type]
    if all(isinstance(v, str) for v in values):
        return {"type": "string", "enum": values}
    if all(isinstance(v, int) for v in values):
        return {"type": "integer", "enum": values}
    if all(isinstance(v, (int, float)) for v in values):
        return {"type": "number", "enum": values}
    return {"enum": values}

def _handle_sequence(py_type: Any, args: tuple) -> JSONSchemaType:
    """
    Convert a sequence type (e.g., List, Sequence) to a JSON schema array type.
    """
    item_type = args[0] if args else Any
    return {"type": "array", "items": map_type_to_schema(item_type)}

def _handle_tuple(args: tuple) -> JSONSchemaType:
    """
    Convert a tuple type to a JSON schema representation.
    Handles both fixed-length and variable-length tuples.
    """
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

def _handle_dataclass(py_type: Any) -> JSONSchemaType:
    """
    Convert a dataclass type to a JSON schema object, including required fields and property types.
    """
    props = {}
    required = []
    dc_fields = fields(py_type)
    type_hints_for_dc = get_type_hints(py_type, include_extras=True)

    for field in dc_fields:
        field_type = type_hints_for_dc.get(field.name, field.type)
        props[field.name] = map_type_to_schema(field_type)
        if _is_required_field(field, field_type):
            required.append(field.name)

    return {
        "type": "object",
        "properties": props,
        "required": sorted(list(set(required))),
        "additionalProperties": False,
    }

def _is_required_field(field: Any, field_type: Any) -> bool:
    """
    Determine if a dataclass field is required based on its default value and type annotation.
    """
    if field.default is not inspect.Parameter.empty:
        return False
    origin = get_origin(field_type)
    args = get_args(field_type)
    return (origin is not Union or type(None) not in args) and (
        origin is Union or origin is not Optional
    )

def _handle_union(args: tuple) -> JSONSchemaType:
    """
    Convert a Union type (including Optional) to a JSON schema using anyOf.
    """
    non_none_args = [a for a in args if a is not type(None)]
    if not non_none_args:
        return {"type": "null"}
    
    schemas = [map_type_to_schema(a) for a in non_none_args]
    if type(None) in args:
        schemas.append({"type": "null"})
    return {"anyOf": schemas}

@runtime_checkable
class TypeHandler(Protocol):
    """
    Protocol for custom type handlers that convert Python types to JSON schema.
    """
    def can_handle(self, py_type: Any) -> bool: ...
    def handle(self, py_type: Any, schema_mapper: Callable[[Any], JSONSchemaType]) -> JSONSchemaType: ...

class BaseTypeHandler:
    def can_handle(self, py_type: Any) -> bool:
        return False

    def handle(self, py_type: Any, schema_mapper: Callable[[Any], JSONSchemaType]) -> JSONSchemaType:
        return {"type": "object"}

class PrimitiveHandler(BaseTypeHandler):
    """
    Handles primitive Python types (str, int, float, bool, None) for JSON schema conversion.
    """
    def can_handle(self, py_type: Any) -> bool:
        return py_type in _PRIMITIVE_TYPE_MAP

    def handle(self, py_type: Any, _: Callable[[Any], JSONSchemaType]) -> JSONSchemaType:
        return _PRIMITIVE_TYPE_MAP[py_type]

class EnumHandler(BaseTypeHandler):
    """
    Handles Python Enum types for JSON schema conversion.
    """
    def can_handle(self, py_type: Any) -> bool:
        return isinstance(py_type, type) and issubclass(py_type, enum.Enum)

    def handle(self, py_type: Any, _: Callable[[Any], JSONSchemaType]) -> JSONSchemaType:
        return _handle_enum(py_type)

class PydanticHandler(BaseTypeHandler):
    """
    Handles Pydantic BaseModel types for JSON schema conversion, supporting both v1 and v2.
    """
    def can_handle(self, py_type: Any) -> bool:
        return (IS_PYDANTIC_AVAILABLE and isinstance(py_type, type) 
                and issubclass(py_type, _BaseModel))

    def handle(self, py_type: Any, _: Callable[[Any], JSONSchemaType]) -> JSONSchemaType:
        schema: Optional[JSONSchemaType] = None
        if _PYDANTIC_V2 and hasattr(py_type, "model_json_schema"):
            schema = py_type.model_json_schema()  # type: ignore
        elif _PYDANTIC_V1 and hasattr(py_type, "schema"):
            schema = py_type.schema()  # type: ignore
        return schema if schema is not None else {"type": "object"}

class SequenceHandler(BaseTypeHandler):
    """
    Handles sequence types (lists, sequences) for JSON schema conversion.
    """
    def can_handle(self, py_type: Any) -> bool:
        origin = get_origin(py_type)
        return (origin in (list, List) or 
                (isinstance(py_type, type) and issubclass(py_type, Sequence) 
                 and not issubclass(py_type, (str, bytes, bytearray))))

    def handle(self, py_type: Any, schema_mapper: Callable[[Any], JSONSchemaType]) -> JSONSchemaType:
        return _handle_sequence(py_type, get_args(py_type))

_type_handlers = [
    PrimitiveHandler(),
    EnumHandler(),
    PydanticHandler(),
    SequenceHandler(),
]

def map_type_to_schema(py_type: Any) -> JSONSchemaType:
    """
    Main entry point for mapping a Python type annotation to a JSON schema type.
    Delegates to registered type handlers and handles common generic types.
    """
    origin = get_origin(py_type)
    args = get_args(py_type)

    for handler in _type_handlers:
        if handler.can_handle(py_type):
            return handler.handle(py_type, map_type_to_schema)

    if origin is Literal:
        return _handle_enum(lambda: args)

    if origin in (tuple, Tuple):
        return _handle_tuple(args)

    if origin in (set, Set, frozenset, FrozenSet):
        return {**_handle_sequence(py_type, args), "uniqueItems": True}

    if is_dataclass(py_type):
        return _handle_dataclass(py_type)

    if origin in (dict, Dict) or (isinstance(py_type, type) and issubclass(py_type, Mapping)):
        if not args or len(args) != 2:
            return {"type": "object", "additionalProperties": map_type_to_schema(Any)}
        key_type, value_type = args
        schema = {"type": "object", "additionalProperties": map_type_to_schema(value_type)}
        if key_type is not str:
            schema["x-key-type"] = str(key_type)
        return schema

    if origin is Union:
        return _handle_union(args)

    if origin is Optional:
        return _handle_union((args[0], type(None))) if args else {"type": "null"}

    if origin in (Final, ClassVar):
        return map_type_to_schema(args[0]) if args else {}

    if py_type is Any:
        return {}

    if isinstance(py_type, TypeVar):
        if constraints := getattr(py_type, "__constraints__", None):
            return {"anyOf": [map_type_to_schema(c) for c in constraints]}
        bound = getattr(py_type, "__bound__", None)
        return map_type_to_schema(bound) if bound and bound is not object else {}

    return {"type": "object"}


def tool(fn_for_tool: Optional[Callable[P, R]] = None, *, return_type: Literal["content", "content_and_artifact"] = "content") -> ThinAgentsTool[P, R]:  # type: ignore
    """
    Decorator to register a function as a ThinAgentsTool, optionally specifying the return type.
    Enforces return type compatibility and attaches a tool_schema method for OpenAPI/JSON schema generation.

    Args:
        fn_for_tool: The function to register as a tool.
        return_type: The return type of the tool.
            - "content": The tool returns only content. This is the default.  
            - "content_and_artifact": The tool returns both content and an artifact, where the artifact is something that can be sent downstream.

    Returns:
        A ThinAgentsTool object that can be used to execute the tool.
    """
    if fn_for_tool is None:
        # return decorator when no function provided
        return lambda fn: tool(fn, return_type=return_type)  # type: ignore
    annotated_desc = ""
    actual_func = fn_for_tool
    if get_origin(fn_for_tool) is Annotated:
        unwrapped_func, *meta = get_args(fn_for_tool)
        actual_func = unwrapped_func
        annotated_desc = next((m for m in meta if isinstance(m, str)), "")

    # enforce return_type annotation compatibility at decoration time
    if return_type == "content_and_artifact":
        sig = inspect.signature(actual_func)
        ret_ann = sig.return_annotation
        # no annotation provided
        if ret_ann is inspect.Signature.empty:
            raise ValueError(
                f"Tool '{actual_func.__name__}' declared return_type='content_and_artifact' but no return annotation found"
            )
        origin = get_origin(ret_ann)
        args = get_args(ret_ann)
        # annotation must be Tuple[...] of length 2
        if origin not in (tuple, Tuple) or len(args) != 2:
            raise ValueError(
                f"Tool '{actual_func.__name__}' declared return_type='content_and_artifact' but return annotation is {ret_ann!r}, expected Tuple[content_type, artifact_type]"
            )

    @functools.wraps(actual_func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
        # call the actual tool function
        result = actual_func(*args, **kwargs)
        # if the tool declares content_and_artifact, enforce a 2-tuple return
        if return_type == "content_and_artifact" and not (isinstance(result, tuple) and len(result) == 2):
            raise ValueError(
                f"Tool '{actual_func.__name__}' declared return_type='content_and_artifact' but returned {result!r}"
            )
        return result

    # store desired return_type on the wrapper
    wrapper.return_type = return_type  # type: ignore

    def tool_schema() -> Dict[str, Any]:
        # generate original function schema
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

        # wrap schema with return_type metadata
        original_schema = {
            "type": "function",
            "function": {
                "name": actual_func.__name__,
                "description": description,
                "parameters": params_schema,
            },
        }
        return {"tool_schema": original_schema, "return_type": return_type}

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
    """
    Generate a JSON schema for a function parameter, including type, title, description, and default value.
    """
    base_type = annotation
    param_description_from_annotated: Optional[str] = None

    if get_origin(annotation) is Annotated:
        actual_type, *metadata = get_args(annotation)
        base_type = actual_type
        param_description_from_annotated = next(
            (m for m in metadata if isinstance(m, str) and not m.startswith(":")),
            None,
        )

    core_type_schema = map_type_to_schema(base_type)
    param_final_schema = core_type_schema.copy()  # Start with base schema

    param_final_schema["title"] = name.replace("_", " ").capitalize()

    if param_description_from_annotated:
        param_final_schema["description"] = param_description_from_annotated
    if param.default is not inspect.Parameter.empty:
        param_final_schema["default"] = param.default

    return param_final_schema


def _is_required_parameter(param: inspect.Parameter, annotation: Any) -> bool:
    """
    Determine if a function parameter is required based on its default value and type annotation.
    """
    if param.default is not inspect.Parameter.empty:
        return False
    current_type_to_check = annotation
    if get_origin(current_type_to_check) is Annotated:
        args = get_args(current_type_to_check)
        if args:
            current_type_to_check = args[0]

    origin = get_origin(current_type_to_check)
    args = get_args(current_type_to_check)

    return (origin is not Union or type(None) not in args) and (
        origin is Union or origin is not Optional
    )
