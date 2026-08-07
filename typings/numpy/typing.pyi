from typing import Any, Generic, TypeAlias, TypeVar

ArrayLike: TypeAlias = Any
_ScalarT = TypeVar("_ScalarT")

class NDArray(Generic[_ScalarT]): ...
