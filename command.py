import argparse
from dataclasses import dataclass, fields, MISSING
from typing import Optional, Any


@dataclass
class Command:
    @staticmethod
    def _get_type(field_type: Any) -> Any:
        if field_type == Optional[int]:
            return int
        elif field_type == Optional[float]:
            return float
        elif field_type == Optional[str]:
            return str
        elif field_type == Optional[list]:
            return list
        return field_type

    @staticmethod
    def _add_positional_argument(field, parser):
        parser.add_argument(field.name, type=Command._get_type(field.type))

    @staticmethod
    def _add_option(field, parser):
        if field.type == bool:
            parser.add_argument(f"--{field.name}", action="store_true")
        elif field.type == Optional[int] or field.type == int:
            parser.add_argument(f"--{field.name}", type=int, default=field.default)
        elif field.type == Optional[float] or field.type == float:
            parser.add_argument(f"--{field.name}", type=float, default=field.default)
        elif field.type == Optional[str] or field.type == str:
            parser.add_argument(f"--{field.name}", type=str, default=field.default)
        elif field.type == Optional[list] or field.type == list:
            parser.add_argument(
                f"--{field.name}",
                nargs="+",
                type=str,
                default=field.default,
            )
        else:
            raise TypeError(f"Unsupported type: {field.type}")

    def parse_args(self):
        parser = argparse.ArgumentParser(description=self.__class__.__doc__)

        for field in fields(self):
            if field.default == MISSING:
                self._add_positional_argument(field, parser)
            else:
                self._add_option(field, parser)

        args = parser.parse_args()
        for field in fields(self):
            setattr(self, field.name, getattr(args, field.name))

    def run(self):
        raise NotImplementedError("Subclasses should implement this method.")
