from __future__ import annotations

from tecore.io.schema import get_schema
from tecore.io.reader import read_csv, validate_df


def cmd_validate(args) -> int:
    schema = get_schema(args.schema)
    df = read_csv(args.input)
    errors = validate_df(df, schema)

    if errors:
        print("INVALID")
        for e in errors:
            print("-", e)
        return 2

    print("OK")
    return 0
