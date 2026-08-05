"""
Recruitment Task - Virtual columns in pandas DataFrame
solution.py

See INTERPRETATION.md for the full reasoning behind the design decisions
(unary +/-, operator precedence, column validation scope, etc).
"""

import re
import pandas as pd


def add_virtual_column(df: pd.DataFrame, role: str, new_column: str) -> pd.DataFrame:
    """
    Return a copy of `df` with an extra column computed from `role`

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. Every column label must contain only letters/underscore
    role : str
        Expression defining the new column, e.g. "quantity * price" or
        "sales-cost+bonus*tax". Supported operators: +, -, *. Any number of
        operands is allowed; multiplication has higher precedence than +/-
        (see INTERPRETATION.md, point 3)
    new_column : str
        Name of the new column. Must contain only letters/underscore

    Returns
    -------
    pandas.DataFrame
        `df` plus `new_column`, or an empty DataFrame if `df`, `role`,
        `new_column`, or the computation itself is invalid
    """

    column_label_pattern = re.compile(r'^[a-zA-Z_]+$')
    # column name, or a single +/-/* operator, in sequence
    role_pattern = re.compile(r'^[a-zA-Z_]+([+\-*][a-zA-Z_]+)*$')
    token_pattern = re.compile(r'[a-zA-Z_]+|[+\-*]')

    if not isinstance(df, pd.DataFrame):
        # debug: df is not a pandas.DataFrame
        return pd.DataFrame()

    if not all(isinstance(c, str) and column_label_pattern.fullmatch(c) for c in df.columns):
        # debug: df contains a column label with a disallowed character (validated for the whole df not just role - see INTERPRETATION.md point 2)
        return pd.DataFrame()

    if not isinstance(new_column, str) or not column_label_pattern.fullmatch(new_column):
        # debug: new_column missing/None or contains a disallowed character
        return pd.DataFrame()

    if not isinstance(role, str):
        # debug: role is not a string
        return pd.DataFrame()

    role_clean = role.replace(" ", "")
    if not role_pattern.fullmatch(role_clean):
        # debug: role has bad syntax - unary/leading/trailing operator, double operator, invalid char, no operands, etc.
        return pd.DataFrame()

    tokens = token_pattern.findall(role_clean)
    columns_used = tokens[0::2]
    operators = tokens[1::2]

    if not all(col in df.columns for col in columns_used):
        # debug: role references a column that doesn't exist in df
        return pd.DataFrame()

    try:
        values = [df[col] for col in columns_used]

        # first pass: fold '*' chains (higher precedence, supports any number of operands)
        i = 0
        while i < len(operators):
            if operators[i] == '*':
                values[i] = values[i] * values[i + 1]
                del values[i + 1]
                del operators[i]
            else:
                i += 1

        # second pass: +/- left to right
        result = values[0]
        for op, val in zip(operators, values[1:]):
            result = result + val if op == '+' else result - val

    except Exception:
        # debug: computation failed e.g. arithmetic on a non-numeric column
        return pd.DataFrame()

    result_df = df.copy()
    result_df[new_column] = result
    return result_df
