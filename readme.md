# Task interpretation and design decisions

Places where the task text left room for interpretation, and the
reasoning behind each decision.

## 1. Unary +/- at the start of `role`

No example in the task or sample tests uses `"-col"` or `"+col"`.
**Decision:** treat a leading `+`/`-` as invalid syntax (empty df).
Adding real support for it would mean guessing an untested requirement.

## 2. Validation scope: all of `df.columns`, not just columns used in `role`

*"If the role or any column label is incorrect..."* is ambiguous about
scope. **Decision:** validate every column in `df.columns`, even ones
not referenced in `role` - the stricter, literal reading.

This also resolves a parsing ambiguity for free: a column named e.g.
`"col-a"` already breaks the letters/underscore rule, so such a `df` is
rejected outright, before `role` is even parsed - no separate
disambiguation logic needed.

## 3. Operator precedence and operand count

Task examples only cover two operands with one operator. **Decision:**
allow any number of operands, with multiplication evaluated before
+/- (standard math order), left to right for equal precedence. This is
an assumption, not a stated requirement.

## 4. Error resilience

Added (not explicitly required, but natural given "if role... is
incorrect -> empty df"): reject non-`DataFrame` input, and wrap the
computation in `try/except` so arithmetic on a non-numeric column (e.g.
the `name` column in the spec's own example) returns an empty df instead
of raising.

## 5. Note on the PDF example

The PDF calls `add_virtual_column(fruits_sales, "quantity * price",
"total")` but the printed output shows column `price_total`, not
`total` - likely a typo in the task text. The implementation uses the
name passed via `new_column`, per the function signature.