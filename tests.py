import unittest
import pandas as pd

from solution import add_virtual_column


class TestRoleExpressionSyntaxValidation(unittest.TestCase):
    """
    `self.df` is empty (no columns at all), so these tests exercise only
    the syntax check on `role` itself (pattern/token shape) - they never
    reach the point where actual df column labels are checked.
    See TestColumnLabelValidationOnWholeDataFrame for tests that validate
    real df.columns labels
    """

    def setUp(self):
        self.df = pd.DataFrame()

    def test_trailing_operator_is_invalid(self):
        result = add_virtual_column(self.df, "sales+", "new_col")
        self.assertTrue(result.empty)

    def test_expression_of_only_operators_is_invalid(self):
        result = add_virtual_column(self.df, "***", "new_col")
        self.assertTrue(result.empty)

    def test_double_multiplication_operator_is_invalid(self):
        # "**" (power operator) is not supported
        result = add_virtual_column(self.df, "sales**cost", "new_col")
        self.assertTrue(result.empty)

    def test_digit_in_role_column_reference_is_invalid(self):
        result = add_virtual_column(self.df, "sales1-cost", "new_col")
        self.assertTrue(result.empty)

    def test_at_sign_in_role_column_reference_is_invalid(self):
        result = add_virtual_column(self.df, "sales@cost", "new_col")
        self.assertTrue(result.empty)

    def test_exclamation_mark_in_role_column_reference_is_invalid(self):
        result = add_virtual_column(self.df, "sa!es+cost", "new_col")
        self.assertTrue(result.empty)

    def test_leading_operator_is_invalid(self):
        result = add_virtual_column(self.df, "*sales", "new_col")
        self.assertTrue(result.empty)

    def test_double_binary_operator_is_invalid(self):
        result = add_virtual_column(self.df, "sales--cost", "new_col")
        self.assertTrue(result.empty)


class TestUnaryOperators(unittest.TestCase):
    """
    Design decision: unary +/- at the start of `role` is treated as an
    invalid expression (not a "negate this column" operation).
    See INTERPRETATION.md point 1
    """

    def setUp(self):
        self.df = pd.DataFrame([[1, 1]] * 2, columns=["label_one", "label_two"])

    def test_unary_minus_is_invalid(self):
        result = add_virtual_column(self.df, "-label_one", "label_three")
        self.assertTrue(result.empty)

    def test_unary_plus_is_invalid(self):
        result = add_virtual_column(self.df, "+label_one", "label_three")
        self.assertTrue(result.empty)

    def test_unary_minus_in_the_middle_is_invalid(self):
        # "label_one - -label_two" after removing spaces: "label_one--label_two"
        result = add_virtual_column(self.df, "label_one - -label_two", "label_three")
        self.assertTrue(result.empty)


class TestColumnLabelValidationOnWholeDataFrame(unittest.TestCase):
    """
    Design decision: ALL column labels in `df` are validated, not only the
    ones referenced in `role`. See INTERPRETATION.md, point 2
    """

    def test_unused_invalid_column_still_rejects_df(self):
        # "bad-col" is not used in role, but it still makes the whole df invalid
        df = pd.DataFrame(
            [[1, 2, 3]] * 2, columns=["label_one", "label_two", "bad-col"]
        )
        result = add_virtual_column(df, "label_one + label_two", "label_three")
        self.assertTrue(result.empty)

    def test_unused_column_with_digit_still_rejects_df(self):
        df = pd.DataFrame(
            [[1, 2, 3]] * 2, columns=["label_one", "label_two", "col1"]
        )
        result = add_virtual_column(df, "label_one + label_two", "label_three")
        self.assertTrue(result.empty)

    def test_all_valid_labels_pass(self):
        df = pd.DataFrame(
            [[1, 2, 3]] * 2, columns=["label_one", "label_two", "label_extra"]
        )
        result = add_virtual_column(df, "label_one + label_two", "label_three")
        self.assertFalse(result.empty)


class TestNewColumnValidation(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame([[1, 1]] * 2, columns=["label_one", "label_two"])

    def test_new_column_with_digit_is_invalid(self):
        result = add_virtual_column(self.df, "label_one + label_two", "label3")
        self.assertTrue(result.empty)

    def test_new_column_with_special_character_is_invalid(self):
        result = add_virtual_column(self.df, "label_one + label_two", "label!")
        self.assertTrue(result.empty)

    def test_new_column_empty_string_is_invalid(self):
        result = add_virtual_column(self.df, "label_one + label_two", "")
        self.assertTrue(result.empty)

    def test_new_column_not_a_string_is_invalid(self):
        result = add_virtual_column(self.df, "label_one + label_two", None)
        self.assertTrue(result.empty)


class TestInputTypeRobustness(unittest.TestCase):
    """
    Robustness against unexpected input types - not explicitly required by
    the task description, but the function should degrade gracefully
    (empty DataFrame) instead of raising. See INTERPRETATION.md point 4
    """

    def test_df_is_none(self):
        result = add_virtual_column(None, "label_one + label_two", "label_three")
        self.assertTrue(result.empty)

    def test_df_is_not_a_dataframe(self):
        result = add_virtual_column(
            {"label_one": [1, 2], "label_two": [3, 4]},
            "label_one + label_two",
            "label_three",
        )
        self.assertTrue(result.empty)

    def test_role_is_not_a_string(self):
        df = pd.DataFrame([[1, 1]] * 2, columns=["label_one", "label_two"])
        result = add_virtual_column(df, None, "label_three")
        self.assertTrue(result.empty)

    def test_non_numeric_column_in_expression_returns_empty(self):
        # arithmetic on a text column should not raise - should degrade to empty df
        df = pd.DataFrame(
            {"name": ["banana", "apple"], "quantity": [10, 3]}
        )
        result = add_virtual_column(df, "name+quantity", "total")
        self.assertTrue(result.empty)


class TestOperatorPrecedenceAndChaining(unittest.TestCase):
    """
    Design decision: `role` supports more than two operands, and
    multiplication has higher precedence than addition/subtraction
    (standard mathematical order). See INTERPRETATION.md point 3
    """

    def setUp(self):
        self.df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
            "d": [7, 8],
        })

    def test_chained_multiplication_three_columns(self):
        # a*b*c -> 1*3*5=15, 2*4*6=48
        result = add_virtual_column(self.df, "a*b*c", "result")
        expected = pd.Series([15, 48], name="result")
        pd.testing.assert_series_equal(result["result"], expected)

    def test_multiplication_before_addition(self):
        # a+b*c -> 1+(3*5)=16, 2+(4*6)=26
        result = add_virtual_column(self.df, "a+b*c", "result")
        expected = pd.Series([16, 26], name="result")
        pd.testing.assert_series_equal(result["result"], expected)

    def test_mixed_precedence_four_columns(self):
        # a-b*c+d -> 1-(3*5)+7=-7, 2-(4*6)+8=-14
        result = add_virtual_column(self.df, "a-b*c+d", "result")
        expected = pd.Series([-7, -14], name="result")
        pd.testing.assert_series_equal(result["result"], expected)


class TestExampleFromSpecification(unittest.TestCase):
    """Recreates the fruits_sales example from the task description."""

    def setUp(self):
        self.df = pd.DataFrame({
            "name": ["banana", "apple"],
            "quantity": [10, 3],
            "price": [10, 1],
        })

    def test_quantity_times_price(self):
        result = add_virtual_column(self.df, "quantity * price", "total")
        self.assertListEqual(list(result["total"]), [100, 3])
        # original columns must be preserved
        self.assertListEqual(list(result["name"]), ["banana", "apple"])


class TestSumOfTwoColumns(unittest.TestCase):

    def test_sum_of_two_columns(self):
        df = pd.DataFrame([[1, 1]] * 2, columns=["label_one", "label_two"])
        df_expected = pd.DataFrame(
            [[1, 1, 2]] * 2, columns=["label_one", "label_two", "label_three"]
        )
        df_result = add_virtual_column(df, "label_one+label_two", "label_three")
        self.assertTrue(
            df_result.equals(df_expected),
            f"The function should sum the columns: label_one and "
            f"label_two.\n\nResult:\n\n{df_result}\n\nExpected:\n\n{df_expected}",
        )

    def test_multiplication_of_two_columns(self):
        df = pd.DataFrame([[1, 1]] * 2, columns=["label_one", "label_two"])
        df_expected = pd.DataFrame(
            [[1, 1, 1]] * 2, columns=["label_one", "label_two", "label_three"]
        )
        df_result = add_virtual_column(df, "label_one * label_two", "label_three")
        self.assertTrue(
            df_result.equals(df_expected),
            f"The function should multiply the columns: label_one and "
            f"label_two.\n\nResult:\n\n{df_result}\n\nExpected:\n\n{df_expected}",
        )

    def test_subtraction_of_two_columns(self):
        df = pd.DataFrame([[1, 1]] * 2, columns=["label_one", "label_two"])
        df_expected = pd.DataFrame(
            [[1, 1, 0]] * 2, columns=["label_one", "label_two", "label_three"]
        )
        df_result = add_virtual_column(df, "label_one - label_two", "label_three")
        self.assertTrue(
            df_result.equals(df_expected),
            f"The function should subtract the columns: label_one and "
            f"label_two.\n\nResult:\n\n{df_result}\n\nExpected:\n\n{df_expected}",
        )

    def test_empty_result_when_invalid_labels(self):
        df = pd.DataFrame([[1, 2]] * 3, columns=["label_one", "label_two"])
        df_result = add_virtual_column(df, "label_one + label_two", "label3")
        self.assertTrue(
            df_result.empty,
            f'Should return an empty df when the "new_column" is invalid.'
            f"\n\nResult:\n\n{df_result}\n\nExpected:\n\nEmpty df",
        )

        df = pd.DataFrame([[1, 2]] * 3, columns=["label-one", "label_two"])
        df_result = add_virtual_column(df, "label-one + label_two", "label")
        self.assertTrue(
            df_result.empty,
            f"Should return an empty df when both df columns and roles are "
            f"invalid.\n\nResult:\n\n{df_result}\n\nExpected:\n\nEmpty df",
        )

        df = pd.DataFrame([[1, 2]] * 3, columns=["label-one", "label_two"])
        df_result = add_virtual_column(df, "label_one + label_two", "label")
        self.assertTrue(
            df_result.empty,
            f"Should return an empty df when a df column is invalid."
            f"\n\nResult:\n\n{df_result}\n\nExpected:\n\nEmpty df",
        )

    def test_empty_result_when_invalid_rules(self):
        df = pd.DataFrame([[1, 1]] * 2, columns=["label_one", "label_two"])
        df_result = add_virtual_column(df, "label_one \\ label_two", "label_three")
        self.assertTrue(
            df_result.empty,
            f"Should return an empty df when the role have invalid character: "
            f"'\\'.\n\nResult:\n\n{df_result}\n\nExpected:\n\nEmpty df",
        )

        df_result = add_virtual_column(df, "label&one + label_two", "label_three")
        self.assertTrue(
            df_result.empty,
            f"Should return an empty df when the role have invalid character: "
            f"'&'.\n\nResult:\n\n{df_result}\n\nExpected:\n\nEmpty df",
        )

        df_result = add_virtual_column(df, "label_five + label_two", "label_three")
        self.assertTrue(
            df_result.empty,
            f"Should return an empty df when the role have a column which "
            f"isn't in the df: 'label_five'.\n\nResult:\n\n{df_result}"
            f"\n\nExpected:\n\nEmpty df",
        )

    def test_when_extra_spaces_in_rules(self):
        df = pd.DataFrame([[1, 1]] * 2, columns=["label_one", "label_two"])
        df_expected = pd.DataFrame(
            [[1, 1, 2]] * 2, columns=["label_one", "label_two", "label_three"]
        )
        df_result = add_virtual_column(df, "label_one+label_two", "label_three")
        self.assertTrue(
            df_result.equals(df_expected),
            f"Should work when the role haven't spaces between the operation "
            f"and the column.\n\nResult:\n\n{df_result}\n\nExpected:\n\n{df_expected}",
        )

        df_result = add_virtual_column(df, "label_one + label_two ", "label_three")
        self.assertTrue(
            df_result.equals(df_expected),
            f"Should work when the role have spaces between the operation "
            f"and the column.\n\nResult:\n\n{df_result}\n\nExpected:\n\n{df_expected}",
        )

        df_result = add_virtual_column(df, " label_one + label_two ", "label_three")
        self.assertTrue(
            df_result.equals(df_expected),
            f"Should work when the role have extra spaces in the "
            f"start/end.\n\nResult:\n\n{df_result}\n\nExpected:\n\n{df_expected}",
        )


if __name__ == "__main__":
    unittest.main()