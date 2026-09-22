import math
import unittest

from gpu_suite.strict_json import (
    DuplicateKeyError,
    NonStandardConstantError,
    StrictJsonError,
    dump_bytes,
    dumps,
    loads,
)


class StrictJsonTests(unittest.TestCase):
    def test_duplicate_keys_are_rejected(self):
        with self.assertRaises(DuplicateKeyError):
            loads('{"a":1,"a":2}')

    def test_nonstandard_constants_are_rejected(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                with self.assertRaises(NonStandardConstantError):
                    loads('{"value":' + constant + "}")

    def test_finite_syntax_that_overflows_binary_float_is_rejected(self):
        with self.assertRaises(StrictJsonError):
            loads('{"value":1e9999}')

    def test_invalid_utf8_is_rejected(self):
        with self.assertRaises(StrictJsonError):
            loads(b'{"value":"\xff"}')

    def test_deterministic_output(self):
        value = {"z": ["日本語", -0.0], "a": 1.5}
        self.assertEqual(dumps(value), '{"a":1.5,"z":["日本語",0.0]}')
        self.assertEqual(
            dump_bytes(value),
            b'{"a":1.5,"z":["\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",0.0]}\n',
        )

    def test_nonfinite_output_is_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(StrictJsonError):
                    dumps({"value": value})


if __name__ == "__main__":
    unittest.main()
