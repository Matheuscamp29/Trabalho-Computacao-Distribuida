import unittest

import logical_clock


class LamportClockTests(unittest.TestCase):
    def setUp(self):
        logical_clock.valor = 0

    def test_send_and_receive_follow_lamport_rule(self):
        self.assertEqual(logical_clock.tick(), 1)
        self.assertEqual(logical_clock.receive(5), 6)
        self.assertEqual(logical_clock.receive(2), 7)
        self.assertEqual(logical_clock.tick(), 8)
        self.assertEqual(logical_clock.atual(), 8)

    def test_negative_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            logical_clock.receive(-1)


if __name__ == "__main__":
    unittest.main()
