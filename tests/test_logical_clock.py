import unittest

from logical_clock import LamportClock


class LamportClockTests(unittest.TestCase):
    def test_send_and_receive_follow_lamport_rule(self):
        clock = LamportClock()
        self.assertEqual(clock.tick(), 1)
        self.assertEqual(clock.receive(5), 6)
        self.assertEqual(clock.receive(2), 7)
        self.assertEqual(clock.tick(), 8)

    def test_negative_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            LamportClock().receive(-1)


if __name__ == "__main__":
    unittest.main()
