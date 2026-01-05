import unittest
from src.util.image_util import extract_time_features

class TestExtractTimeFeatures(unittest.TestCase):

    def test_standard_iso_format(self):
        """Test standard ISO 8601 format."""
        # Test morning
        month, period = extract_time_features("2023-12-01 08:30:00")
        self.assertEqual(month, 12)
        self.assertEqual(period, "morning")

    def test_exif_colon_format(self):
        """Test the common EXIF format using colons for dates (YYYY:MM:DD)."""
        month, period = extract_time_features("2023:05:15 14:20:00")
        self.assertEqual(month, 5)
        self.assertEqual(period, "afternoon")

    def test_time_period_boundaries(self):
        """Verify the edges of each time period category."""
        # 5:00 AM - Start of morning
        _, p1 = extract_time_features("2023-01-01 05:00:00")
        self.assertEqual(p1, "morning")

        # 12:00 PM - Start of afternoon
        _, p2 = extract_time_features("2023-01-01 12:00:00")
        self.assertEqual(p2, "afternoon")

        # 5:00 PM (17:00) - Start of evening
        _, p3 = extract_time_features("2023-01-01 17:00:00")
        self.assertEqual(p3, "evening")

        # 9:00 PM (21:00) - Start of night
        _, p4 = extract_time_features("2023-01-01 21:00:00")
        self.assertEqual(p4, "night")

        # 4:59 AM - Still night
        _, p5 = extract_time_features("2023-01-01 04:59:59")
        self.assertEqual(p5, "night")

    def test_invalid_and_empty_inputs(self):
        """Ensure the function handles bad data gracefully without crashing."""
        # Empty input
        self.assertEqual(extract_time_features(None), (None, None))
        self.assertEqual(extract_time_features(""), (None, None))

        # Completely invalid format
        self.assertEqual(extract_time_features("not-a-date"), (None, None))

        # Partial/Incomplete date
        self.assertEqual(extract_time_features("2023-12"), (None, None))

if __name__ == '__main__':
    unittest.main()