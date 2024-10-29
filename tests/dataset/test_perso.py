import unittest
from ppg_tts.dataset import PersoDataset


class TestPersoDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = PersoDataset("./data/val")


    def testKeys(self):
        test_entry = self.dataset[0]
        self.assertTrue("feature" in test_entry[1])
        self.assertTrue("text" in test_entry[1])
        self.assertTrue("melspectrogram" in test_entry[1])

    def testAscending(self):
        prev = -1

        for data in self.dataset:
            num_frames = data[1]["feature"].shape[-1]
            self.assertGreaterEqual(num_frames, prev, f"{data[0]} is not in sorted order")
            prev = num_frames

if __name__ == '__main__':
    unittest.main()