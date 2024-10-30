import unittest
from ppg_tts.dataset import PersoDatasetBasic


class TestPersoDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = PersoDatasetBasic("./data/val")


    def testKeys(self):
        test_entry = self.dataset[0]
        self.assertTrue("key" in test_entry)
        self.assertTrue("feature" in test_entry)
        self.assertTrue("text" in test_entry)
        self.assertTrue("melspectrogram" in test_entry)

    def testAscending(self):
        prev = -1

        for data in self.dataset:
            num_frames = data["feature"].shape[-1]
            self.assertGreaterEqual(num_frames, prev, f"{data[0]} is not in sorted order")
            prev = num_frames

if __name__ == '__main__':
    unittest.main()