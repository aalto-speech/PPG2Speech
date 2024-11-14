import unittest
from ppg_tts.dataset import PersoDatasetBasic, PersoDatasetWithConditions


class TestPersoDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = PersoDatasetBasic("./data/test")


    def testKeys(self):
        test_entry = self.dataset[0]
        self.assertTrue("key" in test_entry)
        self.assertTrue("feature" in test_entry)
        self.assertTrue("text" in test_entry)

    def testAscending(self):
        prev = -1

        for data in self.dataset:
            num_frames = data["feature"].shape[-1]
            self.assertGreaterEqual(num_frames, prev, f"{data['key']} is not in sorted order")
            prev = num_frames

class TestPersoWithCondition(unittest.TestCase):
    def setUp(self):
        self.dataset = PersoDatasetWithConditions("data/test")

    def testPPGandMel(self):
        for data in self.dataset:
            self.assertEqual(data['ppg'].size(0), data['mel'].size(1), msg=f"{data['key']} have ppg shape {data['ppg'].shape} and have mel shape {data['mel'].shape}")

if __name__ == '__main__':
    TestPersoDataset.run()
    TestPersoWithCondition.run()