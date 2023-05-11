import unittest

import numpy as np
import numpy.testing

from skactiveml.stream.clustering._clu_stream_al import MicroCluster, CluStream


class TestCluStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream = np.random.uniform(size=(10000, 5))
        cls.cluStream = CluStream(10, 1, 100)

        for x in cls.stream:
            cls.cluStream.fit_one(x)

    def test_features(self):
        for i, mc in self.cluStream.micro_clusters.items():
            self.assertEqual(len(mc.x), mc.features["n"])
            np.testing.assert_allclose(sum(mc.x), mc.features["ls_x"])

    def test_radius(self):

        for i, mc in self.cluStream.micro_clusters.items():
            np.testing.assert_allclose(np.std(mc.x, axis=0), np.sqrt(mc.features["M"] / (mc.features["n"])))

