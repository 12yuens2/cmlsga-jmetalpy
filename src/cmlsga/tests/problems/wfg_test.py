import unittest

from cmlsga.problems.wfg import *

class TestWFGShapesTransforms(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestWFGShapesTransforms, self).__init__(*args, **kwargs)

        self.x = [i for i in range(1, 10)]
        self.m = 1
        self.alpha = 2
        self.beta = 0.2

    def test_shape(self):
        linear(self.x, self.m)
        convex(self.x, self.m)
        concave(self.x, self.m)
        mixed(self.x, self.m, self.alpha)
        disc(self.x, self.m, self.alpha, self.beta)


if __name__ == "__main__":
    unittest.main()
