import sys
sys.path.insert(0, "/home/node/.local/lib/python3.11/site-packages")
sys.path.insert(0, "/workspace")
import unittest

if __name__ == "__main__":
    names = sys.argv[1:] or ["tests.test_sourcemap"]
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
