import argparse
import unittest

from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

GRADESCOPE_FP = "/autograder/results/results.json"
LOCAL_FP = "results.json"


def main(fp: str):
    suite = unittest.defaultTestLoader.discover("tests")
    with open(fp, "w") as f:
        JSONTestRunner(visibility="visible", stream=f).run(suite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(LOCAL_FP if args.debug else GRADESCOPE_FP)
