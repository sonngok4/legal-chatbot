#!/usr/bin/env python3
"""
Test runner script for News Chatbot API
Runs all unit tests for the chatbot system
"""

import os
import sys
import unittest
from io import StringIO

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all_tests():
    """Run all unit tests and generate a comprehensive report."""

    print("=" * 80)
    print("NEWS CHATBOT API - UNIT TEST RUNNER")
    print("=" * 80)

    # Test modules to run
    test_modules = ["test_traffic_law_chatbot", "test_healthcare_chatbot", "test_app"]

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load tests from each module
    for module_name in test_modules:
        try:
            tests = loader.loadTestsFromName(module_name)
            suite.addTests(tests)
            print(f"✅ Loaded tests from {module_name}")
        except Exception as e:
            print(f"❌ Failed to load tests from {module_name}: {e}")

    # Run tests with detailed output
    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)

    # Capture test output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, buffer=True)

    # Run the tests
    result = runner.run(suite)

    # Print results
    output = stream.getvalue()
    print(output)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, "skipped") else 0
    passed = total_tests - failures - errors - skipped

    print(f"Total Tests:  {total_tests}")
    print(f"Passed:       {passed} ✅")
    print(f"Failed:       {failures} ❌")
    print(f"Errors:       {errors} ⚠️")
    print(f"Skipped:      {skipped} ⏭️")

    if failures > 0:
        print("\nFAILURES:")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"❌ {test}")
            print(f"   {traceback.split(chr(10))[-2]}")

    if errors > 0:
        print("\nERRORS:")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"⚠️  {test}")
            print(f"   {traceback.split(chr(10))[-2]}")

    # Calculate success rate
    success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    if success_rate == 100:
        print("🎉 ALL TESTS PASSED! 🎉")
    elif success_rate >= 80:
        print("✅ Most tests passed, but some issues need attention.")
    else:
        print("❌ Many tests failed. Please review and fix issues.")

    print("=" * 80)

    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


def run_specific_module(module_name):
    """Run tests for a specific module."""
    print(f"Running tests for {module_name}...")

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Run specific test module
        module_name = sys.argv[1]
        return run_specific_module(module_name)
    else:
        # Run all tests
        return run_all_tests()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
