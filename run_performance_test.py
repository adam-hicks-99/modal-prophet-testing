"""
Standalone script to run Prophet forecasting performance tests on Modal.
Provides easy configuration and detailed performance metrics.
"""

import subprocess
import sys
import argparse
from datetime import datetime


def run_modal_app(n_series: int, batch_size: int, max_parallel: int = 192):
    """
    Run the Modal app with specified parameters.

    Args:
        n_series: Total number of series to forecast
        batch_size: Number of series per batch
        max_parallel: Maximum parallel workers
    """
    print(f"\n{'='*70}")
    print(f"PROPHET FORECASTING PERFORMANCE TEST ON MODAL")
    print(f"{'='*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Total series: {n_series:,}")
    print(f"  Batch size: {batch_size:,}")
    print(f"  Max parallel workers: {max_parallel}")
    print(f"  Data: 36 months historical, 24 months forecast")
    print(f"{'='*70}\n")

    # Build the modal run command
    cmd = [
        "modal", "run",
        "modal_prophet_forecast.py",
        "--n-series", str(n_series),
        "--batch-size", str(batch_size),
        "--max-parallel", str(max_parallel)
    ]

    try:
        # Run the command
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n{'='*70}")
        print(f"Test completed successfully!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"\nError running Modal app: {e}")
        return 1
    except FileNotFoundError:
        print("\nError: Modal CLI not found. Please install it with: pip install modal")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run Prophet forecasting performance test on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full test with 850k series (production scale)
  python run_performance_test.py --full

  # Quick test with 1,000 series
  python run_performance_test.py --quick

  # Medium test with 10,000 series
  python run_performance_test.py --medium

  # Custom test with specific parameters
  python run_performance_test.py --n-series 100000 --batch-size 5000 --max-parallel 192

Notes:
  - Each series has 36 months of historical data
  - Forecasts 24 months into the future
  - Uses Modal for distributed parallel execution
  - Cost is approximately $0.000030 per CPU-second
        """
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test with 850,000 series (default configuration)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with 1,000 series"
    )

    parser.add_argument(
        "--medium",
        action="store_true",
        help="Run medium test with 10,000 series"
    )

    parser.add_argument(
        "--n-series",
        type=int,
        help="Number of series to forecast (custom)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)"
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=192,
        help="Maximum parallel workers (default: 192)"
    )

    args = parser.parse_args()

    # Determine configuration
    if args.full:
        n_series = 850000
        batch_size = 1000
    elif args.quick:
        n_series = 1000
        batch_size = 500
    elif args.medium:
        n_series = 10000
        batch_size = 1000
    elif args.n_series:
        n_series = args.n_series
        batch_size = args.batch_size
    else:
        # Default to full test
        print("No test size specified. Use --full, --quick, --medium, or --n-series")
        print("Running with default: 850,000 series (use --help for options)")
        n_series = 850000
        batch_size = 1000

    max_parallel = args.max_parallel

    # Run the test
    return run_modal_app(n_series, batch_size, max_parallel)


if __name__ == "__main__":
    sys.exit(main())
