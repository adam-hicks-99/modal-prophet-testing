"""
Modal app for running Prophet forecasts at scale.
Optimized for parallel execution across 850k time series.
"""

import modal
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Create Modal app
app = modal.App("prophet-performance-test")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "prophet>=1.1.5",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pystan<3.0",  # Prophet requires pystan 2.x
    )
)


@app.function(
    image=image,
    cpu=1.0,  # 1 CPU per task
    memory=2048,  # 2GB memory per task
    timeout=600,  # 10 minute timeout per task
)
def forecast_single_series(series_data: dict) -> dict:
    """
    Forecast a single time series using Prophet.

    Args:
        series_data: Dictionary containing:
            - series_id: Unique identifier
            - ds: List of dates (as strings)
            - y: List of values

    Returns:
        Dictionary with forecast results and timing info
    """
    from prophet import Prophet

    start_time = time.time()
    series_id = series_data['series_id']

    try:
        # Prepare data
        df = pd.DataFrame({
            'ds': pd.to_datetime(series_data['ds']),
            'y': series_data['y']
        })

        # Initialize and fit Prophet model
        model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )

        model.fit(df)

        # Create future dataframe for 24 months
        future = model.make_future_dataframe(periods=24, freq='MS')

        # Generate forecast
        forecast = model.predict(future)

        # Extract only the forecasted periods (last 24 months)
        forecast_only = forecast.tail(24)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        execution_time = time.time() - start_time

        return {
            'series_id': series_id,
            'success': True,
            'execution_time': execution_time,
            'forecast_dates': forecast_only['ds'].dt.strftime('%Y-%m-%d').tolist(),
            'forecast_values': forecast_only['yhat'].tolist(),
            'forecast_lower': forecast_only['yhat_lower'].tolist(),
            'forecast_upper': forecast_only['yhat_upper'].tolist(),
            'error': None
        }

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            'series_id': series_id,
            'success': False,
            'execution_time': execution_time,
            'forecast_dates': None,
            'forecast_values': None,
            'forecast_lower': None,
            'forecast_upper': None,
            'error': str(e)
        }


@app.function(
    image=image,
    cpu=1.0,
    memory=1024,
    timeout=300,
)
def prepare_series_data(series_id: int, n_periods: int = 36) -> dict:
    """
    Generate data for a single series (runs on Modal).

    Args:
        series_id: Unique identifier for the series
        n_periods: Number of historical periods

    Returns:
        Dictionary with series data ready for forecasting
    """
    # Generate date range
    dates = pd.date_range(start="2021-01-01", periods=n_periods, freq='MS')

    # Generate synthetic data
    np.random.seed(series_id)
    base_value = np.random.uniform(100, 10000)
    trend = np.linspace(0, base_value * 0.3, n_periods)
    seasonality = base_value * 0.2 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)
    noise = np.random.normal(0, base_value * 0.1, n_periods)
    values = np.maximum(base_value + trend + seasonality + noise, 0)

    return {
        'series_id': series_id,
        'ds': dates.strftime('%Y-%m-%d').tolist(),
        'y': values.tolist()
    }


@app.local_entrypoint()
def main(
    n_series: int = 850000,
    batch_size: int = 1000,
    max_parallel: int = 192
):
    """
    Main entry point for running the forecasting job.

    Args:
        n_series: Total number of series to forecast (default 850k)
        batch_size: Number of series to process in each batch
        max_parallel: Maximum parallel workers (default 192)
    """
    print(f"Starting Prophet forecasting job on Modal")
    print(f"Total series: {n_series:,}")
    print(f"Batch size: {batch_size:,}")
    print(f"Max parallel workers: {max_parallel}")
    print(f"=" * 60)

    overall_start = time.time()

    # Generate series IDs
    series_ids = list(range(n_series))

    # Process in batches to manage memory
    total_processed = 0
    total_successful = 0
    total_failed = 0
    all_execution_times = []

    for batch_start in range(0, n_series, batch_size):
        batch_end = min(batch_start + batch_size, n_series)
        batch_ids = series_ids[batch_start:batch_end]

        batch_start_time = time.time()
        print(f"\nProcessing batch {batch_start // batch_size + 1}: "
              f"Series {batch_start:,} to {batch_end:,}")

        # Step 1: Generate data in parallel
        print(f"  Generating data for {len(batch_ids)} series...")
        data_gen_start = time.time()
        series_data_list = list(prepare_series_data.map(batch_ids))
        data_gen_time = time.time() - data_gen_start
        print(f"  Data generation completed in {data_gen_time:.2f}s")

        # Step 2: Run forecasts in parallel
        print(f"  Running forecasts...")
        forecast_start = time.time()
        results = list(forecast_single_series.map(series_data_list))
        forecast_time = time.time() - forecast_start

        # Collect statistics
        batch_successful = sum(1 for r in results if r['success'])
        batch_failed = len(results) - batch_successful
        batch_times = [r['execution_time'] for r in results if r['success']]

        total_processed += len(results)
        total_successful += batch_successful
        total_failed += batch_failed
        all_execution_times.extend(batch_times)

        batch_elapsed = time.time() - batch_start_time

        print(f"  Batch completed in {batch_elapsed:.2f}s")
        print(f"  Forecast execution: {forecast_time:.2f}s")
        print(f"  Successful: {batch_successful}, Failed: {batch_failed}")
        if batch_times:
            print(f"  Avg time per series: {np.mean(batch_times):.3f}s")
            print(f"  Throughput: {len(results) / forecast_time:.1f} series/sec")

    overall_elapsed = time.time() - overall_start

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total series processed: {total_processed:,}")
    print(f"Successful forecasts: {total_successful:,}")
    print(f"Failed forecasts: {total_failed:,}")
    print(f"Success rate: {100 * total_successful / total_processed:.2f}%")
    print(f"\nTotal execution time: {overall_elapsed:.2f}s ({overall_elapsed / 60:.2f} minutes)")

    if all_execution_times:
        print(f"\nPer-series execution time statistics:")
        print(f"  Mean: {np.mean(all_execution_times):.3f}s")
        print(f"  Median: {np.median(all_execution_times):.3f}s")
        print(f"  Min: {np.min(all_execution_times):.3f}s")
        print(f"  Max: {np.max(all_execution_times):.3f}s")
        print(f"  Std Dev: {np.std(all_execution_times):.3f}s")

    print(f"\nOverall throughput: {total_processed / overall_elapsed:.1f} series/sec")

    # Cost estimation (approximate Modal pricing)
    # Modal charges approximately $0.000030 per CPU-second
    total_cpu_seconds = sum(all_execution_times) if all_execution_times else 0
    estimated_cost = total_cpu_seconds * 0.000030
    print(f"\nEstimated cost: ${estimated_cost:.2f}")
    print(f"  (Based on {total_cpu_seconds:.0f} CPU-seconds at $0.000030 per CPU-second)")
    print("=" * 60)
