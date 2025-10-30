"""
Data generation module for creating synthetic time series data.
Generates 850k monthly time series with 3 years (36 months) of historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import pyarrow as pa
import pyarrow.parquet as pq


def generate_single_series(
    series_id: int,
    n_periods: int = 36,
    start_date: str = "2021-01-01"
) -> pd.DataFrame:
    """
    Generate a single time series with trend, seasonality, and noise.

    Args:
        series_id: Unique identifier for the series
        n_periods: Number of historical periods (default 36 months)
        start_date: Starting date for the series

    Returns:
        DataFrame with columns: series_id, ds, y
    """
    # Create date range
    dates = pd.date_range(start=start_date, periods=n_periods, freq='MS')

    # Generate base value (different for each series)
    np.random.seed(series_id)
    base_value = np.random.uniform(100, 10000)

    # Generate trend component
    trend = np.linspace(0, base_value * 0.3, n_periods)

    # Generate seasonal component (12-month seasonality)
    seasonality = base_value * 0.2 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)

    # Generate noise
    noise = np.random.normal(0, base_value * 0.1, n_periods)

    # Combine components
    values = base_value + trend + seasonality + noise

    # Ensure no negative values
    values = np.maximum(values, 0)

    return pd.DataFrame({
        'series_id': series_id,
        'ds': dates,
        'y': values
    })


def generate_batch_series(
    series_ids: List[int],
    n_periods: int = 36,
    start_date: str = "2021-01-01"
) -> pd.DataFrame:
    """
    Generate multiple time series in a batch.

    Args:
        series_ids: List of series IDs to generate
        n_periods: Number of historical periods
        start_date: Starting date for all series

    Returns:
        DataFrame with all series concatenated
    """
    series_list = []
    for series_id in series_ids:
        series_list.append(generate_single_series(series_id, n_periods, start_date))

    return pd.concat(series_list, ignore_index=True)


def generate_all_series_metadata(
    n_series: int = 850000,
    batch_size: int = 10000
) -> List[Tuple[int, int]]:
    """
    Generate metadata for batching series generation.

    Args:
        n_series: Total number of series to generate (default 850k)
        batch_size: Number of series per batch

    Returns:
        List of tuples (start_id, end_id) for each batch
    """
    batches = []
    for i in range(0, n_series, batch_size):
        start_id = i
        end_id = min(i + batch_size, n_series)
        batches.append((start_id, end_id))

    return batches


def save_series_to_parquet(
    df: pd.DataFrame,
    output_path: str
):
    """
    Save series data to Parquet format for efficient storage.

    Args:
        df: DataFrame containing series data
        output_path: Path to save the parquet file
    """
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression='snappy')


def get_series_list(n_series: int = 850000) -> List[int]:
    """
    Get list of all series IDs.

    Args:
        n_series: Total number of series

    Returns:
        List of series IDs from 0 to n_series-1
    """
    return list(range(n_series))


if __name__ == "__main__":
    # Test generation with a small sample
    print("Testing data generation with 10 series...")
    test_series = generate_batch_series(list(range(10)))
    print(f"Generated {len(test_series)} rows")
    print("\nSample data:")
    print(test_series.head(10))
    print("\nData info:")
    print(test_series.info())
