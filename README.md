# Modal Prophet Performance Testing

Performance testing framework for running Prophet time series forecasting at scale using Modal's distributed compute platform.

## Overview

This project tests the performance and cost of running Prophet forecasts on 850,000 monthly time series using Modal's distributed computing platform with 192 cores.

### Test Specifications

- **Number of series**: 850,000 time series
- **Data grain**: Monthly
- **Historical data**: 3 years (36 months)
- **Forecast horizon**: 24 months into the future
- **Parallel compute**: 192 cores on Modal
- **Prophet configuration**: Multiplicative seasonality, yearly seasonality enabled

## Project Structure

```
modal-prophet-testing/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── data_generator.py             # Time series data generation utilities
├── modal_prophet_forecast.py     # Main Modal app for distributed forecasting
└── run_performance_test.py       # Helper script to run tests
```

## Prerequisites

1. **Python 3.11+**
2. **Modal account**: Sign up at https://modal.com
3. **Modal CLI**: Install and authenticate

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and authenticate Modal CLI

```bash
pip install modal
modal token new
```

This will open a browser window to authenticate your Modal account.

## Usage

### Quick Start

Run the full performance test with 850k series:

```bash
python run_performance_test.py --full
```

Or run directly with Modal:

```bash
modal run modal_prophet_forecast.py
```

### Test Configurations

#### Full Test (850k series - Production Scale)
```bash
python run_performance_test.py --full
```

#### Quick Test (1k series - Testing)
```bash
python run_performance_test.py --quick
```

#### Medium Test (10k series - Validation)
```bash
python run_performance_test.py --medium
```

#### Custom Test
```bash
python run_performance_test.py \
  --n-series 100000 \
  --batch-size 5000 \
  --max-parallel 192
```

### Command-Line Options

- `--full`: Run with 850,000 series (default)
- `--quick`: Run with 1,000 series for quick testing
- `--medium`: Run with 10,000 series for validation
- `--n-series N`: Custom number of series to forecast
- `--batch-size N`: Batch size for processing (default: 1000)
- `--max-parallel N`: Maximum parallel workers (default: 192)

## Performance Metrics

The test reports the following metrics:

### Execution Metrics
- **Total execution time**: End-to-end time for all forecasts
- **Per-series execution time**: Mean, median, min, max, std dev
- **Throughput**: Series processed per second
- **Success rate**: Percentage of successful forecasts

### Cost Metrics
- **Total CPU-seconds**: Sum of all computation time
- **Estimated cost**: Based on Modal's pricing (~$0.000030 per CPU-second)

### Example Output

```
==============================================================
FINAL SUMMARY
==============================================================
Total series processed: 850,000
Successful forecasts: 849,847
Failed forecasts: 153
Success rate: 99.98%

Total execution time: 3542.15s (59.04 minutes)

Per-series execution time statistics:
  Mean: 4.823s
  Median: 4.731s
  Min: 2.145s
  Max: 9.876s
  Std Dev: 0.892s

Overall throughput: 239.9 series/sec

Estimated cost: $123.45
  (Based on 4,115,000 CPU-seconds at $0.000030 per CPU-second)
==============================================================
```

## Cost Estimation

### Modal Pricing (Approximate)
- **CPU compute**: ~$0.000030 per CPU-second
- **Memory**: Included with CPU allocation
- **Network/IO**: Minimal for this workload

### Expected Costs for 850k Series

Based on Prophet's typical performance:

| Scenario | Avg Time/Series | Total CPU-seconds | Est. Cost |
|----------|----------------|-------------------|-----------|
| Optimistic | 3s | 2,550,000 | $76.50 |
| Expected | 5s | 4,250,000 | $127.50 |
| Conservative | 7s | 5,950,000 | $178.50 |

**Note**: Actual costs depend on:
- Prophet convergence speed
- Data characteristics
- Modal infrastructure overhead
- Network latency

## Technical Details

### Modal Configuration

- **CPU per task**: 1 CPU
- **Memory per task**: 2GB
- **Timeout per task**: 600 seconds (10 minutes)
- **Parallel execution**: Modal automatically distributes across available workers
- **Image**: Debian Slim with Python 3.11

### Prophet Configuration

```python
Prophet(
    seasonality_mode='multiplicative',
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
)
```

### Data Generation

Synthetic data includes:
- **Base value**: Random baseline between 100 and 10,000
- **Trend component**: Linear growth of 30% over historical period
- **Seasonal component**: 12-month sinusoidal pattern (20% of base)
- **Noise**: Normal distribution (10% of base standard deviation)

## Optimization Tips

### For Faster Execution

1. **Increase batch size**: Reduces overhead between batches
   ```bash
   --batch-size 5000
   ```

2. **Simplify Prophet model**: Disable unnecessary components
   ```python
   Prophet(yearly_seasonality=False)
   ```

3. **Reduce forecast horizon**: If you don't need 24 months
   ```python
   model.make_future_dataframe(periods=12, freq='MS')
   ```

### For Lower Cost

1. **Reduce CPU allocation**: If acceptable performance
   ```python
   cpu=0.5  # Half CPU per task
   ```

2. **Process in smaller batches**: Better cost control but slower
   ```bash
   --batch-size 500
   ```

## Monitoring

### View Modal Dashboard

Visit https://modal.com/apps to see:
- Real-time execution progress
- Active containers and CPU usage
- Costs accumulating
- Logs and errors

### Check Logs

```bash
modal app logs prophet-performance-test
```

## Troubleshooting

### "Modal token not found"
```bash
modal token new
```

### "Out of memory errors"
Increase memory allocation in `modal_prophet_forecast.py`:
```python
@app.function(memory=4096)  # 4GB instead of 2GB
```

### "Timeout errors"
Increase timeout or simplify Prophet model:
```python
@app.function(timeout=900)  # 15 minutes instead of 10
```

### Slow performance
- Check Modal dashboard for container startup times
- Consider increasing batch size
- Verify network connectivity

## Local Testing

Test data generation locally:

```bash
python data_generator.py
```

Test single series forecast locally (requires local Modal setup):

```python
from modal_prophet_forecast import prepare_series_data, forecast_single_series

# This won't work locally without Modal runtime,
# but you can test the logic
```

## Next Steps

1. **Run a quick test** to verify setup:
   ```bash
   python run_performance_test.py --quick
   ```

2. **Run a medium test** to estimate performance:
   ```bash
   python run_performance_test.py --medium
   ```

3. **Run the full test** when ready:
   ```bash
   python run_performance_test.py --full
   ```

4. **Analyze results** and optimize as needed

## Resources

- **Modal Documentation**: https://modal.com/docs
- **Prophet Documentation**: https://facebook.github.io/prophet/
- **Modal Pricing**: https://modal.com/pricing

## License

MIT License - Feel free to use and modify for your needs.
