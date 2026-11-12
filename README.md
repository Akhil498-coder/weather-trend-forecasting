# Weather Trend Forecasting

Akhilesh Kanoj

This is my submission for the PM Accelerator Data Science Intern assessment. I cleaned and explored two and a half
years of daily weather data for 185 capital cities, forecast tomorrow's temperature in six of them, and looked at
anomalies, air quality and regional patterns.

## Dataset

I used the [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)
from Kaggle. It has one weather snapshot per capital per day. My copy runs from 16 May 2024 to 13 October 2026
(170,928 rows, 41 columns).

The CSV is not in this repo because it is large and gets updated. Download it from Kaggle and save it as
`data/GlobalWeatherRepository.csv`.

## How to run

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then run the notebooks in `notebooks/` in order, from 01 to 06. Notebook 01 creates
`data/processed/weather_clean.csv`, which all the other notebooks read. Notebook 03 creates
`data/processed/city_features.csv`, which notebook 04 reads. I used Python 3.13. The whole run takes a few minutes.

## Folder structure

```
data/                    raw CSV (not committed) and data/processed/ (created by the notebooks)
notebooks/               the six notebooks, saved with their outputs
reports/REPORT.md        the written report
reports/figures/         every figure, saved by the notebooks
reports/metrics/         validation and test metrics as CSV
requirements.txt
```

## What I did

**01 Data cleaning.** I kept the 185 locations with at least 500 days of data. The rest were mostly the same
capitals with the country name in another language, so they could not form a time series. I set impossible values
to missing instead of clipping them (79.3 C in Suva, a wind speed of 2,963 km/h, pressure of 3,000 mb). I found that
for one week in July 2026 some cities reported station pressure instead of sea-level pressure, and that on
17 August 2026 about half of the cities had nonsense readings, like a blizzard in London. I removed both. I also
found that the time of the daily snapshot moved about 6 hours earlier over the period, which matters a lot for
temperature, so I kept the local hour as a feature.

**02 EDA.** I looked at temperature against latitude, the two hemispheres, the six forecast cities, rain and
correlations. The main lesson was that a global average temperature is not useful, because the northern and southern
seasons cancel out. That is why I forecast single cities.

**03 Baselines.** I picked six cities from different climates: Moscow, London, New Delhi, Singapore, Canberra and
Buenos Aires. I built features that only use the past (lags, rolling means, time of year, yesterday's weather) and
split the data by time: train until April 2026, 90 days of validation, then the last 90 days as test. I compared
Ridge regression with two baselines: yesterday's value and the same day last year.

**04 More models and an ensemble.** I added Random Forest, Gradient Boosting and SARIMAX. I could not use a seasonal
ARIMA with a 365-day season because there are only about two years of data, so I used Fourier terms for the yearly
cycle. I chose the ensemble weights on the validation window and used the test window only once at the end. I also
compared feature importance with three methods.

**05 Anomaly detection.** I compared a z-score on the deseasonalized temperature with Isolation Forest on several
variables. They mostly flag different days: the z-score finds hot and cold spikes, Isolation Forest finds storms and
heavy rain. I also looked at PM2.5 outliers.

**06 Climate, air quality and spatial patterns.** I compared continents and latitude bands, checked how air quality
moves with the weather inside each city, mapped PM2.5, and measured how fast the link between two cities' weather
fades with distance.

## Results

Test window: 16 July to 13 October 2026, 89 days per city. Mean absolute error in C, lower is better.

| City | Naive (yesterday) | Seasonal naive | Best single model | Weighted ensemble |
|---|---|---|---|---|
| Moscow | 1.96 | 3.87 | 1.77 (Gradient Boosting) | 1.77 |
| London | 2.11 | 2.78 | 1.86 (Gradient Boosting) | 1.88 |
| New Delhi | 1.65 | 3.61 | 1.71 (Random Forest) | 1.74 |
| Singapore | 0.81 | 2.04 | 0.83 (Random Forest) | 0.86 |
| Canberra | 2.40 | 3.65 | 2.13 (Ridge) | 2.11 |
| Buenos Aires | 2.29 | 3.79 | 2.07 (Random Forest) | 2.16 |
| Average | 1.87 | 3.29 | 1.77 (Gradient Boosting) | 1.75 |

- The weighted ensemble, which I picked on the validation window, had the lowest average error, about 6% better
  than "same as yesterday". A block bootstrap says that average gain is real, but city by city it is only clear in
  Moscow, London and Canberra. The test window is too short to tell the best few models apart.
- No model beat the naive baseline in New Delhi or Singapore. In Singapore the readings become almost flat from
  September 2026, so yesterday's value is nearly perfect there.
- SARIMAX was worse than the naive baseline on average. Every model beat the seasonal naive baseline.
- The most useful inputs were yesterday's temperature, pressure and humidity.
- Wind and rain go with cleaner air in most cities, while ozone rises with sunlight. Day-to-day temperature swings
  of two capitals are linked when they are close (correlation 0.59 under 250 km) and unrelated beyond about 2,000 km.

All metrics (MAE, RMSE, sMAPE, R2) for every model and city are in `reports/metrics/` and in
[reports/REPORT.md](reports/REPORT.md).

## Limitations

- About 2.4 years of data is only two yearly cycles. That is not enough for proper seasonal models.
- The test window is a single season (northern late summer, southern late winter), so the results may not hold
  in other months.
- There is one reading per day, and its time of day drifted by about 6 hours. This mixes weather with the daily
  cycle and makes trends between years unreliable.
- The data source seems to have changed a few times (pressure in July 2026, the bad day in August 2026, flat
  readings in some tropical cities from September 2026). The last two are in my test window.
- I first noticed the bad day of 17 August 2026 in the test plots. I then wrote a rule for it that I applied to the
  whole period, and it only catches that day. It still means a cleaning choice was made with test data in view,
  so I documented it in notebook 01 and showed how the naive baseline scores with the day kept.

## About PM Accelerator

Product Manager Accelerator is a program based in Boston, founded in 2020, that supports product managers at every
stage of their career, from students looking for their first PM job to directors moving into leadership. Its
programs include PMA Pro for PM job hunting, an AI PM Bootcamp where teams build and launch a real AI product,
PMA Power Skills, PMA Leader and 1:1 resume reviews. It also publishes hundreds of free PM training videos and
courses on YouTube. More at [pmaccelerator.io](https://www.pmaccelerator.io).
