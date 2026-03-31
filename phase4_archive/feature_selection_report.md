# Method 33: Deep EDA & Feature Selection Report

## 1. Target Variable Analysis
- **Count**: 1024
- **Skew**: 0.0402 (Raw) vs -0.2839 (Log)
- **Kurtosis**: -1.5161 (Raw) vs -1.7310 (Log)
- **Conclusion**: Log-transform not effective (Unlikely). Check data.

## 2. Image Feature Analysis
- **Total Image Features**: 65
- **Redundancy Check**:
    - **Cluster 9**: 3 features (s2_B1_mean, s2_B2_mean, s2_B4_mean...). Keeping **s2_B4_mean**.
    - **Cluster 8**: 5 features (s2_B1_std, s2_B2_std, s2_B3_std...). Keeping **s2_B1_std**.
    - **Cluster 11**: 2 features (s2_B1_median, s2_B2_median...). Keeping **s2_B2_median**.
    - **Cluster 17**: 2 features (s2_B1_min, s2_B2_min...). Keeping **s2_B1_min**.
    - **Cluster 1**: 3 features (s2_B1_max, s2_B2_max, s2_B3_max...). Keeping **s2_B1_max**.
    - **Cluster 10**: 3 features (s2_B3_mean, s2_B5_mean, s2_B9_std...). Keeping **s2_B5_mean**.
    - **Cluster 15**: 3 features (s2_B3_median, s2_B5_median, s2_B12_mean...). Keeping **s2_B12_mean**.
    - **Cluster 16**: 2 features (s2_B3_min, s2_B4_min...). Keeping **s2_B3_min**.
    - **Cluster 2**: 6 features (s2_B4_max, s2_B5_max, s2_B6_max...). Keeping **s2_B8A_max**.
    - **Cluster 19**: 4 features (s2_B5_min, s2_B9_min, s2_B11_min...). Keeping **s2_B5_min**.
    - **Cluster 4**: 5 features (s2_B6_mean, s2_B7_mean, s2_B8_mean...). Keeping **s2_B9_mean**.
    - **Cluster 5**: 4 features (s2_B6_std, s2_B7_std, s2_B8_std...). Keeping **s2_B6_std**.
    - **Cluster 3**: 5 features (s2_B6_median, s2_B7_median, s2_B8_median...). Keeping **s2_B8A_median**.
    - **Cluster 18**: 4 features (s2_B6_min, s2_B7_min, s2_B8_min...). Keeping **s2_B7_min**.
    - **Cluster 14**: 2 features (s2_B11_mean, s2_B11_median...). Keeping **s2_B11_mean**.
    - **Cluster 6**: 2 features (s2_B11_std, s2_B12_std...). Keeping **s2_B11_std**.
    - **Cluster 22**: 4 features (viirs_mean, viirs_std, viirs_median...). Keeping **viirs_max**.

- **Reduced Set**: 23 features (from 65).

## 3. Feature Importance (Random Forest)
| Feature | Importance | Std |
|---|---|---|
| deflated_gdp_usd | 1.8845 | 0.0492 |
| straight_distance_to_capital_km | 0.0250 | 0.0010 |
| us_cpi | 0.0127 | 0.0006 |
| viirs_max | 0.0048 | 0.0002 |
| s2_B8A_median | 0.0032 | 0.0005 |
| s2_B5_min | 0.0024 | 0.0001 |
| seismic_hazard_zone_Moderate | 0.0021 | 0.0002 |
| s2_B7_min | 0.0020 | 0.0002 |
| s2_B9_max | 0.0019 | 0.0002 |
| s2_B12_max | 0.0018 | 0.0001 |
| s2_B9_mean | 0.0017 | 0.0001 |
| s2_B11_max | 0.0016 | 0.0001 |
| s2_B4_median | 0.0014 | 0.0001 |
| s2_B11_std | 0.0013 | 0.0001 |
| s2_B1_max | 0.0013 | 0.0001 |
| s2_B8A_max | 0.0012 | 0.0001 |
| s2_B12_median | 0.0012 | 0.0001 |
| s2_B11_mean | 0.0011 | 0.0001 |
| s2_B6_std | 0.0011 | 0.0001 |
| s2_B3_min | 0.0011 | 0.0001 |
| viirs_min | 0.0010 | 0.0001 |
| s2_B2_median | 0.0010 | 0.0001 |
| s2_B12_mean | 0.0009 | 0.0001 |
| s2_B1_min | 0.0009 | 0.0001 |
| s2_B1_std | 0.0009 | 0.0001 |
| s2_B5_mean | 0.0008 | 0.0001 |
| koppen_climate_zone_Cwb | 0.0008 | 0.0001 |
| s2_B4_mean | 0.0006 | 0.0000 |
| tropical_cyclone_wind_risk_Moderate | 0.0004 | 0.0001 |
| seismic_hazard_zone_High | 0.0002 | 0.0000 |
| flood_risk_class_Yes | 0.0002 | 0.0000 |
| tropical_cyclone_wind_risk_Low | 0.0002 | 0.0000 |
| flood_risk_class_No | 0.0001 | 0.0000 |
| tropical_cyclone_wind_risk_Very High | 0.0001 | 0.0000 |
| koppen_climate_zone_Af | 0.0001 | 0.0000 |
| access_to_port_No | 0.0001 | 0.0000 |
| access_to_airport_No | 0.0001 | 0.0000 |
| access_to_port_Yes | 0.0001 | 0.0000 |
| tornadoes_wind_risk_Low | 0.0001 | 0.0000 |
| tropical_cyclone_wind_risk_High | 0.0001 | 0.0000 |
| tornadoes_wind_risk_Very Low | 0.0001 | 0.0000 |
| access_to_airport_Yes | 0.0001 | 0.0000 |
| koppen_climate_zone_Am | 0.0001 | 0.0000 |
| access_to_railway_No | 0.0000 | 0.0000 |
| koppen_climate_zone_Cfa | 0.0000 | 0.0000 |
| access_to_railway_Yes | 0.0000 | 0.0000 |
| seismic_hazard_zone_Low | 0.0000 | 0.0000 |
| koppen_climate_zone_Dfa | 0.0000 | 0.0000 |
| koppen_climate_zone_Dfb | 0.0000 | 0.0000 |
| koppen_climate_zone_Aw | 0.0000 | 0.0000 |
| access_to_highway_Yes | 0.0000 | 0.0000 |
| tropical_cyclone_wind_risk_0 | 0.0000 | 0.0000 |
| access_to_highway_No | 0.0000 | 0.0000 |
| access_to_port_nan | 0.0000 | 0.0000 |
| flood_risk_class_nan | 0.0000 | 0.0000 |
| seismic_hazard_zone_nan | 0.0000 | 0.0000 |
| access_to_highway_nan | 0.0000 | 0.0000 |
| access_to_railway_nan | 0.0000 | 0.0000 |
| access_to_airport_nan | 0.0000 | 0.0000 |
| tropical_cyclone_wind_risk_nan | 0.0000 | 0.0000 |
| tornadoes_wind_risk_nan | 0.0000 | 0.0000 |
| koppen_climate_zone_nan | 0.0000 | 0.0000 |

- **Selected Features**: 21 (Importance > 0.001)
