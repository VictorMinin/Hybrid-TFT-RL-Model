import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer

def normalize_features(eurusd_df: pd.DataFrame):
    split_date = "2022-01-01"
    train_df = eurusd_df.loc[eurusd_df.index < split_date].copy()
    test_df = eurusd_df.loc[eurusd_df.index >= split_date].copy()
    scalers_path = "scalers.joblib"
    fitted_scalers = {}

    scaling_dict = {
        ("RSI_14_1", "RSI_14_5", "RSI_14_15", "RSI_14_30", "RSI_14_60", "RSI_14_240") : "StandardScaler",
        ("G_Width_1", "G_Width_5", "G_Width_15", "G_Width_30", "G_Width_60", "G_Width_240") : "RobustScaler",
        ("Upper_Slope_1", "Upper_Slope_5", "Upper_Slope_15", "Upper_Slope_30", "Upper_Slope_60", "Upper_Slope_240") : "QuantileTransformer",
        ("Middle_Slope_1", "Middle_Slope_5", "Middle_Slope_15", "Middle_Slope_30", "Middle_Slope_60", "Middle_Slope_240") : "QuantileTransformer",
        ("Lower_Slope_1", "Lower_Slope_5", "Lower_Slope_15", "Lower_Slope_30", "Lower_Slope_60", "Lower_Slope_240") : "QuantileTransformer",
        ("G%_1", "G%_5", "G%_15", "G%_30", "G%_60", "G%_240") : "StandardScaler",
        ("%K_1", "%K_5", "%K_15", "%K_30", "%K_60", "%K_240") : "StandardScaler",
        ("%D_1", "%D_5", "%D_15", "%D_30", "%D_60", "%D_240") : "StandardScaler",
        ("atr_1", "atr_5", "atr_15", "atr_30", "atr_60", "atr_240") : "PowerTransformer",
        ("fast_r_1", "fast_r_5", "fast_r_15", "fast_r_30", "fast_r_60", "fast_r_240") : "StandardScaler",
        ("slow_r_1", "slow_r_5", "slow_r_15", "slow_r_30", "slow_r_60", "slow_r_240") : "StandardScaler",
        ("r_diff_1", "r_diff_5", "r_diff_15", "r_diff_30", "r_diff_60", "r_diff_240") : "StandardScaler",
        ("dmx_signal_1", "dmx_signal_5", "dmx_signal_15", "dmx_signal_30", "dmx_signal_60", "dmx_signal_240") : "PowerTransformer",
        ("jma_slope_1", "jma_slope_5", "jma_slope_15", "jma_slope_30", "jma_slope_60", "jma_slope_240") : "RobustScaler",
        ("lr_slope_1_length_100", "lr_slope_5_length_100", "lr_slope_15_length_100", "lr_slope_30_length_100", "lr_slope_60_length_100", "lr_slope_240_length_100", ) : "RobustScaler",
        ("lr_slope_1_length_20", "lr_slope_5_length_20", "lr_slope_15_length_20", "lr_slope_30_length_20", "lr_slope_60_length_20", "lr_slope_240_length_20", ) : "RobustScaler"
    }

    for column_tuple, scaler_name in scaling_dict.items():
        for column in column_tuple:
            if scaler_name == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_name == "RobustScaler":
                scaler = RobustScaler()
            elif scaler_name == "QuantileTransformer":
                scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(train_df)//10, 1000), 10))
            elif scaler_name == "PowerTransformer":
                scaler = PowerTransformer()

            # Fit only on the training data
            scaler.fit(train_df[[column]])

            fitted_scalers[column] = scaler

            # Transform both the training and testing data
            train_df[column] = scaler.transform(train_df[[column]])
            test_df[column] = scaler.transform(test_df[[column]])

    joblib.dump(fitted_scalers, scalers_path)

    train_df.to_csv('X_train_normalized.csv')
    test_df.to_csv('X_test_normalized.csv')

    return