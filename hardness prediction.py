from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# =====================================================
# 1. XGBOOST CONTROL
# =====================================================

try:
    from xgboost import XGBRegressor
    HAS_XGB = True

except ImportError:
    HAS_XGB = False


# =====================================================
# 2. LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

log = logging.getLogger(__name__)


# =====================================================
# 3. CONFIGURATION
# =====================================================

@dataclass
class Config:

    base_path: str = r"C:\Users\Ayberk\Desktop\isilislem data"

    input_file: str = "raw_tempering_data.csv"

    output_dir: str = "outputs"

    model_path: str = "outputs/rf_final_model.joblib"

    test_size: float = 0.25

    random_state: int = 42

    cv_folds: int = 5

    numeric_features: List[str] = field(
        default_factory=lambda: [
            "Initial_Hardness_HRC",
            "Tempering_Time_s",
            "Tempering_Temperature_C",
            "C", "Mn", "P", "S",
            "Si", "Ni", "Cr", "Mo",
            "V", "Al", "Cu"
        ]
    )

    categorical_features: List[str] = field(
        default_factory=lambda: ["Steel_Type"]
    )

    target: str = "Final_Hardness_HRC"

    physical_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {

            "Initial_Hardness_HRC": (0, 70),
            "Tempering_Time_s": (0, 1e6),
            "Tempering_Temperature_C": (0, 1200),

            "C": (0, 3),
            "Mn": (0, 20),
            "P": (0, 1),
            "S": (0, 1),

            "Si": (0, 5),
            "Ni": (0, 30),
            "Cr": (0, 30),
            "Mo": (0, 10),
            "V": (0, 5),
            "Al": (0, 5),
            "Cu": (0, 10),

            "Final_Hardness_HRC": (0, 70)
        }
    )


# =====================================================
# 4. DATA LOADING
# =====================================================

def load_data(cfg: Config) -> pd.DataFrame:

    file_path = Path(cfg.base_path) / cfg.input_file

    if not file_path.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {file_path}"
        )

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()

    log.info(
        "Veri yüklendi: %d satır, %d sütun",
        df.shape[0],
        df.shape[1]
    )

    rename_columns = {

        "Steel type": "Steel_Type",

        "Initial hardness (HRC) - post quenching":
            "Initial_Hardness_HRC",

        "Tempering time (s)": "Tempering_Time_s",

        "C (%wt)": "C",
        "Mn (%wt)": "Mn",
        "P (%wt)": "P",
        "S (%wt)": "S",
        "Si (%wt)": "Si",
        "Ni (%wt)": "Ni",
        "Cr (%wt)": "Cr",
        "Mo (%wt)": "Mo",
        "V (%wt)": "V",
        "Al (%wt)": "Al",
        "Cu (%wt)": "Cu",

        "Final hardness (HRC) - post tempering":
            "Final_Hardness_HRC"
    }

    df.rename(columns=rename_columns, inplace=True)

    # Sıcaklık sütununu bul
    for col in df.columns:

        if "Tempering temperature" in col:
            df.rename(
                columns={col: "Tempering_Temperature_C"},
                inplace=True
            )

    return df


# =====================================================
# 5. DATA VALIDATION AND CLEANING
# =====================================================

def clean_and_validate(
    df: pd.DataFrame,
    cfg: Config
) -> pd.DataFrame:

    required_columns = (
        cfg.numeric_features
        + cfg.categorical_features
        + [cfg.target]
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"CSV dosyasında eksik sütunlar var: {missing_columns}"
        )

    # Duplicate temizliği
    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    log.info(
        "Duplicate temizliği: %d -> %d",
        before,
        len(df)
    )

    # Sayısal dönüşüm
    for col in cfg.numeric_features + [cfg.target]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Geçersiz fiziksel değerler
    for col, (low, high) in cfg.physical_ranges.items():

        mask = (
            (df[col] < low) |
            (df[col] > high)
        )

        if mask.any():

            log.warning(
                "%s sütununda %d geçersiz değer bulundu.",
                col,
                mask.sum()
            )

            df.loc[mask, col] = np.nan

    # Hedef değişkeni eksik satırları sil
    df = df.dropna(
        subset=[cfg.target]
    ).reset_index(drop=True)

    # Boş çelik türlerini eksik olarak işaretle
    df[cfg.categorical_features] = (
        df[cfg.categorical_features].replace("", np.nan)
    )

    log.info(
        "Temizlik sonrası veri: %d satır",
        len(df)
    )

    if len(df) < cfg.cv_folds:
        raise ValueError(
            "Veri sayısı Cross Validation için yetersiz."
        )

    return df


# =====================================================
# 6. PREPROCESSOR
# =====================================================

def build_preprocessor(cfg: Config):

    numeric_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )

    ])

    categorical_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )

    ])

    preprocessor = ColumnTransformer([

        (
            "num",
            numeric_pipeline,
            cfg.numeric_features
        ),

        (
            "cat",
            categorical_pipeline,
            cfg.categorical_features
        )

    ])

    return preprocessor


# =====================================================
# 7. MODEL CREATION
# =====================================================

def build_models(cfg: Config):

    models = {

        "Linear Regression": Pipeline([

            ("pre", build_preprocessor(cfg)),

            ("model", LinearRegression())

        ]),

        "SVR": Pipeline([

            ("pre", build_preprocessor(cfg)),

            ("model", SVR(
                kernel="rbf",
                C=100,
                gamma="scale"
            ))

        ]),

        "Decision Tree": Pipeline([

            ("pre", build_preprocessor(cfg)),

            ("model", DecisionTreeRegressor(
                max_depth=10,
                min_samples_leaf=5,
                random_state=cfg.random_state
            ))

        ]),

        "Random Forest": Pipeline([

            ("pre", build_preprocessor(cfg)),

            ("model", RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=4,
                random_state=cfg.random_state,
                n_jobs=-1
            ))

        ]),

        "Gradient Boosting": Pipeline([

            ("pre", build_preprocessor(cfg)),

            ("model", GradientBoostingRegressor(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                random_state=cfg.random_state
            ))

        ])

    }

    if HAS_XGB:

        models["XGBoost"] = Pipeline([

            ("pre", build_preprocessor(cfg)),

            ("model", XGBRegressor(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=cfg.random_state,
                n_jobs=-1,
                verbosity=0
            ))

        ])

    return models


# =====================================================
# 8. MODEL TRAINING AND EVALUATION
# =====================================================

def train_and_evaluate(
    models,
    X_train,
    X_test,
    y_train,
    y_test,
    cfg
):

    kf = KFold(
        n_splits=cfg.cv_folds,
        shuffle=True,
        random_state=cfg.random_state
    )

    results = []

    for name, model in models.items():

        log.info("Model eğitiliyor: %s", name)

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)

        test_pred = model.predict(X_test)

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring="r2",
            n_jobs=1
        )

        results.append({

            "Model": name,

            "Train MAE": mean_absolute_error(
                y_train, train_pred
            ),

            "Test MAE": mean_absolute_error(
                y_test, test_pred
            ),

            "Train R2": r2_score(
                y_train, train_pred
            ),

            "Test R2": r2_score(
                y_test, test_pred
            ),

            "CV R2 Mean": cv_scores.mean(),

            "CV R2 Std": cv_scores.std()

        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="CV R2 Mean",
        ascending=False
    ).reset_index(drop=True)

    return results_df


# =====================================================
# 9. RANDOM FOREST HYPERPARAMETER OPTIMIZATION
# =====================================================

def tune_random_forest(
    X_train,
    y_train,
    cfg
):

    pipeline = Pipeline([

        ("pre", build_preprocessor(cfg)),

        ("model", RandomForestRegressor(
            random_state=cfg.random_state,
            n_jobs=-1
        ))

    ])

    param_grid = {

        "model__n_estimators": [300, 500],

        "model__max_depth": [8, 12, None],

        "model__min_samples_leaf": [2, 4, 6]

    }

    cv = KFold(
        n_splits=cfg.cv_folds,
        shuffle=True,
        random_state=cfg.random_state
    )

    grid_search = GridSearchCV(

        estimator=pipeline,

        param_grid=param_grid,

        cv=cv,

        scoring="r2",

        n_jobs=1,

        verbose=1

    )

    log.info("GridSearchCV başlatılıyor...")

    grid_search.fit(X_train, y_train)

    log.info(
        "En iyi parametreler: %s",
        grid_search.best_params_
    )

    log.info(
        "En iyi CV R2: %.4f",
        grid_search.best_score_
    )

    return grid_search.best_estimator_


# =====================================================
# 10. VISUALIZATION
# =====================================================

def save_plots(
    model,
    X_test,
    y_test,
    y_pred,
    output_dir
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Actual vs Predicted
    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.6,
        edgecolor="k"
    )

    minimum = min(y_test.min(), y_pred.min())

    maximum = max(y_test.max(), y_pred.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--"
    )

    plt.xlabel("Gerçek Sertlik (HRC)")

    plt.ylabel("Tahmin Edilen Sertlik (HRC)")

    plt.title("Actual vs Predicted")

    plt.tight_layout()

    plt.savefig(
        output_dir / "actual_vs_predicted.png"
    )

    plt.close()

    # Residual Plot
    residuals = y_test - y_pred

    plt.figure(figsize=(7, 5))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.6,
        edgecolor="k"
    )

    plt.axhline(
        0,
        color="red",
        linestyle="--"
    )

    plt.xlabel("Tahmin Edilen HRC")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig(
        output_dir / "residuals.png"
    )

    plt.close()

    # Feature Importance
    try:

        estimator = model.named_steps["model"]

        if hasattr(estimator, "feature_importances_"):

            feature_names = (
                model.named_steps["pre"]
                .get_feature_names_out()
            )

            importance = pd.Series(
                estimator.feature_importances_,
                index=feature_names
            )

            importance = importance.sort_values(
                ascending=False
            ).head(20)

            plt.figure(figsize=(9, 7))

            importance.sort_values().plot(
                kind="barh"
            )

            plt.xlabel("Importance")

            plt.title("Feature Importance - Top 20")

            plt.tight_layout()

            plt.savefig(
                output_dir / "feature_importance.png"
            )

            plt.close()

    except Exception as error:

        log.warning(
            "Feature importance oluşturulamadı: %s",
            error
        )

    log.info("Grafikler kaydedildi.")


# =====================================================
# 11. SINGLE PREDICTION
# =====================================================

def predict_single(model, cfg):

    print("\n===================================")
    print(" YENİ ISIL İŞLEM SERTLİK TAHMİNİ")
    print("===================================")

    steel_type = input(
        "Çelik türü (Örn: 4140): "
    ).strip()

    if not steel_type:

        raise ValueError(
            "Çelik türü boş bırakılamaz."
        )

    user_input = {

        "Steel_Type": steel_type

    }

    for feature in cfg.numeric_features:

        low, high = cfg.physical_ranges[feature]

        while True:

            try:

                value = float(
                    input(
                        f"{feature} ({low} - {high}): "
                    )
                )

                if not low <= value <= high:

                    print(
                        "Geçersiz aralık. Tekrar deneyin."
                    )

                    continue

                user_input[feature] = value

                break

            except ValueError:

                print(
                    "Lütfen sayısal bir değer girin."
                )

    X_new = pd.DataFrame([user_input])

    prediction = model.predict(X_new)[0]

    print("\n===================================")

    print(
        f"Tahmini Nihai Sertlik: {prediction:.2f} HRC"
    )

    print("===================================")

    # Random Forest belirsizlik tahmini
    estimator = model.named_steps["model"]

    if isinstance(estimator, RandomForestRegressor):

        transformed = (
            model.named_steps["pre"]
            .transform(X_new)
        )

        tree_predictions = np.array([

            tree.predict(transformed)[0]

            for tree in estimator.estimators_

        ])

        lower = np.percentile(
            tree_predictions, 2.5
        )

        upper = np.percentile(
            tree_predictions, 97.5
        )

        print(
            f"Ağaç tahminlerinin %95 aralığı: "
            f"{lower:.2f} - {upper:.2f} HRC"
        )


# =====================================================
# 12. BATCH PREDICTION
# =====================================================

def predict_batch(
    model,
    cfg,
    input_csv
):

    input_path = Path(input_csv)

    if not input_path.exists():

        raise FileNotFoundError(
            f"Batch CSV bulunamadı: {input_path}"
        )

    df = pd.read_csv(input_path)

    df.columns = df.columns.str.strip()

    # Eğitim verisindeki sütun adlarıyla eşleştir
    rename_columns = {

        "Steel type": "Steel_Type",

        "Initial hardness (HRC) - post quenching":
            "Initial_Hardness_HRC",

        "Tempering time (s)": "Tempering_Time_s",

        "C (%wt)": "C",
        "Mn (%wt)": "Mn",
        "P (%wt)": "P",
        "S (%wt)": "S",
        "Si (%wt)": "Si",
        "Ni (%wt)": "Ni",
        "Cr (%wt)": "Cr",
        "Mo (%wt)": "Mo",
        "V (%wt)": "V",
        "Al (%wt)": "Al",
        "Cu (%wt)": "Cu"
    }

    df.rename(
        columns=rename_columns,
        inplace=True
    )

    for col in df.columns:

        if "Tempering temperature" in col:

            df.rename(
                columns={col: "Tempering_Temperature_C"},
                inplace=True
            )

    required = (
        cfg.numeric_features
        + cfg.categorical_features
    )

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Batch CSV eksik sütunlar içeriyor: {missing}"
        )

    X_batch = df[required]

    predictions = model.predict(X_batch)

    df["Predicted_Final_Hardness_HRC"] = predictions

    output_dir = Path(cfg.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / "batch_predictions.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    log.info(
        "Batch tahminleri kaydedildi: %s",
        output_path
    )


# =====================================================
# 13. MAIN FUNCTION
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description="Tempering Hardness Prediction"
    )

    parser.add_argument(
        "--batch-csv",
        type=str,
        default=None,
        help="Toplu tahmin CSV dosyası"
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Random Forest GridSearchCV"
    )

    args = parser.parse_args()

    cfg = Config()

    output_dir = Path(cfg.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 1. Load Data
    log.info("Veri yükleniyor...")

    df = load_data(cfg)

    # 2. Clean Data
    log.info("Veri temizleniyor...")

    df = clean_and_validate(df, cfg)

    # 3. Features and Target
    X = df[
        cfg.numeric_features +
        cfg.categorical_features
    ]

    y = df[cfg.target]

    # 4. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state

    )

    log.info(
        "Train: %d | Test: %d",
        len(X_train),
        len(X_test)
    )

    # 5. Model Creation
    models = build_models(cfg)

    # 6. Model Comparison
    results = train_and_evaluate(

        models,
        X_train,
        X_test,
        y_train,
        y_test,
        cfg

    )

    print("\n========== MODEL COMPARISON ==========\n")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results.to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    # 7. Final Model
    log.info("Final model hazırlanıyor...")

    if args.tune:

        final_model = tune_random_forest(
            X_train,
            y_train,
            cfg
        )

    else:

        final_model = models["Random Forest"]

        final_model.fit(
            X_train,
            y_train
        )

    # 8. Final Evaluation
    y_pred = final_model.predict(X_test)

    test_mae = mean_absolute_error(
        y_test,
        y_pred
    )

    test_r2 = r2_score(
        y_test,
        y_pred
    )

    print("\n========== FINAL MODEL ==========\n")

    print(f"Test MAE: {test_mae:.4f}")

    print(f"Test R2: {test_r2:.4f}")

    # 9. Save Plots
    save_plots(
        final_model,
        X_test,
        y_test,
        y_pred,
        cfg.output_dir
    )

    # 10. Save Model
    model_path = Path(cfg.model_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        final_model,
        model_path
    )

    log.info(
        "Model kaydedildi: %s",
        model_path
    )

    # 11. Prediction
    if args.batch_csv:

        predict_batch(
            final_model,
            cfg,
            args.batch_csv
        )

    else:

        predict_single(
            final_model,
            cfg
        )


# =====================================================
# 14. PROGRAM ENTRY POINT
# =====================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        log.exception(
            "Program çalışırken hata oluştu: %s",
            error
        )

        raise
