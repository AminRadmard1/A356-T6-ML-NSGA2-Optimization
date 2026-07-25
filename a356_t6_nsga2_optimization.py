"""Multi-objective T6 heat-treatment optimization for A356 aluminum alloy.

Install dependencies:
    python -m pip install pandas numpy scikit-learn pymoo matplotlib

Important data note
-------------------
The embedded 30-row dataset is a small, representative demonstration dataset
constructed to reproduce realistic A356-T6 trends. It is suitable for testing
this workflow, but it is not a substitute for traceable specimen-level data
extracted from peer-reviewed papers. Replace ``build_demo_dataset`` with your
validated experimental database before making engineering decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; registers 3D projection
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor


RANDOM_SEED: Final[int] = 42
FEATURE_COLUMNS: Final[list[str]] = [
    "Solution_Temp_C",
    "Solution_Time_h",
    "Aging_Time_h",
]
TARGET_COLUMNS: Final[list[str]] = ["UTS_MPa", "Elongation_pct"]


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for surrogate training and NSGA-II optimization."""

    random_seed: int = RANDOM_SEED
    cv_folds: int = 5
    forest_trees: int = 600
    population_size: int = 200
    generations: int = 300
    output_directory: Path = Path("a356_nsga2_results")
    show_plots: bool = True


class A356T6OptimizationProblem(Problem):
    """Three-objective A356-T6 process optimization problem.

    Decision variables
    ------------------
    x[0] : solution-treatment temperature, 530-545 degC
    x[1] : solution-treatment time, 4-12 h
    x[2] : artificial-aging time, 3-9 h

    Objectives used by pymoo are all minimized:
    f1 = -predicted UTS
    f2 = -predicted elongation
    f3 = total furnace time = solution time + aging time
    """

    def __init__(self, surrogate: MultiOutputRegressor) -> None:
        self.surrogate = surrogate
        super().__init__(
            n_var=3,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.array([530.0, 4.0, 3.0], dtype=float),
            xu=np.array([545.0, 12.0, 9.0], dtype=float),
        )

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, np.ndarray],
        *args: object,
        **kwargs: object,
    ) -> None:
        input_frame = pd.DataFrame(x, columns=FEATURE_COLUMNS)
        predictions = self.surrogate.predict(input_frame)

        predicted_uts = predictions[:, 0]
        predicted_elongation = predictions[:, 1]
        total_furnace_time = x[:, 1] + x[:, 2]

        out["F"] = np.column_stack(
            (-predicted_uts, -predicted_elongation, total_furnace_time)
        )


def build_demo_dataset() -> pd.DataFrame:
    """Return a 30-point demonstration dataset for A356-T6.

    Artificial-aging temperature is assumed to be held within 155-160 degC and
    is therefore not treated as a decision variable. The table intentionally
    includes under-aged, near-peak-aged, over-aged, and excessive-thermal-
    exposure conditions.
    """

    records = [
        # Solution temp, solution time, aging time, UTS, elongation
        (530, 4, 3, 259, 6.9),
        (530, 4, 6, 278, 5.7),
        (530, 6, 5, 284, 6.8),
        (530, 6, 7, 281, 7.0),
        (530, 8, 6, 282, 7.0),
        (530, 10, 9, 257, 8.1),
        (530, 12, 5, 262, 7.5),
        (535, 4, 3, 269, 7.2),
        (535, 4, 6, 289, 6.5),
        (535, 6, 5, 292, 7.1),
        (535, 6, 7, 292, 7.0),
        (535, 8, 6, 294, 7.2),
        (535, 8, 9, 272, 8.2),
        (535, 10, 5, 282, 8.0),
        (535, 12, 7, 268, 7.8),
        (540, 4, 3, 274, 7.5),
        (540, 4, 6, 292, 6.2),
        (540, 6, 5, 292, 7.2),
        (540, 6, 6, 297, 6.9),
        (540, 6, 7, 292, 7.5),
        (540, 8, 5, 292, 7.8),
        (540, 8, 7, 292, 8.2),
        (540, 10, 6, 286, 7.7),
        (540, 12, 9, 257, 8.8),
        (545, 4, 3, 272, 7.7),
        (545, 4, 6, 291, 6.6),
        (545, 6, 5, 291, 7.8),
        (545, 8, 6, 290, 7.8),
        (545, 10, 7, 275, 7.8),
        (545, 12, 9, 247, 7.7),
    ]

    frame = pd.DataFrame(
        records,
        columns=[
            "Solution_Temp_C",
            "Solution_Time_h",
            "Aging_Time_h",
            "UTS_MPa",
            "Elongation_pct",
        ],
    )
    frame["Total_Furnace_Time_h"] = (
        frame["Solution_Time_h"] + frame["Aging_Time_h"]
    )
    return frame


def validate_dataset(data: pd.DataFrame) -> None:
    """Validate columns, missing values, ranges, and furnace-time arithmetic."""

    required = FEATURE_COLUMNS + TARGET_COLUMNS + ["Total_Furnace_Time_h"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if data[required].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")

    bounds = {
        "Solution_Temp_C": (530.0, 545.0),
        "Solution_Time_h": (4.0, 12.0),
        "Aging_Time_h": (3.0, 9.0),
    }
    for column, (lower, upper) in bounds.items():
        if not data[column].between(lower, upper).all():
            raise ValueError(
                f"Values in {column} must remain inside [{lower}, {upper}]."
            )

    expected_time = data["Solution_Time_h"] + data["Aging_Time_h"]
    if not np.allclose(expected_time, data["Total_Furnace_Time_h"]):
        raise ValueError("Total furnace time is inconsistent with process times.")


def build_surrogate(config: OptimizationConfig) -> MultiOutputRegressor:
    """Construct the multi-output random-forest surrogate."""

    base_estimator = RandomForestRegressor(
        n_estimators=config.forest_trees,
        max_depth=None,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        random_state=config.random_seed,
        n_jobs=1,
    )
    return MultiOutputRegressor(base_estimator, n_jobs=-1)


def train_and_evaluate_surrogate(
    data: pd.DataFrame,
    config: OptimizationConfig,
) -> tuple[MultiOutputRegressor, pd.DataFrame, pd.DataFrame]:
    """Evaluate with shuffled K-fold OOF predictions, then refit on all data."""

    features = data[FEATURE_COLUMNS]
    targets = data[TARGET_COLUMNS]

    cross_validator = KFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_seed,
    )

    evaluation_model = build_surrogate(config)
    out_of_fold_predictions = cross_val_predict(
        evaluation_model,
        features,
        targets,
        cv=cross_validator,
        n_jobs=1,
        method="predict",
    )

    r2_values = r2_score(
        targets,
        out_of_fold_predictions,
        multioutput="raw_values",
    )
    rmse_values = np.sqrt(
        mean_squared_error(
            targets,
            out_of_fold_predictions,
            multioutput="raw_values",
        )
    )

    metrics = pd.DataFrame(
        {
            "Output": TARGET_COLUMNS,
            "OOF_R2": r2_values,
            "OOF_RMSE": rmse_values,
        }
    )

    predictions = pd.DataFrame(
        out_of_fold_predictions,
        columns=[f"Predicted_{column}" for column in TARGET_COLUMNS],
        index=data.index,
    )
    evaluation_results = pd.concat(
        [data[FEATURE_COLUMNS + TARGET_COLUMNS], predictions], axis=1
    )

    final_model = build_surrogate(config)
    final_model.fit(features, targets)
    return final_model, metrics, evaluation_results


def run_nsga2(
    surrogate: MultiOutputRegressor,
    config: OptimizationConfig,
) -> pd.DataFrame:
    """Run NSGA-II and return the non-dominated surrogate predictions."""

    problem = A356T6OptimizationProblem(surrogate)
    algorithm = NSGA2(
        pop_size=config.population_size,
        eliminate_duplicates=True,
    )

    result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=("n_gen", config.generations),
        seed=config.random_seed,
        save_history=False,
        verbose=True,
    )

    if result.X is None or result.F is None:
        raise RuntimeError("NSGA-II did not return a Pareto-optimal solution set.")

    pareto = pd.DataFrame(
        {
            "Solution_Temp_C": result.X[:, 0],
            "Solution_Time_h": result.X[:, 1],
            "Aging_Time_h": result.X[:, 2],
            "Predicted_UTS_MPa": -result.F[:, 0],
            "Predicted_Elongation_pct": -result.F[:, 1],
            "Total_Furnace_Time_h": result.F[:, 2],
        }
    )

    # Tree models produce piecewise-constant predictions. Rounding and removing
    # duplicates keeps the reported front compact without changing its meaning.
    pareto = pareto.round(
        {
            "Solution_Temp_C": 3,
            "Solution_Time_h": 3,
            "Aging_Time_h": 3,
            "Predicted_UTS_MPa": 3,
            "Predicted_Elongation_pct": 3,
            "Total_Furnace_Time_h": 3,
        }
    )
    pareto = pareto.drop_duplicates().reset_index(drop=True)
    pareto = pareto.sort_values(
        ["Total_Furnace_Time_h", "Predicted_UTS_MPa"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return pareto


def minmax(values: pd.Series) -> pd.Series:
    """Scale a series to [0, 1], safely handling constant values."""

    span = values.max() - values.min()
    if np.isclose(span, 0.0):
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return (values - values.min()) / span


def select_representative_windows(pareto: pd.DataFrame) -> pd.DataFrame:
    """Select max-strength, max-ductility, and balanced energy-saving windows."""

    if pareto.empty:
        raise ValueError("The Pareto table is empty.")

    working = pareto.copy()
    selected_indices: list[int] = []
    selected_labels: list[str] = []

    strength_index = int(working["Predicted_UTS_MPa"].idxmax())
    selected_indices.append(strength_index)
    selected_labels.append("Maximum Strength")

    remaining = working.drop(index=selected_indices, errors="ignore")
    if remaining.empty:
        ductility_index = strength_index
    else:
        ductility_index = int(remaining["Predicted_Elongation_pct"].idxmax())
    selected_indices.append(ductility_index)
    selected_labels.append("Maximum Ductility")

    strength_score = minmax(working["Predicted_UTS_MPa"])
    ductility_score = minmax(working["Predicted_Elongation_pct"])
    time_saving_score = 1.0 - minmax(working["Total_Furnace_Time_h"])

    # Weights can be modified to reflect plant economics or product priorities.
    working["Balanced_Score"] = (
        0.40 * strength_score
        + 0.35 * ductility_score
        + 0.25 * time_saving_score
    )

    remaining = working.drop(index=selected_indices, errors="ignore")
    if remaining.empty:
        balanced_index = int(working["Balanced_Score"].idxmax())
    else:
        balanced_index = int(remaining["Balanced_Score"].idxmax())
    selected_indices.append(balanced_index)
    selected_labels.append("Balanced Energy-Saving")

    selected = working.loc[selected_indices].copy()
    selected.insert(0, "Process_Window", selected_labels)
    selected = selected.reset_index(drop=True)

    display_columns = [
        "Process_Window",
        "Solution_Temp_C",
        "Solution_Time_h",
        "Aging_Time_h",
        "Predicted_UTS_MPa",
        "Predicted_Elongation_pct",
        "Total_Furnace_Time_h",
        "Balanced_Score",
    ]
    return selected[display_columns]


def plot_pareto_2d(
    pareto: pd.DataFrame,
    output_directory: Path,
    show: bool,
) -> None:
    """Plot UTS versus elongation, with furnace time encoded continuously."""

    figure = plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        pareto["Predicted_UTS_MPa"],
        pareto["Predicted_Elongation_pct"],
        c=pareto["Total_Furnace_Time_h"],
        s=50,
        alpha=0.85,
    )
    plt.xlabel("Predicted UTS (MPa)")
    plt.ylabel("Predicted Elongation (%)")
    plt.title("A356-T6 NSGA-II Pareto Front: Strength vs. Ductility")
    plt.grid(True, alpha=0.3)
    colorbar = figure.colorbar(scatter)
    colorbar.set_label("Total Furnace Time (h)")
    figure.tight_layout()
    figure.savefig(output_directory / "pareto_front_2d.png", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_pareto_3d(
    pareto: pd.DataFrame,
    output_directory: Path,
    show: bool,
) -> None:
    """Plot the complete three-objective Pareto front."""

    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        pareto["Predicted_UTS_MPa"],
        pareto["Predicted_Elongation_pct"],
        pareto["Total_Furnace_Time_h"],
        c=pareto["Solution_Temp_C"],
        s=45,
        alpha=0.85,
    )
    axis.set_xlabel("Predicted UTS (MPa)")
    axis.set_ylabel("Predicted Elongation (%)")
    axis.set_zlabel("Total Furnace Time (h)")
    axis.set_title("A356-T6 Three-Objective NSGA-II Pareto Front")
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.12)
    colorbar.set_label("Solution Temperature (°C)")
    figure.tight_layout()
    figure.savefig(output_directory / "pareto_front_3d.png", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(figure)


def save_results(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    evaluation_results: pd.DataFrame,
    pareto: pd.DataFrame,
    selected_windows: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Persist all model, evaluation, and optimization tables as CSV files."""

    output_directory.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_directory / "a356_demo_dataset.csv", index=False)
    metrics.to_csv(output_directory / "surrogate_metrics.csv", index=False)
    evaluation_results.to_csv(
        output_directory / "out_of_fold_predictions.csv", index=False
    )
    pareto.to_csv(output_directory / "pareto_front.csv", index=False)
    selected_windows.to_csv(
        output_directory / "top_process_windows.csv", index=False
    )


def print_summary(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    pareto: pd.DataFrame,
    selected_windows: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Print a concise terminal report."""

    print("\n" + "=" * 78)
    print("A356-T6 MULTI-OBJECTIVE PROCESS OPTIMIZATION")
    print("=" * 78)
    print(f"Dataset rows: {len(data)}")
    print(f"Non-dominated solutions retained: {len(pareto)}")

    print("\nOut-of-fold surrogate metrics:")
    print(
        metrics.to_string(
            index=False,
            formatters={
                "OOF_R2": "{:.4f}".format,
                "OOF_RMSE": "{:.4f}".format,
            },
        )
    )

    print("\nRepresentative optimal process windows:")
    print(
        selected_windows.to_string(
            index=False,
            formatters={
                "Solution_Temp_C": "{:.2f}".format,
                "Solution_Time_h": "{:.2f}".format,
                "Aging_Time_h": "{:.2f}".format,
                "Predicted_UTS_MPa": "{:.2f}".format,
                "Predicted_Elongation_pct": "{:.2f}".format,
                "Total_Furnace_Time_h": "{:.2f}".format,
                "Balanced_Score": "{:.3f}".format,
            },
        )
    )
    print(f"\nCSV tables and figures saved to: {output_directory.resolve()}")
    print("=" * 78)


def main() -> None:
    """Execute the complete data-to-Pareto workflow."""

    config = OptimizationConfig()
    config.output_directory.mkdir(parents=True, exist_ok=True)

    data = build_demo_dataset()
    validate_dataset(data)

    surrogate, metrics, evaluation_results = train_and_evaluate_surrogate(
        data, config
    )
    pareto = run_nsga2(surrogate, config)
    selected_windows = select_representative_windows(pareto)

    save_results(
        data=data,
        metrics=metrics,
        evaluation_results=evaluation_results,
        pareto=pareto,
        selected_windows=selected_windows,
        output_directory=config.output_directory,
    )
    plot_pareto_2d(pareto, config.output_directory, config.show_plots)
    plot_pareto_3d(pareto, config.output_directory, config.show_plots)
    print_summary(
        data=data,
        metrics=metrics,
        pareto=pareto,
        selected_windows=selected_windows,
        output_directory=config.output_directory,
    )


if __name__ == "__main__":
    main()
