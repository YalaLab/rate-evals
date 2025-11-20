#!/usr/bin/env python3

import pandas as pd
import glob
import os
import json
from pathlib import Path


def analyze_auc_results():
    csv_files = glob.glob("results/*/detailed_results.csv")

    if not csv_files:
        print("No detailed_results.csv files found in results/ subdirectories")
        return

    results = []

    for csv_file in csv_files:
        model_dir = Path(csv_file).parent.name

        try:
            df = pd.read_csv(csv_file)
            avg_auc_calculated = df["auc"].mean()

            # Calculate AUC excluding rows where num_positive is 0
            df_filtered = df[df["num_positive"] > 0]
            avg_auc_filtered = df_filtered["auc"].mean() if len(df_filtered) > 0 else 0.0

            summary_stats_file = Path(csv_file).parent / "summary_stats.json"
            avg_auc_summary = None

            if summary_stats_file.exists():
                try:
                    with open(summary_stats_file, "r") as f:
                        summary_data = json.load(f)
                        avg_auc_summary = summary_data.get("avg_auc")
                except Exception as e:
                    print(f"Error reading {summary_stats_file}: {e}")

            # Assert that calculated and summary AUC values match (within small tolerance)
            if avg_auc_summary is not None:
                diff = abs(avg_auc_calculated - avg_auc_summary)
                assert (
                    diff < 1e-6
                ), f"AUC mismatch for {model_dir}: calculated={avg_auc_calculated:.6f}, summary={avg_auc_summary:.6f}, diff={diff:.6f}"

            results.append(
                {
                    "Model": model_dir,
                    "Average AUC (calculated)": avg_auc_calculated,
                    "Average AUC (filtered)": avg_auc_filtered,
                    "Average AUC (summary)": avg_auc_summary,
                    "Number of Questions": len(df),
                    "Questions with positives": len(df_filtered),
                }
            )

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("Average AUC (calculated)", ascending=False)

        print("\nAverage AUC Results by Model:")
        print("=" * 80)
        print(results_df.to_string(index=False, float_format="%.4f"))

        print(f"\n\nSummary:")
        print(f"Total models analyzed: {len(results_df)}")
        print(
            f"Best performing model: {results_df.iloc[0]['Model']} (AUC: {results_df.iloc[0]['Average AUC (calculated)']:.4f})"
        )
        print(
            f"Worst performing model: {results_df.iloc[-1]['Model']} (AUC: {results_df.iloc[-1]['Average AUC (calculated)']:.4f})"
        )

    else:
        print("No valid results found")


if __name__ == "__main__":
    analyze_auc_results()
