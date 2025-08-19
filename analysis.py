import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_dataset(csv_path: str = "Tesla.csv") -> pd.DataFrame:
	if os.path.exists(csv_path):
		print(f"Loading dataset from {csv_path}")
		df = pd.read_csv(csv_path)
	else:
		print("Tesla.csv not found. Downloading TSLA data via yfinance...")
		try:
			import yfinance as yf
		except ImportError:
			raise SystemExit("yfinance is required to fetch data when Tesla.csv is missing. Please install dependencies.")
		# Download max history
		yf_df = yf.Ticker("TSLA").history(period="max").reset_index()
		# Standardize to expected columns: Date, Close/Last, Volume
		yf_df = yf_df.rename(columns={"Date": "Date", "Close": "Close/Last", "Volume": "Volume"})
		yf_df["Date"] = pd.to_datetime(yf_df["Date"]) 
		# Save a local CSV for reproducibility
		yf_df.to_csv(csv_path, index=False)
		print(f"Downloaded and saved to {csv_path}")
		df = yf_df
	return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
	# Ensure expected columns exist, try to map common alternatives
	columns = {c.lower(): c for c in df.columns}
	# Map price column
	price_col = None
	for candidate in ["close/last", "close", "adj close", "adj_close", "adj. close"]:
		if candidate in columns:
			price_col = columns[candidate]
			break
	if price_col is None:
		raise ValueError("Could not find a closing price column in the dataset.")
	if "Close/Last" not in df.columns:
		df = df.rename(columns={price_col: "Close/Last"})

	# Map date column
	date_col = None
	for candidate in ["date", "datetime"]:
		if candidate in columns:
			date_col = columns[candidate]
			break
	if date_col is None:
		raise ValueError("Could not find a date column in the dataset.")
	if "Date" not in df.columns:
		df = df.rename(columns={date_col: "Date"})

	# Map volume column if exists
	if "Volume" not in df.columns:
		for candidate in ["volume", "vol"]:
			if candidate in columns:
				df = df.rename(columns={columns[candidate]: "Volume"})
				break

	# Clean Close/Last: remove $ and cast to float if needed
	if df["Close/Last"].dtype == object:
		df["Close/Last"] = df["Close/Last"].astype(str).str.replace("$", "", regex=True)
		df["Close/Last"] = pd.to_numeric(df["Close/Last"], errors="coerce")

	# Coerce Date to datetime
	df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

	# Drop rows with missing essential fields
	df = df.dropna(subset=["Date", "Close/Last"]).copy()

	# Sort by date
	df = df.sort_values("Date").reset_index(drop=True)
	return df


def ensure_plots_dir(path: str = "plots") -> str:
	os.makedirs(path, exist_ok=True)
	return path


def plot_price_over_time(df: pd.DataFrame, out_dir: str):
	plt.figure(figsize=(10, 5))
	plt.plot(df["Date"], df["Close/Last"], label="Tesla Closing Price")
	plt.xlabel("Date")
	plt.ylabel("Price (USD)")
	plt.title("Tesla Stock Closing Price Over Time")
	plt.legend()
	out_path = os.path.join(out_dir, "price_over_time.png")
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()
	print(f"Saved plot: {out_path}")


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
	df["Daily Return"] = df["Close/Last"].pct_change() * 100
	return df


def plot_daily_returns_hist(df: pd.DataFrame, out_dir: str):
	plt.figure(figsize=(8, 5))
	plt.hist(df["Daily Return"].dropna(), bins=50, edgecolor="black")
	plt.xlabel("Daily Return (%)")
	plt.ylabel("Frequency (How Often it Occurred)")
	plt.title("Tesla Daily Returns Distribution")
	out_path = os.path.join(out_dir, "daily_returns_hist.png")
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()
	print(f"Saved plot: {out_path}")


def add_cumulative_return(df: pd.DataFrame) -> pd.DataFrame:
	df["Cumulative Return"] = (1 + df["Daily Return"]/100).cumprod()
	return df


def plot_cumulative_return(df: pd.DataFrame, out_dir: str):
	plt.figure(figsize=(10, 5))
	plt.plot(df["Date"], df["Cumulative Return"], label="Cumulative Return")
	plt.xlabel("Date")
	plt.ylabel("Growth of $1 Investment")
	plt.title("Tesla Cumulative Return Over Time")
	plt.legend()
	out_path = os.path.join(out_dir, "cumulative_return.png")
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()
	print(f"Saved plot: {out_path}")


def add_moving_average(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
	df[f"{window}d MA"] = df["Close/Last"].rolling(window=window).mean()
	return df


def plot_price_with_ma(df: pd.DataFrame, out_dir: str, window: int = 20):
	plt.figure(figsize=(10, 5))
	plt.plot(df["Date"], df["Close/Last"], label="Close Price")
	plt.plot(df["Date"], df[f"{window}d MA"], label=f"{window}-day Moving Average")
	plt.xlabel("Date")
	plt.ylabel("Price (USD)")
	plt.title(f"Tesla with {window}-Day Moving Average")
	plt.legend()
	out_path = os.path.join(out_dir, f"price_with_{window}d_MA.png")
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()
	print(f"Saved plot: {out_path}")


def compute_correlation_price_volume(df: pd.DataFrame) -> float:
	if "Volume" not in df.columns:
		print("Volume column not available; correlation cannot be computed.")
		return np.nan
	return df["Close/Last"].corr(df["Volume"])


def main():
	print("--- Step 0: Import required libraries ---")
	# Already imported above

	print("--- Step 1: Read the Tesla stock dataset from a CSV file ---")
	df = load_dataset("Tesla.csv")
	print("Columns available in the dataset:", df.columns.tolist())
	print("-" * 50)

	print("--- Step 2: Clean the data ---")
	df = clean_dataset(df)
	print("First 5 rows of the cleaned and sorted dataset:")
	print(df.head())
	print("-" * 50)

	print("--- Step 3: Explore the dataset ---")
	print("Shape of dataset (rows, columns):", df.shape)
	print("Columns:", df.columns.tolist())
	print("\nSummary statistics:")
	print(df.describe())
	print("\nDataFrame information:")
	# Capture df.info() output as string
	from io import StringIO
	buf = StringIO()
	df.info(buf=buf)
	print(buf.getvalue())
	print("-" * 50)

	plots_dir = ensure_plots_dir()

	print("--- Step 4: Plot stock closing prices over time ---")
	plot_price_over_time(df, plots_dir)
	print("-" * 50)

	print("--- Step 5: Calculate daily returns ---")
	df = add_daily_returns(df)
	print("First 5 rows showing calculated daily returns:")
	print(df[["Date", "Close/Last", "Daily Return"]].head())
	print("-" * 50)

	print("--- Step 6: Create a histogram of daily returns ---")
	plot_daily_returns_hist(df, plots_dir)
	print("-" * 50)

	print("--- Step 7: Calculate summary statistics of returns ---")
	mean_return = df["Daily Return"].mean()
	std_return = df["Daily Return"].std()
	print("Average Daily Return (%):", mean_return)
	print("Risk (Volatility = Std Dev) (%):", std_return)
	print("-" * 50)

	print("--- Step 8: Calculate cumulative returns ---")
	df = add_cumulative_return(df)
	plot_cumulative_return(df, plots_dir)
	print("-" * 50)

	print("--- Step 9: Calculate and plot a 20-day Moving Average ---")
	df = add_moving_average(df, window=20)
	plot_price_with_ma(df, plots_dir, window=20)
	print("-" * 50)

	print("--- Step 10: Check the correlation between price and volume ---")
	correlation = compute_correlation_price_volume(df)
	print("Correlation between Price and Volume:", correlation)

	# Save processed dataset
	out_csv = os.path.join("plots", "processed_tesla.csv")
	df.to_csv(out_csv, index=False)
	print(f"Saved processed dataset: {out_csv}")


if __name__ == "__main__":
	main()