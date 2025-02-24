import pandas as pd
import numpy as np
from datetime import datetime

def load_data(data_file='data.csv', payments_file='payments.csv', verbose=False):
    """
    Load and prepare the input data files for the MCA Portfolio Simulator.
    
    Args:
        data_file: Path to the data.csv file (contains deal information)
        payments_file: Path to the payments.csv file (contains payment transactions)
        verbose: Whether to print information about the loaded data
        
    Returns:
        Tuple of (deals_df, payments_df)
    """
    if verbose:
        print(f"Loading data from files: {data_file} and {payments_file}")
    
    # Load deals data (data.csv)
    deals_df = pd.read_csv(
        data_file,
        dtype={
            'Funded ID': str,
            'Product': str,
            'Commission Cost %': float,  # Add the new column
        },
        parse_dates=['Initial Funding Date'],
        thousands=','  # Handle numbers with commas
    )
    
    # Convert monetary columns to numeric
    monetary_cols = ['Total Original Balance', 'Total Original RTR', 
                     'Past Due Amount', 'Total Principal Net Charge Off']
    for col in monetary_cols:
        deals_df[col] = pd.to_numeric(deals_df[col], errors='coerce')
        
    # Convert Owner FICO to numeric
    deals_df['Owner FICO'] = pd.to_numeric(deals_df['Owner FICO'], errors='coerce')
    
    # Ensure Commission Cost % is numeric (though it should already be from dtype)
    deals_df['Commission Cost %'] = pd.to_numeric(deals_df['Commission Cost %'], errors='coerce')
    
    # Add Vintage-M column based on Initial Funding Date
    deals_df['Vintage-M'] = deals_df['Initial Funding Date'].dt.strftime('%Y-%m')
    
    # Load payments data (payments.csv)
    payments_df = pd.read_csv(
        payments_file,
        dtype={
            'Vintage': str,
            'Vintage-M': str,
            'Funded ID': str,
            'Transaction Description': str
        },
        parse_dates=['Funded Date', 'Transaction Date'],
        thousands=','  # Handle numbers with commas
    )
    
    # Convert Transaction Amount to numeric
    payments_df['Transaction Amount'] = pd.to_numeric(payments_df['Transaction Amount'], errors='coerce')
    
    # Store original Transaction Amount values for comparison
    payments_df['Original Transaction Amount'] = payments_df['Transaction Amount']
    
    # Recalculate initial cash outlay values based on Commission Cost %
    # Create a mapping of Funded ID to Commission Cost % and Total Original Balance
    commission_map = deals_df[['Funded ID', 'Commission Cost %', 'Total Original Balance']].set_index('Funded ID')
    
    # Add commission and balance columns to payments dataframe for calculation
    payments_df = payments_df.merge(
        commission_map, 
        on='Funded ID', 
        how='left'
    )
    
    # Apply the formula for initial cash outlay transactions
    # Formula: =-(1+'Commission Cost %')*('Total Original Balance')
    initial_outlay_mask = (payments_df['Transaction Description'] == 'Initial cash outlay')
    if initial_outlay_mask.any():
        # Calculate new transaction amounts for initial cash outlays
        payments_df.loc[initial_outlay_mask, 'Transaction Amount'] = -(
            (1 + payments_df.loc[initial_outlay_mask, 'Commission Cost %']) * 
            payments_df.loc[initial_outlay_mask, 'Total Original Balance']
        )
        
        if verbose:
            n_updated = initial_outlay_mask.sum()
            print(f"\nRecalculated {n_updated} initial cash outlay transactions based on Commission Cost %")
    
    # Drop the temporary columns used for calculation if we don't need them anymore
    # Keep them for now to enable difference analysis
    # payments_df = payments_df.drop(['Commission Cost %', 'Total Original Balance'], axis=1, errors='ignore')
    
    if verbose:
        print(f"Loaded {len(deals_df)} deals and {len(payments_df)} payment transactions")
        
        # Print summary of data types
        print("\nDeals DataFrame data types:")
        print(deals_df.dtypes)
        
        # Print summary of Commission Cost % column
        print("\nCommission Cost % statistics:")
        print(deals_df['Commission Cost %'].describe())
        
        print("\nPayments DataFrame data types:")
        print(payments_df.dtypes)
        
        # Print basic statistics for monetary columns
        print("\nDeal size statistics:")
        print(deals_df['Total Original Balance'].describe())
        
        # Print transaction type distribution
        print("\nTransaction type distribution:")
        print(payments_df['Transaction Description'].value_counts())
    
    return deals_df, payments_df


def test_data_integrity(deals_df, payments_df):
    """
    Run basic integrity checks on the loaded data.
    
    Args:
        deals_df: DataFrame containing deal information
        payments_df: DataFrame containing payment transactions
        
    Returns:
        bool: True if all critical checks pass, False otherwise
    """
    # Check for required columns in deals_df
    required_deal_cols = ['Funded ID', 'Product', 'Initial Funding Date', 
                          'Total Original Balance', 'Total Original RTR', 'Commission Cost %']
    
    for col in required_deal_cols:
        if col not in deals_df.columns:
            print(f"ERROR: Required column '{col}' missing from deals data")
            return False
    
    # Check for required columns in payments_df
    required_payment_cols = ['Funded ID', 'Transaction Date', 
                             'Transaction Amount', 'Transaction Description']
    
    for col in required_payment_cols:
        if col not in payments_df.columns:
            print(f"ERROR: Required column '{col}' missing from payments data")
            return False
    
    # Check for data type correctness
    if not pd.api.types.is_datetime64_dtype(deals_df['Initial Funding Date']):
        print("ERROR: 'Initial Funding Date' is not correctly formatted as datetime")
        return False
    
    if not pd.api.types.is_datetime64_dtype(payments_df['Transaction Date']):
        print("ERROR: 'Transaction Date' is not correctly formatted as datetime")
        return False
    
    # Check for missing values in critical columns
    critical_cols = {
        'deals': ['Funded ID', 'Total Original Balance', 'Total Original RTR', 'Commission Cost %'],
        'payments': ['Funded ID', 'Transaction Amount', 'Transaction Description']
    }
    
    for df_name, cols in critical_cols.items():
        df = deals_df if df_name == 'deals' else payments_df
        for col in cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                print(f"WARNING: {missing_count} missing values in {df_name} column '{col}'")
    
    # Check for data consistency between datasets
    deal_ids = set(deals_df['Funded ID'])
    payment_ids = set(payments_df['Funded ID'])
    common_ids = deal_ids.intersection(payment_ids)
    
    print(f"Found {len(common_ids)} deals with matching payment records")
    if len(common_ids) == 0:
        print("ERROR: No matching IDs between deals and payments data")
        return False
        
    # Check for missing values in Owner FICO
    missing_owner_fico = deals_df['Owner FICO'].isna().sum()
    if missing_owner_fico > 0:
        print(f"WARNING: {missing_owner_fico} missing values in Owner FICO")
    
    # Check for Commission Cost % values outside expected range (0-0.15)
    commission_range_issue = deals_df[(deals_df['Commission Cost %'] < 0) | 
                                    (deals_df['Commission Cost %'] > 0.20)].shape[0]
    if commission_range_issue > 0:
        print(f"WARNING: {commission_range_issue} records have Commission Cost % outside expected range (0-0.20)")
    
    # All critical checks passed
    return True


def analyze_outlay_difference(payments_df):
    """
    Analyze the difference between original and recalculated initial cash outlay values.
    
    Args:
        payments_df: DataFrame containing payment transactions with both original and recalculated values
        
    Returns:
        DataFrame with difference analysis
    """
    # Filter for initial cash outlay transactions
    outlay_df = payments_df[payments_df['Transaction Description'] == 'Initial cash outlay'].copy()
    
    if len(outlay_df) == 0:
        print("No initial cash outlay transactions found.")
        return None
    
    # Calculate difference between original and new values
    outlay_df['Difference'] = outlay_df['Transaction Amount'] - outlay_df['Original Transaction Amount']
    outlay_df['Difference %'] = (outlay_df['Difference'] / outlay_df['Original Transaction Amount'].abs()) * 100
    
    # Summary statistics
    print("\n=== Initial Cash Outlay Recalculation Analysis ===")
    print(f"Total transactions affected: {len(outlay_df)}")
    
    # Overall statistics
    print("\nOverall Difference Statistics:")
    print(f"Total original outlay: ${outlay_df['Original Transaction Amount'].sum():,.2f}")
    print(f"Total recalculated outlay: ${outlay_df['Transaction Amount'].sum():,.2f}")
    print(f"Total difference: ${outlay_df['Difference'].sum():,.2f}")
    print(f"Average difference: ${outlay_df['Difference'].mean():,.2f}")
    print(f"Average difference %: {outlay_df['Difference %'].mean():.2f}%")
    print(f"Min difference: ${outlay_df['Difference'].min():,.2f}")
    print(f"Max difference: ${outlay_df['Difference'].max():,.2f}")
    
    # Distribution of differences
    print("\nDifference Distribution:")
    bins = [-float('inf'), -10000, -5000, -1000, -500, -100, 0, 100, 500, 1000, 5000, 10000, float('inf')]
    labels = ['< -$10K', '-$10K to -$5K', '-$5K to -$1K', '-$1K to -$500', '-$500 to -$100', 
              '-$100 to $0', '$0 to $100', '$100 to $500', '$500 to $1K', '$1K to $5K', 
              '$5K to $10K', '> $10K']
    outlay_df['Difference Bucket'] = pd.cut(outlay_df['Difference'], bins=bins, labels=labels)
    
    bucket_counts = outlay_df['Difference Bucket'].value_counts().sort_index()
    for bucket, count in bucket_counts.items():
        print(f"  {bucket}: {count} transactions ({count/len(outlay_df)*100:.1f}%)")
    
    # Create a summary dataframe for return
    # Group by Funded ID to show difference by deal
    deal_summary = outlay_df.groupby('Funded ID').agg({
        'Original Transaction Amount': 'sum',
        'Transaction Amount': 'sum',
        'Difference': 'sum',
        'Commission Cost %': 'first',
        'Total Original Balance': 'first'
    }).reset_index()
    
    deal_summary['Difference %'] = (deal_summary['Difference'] / deal_summary['Original Transaction Amount'].abs()) * 100
    deal_summary = deal_summary.sort_values('Difference', ascending=False)
    
    return deal_summary


if __name__ == "__main__":
    # Example usage
    deals_df, payments_df = load_data(verbose=True)
    
    # Test data integrity
    if test_data_integrity(deals_df, payments_df):
        print("Data integrity checks passed!")
    else:
        print("Data integrity checks failed!")
        
    # Analyze difference in initial cash outlay calculations
    deal_summary = analyze_outlay_difference(payments_df)
    
    if deal_summary is not None:
        print("\nTop 10 deals with largest absolute difference:")
        print(deal_summary.head(10)[['Funded ID', 'Original Transaction Amount', 'Transaction Amount', 
                                   'Difference', 'Difference %', 'Commission Cost %']])