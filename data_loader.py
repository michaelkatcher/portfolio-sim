#!/usr/bin/env python3
"""
Data Loader for MCA Portfolio Simulator

This module handles loading and preprocessing of deal and payment data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

from common_utils import setup_logging, DataError
from constants import DEAL_COLUMNS, PAYMENT_COLUMNS, TRANSACTION_TYPES, VALIDATION

# Set up logger
logger = setup_logging(__name__)


def load_data(data_file='data.csv', payments_file='payments.csv', verbose=False):
    """
    Load and prepare the input data files for the MCA Portfolio Simulator.
    
    Args:
        data_file: Path to the data.csv file (contains deal information)
        payments_file: Path to the payments.csv file (contains payment transactions)
        verbose: Whether to print information about the loaded data
        
    Returns:
        Tuple of (deals_df, payments_df)
        
    Raises:
        DataError: If data files can't be loaded or processed
        FileNotFoundError: If input files don't exist
    """
    # Check if files exist
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Deals data file not found: {data_file}")
        
    if not os.path.exists(payments_file):
        raise FileNotFoundError(f"Payments data file not found: {payments_file}")
    
    if verbose:
        logger.info(f"Loading data from files: {data_file} and {payments_file}")
    
    try:
        # Load deals data (data.csv)
        deals_df = pd.read_csv(
            data_file,
            dtype={
                DEAL_COLUMNS['ID']: str,
                DEAL_COLUMNS['PRODUCT']: str,
                DEAL_COLUMNS['COMMISSION']: float,
            },
            parse_dates=[DEAL_COLUMNS['FUNDING_DATE']],
            thousands=','  # Handle numbers with commas
        )
        
        # Check if the file was loaded properly
        if deals_df.empty:
            raise DataError(f"Deals data file is empty: {data_file}")
    except Exception as e:
        raise DataError(f"Error loading deals data: {str(e)}") from e
    
    try:
        # Load payments data (payments.csv)
        payments_df = pd.read_csv(
            payments_file,
            dtype={
                'Vintage': str,
                PAYMENT_COLUMNS['VINTAGE_MONTH']: str,
                PAYMENT_COLUMNS['ID']: str,
                PAYMENT_COLUMNS['DESCRIPTION']: str
            },
            parse_dates=['Funded Date', PAYMENT_COLUMNS['DATE']],
            thousands=','  # Handle numbers with commas
        )
        
        # Check if the file was loaded properly
        if payments_df.empty:
            raise DataError(f"Payments data file is empty: {payments_file}")
    except Exception as e:
        raise DataError(f"Error loading payments data: {str(e)}") from e
    
    try:
        # Process deals data
        _process_deals_data(deals_df, verbose)
        
        # Process payments data
        _process_payments_data(payments_df, deals_df, verbose)
        
        # Validate data integrity
        if not test_data_integrity(deals_df, payments_df, log_warnings=verbose):
            logger.warning("Data integrity check failed. Results may be unreliable.")
        
        if verbose:
            logger.info(f"Loaded {len(deals_df)} deals and {len(payments_df)} payment transactions")
        
        return deals_df, payments_df
    
    except Exception as e:
        raise DataError(f"Error processing data: {str(e)}") from e


def _process_deals_data(deals_df, verbose=False):
    """
    Process deals data after loading.
    
    Args:
        deals_df: DataFrame containing deal information
        verbose: Whether to log detailed information
    """
    # Convert monetary columns to numeric
    monetary_cols = [DEAL_COLUMNS['BALANCE'], DEAL_COLUMNS['RTR'], 
                     'Past Due Amount', 'Total Principal Net Charge Off']
    for col in monetary_cols:
        if col in deals_df.columns:
            deals_df[col] = pd.to_numeric(deals_df[col], errors='coerce')
        elif verbose:
            logger.warning(f"Column '{col}' not found in deals data")
    
    # Convert Owner FICO to numeric
    if DEAL_COLUMNS['FICO'] in deals_df.columns:
        deals_df[DEAL_COLUMNS['FICO']] = pd.to_numeric(deals_df[DEAL_COLUMNS['FICO']], errors='coerce')
    
    # Ensure Commission Cost % is numeric
    if DEAL_COLUMNS['COMMISSION'] in deals_df.columns:
        deals_df[DEAL_COLUMNS['COMMISSION']] = pd.to_numeric(deals_df[DEAL_COLUMNS['COMMISSION']], errors='coerce')
    
    # Add Vintage-M column based on Initial Funding Date
    if DEAL_COLUMNS['FUNDING_DATE'] in deals_df.columns:
        deals_df[DEAL_COLUMNS['VINTAGE_MONTH']] = deals_df[DEAL_COLUMNS['FUNDING_DATE']].dt.strftime('%Y-%m')
    
    if verbose:
        # Log data types
        logger.info("\nDeals DataFrame data types:")
        for col, dtype in deals_df.dtypes.items():
            logger.info(f"  {col}: {dtype}")
        
        # Log basic statistics
        if DEAL_COLUMNS['COMMISSION'] in deals_df.columns:
            logger.info("\nCommission Cost % statistics:")
            logger.info(deals_df[DEAL_COLUMNS['COMMISSION']].describe())
        
        if DEAL_COLUMNS['BALANCE'] in deals_df.columns:
            logger.info("\nDeal size statistics:")
            logger.info(deals_df[DEAL_COLUMNS['BALANCE']].describe())


def _process_payments_data(payments_df, deals_df, verbose=False):
    """
    Process payments data after loading.
    
    Args:
        payments_df: DataFrame containing payment transactions
        deals_df: DataFrame containing deal information
        verbose: Whether to log detailed information
    """
    # Convert Transaction Amount to numeric
    if PAYMENT_COLUMNS['AMOUNT'] in payments_df.columns:
        payments_df[PAYMENT_COLUMNS['AMOUNT']] = pd.to_numeric(payments_df[PAYMENT_COLUMNS['AMOUNT']], errors='coerce')
    
    # Store original Transaction Amount values for comparison
    payments_df['Original Transaction Amount'] = payments_df[PAYMENT_COLUMNS['AMOUNT']]
    
    # Recalculate initial cash outlay values based on Commission Cost %
    try:
        _recalculate_cash_outlays(payments_df, deals_df, verbose)
    except Exception as e:
        logger.warning(f"Error recalculating cash outlays: {str(e)}")
    
    if verbose:
        # Log data types
        logger.info("\nPayments DataFrame data types:")
        for col, dtype in payments_df.dtypes.items():
            logger.info(f"  {col}: {dtype}")
        
        # Log transaction types
        if PAYMENT_COLUMNS['DESCRIPTION'] in payments_df.columns:
            logger.info("\nTransaction type distribution:")
            logger.info(payments_df[PAYMENT_COLUMNS['DESCRIPTION']].value_counts())


def _recalculate_cash_outlays(payments_df, deals_df, verbose=False):
    """
    Recalculate initial cash outlay values based on Commission Cost %.
    
    Args:
        payments_df: DataFrame containing payment transactions
        deals_df: DataFrame containing deal information
        verbose: Whether to log detailed information
    """
    # Check required columns
    required_cols = [DEAL_COLUMNS['ID'], DEAL_COLUMNS['COMMISSION'], DEAL_COLUMNS['BALANCE']]
    if not all(col in deals_df.columns for col in required_cols):
        if verbose:
            logger.warning("Missing required columns in deals data for cash outlay recalculation")
        return
    
    # Create a mapping of Funded ID to Commission Cost % and Total Original Balance
    commission_map = deals_df[required_cols].set_index(DEAL_COLUMNS['ID'])
    
    # Add commission and balance columns to payments dataframe for calculation
    payments_df = payments_df.merge(
        commission_map, 
        on=PAYMENT_COLUMNS['ID'], 
        how='left'
    )
    
    # Apply the formula for initial cash outlay transactions
    # Formula: =-(1+'Commission Cost %')*('Total Original Balance')
    initial_outlay_mask = (payments_df[PAYMENT_COLUMNS['DESCRIPTION']] == TRANSACTION_TYPES['INITIAL_OUTLAY'])
    
    if initial_outlay_mask.any():
        # Calculate new transaction amounts for initial cash outlays
        payments_df.loc[initial_outlay_mask, PAYMENT_COLUMNS['AMOUNT']] = -(
            (1 + payments_df.loc[initial_outlay_mask, DEAL_COLUMNS['COMMISSION']]) * 
            payments_df.loc[initial_outlay_mask, DEAL_COLUMNS['BALANCE']]
        )
        
        if verbose:
            n_updated = initial_outlay_mask.sum()
            logger.info(f"\nRecalculated {n_updated} initial cash outlay transactions based on Commission Cost %")


def test_data_integrity(deals_df, payments_df, log_warnings=True):
    """
    Run basic integrity checks on the loaded data.
    
    Args:
        deals_df: DataFrame containing deal information
        payments_df: DataFrame containing payment transactions
        log_warnings: Whether to log warnings
        
    Returns:
        bool: True if all critical checks pass, False otherwise
    """
    critical_failure = False
    
    # Check for required columns in deals_df
    required_deal_cols = [
        DEAL_COLUMNS['ID'], 
        DEAL_COLUMNS['PRODUCT'], 
        DEAL_COLUMNS['FUNDING_DATE'], 
        DEAL_COLUMNS['BALANCE'],
        DEAL_COLUMNS['RTR'],
        DEAL_COLUMNS['COMMISSION']
    ]
    
    for col in required_deal_cols:
        if col not in deals_df.columns:
            logger.error(f"Required column '{col}' missing from deals data")
            critical_failure = True
    
    # Check for required columns in payments_df
    required_payment_cols = [
        PAYMENT_COLUMNS['ID'], 
        PAYMENT_COLUMNS['DATE'], 
        PAYMENT_COLUMNS['AMOUNT'], 
        PAYMENT_COLUMNS['DESCRIPTION']
    ]
    
    for col in required_payment_cols:
        if col not in payments_df.columns:
            logger.error(f"Required column '{col}' missing from payments data")
            critical_failure = True
    
    if critical_failure:
        return False
    
    # Check for data type correctness
    if DEAL_COLUMNS['FUNDING_DATE'] in deals_df.columns and not pd.api.types.is_datetime64_dtype(deals_df[DEAL_COLUMNS['FUNDING_DATE']]):
        logger.error(f"'{DEAL_COLUMNS['FUNDING_DATE']}' is not correctly formatted as datetime")
        return False
    
    if PAYMENT_COLUMNS['DATE'] in payments_df.columns and not pd.api.types.is_datetime64_dtype(payments_df[PAYMENT_COLUMNS['DATE']]):
        logger.error(f"'{PAYMENT_COLUMNS['DATE']}' is not correctly formatted as datetime")
        return False
    
    # Check for missing values in critical columns
    critical_cols = {
        'deals': [DEAL_COLUMNS['ID'], DEAL_COLUMNS['BALANCE'], DEAL_COLUMNS['RTR'], DEAL_COLUMNS['COMMISSION']],
        'payments': [PAYMENT_COLUMNS['ID'], PAYMENT_COLUMNS['AMOUNT'], PAYMENT_COLUMNS['DESCRIPTION']]
    }
    
    for df_name, cols in critical_cols.items():
        df = deals_df if df_name == 'deals' else payments_df
        for col in cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0 and log_warnings:
                    logger.warning(f"{missing_count} missing values in {df_name} column '{col}'")
    
    # Check for data consistency between datasets
    deal_ids = set(deals_df[DEAL_COLUMNS['ID']])
    payment_ids = set(payments_df[PAYMENT_COLUMNS['ID']])
    common_ids = deal_ids.intersection(payment_ids)
    
    logger.info(f"Found {len(common_ids)} deals with matching payment records")
    if len(common_ids) == 0:
        logger.error("No matching IDs between deals and payments data")
        return False
    
    # Check for missing values in Owner FICO
    if DEAL_COLUMNS['FICO'] in deals_df.columns:
        missing_owner_fico = deals_df[DEAL_COLUMNS['FICO']].isna().sum()
        if missing_owner_fico > 0 and log_warnings:
            logger.warning(f"{missing_owner_fico} missing values in Owner FICO")
    
    # Check for Commission Cost % values outside expected range
    if DEAL_COLUMNS['COMMISSION'] in deals_df.columns:
        commission_range_issue = deals_df[(deals_df[DEAL_COLUMNS['COMMISSION']] < VALIDATION['COMMISSION_MIN']) | 
                                         (deals_df[DEAL_COLUMNS['COMMISSION']] > VALIDATION['COMMISSION_MAX'])].shape[0]
        if commission_range_issue > 0 and log_warnings:
            logger.warning(f"{commission_range_issue} records have Commission Cost % outside expected range ({VALIDATION['COMMISSION_MIN']}-{VALIDATION['COMMISSION_MAX']})")
    
    # All critical checks passed
    return True


def analyze_outlay_difference(payments_df):
    """
    Analyze the difference between original and recalculated initial cash outlay values.
    
    Args:
        payments_df: DataFrame containing payment transactions with both original and recalculated values
        
    Returns:
        DataFrame with difference analysis or None if no outlays found
    """
    required_cols = [PAYMENT_COLUMNS['DESCRIPTION'], PAYMENT_COLUMNS['AMOUNT'], 'Original Transaction Amount', 
                     DEAL_COLUMNS['COMMISSION'], DEAL_COLUMNS['BALANCE']]
    
    missing_cols = [col for col in required_cols if col not in payments_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns for analysis: {missing_cols}")
        return None
    
    # Filter for initial cash outlay transactions
    outlay_df = payments_df[payments_df[PAYMENT_COLUMNS['DESCRIPTION']] == TRANSACTION_TYPES['INITIAL_OUTLAY']].copy()
    
    if len(outlay_df) == 0:
        logger.info("No initial cash outlay transactions found.")
        return None
    
    try:
        # Calculate difference between original and new values
        outlay_df['Difference'] = outlay_df[PAYMENT_COLUMNS['AMOUNT']] - outlay_df['Original Transaction Amount']
        outlay_df['Difference %'] = (outlay_df['Difference'] / outlay_df['Original Transaction Amount'].abs()) * 100
        
        # Summary statistics
        logger.info("\n=== Initial Cash Outlay Recalculation Analysis ===")
        logger.info(f"Total transactions affected: {len(outlay_df)}")
        
        # Overall statistics
        logger.info("\nOverall Difference Statistics:")
        logger.info(f"Total original outlay: ${outlay_df['Original Transaction Amount'].sum():,.2f}")
        logger.info(f"Total recalculated outlay: ${outlay_df[PAYMENT_COLUMNS['AMOUNT']].sum():,.2f}")
        logger.info(f"Total difference: ${outlay_df['Difference'].sum():,.2f}")
        logger.info(f"Average difference: ${outlay_df['Difference'].mean():,.2f}")
        logger.info(f"Average difference %: {outlay_df['Difference %'].mean():.2f}%")
        logger.info(f"Min difference: ${outlay_df['Difference'].min():,.2f}")
        logger.info(f"Max difference: ${outlay_df['Difference'].max():,.2f}")
        
        # Distribution of differences
        logger.info("\nDifference Distribution:")
        bins = [-float('inf'), -10000, -5000, -1000, -500, -100, 0, 100, 500, 1000, 5000, 10000, float('inf')]
        labels = ['< -$10K', '-$10K to -$5K', '-$5K to -$1K', '-$1K to -$500', '-$500 to -$100', 
                  '-$100 to $0', '$0 to $100', '$100 to $500', '$500 to $1K', '$1K to $5K', 
                  '$5K to $10K', '> $10K']
        outlay_df['Difference Bucket'] = pd.cut(outlay_df['Difference'], bins=bins, labels=labels)
        
        bucket_counts = outlay_df['Difference Bucket'].value_counts().sort_index()
        for bucket, count in bucket_counts.items():
            logger.info(f"  {bucket}: {count} transactions ({count/len(outlay_df)*100:.1f}%)")
        
        # Create a summary dataframe for return
        # Group by Funded ID to show difference by deal
        deal_summary = outlay_df.groupby(PAYMENT_COLUMNS['ID']).agg({
            'Original Transaction Amount': 'sum',
            PAYMENT_COLUMNS['AMOUNT']: 'sum',
            'Difference': 'sum',
            DEAL_COLUMNS['COMMISSION']: 'first',
            DEAL_COLUMNS['BALANCE']: 'first'
        }).reset_index()
        
        deal_summary['Difference %'] = (deal_summary['Difference'] / deal_summary['Original Transaction Amount'].abs()) * 100
        deal_summary = deal_summary.sort_values('Difference', ascending=False)
        
        return deal_summary
        
    except Exception as e:
        logger.error(f"Error analyzing outlay differences: {str(e)}")
        return None