#!/usr/bin/env python3
"""
Deal Selector for MCA Portfolio Simulator

This module handles the selection of deals based on various criteria.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def select_deals(deals_df, 
                 product_types=None, 
                 credit_tiers=None, 
                 fico_range=None, 
                 deal_size_range=None, 
                 vintage_range=None,
                 verbose=False):
    """
    Select a subset of deals based on specified criteria.
    
    Args:
        deals_df: DataFrame containing deal information
        product_types: List of product types to include (e.g., ['RBF', 'SuperB'])
        credit_tiers: List of credit tiers to include (e.g., ['A', 'B', 'C'])
        fico_range: Tuple with (min_fico, max_fico) to include
        deal_size_range: Tuple with (min_size, max_size) to include
        vintage_range: Tuple with (start_date, end_date) to include - can be strings ('YYYY-MM-DD') or datetime objects
        verbose: Whether to print information about the selection process
        
    Returns:
        DataFrame containing the selected deals
    """
    if verbose:
        print(f"Selecting deals from {len(deals_df)} total deals")
        print(f"Selection criteria:")
        print(f"  Product Types: {product_types}")
        print(f"  Credit Tiers: {credit_tiers}")
        print(f"  FICO Range: {fico_range}")
        print(f"  Deal Size Range: {deal_size_range}")
        print(f"  Vintage Range: {vintage_range}")
    
    # Start with a copy of all deals
    selected_deals = deals_df.copy()
    
    # Track the impact of each filter
    filter_impacts = {}
    original_count = len(selected_deals)
    
    # Filter by product type
    if product_types is not None and len(product_types) > 0:
        if 'Product' in selected_deals.columns:
            before_count = len(selected_deals)
            selected_deals = selected_deals[selected_deals['Product'].isin(product_types)]
            filter_impacts['Product Type'] = before_count - len(selected_deals)
        else:
            print("WARNING: 'Product' column not found, skipping product type filter")
    
    # Filter by credit tier
    if credit_tiers is not None and len(credit_tiers) > 0:
        if 'Credit_Tier' in selected_deals.columns:
            before_count = len(selected_deals)
            selected_deals = selected_deals[selected_deals['Credit_Tier'].isin(credit_tiers)]
            filter_impacts['Credit Tier'] = before_count - len(selected_deals)
        else:
            print("WARNING: 'Credit_Tier' column not found, skipping credit tier filter")
    
    # Filter by FICO score
    if fico_range is not None and len(fico_range) == 2:
        if 'Owner FICO' in selected_deals.columns:
            min_fico, max_fico = fico_range
            before_count = len(selected_deals)
            selected_deals = selected_deals[
                (selected_deals['Owner FICO'] >= min_fico) & 
                (selected_deals['Owner FICO'] <= max_fico)
            ]
            filter_impacts['FICO Range'] = before_count - len(selected_deals)
        else:
            print("WARNING: 'Owner FICO' column not found, skipping FICO filter")
    
    # Filter by deal size
    if deal_size_range is not None and len(deal_size_range) == 2:
        if 'Total Original Balance' in selected_deals.columns:
            min_size, max_size = deal_size_range
            before_count = len(selected_deals)
            selected_deals = selected_deals[
                (selected_deals['Total Original Balance'] >= min_size) & 
                (selected_deals['Total Original Balance'] <= max_size)
            ]
            filter_impacts['Deal Size'] = before_count - len(selected_deals)
        else:
            print("WARNING: 'Total Original Balance' column not found, skipping deal size filter")
    
    # Filter by vintage (funding date)
    if vintage_range is not None and len(vintage_range) == 2:
        if 'Initial Funding Date' in selected_deals.columns:
            start_date, end_date = vintage_range
            
            # Convert string dates to datetime if needed
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            before_count = len(selected_deals)
            selected_deals = selected_deals[
                (selected_deals['Initial Funding Date'] >= start_date) & 
                (selected_deals['Initial Funding Date'] <= end_date)
            ]
            filter_impacts['Vintage'] = before_count - len(selected_deals)
        else:
            print("WARNING: 'Initial Funding Date' column not found, skipping vintage filter")
    
    # Print summary if verbose
    if verbose:
        print(f"\nSelection results:")
        print(f"  Original deals: {original_count}")
        print(f"  Selected deals: {len(selected_deals)} ({len(selected_deals)/original_count*100:.2f}%)")
        print(f"  Excluded deals: {original_count - len(selected_deals)}")
        
        print("\nImpact of each filter:")
        for filter_name, impact in filter_impacts.items():
            print(f"  {filter_name}: -{impact} deals")
        
        if len(selected_deals) > 0:
            print("\nSelected deal statistics:")
            if 'Total Original Balance' in selected_deals.columns:
                bal = selected_deals['Total Original Balance']
                print(f"  Total balance: ${bal.sum():,.2f}")
                print(f"  Average deal size: ${bal.mean():,.2f}")
                print(f"  Min deal size: ${bal.min():,.2f}")
                print(f"  Max deal size: ${bal.max():,.2f}")
            
            if 'Product' in selected_deals.columns:
                print("\nProduct distribution in selected deals:")
                product_counts = selected_deals['Product'].value_counts()
                for product, count in product_counts.items():
                    print(f"  {product}: {count} deals ({count/len(selected_deals)*100:.2f}%)")
    
    return selected_deals


def validate_selection_criteria(deals_df, product_types=None, credit_tiers=None, 
                               fico_range=None, deal_size_range=None, vintage_range=None):
    """
    Validate selection criteria against the available data.
    
    Args:
        deals_df: DataFrame containing deal information
        product_types, credit_tiers, etc.: Selection criteria
        
    Returns:
        dict: Validation results with issues and warnings
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check product types
    if product_types is not None:
        if 'Product' not in deals_df.columns:
            results['errors'].append("'Product' column not found in data")
            results['valid'] = False
        else:
            available_products = deals_df['Product'].unique()
            invalid_products = [p for p in product_types if p not in available_products]
            if invalid_products:
                results['warnings'].append(f"Products not found in data: {invalid_products}")
    
    # Check credit tiers
    if credit_tiers is not None:
        if 'Credit_Tier' not in deals_df.columns:
            results['errors'].append("'Credit_Tier' column not found in data")
            results['valid'] = False
        else:
            available_tiers = deals_df['Credit_Tier'].unique()
            invalid_tiers = [t for t in credit_tiers if t not in available_tiers]
            if invalid_tiers:
                results['warnings'].append(f"Credit tiers not found in data: {invalid_tiers}")
    
    # Check FICO range
    if fico_range is not None:
        if 'Owner FICO' not in deals_df.columns:
            results['errors'].append("'Owner FICO' column not found in data")
            results['valid'] = False
        else:
            if not isinstance(fico_range, tuple) or len(fico_range) != 2:
                results['errors'].append("FICO range must be a tuple of (min, max)")
                results['valid'] = False
            else:
                min_fico, max_fico = fico_range
                data_min = deals_df['Owner FICO'].min()
                data_max = deals_df['Owner FICO'].max()
                
                if min_fico > max_fico:
                    results['errors'].append(f"Invalid FICO range: min ({min_fico}) > max ({max_fico})")
                    results['valid'] = False
                
                if min_fico < data_min:
                    results['warnings'].append(f"Min FICO ({min_fico}) is below data minimum ({data_min})")
                
                if max_fico > data_max:
                    results['warnings'].append(f"Max FICO ({max_fico}) is above data maximum ({data_max})")
    
    # Check deal size range
    if deal_size_range is not None:
        if 'Total Original Balance' not in deals_df.columns:
            results['errors'].append("'Total Original Balance' column not found in data")
            results['valid'] = False
        else:
            if not isinstance(deal_size_range, tuple) or len(deal_size_range) != 2:
                results['errors'].append("Deal size range must be a tuple of (min, max)")
                results['valid'] = False
            else:
                min_size, max_size = deal_size_range
                
                if min_size > max_size:
                    results['errors'].append(f"Invalid deal size range: min ({min_size}) > max ({max_size})")
                    results['valid'] = False
    
    # Check vintage range
    if vintage_range is not None:
        if 'Initial Funding Date' not in deals_df.columns:
            results['errors'].append("'Initial Funding Date' column not found in data")
            results['valid'] = False
        else:
            if not isinstance(vintage_range, tuple) or len(vintage_range) != 2:
                results['errors'].append("Vintage range must be a tuple of (start_date, end_date)")
                results['valid'] = False
            else:
                start_date, end_date = vintage_range
                
                # Convert string dates to datetime if needed
                if isinstance(start_date, str):
                    try:
                        start_date = pd.to_datetime(start_date)
                    except:
                        results['errors'].append(f"Invalid start date format: {start_date}")
                        results['valid'] = False
                
                if isinstance(end_date, str):
                    try:
                        end_date = pd.to_datetime(end_date)
                    except:
                        results['errors'].append(f"Invalid end date format: {end_date}")
                        results['valid'] = False
                
                if results['valid'] and start_date > end_date:
                    results['errors'].append(f"Invalid vintage range: start date ({start_date}) > end date ({end_date})")
                    results['valid'] = False
    
    return results


if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    
    deals_df, _ = load_data()
    
    # Run tests
    test_deal_selection(deals_df)
    
    # Example selection
    print("\nExample selection:")
    selected_deals = select_deals(
        deals_df,
        product_types=['RBF'],
        deal_size_range=(100000, 500000),
        vintage_range=('2023-01-01', '2023-12-31'),
        verbose=True
    )