#!/usr/bin/env python3
"""
MCA Portfolio Simulator - Comprehensive Analysis Runner

This script performs a comprehensive portfolio analysis with the MCA Portfolio Simulator
and exports the results to Excel files according to the project requirements.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import argparse

# Import modules
from data_loader import load_data
from deal_selector import select_deals
from portfolio_simulator import Portfolio, AllocationPolicy, FeeSchedule, compare_portfolios
from excel_exporter import export_portfolio_to_excel


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='MCA Portfolio Simulator')
    
    parser.add_argument('--data-file', default='data.csv',
                        help='Path to the deal data CSV file')
    parser.add_argument('--payments-file', default='payments.csv',
                        help='Path to the payments CSV file')
    parser.add_argument('--output-dir', default='output',
                        help='Directory to save output files')
    parser.add_argument('--product-types', nargs='+',
                        help='Filter by product types (e.g., RBF SuperB)')
    parser.add_argument('--min-size', type=float,
                        help='Minimum deal size')
    parser.add_argument('--max-size', type=float,
                        help='Maximum deal size')
    parser.add_argument('--vintage-start', 
                        help='Start date for vintage filter (YYYY-MM-DD)')
    parser.add_argument('--vintage-end', 
                        help='End date for vintage filter (YYYY-MM-DD)')
    parser.add_argument('--allocation-pct', type=float, default=0.25,
                        help='Allocation percentage (0-1)')
    parser.add_argument('--min-allocation', type=float, default=100000,
                        help='Minimum allocation amount')
    parser.add_argument('--max-allocation', type=float, default=1500000,
                        help='Maximum allocation amount')
    parser.add_argument('--fee-pct', type=float, default=0.02,
                        help='Management fee percentage (0-1)')
    parser.add_argument('--output-file', default='portfolio_analysis.xlsx',
                        help='Output Excel file name')
    parser.add_argument('--verbose', action='store_true',
                        help='Display verbose output')
    
    return parser.parse_args()


def ensure_directory(directory):
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def run_single_portfolio(args):
    """Run a single portfolio simulation based on command-line arguments."""
    # Load data
    print(f"Loading data from {args.data_file} and {args.payments_file}...")
    deals_df, payments_df = load_data(
        data_file=args.data_file,
        payments_file=args.payments_file,
        verbose=args.verbose
    )
    
    # Prepare selection criteria
    selection_criteria = {}
    if args.product_types:
        selection_criteria['product_types'] = args.product_types
    
    if args.min_size is not None and args.max_size is not None:
        selection_criteria['deal_size_range'] = (args.min_size, args.max_size)
    
    if args.vintage_start and args.vintage_end:
        selection_criteria['vintage_range'] = (args.vintage_start, args.vintage_end)
    
    # Create allocation policy
    allocation_policy = AllocationPolicy(
        percentage=args.allocation_pct,
        min_amount=args.min_allocation,
        max_amount=args.max_allocation
    )
    
    # Create fee schedule
    fee_schedule = FeeSchedule(
        percentage=args.fee_pct,
        apply_after_principal=True
    )
    
    # Create portfolio name based on criteria
    portfolio_name = "Portfolio Analysis"
    if 'product_types' in selection_criteria:
        portfolio_name += f" - {'/'.join(selection_criteria['product_types'])}"
    if 'vintage_range' in selection_criteria:
        start, end = selection_criteria['vintage_range']
        portfolio_name += f" ({start} to {end})"
    
    # Create portfolio
    print(f"Creating portfolio: {portfolio_name}")
    portfolio = Portfolio(
        name=portfolio_name,
        selection_criteria=selection_criteria,
        allocation_policy=allocation_policy,
        fee_schedule=fee_schedule
    )
    
    # Simulate portfolio
    print("Simulating portfolio...")
    portfolio.simulate(deals_df, payments_df, verbose=args.verbose)
    
    # Ensure output directory exists
    ensure_directory(args.output_dir)
    output_path = os.path.join(args.output_dir, args.output_file)
    
    # Export to Excel
    print(f"Exporting results to {output_path}...")
    export_portfolio_to_excel(portfolio, output_path, verbose=args.verbose)
    
    print(f"\nPortfolio analysis complete!")
    print(f"Results saved to: {os.path.abspath(output_path)}")
    print("\nPortfolio Summary:")
    print(portfolio)
    
    return portfolio


def run_comparative_analysis():
    """Run a comprehensive comparative analysis of multiple portfolio scenarios."""
    print("=" * 80)
    print("MCA Portfolio Simulator - Comparative Analysis")
    print("=" * 80)
    
    # Load data
    print("\nLoading data files...")
    deals_df, payments_df = load_data(verbose=True)
    
    # Output directory
    output_dir = "comparative_analysis"
    ensure_directory(output_dir)
    
    # Define scenarios for comparison
    # 1. Comparing different product types
    product_portfolios = []
    for product in ['RBF', 'SuperB']:
        portfolio = Portfolio(
            name=f"{product} Products (2023)",
            selection_criteria={
                'product_types': [product],
                'vintage_range': ('2023-01-01', '2023-12-31')
            },
            allocation_policy=AllocationPolicy(
                percentage=0.25,
                min_amount=100000,
                max_amount=1500000
            ),
            fee_schedule=FeeSchedule(
                percentage=0.02,
                apply_after_principal=True
            )
        )
        product_portfolios.append(portfolio)
    
    # 2. Comparing different allocation policies
    allocation_portfolios = []
    for pct, name in [(0.1, "10%"), (0.25, "25%"), (0.5, "50%")]:
        portfolio = Portfolio(
            name=f"Allocation {name} (2023)",
            selection_criteria={
                'vintage_range': ('2023-01-01', '2023-12-31')
            },
            allocation_policy=AllocationPolicy(
                percentage=pct,
                min_amount=100000,
                max_amount=1500000
            ),
            fee_schedule=FeeSchedule(
                percentage=0.02,
                apply_after_principal=True
            )
        )
        allocation_portfolios.append(portfolio)
    
    # 3. Comparing different fee structures
    fee_portfolios = []
    for fee, name in [(0.0, "0%"), (0.02, "2%"), (0.05, "5%")]:
        portfolio = Portfolio(
            name=f"Fee {name} (2023)",
            selection_criteria={
                'vintage_range': ('2023-01-01', '2023-12-31')
            },
            allocation_policy=AllocationPolicy(
                percentage=0.25,
                min_amount=100000,
                max_amount=1500000
            ),
            fee_schedule=FeeSchedule(
                percentage=fee,
                apply_after_principal=True
            )
        )
        fee_portfolios.append(portfolio)
    
    # 4. Comparing different vintage periods
    vintage_portfolios = []
    for period, name in [
        (('2023-01-01', '2023-03-31'), "Q1 2023"),
        (('2023-04-01', '2023-06-30'), "Q2 2023"),
        (('2023-07-01', '2023-09-30'), "Q3 2023"),
        (('2023-10-01', '2023-12-31'), "Q4 2023")
    ]:
        portfolio = Portfolio(
            name=f"Vintage {name}",
            selection_criteria={
                'vintage_range': period
            },
            allocation_policy=AllocationPolicy(
                percentage=0.25,
                min_amount=100000,
                max_amount=1500000
            ),
            fee_schedule=FeeSchedule(
                percentage=0.02,
                apply_after_principal=True
            )
        )
        vintage_portfolios.append(portfolio)
    
    # Combine all portfolios for simulation
    all_portfolios = {
        "product": product_portfolios,
        "allocation": allocation_portfolios,
        "fee": fee_portfolios,
        "vintage": vintage_portfolios
    }
    
    # Simulate all portfolios
    for category, portfolios in all_portfolios.items():
        print(f"\nSimulating {category.capitalize()} Comparison Portfolios:")
        for i, portfolio in enumerate(portfolios):
            print(f"  {i+1}. {portfolio.name}...")
            portfolio.simulate(deals_df, payments_df)
            
            # Export individual portfolio results
            file_name = f"{category}_{i+1}_{portfolio.name.replace(' ', '_').replace('(', '').replace(')', '')}.xlsx"
            file_path = os.path.join(output_dir, file_name)
            export_portfolio_to_excel(portfolio, file_path)
    
    # Create comparison Excel files
    for category, portfolios in all_portfolios.items():
        comparison_df = compare_portfolios(portfolios)
        
        # Save comparison to CSV
        comparison_path = os.path.join(output_dir, f"{category}_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n{category.capitalize()} comparison saved to {comparison_path}")
        
        # Display comparison table
        print(f"\n{category.capitalize()} Portfolio Comparison:")
        print("-" * 100)
        print(comparison_df.to_string())
    
    print("\nComparative analysis complete!")
    print(f"All results saved to: {os.path.abspath(output_dir)}")


def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Run portfolio simulation
    run_single_portfolio(args)
    
    # Uncomment to run comparative analysis
    # run_comparative_analysis()


if __name__ == "__main__":
    main()