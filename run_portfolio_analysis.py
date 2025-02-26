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
import sys
import traceback
import logging

# Import modules
from data_loader import load_data
from deal_selector import select_deals
from portfolio_simulator import Portfolio, AllocationPolicy, FeeSchedule
from excel_exporter import export_portfolio_to_excel
from common_utils import setup_logging, MCASimulatorError, ConfigurationError, DataError, SimulationError, ExportError

# Set up logger
logger = None

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
    parser.add_argument('--compare', action='store_true',
                        help='Run comparative analysis instead of single portfolio')
	# Add log file argument
    parser.add_argument('--log-file', 
                        help='Log file name (defaults to same name as output file)')
	
    return parser.parse_args()


def ensure_directory(directory):
    """
    Ensure the output directory exists.
    
    Args:
        directory: Directory path to create if it doesn't exist
        
    Raises:
        OSError: If directory creation fails
    """
    # Convert relative paths to be relative to the script directory
    if not os.path.isabs(directory):
        directory = os.path.join(get_script_dir(), directory)
        
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            raise OSError(f"Failed to create directory {directory}: {str(e)}")
    
    return directory  # Return the full path


def validate_arguments(args):
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        ConfigurationError: If arguments are invalid
    """
    # Remove the file path checks since we'll handle file paths in data_loader.py
    # No need to check if files exist here anymore
    
    # Check numeric ranges
    if args.allocation_pct is not None and (args.allocation_pct <= 0 or args.allocation_pct > 1):
        raise ConfigurationError(f"Allocation percentage must be between 0 and 1, got {args.allocation_pct}")
    
    if args.fee_pct is not None and (args.fee_pct < 0 or args.fee_pct > 1):
        raise ConfigurationError(f"Fee percentage must be between 0 and 1, got {args.fee_pct}")
    
    if args.min_size is not None and args.min_size < 0:
        raise ConfigurationError(f"Minimum deal size cannot be negative, got {args.min_size}")
    
    if args.max_size is not None and args.max_size < 0:
        raise ConfigurationError(f"Maximum deal size cannot be negative, got {args.max_size}")
    
    if args.min_size is not None and args.max_size is not None and args.min_size > args.max_size:
        raise ConfigurationError(f"Minimum deal size ({args.min_size}) cannot be greater than maximum deal size ({args.max_size})")
    
    if args.min_allocation is not None and args.min_allocation < 0:
        raise ConfigurationError(f"Minimum allocation cannot be negative, got {args.min_allocation}")
        
    if args.max_allocation is not None and args.max_allocation < 0:
        raise ConfigurationError(f"Maximum allocation cannot be negative, got {args.max_allocation}")
    
    if args.min_allocation is not None and args.max_allocation is not None and args.min_allocation > args.max_allocation:
        raise ConfigurationError(f"Minimum allocation ({args.min_allocation}) cannot be greater than maximum allocation ({args.max_allocation})")
    
    # Check date ranges
    if args.vintage_start and args.vintage_end:
        try:
            start_date = pd.to_datetime(args.vintage_start)
            end_date = pd.to_datetime(args.vintage_end)
            if start_date > end_date:
                raise ConfigurationError(f"Vintage start date ({args.vintage_start}) cannot be after end date ({args.vintage_end})")
        except ValueError as e:
            raise ConfigurationError(f"Invalid date format: {str(e)}")

def get_script_dir():
    """Get the directory where the script is located."""
    return os.path.dirname(os.path.abspath(__file__))

def run_single_portfolio(args):
    """
    Run a single portfolio simulation based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Portfolio instance or None if simulation fails
        
    Raises:
        Various exceptions depending on the failure point
    """
    # Validate arguments
    validate_arguments(args)
    
    # Load data
    logger.info(f"Loading data from {args.data_file} and {args.payments_file}...")
    deals_df, payments_df = load_data(
        data_file=args.data_file,
        payments_file=args.payments_file,
        verbose=args.verbose
    )
    # Prepare selection criteria
    selection_criteria = {}
    if args.product_types:
        selection_criteria['product_types'] = args.product_types
        logger.info(f"Using product types filter: {args.product_types}")
    
    # Modified code to handle single min or max size
    if args.min_size is not None or args.max_size is not None:
        min_size = args.min_size if args.min_size is not None else 0
        max_size = args.max_size if args.max_size is not None else float('inf')
        selection_criteria['deal_size_range'] = (min_size, max_size)
        
        if args.max_size is None:
            logger.info(f"Using deal size range: ${min_size:,.2f} and above")
        elif args.min_size is None:
            logger.info(f"Using deal size range: up to ${max_size:,.2f}")
        else:
            logger.info(f"Using deal size range: ${min_size:,.2f} to ${max_size:,.2f}")
    
    if args.vintage_start and args.vintage_end:
        selection_criteria['vintage_range'] = (args.vintage_start, args.vintage_end)
        logger.info(f"Using vintage range: {args.vintage_start} to {args.vintage_end}")
    
    # Create allocation policy
    allocation_policy = AllocationPolicy(
        percentage=args.allocation_pct,
        min_amount=args.min_allocation,
        max_amount=args.max_allocation
    )
    logger.info(f"Using allocation policy: {allocation_policy}")
    
    # Create fee schedule
    fee_schedule = FeeSchedule(
        percentage=args.fee_pct,
        apply_after_principal=True
    )
    logger.info(f"Using fee schedule: {fee_schedule}")
    
    # Create portfolio name based on criteria
    portfolio_name = "Portfolio Analysis"
    if 'product_types' in selection_criteria:
        portfolio_name += f" - {'/'.join(selection_criteria['product_types'])}"
    if 'vintage_range' in selection_criteria:
        start, end = selection_criteria['vintage_range']
        portfolio_name += f" ({start} to {end})"
    
    # Create portfolio
    logger.info(f"Creating portfolio: {portfolio_name}")
    portfolio = Portfolio(
        name=portfolio_name,
        selection_criteria=selection_criteria,
        allocation_policy=allocation_policy,
        fee_schedule=fee_schedule
    )
    
    # Simulate portfolio
    logger.info("Simulating portfolio...")
    portfolio.simulate(deals_df, payments_df, verbose=args.verbose)
    
    if portfolio.cashflows is None or portfolio.cashflows.empty:
        logger.warning("No cashflows generated for portfolio.")
    
    # Ensure output directory exists
    output_dir = ensure_directory(args.output_dir)  
    output_path = os.path.join(output_dir, args.output_file)
    
    # Export to Excel
    logger.info(f"Exporting results to {output_path}...")
    export_portfolio_to_excel(portfolio, output_path, verbose=args.verbose)
    
    logger.info(f"\nPortfolio analysis complete!")
    logger.info(f"Results saved to: {os.path.abspath(output_path)}")
    logger.info("\nPortfolio Summary:")
    logger.info(portfolio)
    
    return portfolio


def run_comparative_analysis(args):
    """
    Generate multiple portfolio scenarios for comparison.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        List of generated portfolios
        
    Raises:
        Various exceptions depending on the failure point
    """
    logger.info("=" * 80)
    logger.info("MCA Portfolio Simulator - Multiple Portfolio Scenarios")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\nLoading data files...")
    deals_df, payments_df = load_data(
        data_file=args.data_file,
        payments_file=args.payments_file,
        verbose=True
    )
    
    # Output directory
    output_dir = args.output_dir
    ensure_directory(output_dir)
    
    # Define scenarios to generate
    scenarios = [
        # Different product types
        {
            'name': "RBF Products (2023)",
            'selection_criteria': {
                'product_types': ['RBF'],
                'vintage_range': ('2023-01-01', '2023-12-31')
            },
            'allocation_policy': AllocationPolicy(percentage=0.25, min_amount=100000, max_amount=1500000),
            'fee_schedule': FeeSchedule(percentage=0.02, apply_after_principal=True)
        },
        {
            'name': "SuperB Products (2023)",
            'selection_criteria': {
                'product_types': ['SuperB'],
                'vintage_range': ('2023-01-01', '2023-12-31')
            },
            'allocation_policy': AllocationPolicy(percentage=0.25, min_amount=100000, max_amount=1500000),
            'fee_schedule': FeeSchedule(percentage=0.02, apply_after_principal=True)
        },
        
        # Different allocation policies
        {
            'name': "Allocation 10% (2023)",
            'selection_criteria': {
                'vintage_range': ('2023-01-01', '2023-12-31')
            },
            'allocation_policy': AllocationPolicy(percentage=0.1, min_amount=100000, max_amount=1500000),
            'fee_schedule': FeeSchedule(percentage=0.02, apply_after_principal=True)
        },
        {
            'name': "Allocation 50% (2023)",
            'selection_criteria': {
                'vintage_range': ('2023-01-01', '2023-12-31')
            },
            'allocation_policy': AllocationPolicy(percentage=0.5, min_amount=100000, max_amount=1500000),
            'fee_schedule': FeeSchedule(percentage=0.02, apply_after_principal=True)
        },
        
        # Different fee structures
        {
            'name': "No Fee (2023)",
            'selection_criteria': {
                'vintage_range': ('2023-01-01', '2023-12-31')
            },
            'allocation_policy': AllocationPolicy(percentage=0.25, min_amount=100000, max_amount=1500000),
            'fee_schedule': FeeSchedule(percentage=0.0, apply_after_principal=True)
        },
        {
            'name': "5% Fee (2023)",
            'selection_criteria': {
                'vintage_range': ('2023-01-01', '2023-12-31')
            },
            'allocation_policy': AllocationPolicy(percentage=0.25, min_amount=100000, max_amount=1500000),
            'fee_schedule': FeeSchedule(percentage=0.05, apply_after_principal=True)
        },
        
        # Different vintage periods
        {
            'name': "Q1 2023 Vintage",
            'selection_criteria': {
                'vintage_range': ('2023-01-01', '2023-03-31')
            },
            'allocation_policy': AllocationPolicy(percentage=0.25, min_amount=100000, max_amount=1500000),
            'fee_schedule': FeeSchedule(percentage=0.02, apply_after_principal=True)
        },
        {
            'name': "Q4 2023 Vintage",
            'selection_criteria': {
                'vintage_range': ('2023-10-01', '2023-12-31')
            },
            'allocation_policy': AllocationPolicy(percentage=0.25, min_amount=100000, max_amount=1500000),
            'fee_schedule': FeeSchedule(percentage=0.02, apply_after_principal=True)
        }
    ]
    
    # Store all portfolios
    generated_portfolios = []
    
    # Run each scenario
    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"\nSimulating scenario {i}/{len(scenarios)}: {scenario['name']}")
        
        try:
            # Create portfolio
            portfolio = Portfolio(
                name=scenario['name'],
                selection_criteria=scenario['selection_criteria'],
                allocation_policy=scenario['allocation_policy'],
                fee_schedule=scenario['fee_schedule']
            )
            
            # Simulate
            portfolio.simulate(deals_df, payments_df)
            
            # Export to Excel
            file_name = f"{scenario['name'].replace(' ', '_').replace('(', '').replace(')', '')}.xlsx"
            file_path = os.path.join(output_dir, file_name)
            export_portfolio_to_excel(portfolio, file_path)
            logger.info(f"  Results exported to: {file_path}")
            
            # Store portfolio
            generated_portfolios.append(portfolio)
            
        except Exception as e:
            logger.error(f"Error processing scenario '{scenario['name']}': {str(e)}")
            logger.debug(traceback.format_exc())
    
    logger.info("\nAll portfolio scenarios complete!")
    logger.info(f"Results saved to: {os.path.abspath(output_dir)}")
    
    return generated_portfolios

def main():
    """Main entry point."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Set log file name (based on output file if not specified)
        log_file = args.log_file
        if not log_file and args.output_file:
            # Use the output filename base with .log extension
            base_name = os.path.splitext(args.output_file)[0]
            log_file = f"{base_name}.log"
        
        # Configure logger with the log file
        global logger
        logger = setup_logging(__name__, log_to_file=True, log_file=log_file)
        
        # Configure log level based on verbose flag
        log_level = logging.DEBUG if args.verbose else logging.INFO
        for handler in logger.handlers:
            handler.setLevel(log_level)
        
        logger.info("Starting MCA Portfolio Simulator")
        
        # Run portfolio simulation
        if args.compare:
            run_comparative_analysis(args)
        else:
            run_single_portfolio(args)
        
        return 0
        
    except FileNotFoundError as e:
        if logger:
            logger.error(f"File not found: {str(e)}")
        else:
            print(f"File not found: {str(e)}")
        return 1
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        return 2
        
    except DataError as e:
        logger.error(f"Data error: {str(e)}")
        return 3
        
    except SimulationError as e:
        logger.error(f"Simulation error: {str(e)}")
        return 4
        
    except ExportError as e:
        logger.error(f"Export error: {str(e)}")
        return 5
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 99


if __name__ == "__main__":
    sys.exit(main())