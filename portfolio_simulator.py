#!/usr/bin/env python3
"""
Portfolio Simulator for MCA Portfolio Simulator

This module handles the simulation of MCA portfolios based on deal selection criteria,
allocation policies, and fee schedules.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from deal_selector import select_deals, validate_selection_criteria


class AllocationPolicy:
    """Defines how much of each deal to include in the portfolio."""
    
    def __init__(self, percentage=0.1, min_amount=None, max_amount=None):
        """
        Initialize the allocation policy.
        
        Args:
            percentage: Percentage of each deal to allocate (0.0 to 1.0)
            min_amount: Minimum allocation amount (or None for no minimum)
            max_amount: Maximum allocation amount (or None for no maximum)
        """
        if percentage <= 0 or percentage > 1:
            raise ValueError("Percentage must be greater than 0 and less than or equal to 1")
        
        self.percentage = percentage
        self.min_amount = min_amount
        self.max_amount = max_amount
    
    def calculate_allocation(self, deal_size):
        """
        Calculate the allocation amount for a deal.
        
        Args:
            deal_size: Size of the deal
            
        Returns:
            Allocation amount
        """
        allocation = deal_size * self.percentage
        
        if self.min_amount is not None:
            allocation = max(self.min_amount, allocation)
        
        if self.max_amount is not None:
            allocation = min(self.max_amount, allocation)
        
        return allocation
    
    def __str__(self):
        """Return a string representation of the allocation policy."""
        result = f"{self.percentage:.1%} of each deal"
        
        if self.min_amount is not None:
            result += f", minimum ${self.min_amount:,.2f}"
        
        if self.max_amount is not None:
            result += f", maximum ${self.max_amount:,.2f}"
        
        return result


class FeeSchedule:
    """Defines how fees are applied to portfolio cashflows."""
    
    def __init__(self, percentage=0.0, apply_after_principal=True):
        """
        Initialize the fee schedule.
        
        Args:
            percentage: Fee percentage to apply to cashflows (0.0 to 1.0)
            apply_after_principal: Whether to apply fees only after principal is repaid
        """
        if percentage < 0 or percentage > 1:
            raise ValueError("Fee percentage must be between 0 and 1")
        
        self.percentage = percentage
        self.apply_after_principal = apply_after_principal
    
    def calculate_fee(self, cashflow_amount, principal_repaid=True):
        """
        Calculate the fee for a cashflow.
        
        Args:
            cashflow_amount: Amount of the cashflow
            principal_repaid: Whether principal has been repaid
            
        Returns:
            Fee amount
        """
        if cashflow_amount <= 0:
            return 0  # No fee on outgoing cashflows
        
        if self.apply_after_principal and not principal_repaid:
            return 0  # No fee when principal is not yet repaid
        
        return cashflow_amount * self.percentage
    
    def __str__(self):
        """Return a string representation of the fee schedule."""
        if self.percentage == 0:
            return "No fees"
        
        result = f"{self.percentage:.1%} fee"
        
        if self.apply_after_principal:
            result += " after principal repayment"
        else:
            result += " on all cashflows"
        
        return result


class Portfolio:
    """Represents a simulated MCA portfolio."""
    
    def __init__(self, name, selection_criteria=None, allocation_policy=None, fee_schedule=None, 
                seasoned_date='2024-01-31'):
        """
        Initialize a portfolio.
        
        Args:
            name: Name of the portfolio
            selection_criteria: Dict with criteria for selecting deals
            allocation_policy: AllocationPolicy instance
            fee_schedule: FeeSchedule instance
            seasoned_date: Date to use for seasoned metrics (deals funded before this date)
        """
        self.name = name
        self.selection_criteria = selection_criteria or {}
        self.allocation_policy = allocation_policy or AllocationPolicy()
        self.fee_schedule = fee_schedule or FeeSchedule()
        self.seasoned_date = pd.to_datetime(seasoned_date)
        
        # These will be populated during simulation
        self.selected_deals = None
        self.deal_allocations = None
        self.cashflows = None
        self.metrics = None
    
    def simulate(self, deals_df, payments_df, verbose=False):
        """
        Simulate the portfolio based on the provided data.
        
        Args:
            deals_df: DataFrame containing deal information
            payments_df: DataFrame containing payment transactions
            verbose: Whether to print information during simulation
            
        Returns:
            self (for method chaining)
        """
        if verbose:
            print(f"Simulating portfolio: {self.name}")
            print(f"Selection criteria: {self.selection_criteria}")
            print(f"Allocation policy: {self.allocation_policy}")
            print(f"Fee schedule: {self.fee_schedule}")
        
        # Step 1: Select deals based on criteria
        self.selected_deals = select_deals(deals_df, **self.selection_criteria, verbose=verbose)
        
        # Step 2: Calculate allocations for each deal
        self._calculate_allocations(verbose)
        
        # Step 3: Generate cashflows
        self._generate_cashflows(payments_df, verbose)
        
        # Step 4: Calculate portfolio metrics
        self._calculate_metrics(verbose)
        
        return self
    
    def _calculate_allocations(self, verbose=False):
        """Calculate allocations for each selected deal."""
        if self.selected_deals is None:
            raise ValueError("No deals selected. Run simulate() first.")
        
        # Create a copy of selected deals
        self.deal_allocations = self.selected_deals.copy()
        
        # Calculate allocation for each deal
        self.deal_allocations['Allocation Amount'] = self.deal_allocations['Total Original Balance'].apply(
            self.allocation_policy.calculate_allocation
        )
        
        # Calculate allocation percentage
        self.deal_allocations['Allocation Percentage'] = (
            self.deal_allocations['Allocation Amount'] / self.deal_allocations['Total Original Balance']
        )
        
        # Calculate allocated RTR
        self.deal_allocations['Allocated RTR'] = (
            self.deal_allocations['Total Original RTR'] * self.deal_allocations['Allocation Percentage']
        )
        
        if verbose:
            total_allocation = self.deal_allocations['Allocation Amount'].sum()
            avg_percentage = self.deal_allocations['Allocation Percentage'].mean() * 100
            
            print(f"\nAllocation summary:")
            print(f"  Total deals: {len(self.deal_allocations)}")
            print(f"  Total allocation: ${total_allocation:,.2f}")
            print(f"  Average allocation: ${self.deal_allocations['Allocation Amount'].mean():,.2f}")
            print(f"  Average allocation percentage: {avg_percentage:.2f}%")
    
    def _generate_cashflows(self, payments_df, verbose=False):
        """Generate cashflows for the portfolio based on allocations."""
        if self.deal_allocations is None:
            raise ValueError("No allocations calculated. Run simulate() first.")
        
        # Get all payments for the selected deals
        deal_ids = self.deal_allocations['Funded ID'].tolist()
        deal_payments = payments_df[payments_df['Funded ID'].isin(deal_ids)].copy()
        
        if verbose:
            print(f"\nGenerating cashflows from {len(deal_payments)} payment records")
        
        # Create a mapping of deal IDs to allocation percentages
        allocation_map = self.deal_allocations[['Funded ID', 'Allocation Percentage']].set_index('Funded ID')['Allocation Percentage'].to_dict()
        
        # Add allocation percentage column using vectorized mapping
        deal_payments['Allocation Percentage'] = deal_payments['Funded ID'].map(allocation_map)
        
        # Apply the allocation percentage to each payment (vectorized)
        deal_payments['Portfolio Amount'] = deal_payments['Transaction Amount'] * deal_payments['Allocation Percentage']
        
        # Create a mapping of deal IDs to allocated amounts (for tracking principal repayment)
        principal_map = self.deal_allocations[['Funded ID', 'Allocation Amount']].set_index('Funded ID')['Allocation Amount'].to_dict()
        
        # Add running totals to track principal repayment - vectorized approach
        deal_payments = deal_payments.sort_values(['Funded ID', 'Transaction Date'])
        
        # Use transform to create cumulative sums within each deal group
        # But we only want to count positive amounts (incoming cashflows)
        incoming_mask = deal_payments['Portfolio Amount'] > 0
        deal_payments['Incoming Amount'] = deal_payments['Portfolio Amount'].where(incoming_mask, 0)
        deal_payments['Cumulative Incoming'] = deal_payments.groupby('Funded ID')['Incoming Amount'].transform('cumsum')
        
        # Track which payments are after principal repayment (vectorized)
        # Create a Series mapping each deal to its principal amount
        deal_payments['Deal Principal'] = deal_payments['Funded ID'].map(
            {k: abs(v) for k, v in principal_map.items()}
        )
        
        # Determine if principal has been repaid
        deal_payments['Principal Repaid'] = deal_payments['Cumulative Incoming'] >= deal_payments['Deal Principal']
        
        # Calculate fees (vectorized)
        if self.fee_schedule.apply_after_principal:
            # Only apply fees to incoming cashflows after principal repayment
            fee_mask = (deal_payments['Portfolio Amount'] > 0) & deal_payments['Principal Repaid']
        else:
            # Apply fees to all incoming cashflows
            fee_mask = deal_payments['Portfolio Amount'] > 0
            
        deal_payments['Fee Amount'] = 0.0  # Initialize to zero
        deal_payments.loc[fee_mask, 'Fee Amount'] = deal_payments.loc[fee_mask, 'Portfolio Amount'] * self.fee_schedule.percentage
        
        # Calculate net amount
        deal_payments['Net Amount'] = deal_payments['Portfolio Amount'] - deal_payments['Fee Amount']
        
        # Store the cashflows
        self.cashflows = deal_payments[['Funded ID', 'Transaction Date', 'Transaction Description', 
                                        'Portfolio Amount', 'Fee Amount', 'Net Amount', 'Principal Repaid']]
        
        if verbose:
            total_in = self.cashflows[self.cashflows['Portfolio Amount'] > 0]['Portfolio Amount'].sum()
            total_out = abs(self.cashflows[self.cashflows['Portfolio Amount'] < 0]['Portfolio Amount'].sum())
            total_fees = self.cashflows['Fee Amount'].sum()
            
            print(f"Cashflow summary:")
            print(f"  Total inflows: ${total_in:,.2f}")
            print(f"  Total outflows: ${total_out:,.2f}")
            print(f"  Total fees: ${total_fees:,.2f}")
            print(f"  Net cashflow: ${total_in - total_out - total_fees:,.2f}")
    
    def _calculate_metrics(self, verbose=False):
        """Calculate performance metrics for the portfolio."""
        if self.cashflows is None:
            raise ValueError("No cashflows generated. Run simulate() first.")
        
        # Prepare metrics dictionary
        self.metrics = {
            'all': {},
            'seasoned': {}
        }
        
        # Calculate metrics for all deals
        self._calculate_metrics_for_subset(
            'all', 
            self.cashflows, 
            verbose
        )
        
        # Calculate metrics for seasoned deals only if we have funding date information
        if self.deal_allocations is not None and 'Initial Funding Date' in self.deal_allocations.columns:
            # Get seasoned deal IDs
            seasoned_deals = self.deal_allocations[
                self.deal_allocations['Initial Funding Date'] < self.seasoned_date
            ]['Funded ID'].tolist()
            
            # Filter cashflows for seasoned deals
            seasoned_cashflows = self.cashflows[self.cashflows['Funded ID'].isin(seasoned_deals)]
            
            if len(seasoned_cashflows) > 0:
                self._calculate_metrics_for_subset(
                    'seasoned', 
                    seasoned_cashflows, 
                    verbose
                )
            else:
                if verbose:
                    print("\nNo seasoned deals found based on date criteria.")
                # Initialize empty metrics for seasoned deals
                self.metrics['seasoned'] = {
                    'total_invested': 0,
                    'total_returned': 0,
                    'net_cashflow': 0,
                    'moic': float('nan'),
                    'irr': float('nan'),
                    'deal_count': 0
                }
        
    def _calculate_metrics_for_subset(self, metrics_key, cashflows_subset, verbose=False):
        """
        Calculate metrics for a subset of cashflows.
        
        Args:
            metrics_key: Key to store metrics under ('all' or 'seasoned')
            cashflows_subset: DataFrame of cashflows to analyze
            verbose: Whether to print detailed information
        """
        # Calculate by date for IRR calculation
        cf_by_date = cashflows_subset.groupby('Transaction Date')['Net Amount'].sum().reset_index()
        cf_by_date = cf_by_date.sort_values('Transaction Date')
        
        # Get unique deal count
        deal_count = cashflows_subset['Funded ID'].nunique()
        
        # Basic metrics
        total_in = cashflows_subset[cashflows_subset['Net Amount'] > 0]['Net Amount'].sum()
        total_out = abs(cashflows_subset[cashflows_subset['Net Amount'] < 0]['Net Amount'].sum())
        
        if total_out > 0:
            moic = total_in / total_out
        else:
            moic = float('nan')
        
        self.metrics[metrics_key] = {
            'total_invested': total_out,
            'total_returned': total_in,
            'net_cashflow': total_in - total_out,
            'moic': moic,
            'deal_count': deal_count
        }
        
        # Calculate IRR if possible
        cf_amounts = cf_by_date['Net Amount'].tolist()
        if len(cf_amounts) > 1 and not all(amt >= 0 for amt in cf_amounts):
            try:
                # Use numpy's IRR function
                irr = np.irr(cf_amounts)
                # Annualize the IRR
                annual_irr = (1 + irr) ** 365 - 1
                self.metrics[metrics_key]['irr'] = annual_irr
            except:
                self.metrics[metrics_key]['irr'] = float('nan')
        else:
            self.metrics[metrics_key]['irr'] = float('nan')
        
        if verbose:
            category = "All deals" if metrics_key == 'all' else f"Seasoned deals (before {self.seasoned_date.strftime('%Y-%m-%d')})"
            print(f"\n{category} performance metrics:")
            print(f"  Deal count: {self.metrics[metrics_key]['deal_count']}")
            print(f"  Total invested: ${self.metrics[metrics_key]['total_invested']:,.2f}")
            print(f"  Total returned: ${self.metrics[metrics_key]['total_returned']:,.2f}")
            print(f"  Net cashflow: ${self.metrics[metrics_key]['net_cashflow']:,.2f}")
            print(f"  MOIC: {self.metrics[metrics_key]['moic']:.2f}x")
            
            if not np.isnan(self.metrics[metrics_key].get('irr', float('nan'))):
                print(f"  IRR: {self.metrics[metrics_key]['irr']*100:.2f}%")
            else:
                print("  IRR: Unable to calculate")
    
    def get_summary(self):
        """
        Get a summary of the portfolio simulation.
        
        Returns:
            Dict with portfolio summary information
        """
        if self.metrics is None:
            raise ValueError("No metrics calculated. Run simulate() first.")
        
        return {
            'name': self.name,
            'all_deals': {
                'deal_count': self.metrics['all'].get('deal_count', 0),
                'total_invested': self.metrics['all']['total_invested'],
                'total_returned': self.metrics['all']['total_returned'],
                'net_cashflow': self.metrics['all']['net_cashflow'],
                'moic': self.metrics['all']['moic'],
                'irr': self.metrics['all'].get('irr', float('nan')),
            },
            'seasoned_deals': {
                'deal_count': self.metrics['seasoned'].get('deal_count', 0),
                'total_invested': self.metrics['seasoned'].get('total_invested', 0),
                'total_returned': self.metrics['seasoned'].get('total_returned', 0),
                'net_cashflow': self.metrics['seasoned'].get('net_cashflow', 0),
                'moic': self.metrics['seasoned'].get('moic', float('nan')),
                'irr': self.metrics['seasoned'].get('irr', float('nan')),
                'seasoned_date': self.seasoned_date.strftime('%Y-%m-%d')
            },
            'allocation_policy': str(self.allocation_policy),
            'fee_schedule': str(self.fee_schedule)
        }
    
    def __str__(self):
        """Return a string representation of the portfolio."""
        if self.metrics is None:
            return f"Portfolio: {self.name} (not simulated)"
        
        summary = self.get_summary()
        
        # Format IRR for display
        all_irr = f"{summary['all_deals']['irr']*100:.2f}%" if not np.isnan(summary['all_deals']['irr']) else "N/A"
        seasoned_irr = f"{summary['seasoned_deals']['irr']*100:.2f}%" if not np.isnan(summary['seasoned_deals']['irr']) else "N/A"
        
        return (
            f"Portfolio: {summary['name']}\n\n"
            f"ALL DEALS:\n"
            f"  Deals: {summary['all_deals']['deal_count']}\n"
            f"  Investment: ${summary['all_deals']['total_invested']:,.2f}\n"
            f"  Returned: ${summary['all_deals']['total_returned']:,.2f}\n"
            f"  Net: ${summary['all_deals']['net_cashflow']:,.2f}\n"
            f"  MOIC: {summary['all_deals']['moic']:.2f}x\n"
            f"  IRR: {all_irr}\n\n"
            f"SEASONED DEALS (before {summary['seasoned_deals']['seasoned_date']}):\n"
            f"  Deals: {summary['seasoned_deals']['deal_count']}\n"
            f"  Investment: ${summary['seasoned_deals']['total_invested']:,.2f}\n"
            f"  Returned: ${summary['seasoned_deals']['total_returned']:,.2f}\n"
            f"  Net: ${summary['seasoned_deals']['net_cashflow']:,.2f}\n"
            f"  MOIC: {summary['seasoned_deals']['moic']:.2f}x\n"
            f"  IRR: {seasoned_irr}\n"
        )


def compare_portfolios(portfolios, metrics_type='all'):
    """
    Compare multiple portfolio simulations.
    
    Args:
        portfolios: List of Portfolio instances
        metrics_type: Type of metrics to compare ('all' or 'seasoned')
        
    Returns:
        DataFrame with comparison of portfolios
    """
    if not portfolios:
        raise ValueError("No portfolios provided")
    
    if metrics_type not in ['all', 'seasoned']:
        raise ValueError("metrics_type must be 'all' or 'seasoned'")
    
    # Get summary for each portfolio
    summaries = []
    for portfolio in portfolios:
        try:
            summary = portfolio.get_summary()
            
            # Add portfolio name
            metrics_data = summary[f'{metrics_type}_deals'].copy()
            metrics_data['name'] = summary['name']
            metrics_data['allocation_policy'] = summary['allocation_policy']
            metrics_data['fee_schedule'] = summary['fee_schedule']
            
            summaries.append(metrics_data)
        except ValueError:
            # Skip portfolios that haven't been simulated
            continue
    
    if not summaries:
        raise ValueError("None of the provided portfolios have been simulated")
    
    # Create a DataFrame for comparison
    df = pd.DataFrame(summaries)
    
    # Reorder and format columns
    columns = ['name', 'deal_count', 'total_invested', 'total_returned', 
               'net_cashflow', 'moic', 'irr', 'allocation_policy', 'fee_schedule']
    
    if metrics_type == 'seasoned':
        columns.append('seasoned_date')
    
    # Ensure all columns exist
    available_columns = [col for col in columns if col in df.columns]
    
    df = df[available_columns]
    
    # Rename columns for display
    column_mapping = {
        'name': 'Name', 
        'deal_count': 'Deals', 
        'total_invested': 'Invested', 
        'total_returned': 'Returned', 
        'net_cashflow': 'Net Cashflow', 
        'moic': 'MOIC', 
        'irr': 'IRR', 
        'allocation_policy': 'Allocation Policy', 
        'fee_schedule': 'Fee Schedule',
        'seasoned_date': 'Seasoned Date'
    }
    
    df.columns = [column_mapping.get(col, col) for col in df.columns]
    
    # Format numeric columns
    if 'IRR' in df.columns:
        df['IRR'] = df['IRR'] * 100  # Convert to percentage
    
    return df


if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    
    # Load data
    deals_df, payments_df = load_data()
    
    # Create a portfolio
    portfolio = Portfolio(
        name="Sample Portfolio",
        selection_criteria={
            'product_types': ['RBF'],
            'deal_size_range': (100000, 500000),
            'vintage_range': ('2023-01-01', '2023-12-31')
        },
        allocation_policy=AllocationPolicy(
            percentage=0.1, 
            min_amount=50000, 
            max_amount=200000
        ),
        fee_schedule=FeeSchedule(
            percentage=0.02,
            apply_after_principal=True
        )
    )
    
    # Simulate the portfolio
    portfolio.simulate(deals_df, payments_df, verbose=True)
    
    # Print portfolio summary
    print("\nPortfolio Summary:")
    print(portfolio)