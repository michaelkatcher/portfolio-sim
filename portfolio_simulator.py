#!/usr/bin/env python3
"""
Portfolio Simulator for MCA Portfolio Simulator

This module handles the simulation of MCA portfolios based on deal selection criteria,
allocation policies, and fee schedules.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from deal_selector import select_deals, validate_selection_criteria
from common_utils import setup_logging, SimulationError, ConfigurationError
from constants import DEAL_COLUMNS, PAYMENT_COLUMNS, TRANSACTION_TYPES, PORTFOLIO, VALIDATION

# Set up logger
logger = setup_logging(__name__)


class AllocationPolicy:
    """Defines how much of each deal to include in the portfolio."""
    
    def __init__(self, percentage=0.1, min_amount=None, max_amount=None):
        """
        Initialize the allocation policy.
        
        Args:
            percentage: Percentage of each deal to allocate (0.0 to 1.0)
            min_amount: Minimum allocation amount (or None for no minimum)
            max_amount: Maximum allocation amount (or None for no maximum)
            
        Raises:
            ConfigurationError: If percentage is not in the valid range
        """
        # Validate inputs
        if percentage <= 0 or percentage > 1:
            raise ConfigurationError("Allocation percentage must be greater than 0 and less than or equal to 1")
        
        if min_amount is not None and min_amount < 0:
            raise ConfigurationError("Minimum allocation amount cannot be negative")
            
        if max_amount is not None and max_amount < 0:
            raise ConfigurationError("Maximum allocation amount cannot be negative")
            
        if min_amount is not None and max_amount is not None and min_amount > max_amount:
            raise ConfigurationError("Minimum allocation amount cannot be greater than maximum")
        
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
        if deal_size <= 0:
            logger.warning(f"Received non-positive deal size: {deal_size}. Using 0 allocation.")
            return 0
            
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
            
        Raises:
            ConfigurationError: If percentage is not in the valid range
        """
        # Validate input
        if percentage < VALIDATION['FEE_MIN'] or percentage > VALIDATION['FEE_MAX']:
            raise ConfigurationError(f"Fee percentage must be between {VALIDATION['FEE_MIN']} and {VALIDATION['FEE_MAX']}")
        
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
                seasoned_date=PORTFOLIO['DEFAULT_SEASONED_DATE']):
        """
        Initialize a portfolio.
        
        Args:
            name: Name of the portfolio
            selection_criteria: Dict with criteria for selecting deals
            allocation_policy: AllocationPolicy instance
            fee_schedule: FeeSchedule instance
            seasoned_date: Date to use for seasoned metrics (deals funded before this date)
            
        Raises:
            ConfigurationError: If inputs are invalid
        """
        # Validate inputs
        if not name:
            raise ConfigurationError("Portfolio name cannot be empty")
        
        # Convert date string to datetime if needed
        try:
            self.seasoned_date = pd.to_datetime(seasoned_date)
        except Exception as e:
            raise ConfigurationError(f"Invalid seasoned date format: {e}")
        
        self.name = name
        self.selection_criteria = selection_criteria or {}
        
        # Use provided policy or create default
        if allocation_policy is None:
            allocation_policy = AllocationPolicy()
        elif not isinstance(allocation_policy, AllocationPolicy):
            raise ConfigurationError("allocation_policy must be an instance of AllocationPolicy")
            
        # Use provided schedule or create default
        if fee_schedule is None:
            fee_schedule = FeeSchedule()
        elif not isinstance(fee_schedule, FeeSchedule):
            raise ConfigurationError("fee_schedule must be an instance of FeeSchedule")
        
        self.allocation_policy = allocation_policy
        self.fee_schedule = fee_schedule
        
        # These will be populated during simulation
        self.selected_deals = None
        self.deal_allocations = None
        self.cashflows = None
    
    def simulate(self, deals_df, payments_df, verbose=False):
        """
        Simulate the portfolio based on the provided data.
        
        Args:
            deals_df: DataFrame containing deal information
            payments_df: DataFrame containing payment transactions
            verbose: Whether to print information during simulation
            
        Returns:
            self (for method chaining)
            
        Raises:
            SimulationError: If the simulation fails
            ValueError: If input data is invalid
        """
        # Validate input dataframes
        if not isinstance(deals_df, pd.DataFrame) or deals_df.empty:
            raise ValueError("deals_df must be a non-empty pandas DataFrame")
            
        if not isinstance(payments_df, pd.DataFrame) or payments_df.empty:
            raise ValueError("payments_df must be a non-empty pandas DataFrame")
        
        # Verify required columns
        required_deal_cols = [DEAL_COLUMNS['ID'], DEAL_COLUMNS['BALANCE'], DEAL_COLUMNS['RTR']]
        required_payment_cols = [PAYMENT_COLUMNS['ID'], PAYMENT_COLUMNS['DATE'], 
                                PAYMENT_COLUMNS['AMOUNT'], PAYMENT_COLUMNS['DESCRIPTION']]
        
        missing_deal_cols = [col for col in required_deal_cols if col not in deals_df.columns]
        if missing_deal_cols:
            raise ValueError(f"Missing required columns in deals_df: {missing_deal_cols}")
            
        missing_payment_cols = [col for col in required_payment_cols if col not in payments_df.columns]
        if missing_payment_cols:
            raise ValueError(f"Missing required columns in payments_df: {missing_payment_cols}")
        
        try:
            if verbose:
                logger.info(f"Simulating portfolio: {self.name}")
                logger.info(f"Selection criteria: {self.selection_criteria}")
                logger.info(f"Allocation policy: {self.allocation_policy}")
                logger.info(f"Fee schedule: {self.fee_schedule}")
            
            # Step 1: Select deals based on criteria
            self.selected_deals = select_deals(deals_df, **self.selection_criteria, verbose=verbose)
            
            if self.selected_deals.empty:
                logger.warning("No deals selected based on criteria.")
                return self
            
            # Step 2: Calculate allocations for each deal
            self._calculate_allocations(verbose)
            
            # Step 3: Generate cashflows
            self._generate_cashflows(payments_df, verbose)
            
            return self
            
        except Exception as e:
            # Wrap any unexpected exceptions
            raise SimulationError(f"Portfolio simulation failed: {str(e)}") from e
    
    def _calculate_allocations(self, verbose=False):
        """
        Calculate allocations for each selected deal.
        
        Args:
            verbose: Whether to print detailed information
            
        Raises:
            SimulationError: If allocation calculation fails
        """
        if self.selected_deals is None:
            raise SimulationError("No deals selected. Run simulate() first.")
        
        try:
            # Create a copy of selected deals
            self.deal_allocations = self.selected_deals.copy()
            
            # Calculate allocation for each deal
            self.deal_allocations['Allocation Amount'] = self.deal_allocations[DEAL_COLUMNS['BALANCE']].apply(
                self.allocation_policy.calculate_allocation
            )
            
            # Calculate allocation percentage
            self.deal_allocations['Allocation Percentage'] = (
                self.deal_allocations['Allocation Amount'] / self.deal_allocations[DEAL_COLUMNS['BALANCE']]
            )
            
            # Calculate allocated RTR
            self.deal_allocations['Allocated RTR'] = (
                self.deal_allocations[DEAL_COLUMNS['RTR']] * self.deal_allocations['Allocation Percentage']
            )
            
            if verbose:
                total_allocation = self.deal_allocations['Allocation Amount'].sum()
                avg_percentage = self.deal_allocations['Allocation Percentage'].mean() * 100
                
                logger.info(f"\nAllocation summary:")
                logger.info(f"  Total deals: {len(self.deal_allocations)}")
                logger.info(f"  Total allocation: ${total_allocation:,.2f}")
                logger.info(f"  Average allocation: ${self.deal_allocations['Allocation Amount'].mean():,.2f}")
                logger.info(f"  Average allocation percentage: {avg_percentage:.2f}%")
                
        except Exception as e:
            raise SimulationError(f"Error calculating allocations: {str(e)}") from e
    
    def _generate_cashflows(self, payments_df, verbose=False):
        """
        Generate cashflows for the portfolio based on allocations.
        
        Args:
            payments_df: DataFrame containing payment transactions
            verbose: Whether to print detailed information
            
        Raises:
            SimulationError: If cashflow generation fails
        """
        if self.deal_allocations is None:
            raise SimulationError("No allocations calculated. Run simulate() first.")
        
        try:
            # Step 1: Get payments for selected deals
            deal_payments = self._get_deal_payments(payments_df, verbose)
            
            if deal_payments.empty:
                logger.warning("No payment records found for selected deals.")
                self.cashflows = pd.DataFrame(columns=[
                    PAYMENT_COLUMNS['ID'], PAYMENT_COLUMNS['DATE'], PAYMENT_COLUMNS['DESCRIPTION'], 
                    'Portfolio Amount', 'Fee Amount', 'Net Amount', 'Principal Repaid'
                ])
                return
            
            # Step 2: Apply allocation percentages
            deal_payments = self._apply_allocation_percentages(deal_payments)
            
            # Step 3: Track principal repayment
            deal_payments = self._track_principal_repayment(deal_payments)
            
            # Step 4: Calculate fees
            deal_payments = self._calculate_fees(deal_payments)
            
            # Step 5: Finalize cashflows
            self._finalize_cashflows(deal_payments, verbose)
            
        except Exception as e:
            raise SimulationError(f"Error generating cashflows: {str(e)}") from e

    def _get_deal_payments(self, payments_df, verbose=False):
        """
        Get payments for the selected deals.
        
        Args:
            payments_df: DataFrame containing payment transactions
            verbose: Whether to print detailed information
            
        Returns:
            DataFrame with payment data for selected deals
        """
        deal_ids = self.deal_allocations[DEAL_COLUMNS['ID']].tolist()
        
        # Add debug statements
        if verbose:
            logger.info(f"Looking for payments for {len(deal_ids)} deals")
            logger.info(f"Sample deal IDs: {deal_ids[:5] if len(deal_ids) >= 5 else deal_ids}")
            logger.info(f"Total payment records: {len(payments_df)}")
            
            # Check for exact matches
            matching_payments = payments_df[payments_df[PAYMENT_COLUMNS['ID']].isin(deal_ids)]
            logger.info(f"Found {len(matching_payments)} payment records for the selected deals")
            
            # Show transaction types
            if not matching_payments.empty:
                logger.info(f"Transaction types: {matching_payments[PAYMENT_COLUMNS['DESCRIPTION']].value_counts().to_dict()}")
        
        deal_payments = payments_df[payments_df[PAYMENT_COLUMNS['ID']].isin(deal_ids)].copy()
        return deal_payments

    def _apply_allocation_percentages(self, deal_payments):
        """
        Apply allocation percentages to deal payments.
        
        Args:
            deal_payments: DataFrame with payment data
            
        Returns:
            DataFrame with allocation percentages applied
        """
        # Create a mapping of deal IDs to allocation percentages
        allocation_map = self.deal_allocations[[DEAL_COLUMNS['ID'], 'Allocation Percentage']].set_index(DEAL_COLUMNS['ID'])['Allocation Percentage'].to_dict()
        
        # Add allocation percentage column using vectorized mapping
        deal_payments['Allocation Percentage'] = deal_payments[PAYMENT_COLUMNS['ID']].map(allocation_map)
        
        # Apply the allocation percentage to each payment (vectorized)
        deal_payments['Portfolio Amount'] = deal_payments[PAYMENT_COLUMNS['AMOUNT']] * deal_payments['Allocation Percentage']
        
        return deal_payments

    def _track_principal_repayment(self, deal_payments):
        """
        Track principal repayment for each deal.
        
        Args:
            deal_payments: DataFrame with payment data
            
        Returns:
            DataFrame with principal repayment tracking
        """
        # Create a mapping of deal IDs to allocated amounts
        principal_map = self.deal_allocations[[DEAL_COLUMNS['ID'], 'Allocation Amount']].set_index(DEAL_COLUMNS['ID'])['Allocation Amount'].to_dict()
        
        # Sort by deal and date
        deal_payments = deal_payments.sort_values([PAYMENT_COLUMNS['ID'], PAYMENT_COLUMNS['DATE']])
        
        # Track incoming cashflows
        incoming_mask = deal_payments['Portfolio Amount'] > 0
        deal_payments['Incoming Amount'] = deal_payments['Portfolio Amount'].where(incoming_mask, 0)
        deal_payments['Cumulative Incoming'] = deal_payments.groupby(PAYMENT_COLUMNS['ID'])['Incoming Amount'].transform('cumsum')
        
        # Map each deal to its principal amount
        deal_payments['Deal Principal'] = deal_payments[PAYMENT_COLUMNS['ID']].map(
            {k: abs(v) for k, v in principal_map.items()}
        )
        
        # Determine if principal has been repaid
        deal_payments['Principal Repaid'] = deal_payments['Cumulative Incoming'] >= deal_payments['Deal Principal']
        
        return deal_payments

    def _calculate_fees(self, deal_payments):
        """
        Calculate fees for each cashflow.
        
        Args:
            deal_payments: DataFrame with payment data
            
        Returns:
            DataFrame with fees calculated
        """
        # Initialize fee column
        deal_payments['Fee Amount'] = 0.0
        
        # Determine which cashflows should have fees applied
        if self.fee_schedule.apply_after_principal:
            # Only apply fees to incoming cashflows after principal repayment
            fee_mask = (deal_payments['Portfolio Amount'] > 0) & deal_payments['Principal Repaid']
        else:
            # Apply fees to all incoming cashflows
            fee_mask = deal_payments['Portfolio Amount'] > 0
        
        # Calculate fees for applicable cashflows
        deal_payments.loc[fee_mask, 'Fee Amount'] = deal_payments.loc[fee_mask, 'Portfolio Amount'] * self.fee_schedule.percentage
        
        # Calculate net amount
        deal_payments['Net Amount'] = deal_payments['Portfolio Amount'] - deal_payments['Fee Amount']
        
        return deal_payments

    def _finalize_cashflows(self, deal_payments, verbose=False):
        """
        Store the final cashflows and display summary if verbose.
        
        Args:
            deal_payments: DataFrame with payment data
            verbose: Whether to print detailed information
        """
        # Select required columns for the cashflows dataframe
        self.cashflows = deal_payments[[PAYMENT_COLUMNS['ID'], PAYMENT_COLUMNS['DATE'], PAYMENT_COLUMNS['DESCRIPTION'], 
                                      'Portfolio Amount', 'Fee Amount', 'Net Amount', 'Principal Repaid']]
        
        if verbose:
            total_in = self.cashflows[self.cashflows['Portfolio Amount'] > 0]['Portfolio Amount'].sum()
            total_out = abs(self.cashflows[self.cashflows['Portfolio Amount'] < 0]['Portfolio Amount'].sum())
            total_fees = self.cashflows['Fee Amount'].sum()
            
            logger.info(f"Cashflow summary:")
            logger.info(f"  Total inflows: ${total_in:,.2f}")
            logger.info(f"  Total outflows: ${total_out:,.2f}")
            logger.info(f"  Total fees: ${total_fees:,.2f}")
            logger.info(f"  Net cashflow: ${total_in - total_out - total_fees:,.2f}")
    
    def get_summary(self):
        """
        Get a summary of the portfolio configuration.
        
        Returns:
            Dict with portfolio configuration information
            
        Raises:
            SimulationError: If portfolio has not been simulated
        """
        if self.deal_allocations is None:
            raise SimulationError("Portfolio has not been simulated. Run simulate() first.")
            
        return {
            'name': self.name,
            'selection_criteria': self.selection_criteria,
            'allocation_policy': str(self.allocation_policy),
            'fee_schedule': str(self.fee_schedule),
            'seasoned_date': self.seasoned_date.strftime('%Y-%m-%d'),
            'deal_count': len(self.deal_allocations) if self.deal_allocations is not None else 0,
            'total_allocation': self.deal_allocations['Allocation Amount'].sum() if self.deal_allocations is not None else 0
        }
    
    def __str__(self):
        """Return a string representation of the portfolio."""
        if self.selected_deals is None:
            return f"Portfolio: {self.name} (not simulated)"
        
        if self.deal_allocations is None:
            return f"Portfolio: {self.name} (no allocations calculated)"
        
        total_allocation = self.deal_allocations['Allocation Amount'].sum()
        deal_count = len(self.deal_allocations)
        
        return (
            f"Portfolio: {self.name}\n\n"
            f"Selection Criteria: {self.selection_criteria}\n"
            f"Allocation Policy: {self.allocation_policy}\n"
            f"Fee Schedule: {self.fee_schedule}\n\n"
            f"PORTFOLIO SUMMARY:\n"
            f"  Deals: {deal_count}\n"
            f"  Total Allocation: ${total_allocation:,.2f}\n"
        )