"""
Data Quality and Validation Module
Implements comprehensive data quality checks, validation rules, and data cleaning
Ensures data integrity throughout the transformation pipeline
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityCheckType(Enum):
    """Types of quality checks"""
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"


class QualityCheckSeverity(Enum):
    """Severity levels for quality issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityRule:
    """Definition of a data quality rule"""
    name: str
    description: str
    check_type: QualityCheckType
    severity: QualityCheckSeverity
    check_function: Callable
    threshold: float = 0.95  # Minimum pass rate
    columns: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class QualityCheckResult:
    """Result of a quality check"""
    rule_name: str
    check_type: QualityCheckType
    severity: QualityCheckSeverity
    passed: bool
    pass_rate: float
    failed_count: int
    total_count: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    dataset_name: str
    total_rows: int
    total_columns: int
    check_results: List[QualityCheckResult] = field(default_factory=list)
    overall_score: float = 0.0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_result(self, result: QualityCheckResult):
        """Add a quality check result"""
        self.check_results.append(result)
        
        # Update issue counts
        if not result.passed:
            if result.severity == QualityCheckSeverity.CRITICAL:
                self.critical_issues += 1
            elif result.severity == QualityCheckSeverity.HIGH:
                self.high_issues += 1
            elif result.severity == QualityCheckSeverity.MEDIUM:
                self.medium_issues += 1
            elif result.severity == QualityCheckSeverity.LOW:
                self.low_issues += 1
    
    def calculate_overall_score(self):
        """Calculate overall data quality score"""
        if not self.check_results:
            self.overall_score = 0.0
            return
        
        # Weighted scoring based on severity
        weights = {
            QualityCheckSeverity.CRITICAL: 1.0,
            QualityCheckSeverity.HIGH: 0.8,
            QualityCheckSeverity.MEDIUM: 0.6,
            QualityCheckSeverity.LOW: 0.4,
            QualityCheckSeverity.INFO: 0.2
        }
        
        total_weight = 0
        weighted_score = 0
        
        for result in self.check_results:
            weight = weights[result.severity]
            total_weight += weight
            weighted_score += result.pass_rate * weight
        
        self.overall_score = weighted_score / total_weight if total_weight > 0 else 0.0


class DataQualityChecker:
    """Main data quality checker class"""
    
    def __init__(self):
        self.rules: List[QualityRule] = []
        self._initialize_default_rules()
        logger.info("Data Quality Checker initialized")
    
    def _initialize_default_rules(self):
        """Initialize default quality rules"""
        
        # Completeness rules
        self.add_rule(QualityRule(
            name="no_null_values",
            description="Check for null/missing values",
            check_type=QualityCheckType.COMPLETENESS,
            severity=QualityCheckSeverity.HIGH,
            check_function=self._check_completeness,
            threshold=0.95
        ))
        
        self.add_rule(QualityRule(
            name="no_empty_strings",
            description="Check for empty string values",
            check_type=QualityCheckType.COMPLETENESS,
            severity=QualityCheckSeverity.MEDIUM,
            check_function=self._check_empty_strings,
            threshold=0.98
        ))
        
        # Validity rules
        self.add_rule(QualityRule(
            name="valid_email_format",
            description="Check email format validity",
            check_type=QualityCheckType.VALIDITY,
            severity=QualityCheckSeverity.HIGH,
            check_function=self._check_email_format,
            threshold=0.99
        ))
        
        self.add_rule(QualityRule(
            name="valid_date_format",
            description="Check date format validity",
            check_type=QualityCheckType.VALIDITY,
            severity=QualityCheckSeverity.HIGH,
            check_function=self._check_date_format,
            threshold=0.99
        ))
        
        self.add_rule(QualityRule(
            name="valid_rating_range",
            description="Check rating values are within valid range",
            check_type=QualityCheckType.VALIDITY,
            severity=QualityCheckSeverity.CRITICAL,
            check_function=self._check_rating_range,
            threshold=0.99
        ))
        
        # Consistency rules
        self.add_rule(QualityRule(
            name="consistent_data_types",
            description="Check data type consistency",
            check_type=QualityCheckType.CONSISTENCY,
            severity=QualityCheckSeverity.HIGH,
            check_function=self._check_data_types,
            threshold=0.99
        ))
        
        # Uniqueness rules
        self.add_rule(QualityRule(
            name="unique_identifiers",
            description="Check uniqueness of identifier columns",
            check_type=QualityCheckType.UNIQUENESS,
            severity=QualityCheckSeverity.CRITICAL,
            check_function=self._check_uniqueness,
            threshold=1.0
        ))
        
        # Timeliness rules
        self.add_rule(QualityRule(
            name="reasonable_dates",
            description="Check dates are within reasonable range",
            check_type=QualityCheckType.TIMELINESS,
            severity=QualityCheckSeverity.MEDIUM,
            check_function=self._check_date_range,
            threshold=0.95
        ))
    
    def add_rule(self, rule: QualityRule):
        """Add a quality rule"""
        self.rules.append(rule)
        logger.debug(f"Added quality rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a quality rule"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        logger.debug(f"Removed quality rule: {rule_name}")
    
    def check_data_quality(self, df: pd.DataFrame, dataset_name: str = "dataset") -> DataQualityReport:
        """Perform comprehensive data quality checks"""
        logger.info(f"Starting data quality checks for {dataset_name}")
        
        if df is None:
            logger.error(f"DataFrame is None for dataset {dataset_name}")
            return DataQualityReport(
                dataset_name=dataset_name,
                total_rows=0,
                total_columns=0
            )
        
        report = DataQualityReport(
            dataset_name=dataset_name,
            total_rows=len(df),
            total_columns=len(df.columns)
        )
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                result = self._execute_rule(df, rule)
                report.add_result(result)
                
                if not result.passed:
                    logger.warning(f"Quality check failed: {rule.name} - {result.details}")
                
            except Exception as e:
                logger.error(f"Error executing rule {rule.name}: {e}")
                # Add failed result
                result = QualityCheckResult(
                    rule_name=rule.name,
                    check_type=rule.check_type,
                    severity=rule.severity,
                    passed=False,
                    pass_rate=0.0,
                    failed_count=len(df),
                    total_count=len(df),
                    details={"error": str(e)}
                )
                report.add_result(result)
        
        report.calculate_overall_score()
        logger.info(f"Data quality check completed. Overall score: {report.overall_score:.2f}")
        
        return report
    
    def _execute_rule(self, df: pd.DataFrame, rule: QualityRule) -> QualityCheckResult:
        """Execute a single quality rule"""
        try:
            # Execute the check function
            passed_mask, details = rule.check_function(df, rule)
            
            # Calculate metrics
            if isinstance(passed_mask, bool):
                # Global check result
                passed = passed_mask
                pass_rate = 1.0 if passed else 0.0
                failed_count = 0 if passed else len(df)
                total_count = len(df)
            else:
                # Row-level check result
                total_count = len(passed_mask)
                passed_count = passed_mask.sum()
                failed_count = total_count - passed_count
                pass_rate = passed_count / total_count if total_count > 0 else 0.0
                passed = pass_rate >= rule.threshold
            
            return QualityCheckResult(
                rule_name=rule.name,
                check_type=rule.check_type,
                severity=rule.severity,
                passed=passed,
                pass_rate=pass_rate,
                failed_count=failed_count,
                total_count=total_count,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Error in rule execution {rule.name}: {e}")
            raise
    
    # Quality check functions
    def _check_completeness(self, df: pd.DataFrame, rule: QualityRule) -> Tuple[pd.Series, Dict]:
        """Check for missing values"""
        if rule.columns:
            columns_to_check = [col for col in rule.columns if col in df.columns]
        else:
            columns_to_check = df.columns.tolist()
        
        if not columns_to_check:
            return pd.Series([True] * len(df)), {"message": "No columns to check"}
        
        # Check for null values across specified columns
        not_null_mask = df[columns_to_check].notnull().all(axis=1)
        
        details = {
            "columns_checked": columns_to_check,
            "null_counts": df[columns_to_check].isnull().sum().to_dict(),
            "total_nulls": df[columns_to_check].isnull().sum().sum()
        }
        
        return not_null_mask, details
    
    def _check_empty_strings(self, df: pd.DataFrame, rule: QualityRule) -> Tuple[pd.Series, Dict]:
        """Check for empty string values"""
        if rule.columns:
            columns_to_check = [col for col in rule.columns if col in df.columns]
        else:
            # Only check string columns
            columns_to_check = df.select_dtypes(include=['object']).columns.tolist()
        
        if not columns_to_check:
            return pd.Series([True] * len(df)), {"message": "No string columns to check"}
        
        # Check for empty strings
        not_empty_mask = pd.Series([True] * len(df))
        empty_counts = {}
        
        for col in columns_to_check:
            col_not_empty = ~(df[col].astype(str).str.strip() == "")
            not_empty_mask &= col_not_empty
            empty_counts[col] = (~col_not_empty).sum()
        
        details = {
            "columns_checked": columns_to_check,
            "empty_counts": empty_counts,
            "total_empty": sum(empty_counts.values())
        }
        
        return not_empty_mask, details
    
    def _check_email_format(self, df: pd.DataFrame, rule: QualityRule) -> Tuple[pd.Series, Dict]:
        """Check email format validity"""
        email_columns = []
        
        if rule.columns:
            email_columns = [col for col in rule.columns if col in df.columns]
        else:
            # Auto-detect email columns
            email_columns = [col for col in df.columns if 'email' in col.lower()]
        
        if not email_columns:
            return pd.Series([True] * len(df)), {"message": "No email columns found"}
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        valid_mask = pd.Series([True] * len(df))
        invalid_counts = {}
        
        for col in email_columns:
            col_valid = df[col].astype(str).str.match(email_pattern, na=False)
            valid_mask &= col_valid
            invalid_counts[col] = (~col_valid).sum()
        
        details = {
            "columns_checked": email_columns,
            "invalid_counts": invalid_counts,
            "pattern_used": email_pattern
        }
        
        return valid_mask, details
    
    def _check_date_format(self, df: pd.DataFrame, rule: QualityRule) -> Tuple[pd.Series, Dict]:
        """Check date format validity"""
        date_columns = []
        
        if rule.columns:
            date_columns = [col for col in rule.columns if col in df.columns]
        else:
            # Auto-detect date columns
            date_columns = [col for col in df.columns 
                          if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated'])]
        
        if not date_columns:
            return pd.Series([True] * len(df)), {"message": "No date columns found"}
        
        valid_mask = pd.Series([True] * len(df))
        invalid_counts = {}
        
        for col in date_columns:
            try:
                pd.to_datetime(df[col], errors='coerce')
                col_valid = pd.to_datetime(df[col], errors='coerce').notnull()
                valid_mask &= col_valid
                invalid_counts[col] = (~col_valid).sum()
            except Exception:
                invalid_counts[col] = len(df)
                valid_mask = pd.Series([False] * len(df))
        
        details = {
            "columns_checked": date_columns,
            "invalid_counts": invalid_counts
        }
        
        return valid_mask, details
    
    def _check_rating_range(self, df: pd.DataFrame, rule: QualityRule) -> Tuple[pd.Series, Dict]:
        """Check rating values are within valid range (1-5)"""
        rating_columns = []
        
        if rule.columns:
            rating_columns = [col for col in rule.columns if col in df.columns]
        else:
            # Auto-detect rating columns
            rating_columns = [col for col in df.columns if 'rating' in col.lower()]
        
        if not rating_columns:
            return pd.Series([True] * len(df)), {"message": "No rating columns found"}
        
        valid_mask = pd.Series([True] * len(df))
        invalid_counts = {}
        
        for col in rating_columns:
            # Check if rating is between 1 and 5
            col_valid = (df[col] >= 1) & (df[col] <= 5) & df[col].notnull()
            valid_mask &= col_valid
            invalid_counts[col] = (~col_valid).sum()
        
        details = {
            "columns_checked": rating_columns,
            "invalid_counts": invalid_counts,
            "valid_range": "[1, 5]"
        }
        
        return valid_mask, details
    
    def _check_data_types(self, df: pd.DataFrame, rule: QualityRule) -> Tuple[bool, Dict]:
        """Check data type consistency"""
        type_issues = []
        
        for col in df.columns:
            # Check for mixed types in object columns
            if df[col].dtype == 'object':
                unique_types = set(type(val).__name__ for val in df[col].dropna().iloc[:100])
                if len(unique_types) > 1:
                    type_issues.append({
                        "column": col,
                        "types_found": list(unique_types)
                    })
        
        passed = len(type_issues) == 0
        
        details = {
            "type_issues": type_issues,
            "total_issues": len(type_issues)
        }
        
        return passed, details
    
    def _check_uniqueness(self, df: pd.DataFrame, rule: QualityRule) -> Tuple[pd.Series, Dict]:
        """Check uniqueness of identifier columns"""
        id_columns = []
        
        if rule.columns:
            id_columns = [col for col in rule.columns if col in df.columns]
        else:
            # Auto-detect ID columns
            id_columns = [col for col in df.columns if col.lower().endswith('_id') or col.lower() == 'id']
        
        if not id_columns:
            return pd.Series([True] * len(df)), {"message": "No ID columns found"}
        
        unique_mask = pd.Series([True] * len(df))
        duplicate_counts = {}
        
        for col in id_columns:
            col_unique = ~df[col].duplicated()
            unique_mask &= col_unique
            duplicate_counts[col] = df[col].duplicated().sum()
        
        details = {
            "columns_checked": id_columns,
            "duplicate_counts": duplicate_counts,
            "total_duplicates": sum(duplicate_counts.values())
        }
        
        return unique_mask, details
    
    def _check_date_range(self, df: pd.DataFrame, rule: QualityRule) -> Tuple[pd.Series, Dict]:
        """Check dates are within reasonable range"""
        date_columns = []
        
        if rule.columns:
            date_columns = [col for col in rule.columns if col in df.columns]
        else:
            # Auto-detect date columns
            date_columns = [col for col in df.columns 
                          if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated'])]
        
        if not date_columns:
            return pd.Series([True] * len(df)), {"message": "No date columns found"}
        
        # Define reasonable date range (1900 to 1 year in the future)
        min_date = datetime(1900, 1, 1)
        max_date = datetime.now() + timedelta(days=365)
        
        valid_mask = pd.Series([True] * len(df))
        invalid_counts = {}
        
        for col in date_columns:
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                col_valid = (dates >= min_date) & (dates <= max_date) & dates.notnull()
                valid_mask &= col_valid
                invalid_counts[col] = (~col_valid).sum()
            except Exception:
                invalid_counts[col] = len(df)
                valid_mask = pd.Series([False] * len(df))
        
        details = {
            "columns_checked": date_columns,
            "invalid_counts": invalid_counts,
            "valid_range": f"[{min_date.date()}, {max_date.date()}]"
        }
        
        return valid_mask, details


class DataCleaner:
    """Data cleaning and preprocessing utilities"""
    
    def __init__(self):
        logger.info("Data Cleaner initialized")
    
    def clean_dataset(self, df: pd.DataFrame, cleaning_config: Dict[str, Any] = None) -> pd.DataFrame:
        """Apply comprehensive data cleaning"""
        if df is None:
            logger.error("DataFrame is None, cannot perform cleaning")
            return pd.DataFrame()
        
        logger.info(f"Starting data cleaning for dataset with {len(df)} rows")
        
        cleaned_df = df.copy()
        cleaning_config = cleaning_config or {}
        
        # Remove duplicates
        if cleaning_config.get('remove_duplicates', True):
            cleaned_df = self.remove_duplicates(cleaned_df)
        
        # Handle missing values
        if cleaning_config.get('handle_missing', True):
            cleaned_df = self.handle_missing_values(cleaned_df, cleaning_config.get('missing_strategy', {}))
        
        # Clean text columns
        if cleaning_config.get('clean_text', True):
            cleaned_df = self.clean_text_columns(cleaned_df)
        
        # Standardize formats
        if cleaning_config.get('standardize_formats', True):
            cleaned_df = self.standardize_formats(cleaned_df)
        
        # Remove outliers
        if cleaning_config.get('remove_outliers', False):
            cleaned_df = self.remove_outliers(cleaned_df, cleaning_config.get('outlier_config', {}))
        
        logger.info(f"Data cleaning completed. Final dataset: {len(cleaned_df)} rows")
        return cleaned_df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_count = len(df)
        cleaned_df = df.drop_duplicates(subset=subset)
        removed_count = initial_count - len(cleaned_df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
        
        return cleaned_df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """Handle missing values based on strategy"""
        strategy = strategy or {}
        cleaned_df = df.copy()
        
        for column in cleaned_df.columns:
            if cleaned_df[column].isnull().any():
                col_strategy = strategy.get(column, 'auto')
                
                if col_strategy == 'drop':
                    cleaned_df = cleaned_df.dropna(subset=[column])
                elif col_strategy == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif col_strategy == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif col_strategy == 'mode':
                    mode_value = cleaned_df[column].mode().iloc[0] if not cleaned_df[column].mode().empty else 'Unknown'
                    cleaned_df[column].fillna(mode_value, inplace=True)
                elif col_strategy == 'auto':
                    # Auto-select strategy based on data type
                    if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                    else:
                        cleaned_df[column].fillna('Unknown', inplace=True)
                else:
                    # Use provided value
                    cleaned_df[column].fillna(col_strategy, inplace=True)
        
        return cleaned_df
    
    def clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns"""
        cleaned_df = df.copy()
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        
        for column in text_columns:
            if cleaned_df[column].dtype == 'object':
                # Remove extra whitespace
                cleaned_df[column] = cleaned_df[column].astype(str).str.strip()
                
                # Replace multiple spaces with single space
                cleaned_df[column] = cleaned_df[column].str.replace(r'\s+', ' ', regex=True)
                
                # Handle common text issues
                cleaned_df[column] = cleaned_df[column].str.replace(r'[^\w\s.,!?-]', '', regex=True)
                
                # Convert empty strings to NaN
                cleaned_df[column] = cleaned_df[column].replace('', np.nan)
        
        return cleaned_df
    
    def standardize_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data formats"""
        cleaned_df = df.copy()
        
        # Standardize date columns
        date_columns = [col for col in cleaned_df.columns 
                       if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated'])]
        
        for col in date_columns:
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
            except Exception:
                logger.warning(f"Could not convert {col} to datetime")
        
        # Standardize email columns
        email_columns = [col for col in cleaned_df.columns if 'email' in col.lower()]
        for col in email_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
        
        return cleaned_df
    
    def remove_outliers(self, df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        config = config or {}
        cleaned_df = df.copy()
        
        numerical_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for column in numerical_columns:
            if column in config.get('exclude_columns', []):
                continue
            
            Q1 = cleaned_df[column].quantile(0.25)
            Q3 = cleaned_df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers
            outlier_mask = (cleaned_df[column] < lower_bound) | (cleaned_df[column] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                cleaned_df = cleaned_df[~outlier_mask]
                logger.info(f"Removed {outlier_count} outliers from column {column}")
        
        return cleaned_df


def main():
    """Demonstrate data quality and validation functionality"""
    logger.info("🔍 Demonstrating Data Quality and Validation Module")
    
    # Create sample data with quality issues
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = []
    for i in range(n_samples):
        # Introduce various quality issues
        sample_data.append({
            'review_id': i + 1 if np.random.random() > 0.01 else None,  # 1% missing IDs
            'user_id': np.random.randint(1, 101) if np.random.random() > 0.02 else None,  # 2% missing user IDs
            'product_id': np.random.randint(1, 51),
            'rating': np.random.choice([1, 2, 3, 4, 5, 6, 0]) if np.random.random() > 0.05 else None,  # Invalid ratings
            'review_text': f"This product is great! " * np.random.randint(1, 5) if np.random.random() > 0.03 else "",  # Empty text
            'user_email': f"user{i}@example.com" if np.random.random() > 0.05 else "invalid-email",  # Invalid emails
            'created_at': datetime.now() - timedelta(days=np.random.randint(-10, 365)) if np.random.random() > 0.02 else None,  # Future dates
            'helpful_votes': np.random.randint(0, 20) if np.random.random() > 0.01 else None
        })
    
    # Add some duplicates
    for i in range(10):
        sample_data.append(sample_data[i].copy())
    
    df = pd.DataFrame(sample_data)
    
    print(f"\n📊 Original dataset shape: {df.shape}")
    print(f"Sample data quality issues introduced:")
    print(f"  • Missing values: ~1-3% per column")
    print(f"  • Invalid ratings: ~5%")
    print(f"  • Invalid emails: ~5%")
    print(f"  • Empty text: ~3%")
    print(f"  • Future dates: ~2%")
    print(f"  • Duplicates: 10 rows")
    
    # Initialize quality checker
    quality_checker = DataQualityChecker()
    
    # Configure rules for specific columns
    for rule in quality_checker.rules:
        if rule.name == "valid_rating_range":
            rule.columns = ['rating']
        elif rule.name == "valid_email_format":
            rule.columns = ['user_email']
        elif rule.name == "unique_identifiers":
            rule.columns = ['review_id']
        elif rule.name == "valid_date_format":
            rule.columns = ['created_at']
        elif rule.name == "reasonable_dates":
            rule.columns = ['created_at']
    
    # Perform quality checks
    print("\n🔍 Performing data quality checks...")
    quality_report = quality_checker.check_data_quality(df, "Product Reviews")
    
    # Display quality report
    print(f"\n📋 Data Quality Report:")
    print(f"  • Dataset: {quality_report.dataset_name}")
    print(f"  • Total rows: {quality_report.total_rows:,}")
    print(f"  • Total columns: {quality_report.total_columns}")
    print(f"  • Overall quality score: {quality_report.overall_score:.2f}")
    print(f"  • Critical issues: {quality_report.critical_issues}")
    print(f"  • High issues: {quality_report.high_issues}")
    print(f"  • Medium issues: {quality_report.medium_issues}")
    print(f"  • Low issues: {quality_report.low_issues}")
    
    print(f"\n📊 Quality Check Results:")
    for result in quality_report.check_results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  • {result.rule_name}: {status} ({result.pass_rate:.1%})")
        if not result.passed and result.details:
            if 'total_nulls' in result.details:
                print(f"    - Total nulls: {result.details['total_nulls']}")
            if 'invalid_counts' in result.details:
                print(f"    - Invalid values: {sum(result.details['invalid_counts'].values())}")
            if 'total_duplicates' in result.details:
                print(f"    - Duplicates: {result.details['total_duplicates']}")
    
    # Data cleaning demonstration
    print(f"\n🧹 Performing data cleaning...")
    data_cleaner = DataCleaner()
    
    cleaning_config = {
        'remove_duplicates': True,
        'handle_missing': True,
        'clean_text': True,
        'standardize_formats': True,
        'missing_strategy': {
            'rating': 'median',
            'user_email': 'drop',
            'review_text': 'Unknown'
        }
    }
    
    cleaned_df = data_cleaner.clean_dataset(df, cleaning_config)
    
    print(f"\n📈 Cleaning Results:")
    print(f"  • Original rows: {len(df):,}")
    print(f"  • Cleaned rows: {len(cleaned_df):,}")
    print(f"  • Rows removed: {len(df) - len(cleaned_df):,}")
    
    # Re-run quality checks on cleaned data
    print(f"\n🔍 Re-checking quality after cleaning...")
    cleaned_quality_report = quality_checker.check_data_quality(cleaned_df, "Cleaned Product Reviews")
    
    print(f"\n📊 Cleaned Data Quality Report:")
    print(f"  • Overall quality score: {cleaned_quality_report.overall_score:.2f} (was {quality_report.overall_score:.2f})")
    print(f"  • Critical issues: {cleaned_quality_report.critical_issues} (was {quality_report.critical_issues})")
    print(f"  • High issues: {cleaned_quality_report.high_issues} (was {quality_report.high_issues})")
    print(f"  • Medium issues: {cleaned_quality_report.medium_issues} (was {quality_report.medium_issues})")
    
    # Show improvement summary
    print(f"\n📈 Quality Improvement Summary:")
    score_improvement = cleaned_quality_report.overall_score - quality_report.overall_score
    print(f"  • Quality score improved by: {score_improvement:.2f}")
    
    critical_improvement = quality_report.critical_issues - cleaned_quality_report.critical_issues
    high_improvement = quality_report.high_issues - cleaned_quality_report.high_issues
    
    print(f"  • Critical issues resolved: {critical_improvement}")
    print(f"  • High issues resolved: {high_improvement}")
    
    print("\n✅ Data quality and validation demonstration completed!")


if __name__ == "__main__":
    main()