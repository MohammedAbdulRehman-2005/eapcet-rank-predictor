"""
EAPCET Analytics System - Configuration File
Centralized configuration for easy customization.
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Total number of candidates (for percentile calculation)
# Updated with official TS EAPCET 2025 data
TOTAL_CANDIDATES = 151_779  # Qualified candidates (2025 Engineering Stream)
TOTAL_APPEARED = 207_190    # Total appeared candidates
QUALIFYING_PERCENTAGE = 73.26  # Qualification rate

# Model hyperparameters
MODEL_CONFIG = {
    'n_estimators': 100,        # Number of boosting stages
    'max_depth': 5,             # Maximum depth of trees
    'learning_rate': 0.1,       # Learning rate for boosting
    'random_state': 42,         # Random seed for reproducibility
    'test_size': 0.2,          # Train-test split ratio
}

# Quantiles to train (for confidence intervals)
QUANTILES = [0.10, 0.50, 0.90]  # 10th, 50th (median), 90th percentile


# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================

# Percentile thresholds for performance labels
PERFORMANCE_THRESHOLDS = {
    'excellent': 85,    # > 85% = Excellent
    'good': 60,         # 60-85% = Good
    'average': 30,      # 30-60% = Average
                        # < 30% = Below Average
}

# Performance labels and helper text
PERFORMANCE_LABELS = {
    'below_average': {
        'label': 'Below Average',
        'emoji': '💪',
        'helper': 'Room for improvement!',
        'color': '#e74c3c'  # Red
    },
    'average': {
        'label': 'Average',
        'emoji': '📈',
        'helper': 'Keep practicing to improve!',
        'color': '#f39c12'  # Orange
    },
    'good': {
        'label': 'Good',
        'emoji': '👍',
        'helper': 'Great work, keep it up!',
        'color': '#3498db'  # Blue
    },
    'excellent': {
        'label': 'Excellent',
        'emoji': '🌟',
        'helper': 'Outstanding performance!',
        'color': '#27ae60'  # Green
    }
}


# ============================================================================
# EXAM PARAMETERS
# ============================================================================

# Maximum possible score
MAX_SCORE = 160

# Total number of questions in exam
TOTAL_QUESTIONS = 160

# Available exam years
AVAILABLE_YEARS = [2025, 2024, 2023, 2022, 2021]

# Marking scheme (for reference)
MARKING_SCHEME = {
    'correct': +1,      # Marks per correct answer
    'wrong': 0,         # No negative marking
    'unattempted': 0    # Marks for unattempted questions
}


# ============================================================================
# REFERENCE BENCHMARKS
# ============================================================================

# National average score (as percentage of max score)
NATIONAL_AVG_PERCENTAGE = 32.0  # 32% of 300 = 96

# Calculate actual national average score
NATIONAL_AVG_SCORE = int((NATIONAL_AVG_PERCENTAGE / 100) * MAX_SCORE)

# Topper score (always 100%)
TOPPER_SCORE = MAX_SCORE


# ============================================================================
# UI CONFIGURATION
# ============================================================================

# Streamlit page config
PAGE_CONFIG = {
    'page_title': 'EAPCET Analytics Dashboard',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Color scheme
COLORS = {
    'primary': '#3498db',      # Blue
    'secondary': '#2ecc71',    # Green
    'warning': '#f39c12',      # Orange
    'danger': '#e74c3c',       # Red
    'background': '#ffffff',   # White
    'text': '#2c3e50',         # Dark gray
    'muted': '#95a5a6'         # Light gray
}

# Dashboard sections to display
DASHBOARD_SECTIONS = {
    'performance_summary': True,
    'performance_comparison': True,
    'detailed_insights': True,
    'recommendations': True,
    'disclaimer': True
}


# ============================================================================
# FILE PATHS
# ============================================================================

# Data files
DATASET_PATH = 'eapcet_synthetic_dataset_2021_2025.csv'
MODEL_PATH = 'eapcet_rank_model.pkl'
METADATA_PATH = 'eapcet_synthetic_dataset_2021_2025_metadata.txt'

# Output directory for exports (if implemented)
OUTPUT_DIR = 'outputs'


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}


# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

# Input validation ranges
INPUT_VALIDATION = {
    'score': {
        'min': 0,
        'max': MAX_SCORE,
        'default': 4
    },
    'attempted': {
        'min': 0,
        'max': TOTAL_QUESTIONS,
        'default': 4
    },
    'correct': {
        'min': 0,
        'max': TOTAL_QUESTIONS,  # Will be capped to attempted in runtime
        'default': 0
    }
}


# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable/disable features
FEATURES = {
    'show_confidence_interval': True,   # Show rank confidence intervals
    'show_attempt_rate': True,          # Show attempt rate metric
    'show_recommendations': True,       # Show improvement recommendations
    'enable_export': False,             # Enable PDF/Excel export (future)
    'enable_comparison': False,         # Enable multi-student comparison (future)
    'debug_mode': False                 # Show debug information
}


# ============================================================================
# RECOMMENDATION THRESHOLDS
# ============================================================================

# When to show improvement recommendations
RECOMMENDATION_THRESHOLDS = {
    'low_percentile': 60,          # Show recommendations if percentile < 60%
    'low_accuracy': 50,            # Show accuracy tips if accuracy < 50%
    'low_attempt_rate': 75,        # Show attempt tips if attempt rate < 75%
    'high_wrong_answers': 40       # Show accuracy tips if wrong > 40
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_performance_category(percentile: float) -> str:
    """
    Get performance category based on percentile.
    
    Args:
        percentile: Percentile score (0-100)
        
    Returns:
        Category string: 'below_average', 'average', 'good', or 'excellent'
    """
    if percentile >= PERFORMANCE_THRESHOLDS['excellent']:
        return 'excellent'
    elif percentile >= PERFORMANCE_THRESHOLDS['good']:
        return 'good'
    elif percentile >= PERFORMANCE_THRESHOLDS['average']:
        return 'average'
    else:
        return 'below_average'


def calculate_national_avg_score() -> int:
    """Calculate national average score from percentage."""
    return int((NATIONAL_AVG_PERCENTAGE / 100) * MAX_SCORE)


def validate_input(value: float, param_type: str) -> float:
    """
    Validate input value against defined ranges.
    
    Args:
        value: Input value to validate
        param_type: Type of parameter ('score', 'attempted', 'correct')
        
    Returns:
        Validated value within allowed range
    """
    if param_type not in INPUT_VALIDATION:
        return value
    
    config = INPUT_VALIDATION[param_type]
    return max(config['min'], min(config['max'], value))


# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    # Print configuration summary
    print("=" * 70)
    print("EAPCET ANALYTICS SYSTEM - CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\nModel Configuration:")
    print(f"  Total Candidates: {TOTAL_CANDIDATES:,}")
    print(f"  Quantiles: {QUANTILES}")
    print(f"  Test Size: {MODEL_CONFIG['test_size']}")
    
    print(f"\nPerformance Thresholds:")
    for category, threshold in PERFORMANCE_THRESHOLDS.items():
        print(f"  {category.title()}: >{threshold}%")
    
    print(f"\nExam Parameters:")
    print(f"  Max Score: {MAX_SCORE}")
    print(f"  Total Questions: {TOTAL_QUESTIONS}")
    print(f"  Available Years: {AVAILABLE_YEARS}")
    
    print(f"\nBenchmarks:")
    print(f"  National Average: {NATIONAL_AVG_SCORE} ({NATIONAL_AVG_PERCENTAGE}%)")
    print(f"  Topper Score: {TOPPER_SCORE} (100%)")
    
    print(f"\nFeatures Enabled:")
    for feature, enabled in FEATURES.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {feature}")
    
    print("\n" + "=" * 70)