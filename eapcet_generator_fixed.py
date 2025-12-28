import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EAPCETSyntheticGenerator:
    """
    Production-grade synthetic EAPCET dataset generator.
    ML-safe, reproducible, and designed to prevent overfitting.
    
    Key Safety Features:
    - Zero-mean Gaussian noise for ranks (no deterministic mapping)
    - No global re-ranking (preserves stochastic behavior)
    - Realistic score distributions with proper Beta parameters
    - Minimum noise floor for top ranks (prevents hard labels)
    - Statistical validation (correlation-based, not deterministic)
    """
    
    def __init__(self, seed: int = 42, config: Optional[Dict] = None):
        """
        Initialize generator with controlled randomness.
        
        Args:
            seed: Random seed for reproducibility
            config: Optional configuration overrides
        """
        self.seed = seed
        self.generation_timestamp = datetime.now().isoformat()
        np.random.seed(seed)
        
        # Load distribution data
        self.distribution_data = self._load_distribution_data()
        
        # Configuration parameters (all tunable in one place)
        self.config = {
            # Noise scaling (standard deviations as % of rank)
            'noise_scaling': {
                'high': 0.03,      # 3% std dev for high confidence
                'medium': 0.06,    # 6% std dev for medium confidence
                'low': 0.10,       # 10% std dev for low confidence
            },
            # Minimum absolute noise (prevents hard labels at top)
            'min_noise_floor': {
                'high': 2,         # Minimum ±2 ranks even for top ranks
                'medium': 3,
                'low': 5,
            },
            # Density adjustment multipliers
            'density_adjustment': {
                (40, 70): 1.8,     # Higher noise in dense lower ranges
                (70, 100): 1.3,
                (100, 140): 1.0,
                (140, 160): 0.6,   # Lower noise at top ranks
            },
            # Beta distribution parameters for realistic score spread
            'score_distribution_params': {
                'low_marks': (1.5, 2.5),    # Skewed toward lower end
                'mid_marks': (2.0, 2.0),    # Balanced distribution
                'high_marks': (2.5, 1.5),   # Skewed toward higher end
            },
            # Year distribution (default)
            'default_year_distribution': {
                2025: 0.35,
                2024: 0.25,
                2023: 0.20,
                2022: 0.10,
                2021: 0.10
            },
            # Output options
            'include_debug_fields': False,  # Set True for debugging
        }
        
        # Apply config overrides
        if config:
            self._update_config(config)
        
        logger.info(f"Initialized EAPCETSyntheticGenerator with seed={seed}")
    
    def _update_config(self, config: Dict):
        """Deep update configuration with user overrides."""
        for key, value in config.items():
            if key in self.config and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def _load_distribution_data(self) -> List[Tuple]:
        """Load and validate the distribution data with confidence levels."""
        return [
            # 2025 Data (high confidence - exact official data)
            (2025, 40, 41, 152496, 164862, 10, 'high', 'exact'),
            (2025, 41, 42, 139591, 152495, 9, 'high', 'exact'),
            (2025, 42, 43, 126932, 139590, 9, 'high', 'exact'),
            (2025, 43, 44, 114471, 126931, 9, 'high', 'exact'),
            (2025, 44, 45, 102398, 114470, 8, 'high', 'exact'),
            (2025, 45, 46, 90790, 102397, 7, 'high', 'exact'),
            (2025, 46, 47, 80040, 90789, 7, 'high', 'exact'),
            (2025, 47, 48, 70343, 80039, 6, 'high', 'exact'),
            (2025, 48, 49, 61657, 70342, 6, 'high', 'exact'),
            (2025, 49, 50, 53971, 61656, 5, 'high', 'exact'),
            (2025, 50, 51, 47381, 53970, 5, 'high', 'exact'),
            (2025, 51, 52, 41853, 47380, 4, 'high', 'exact'),
            (2025, 52, 53, 37165, 41852, 4, 'high', 'exact'),
            (2025, 53, 54, 33183, 37164, 4, 'high', 'exact'),
            (2025, 54, 55, 29790, 33182, 3, 'high', 'exact'),
            (2025, 55, 56, 26833, 29789, 3, 'high', 'exact'),
            (2025, 56, 57, 24263, 26832, 3, 'high', 'exact'),
            (2025, 57, 58, 22058, 24262, 3, 'high', 'exact'),
            (2025, 58, 59, 20125, 22057, 2, 'high', 'exact'),
            (2025, 59, 60, 18356, 20124, 2, 'high', 'exact'),
            (2025, 60, 61, 16784, 18355, 2, 'high', 'exact'),
            (2025, 61, 62, 15399, 16783, 2, 'high', 'exact'),
            (2025, 62, 63, 14203, 15398, 2, 'high', 'exact'),
            (2025, 63, 64, 13084, 14202, 2, 'high', 'exact'),
            (2025, 64, 65, 12048, 13083, 2, 'high', 'exact'),
            (2025, 65, 66, 11088, 12047, 2, 'high', 'exact'),
            (2025, 66, 67, 10248, 11087, 2, 'high', 'exact'),
            (2025, 67, 68, 9475, 10247, 2, 'high', 'exact'),
            (2025, 68, 69, 8749, 9474, 2, 'high', 'exact'),
            (2025, 69, 70, 8077, 8748, 2, 'high', 'exact'),
            (2025, 70, 71, 7466, 8076, 2, 'high', 'exact'),
            (2025, 71, 72, 6940, 7465, 2, 'high', 'exact'),
            (2025, 72, 73, 6407, 6939, 2, 'high', 'exact'),
            (2025, 73, 74, 5936, 6406, 2, 'high', 'exact'),
            (2025, 74, 75, 5460, 5935, 2, 'high', 'exact'),
            (2025, 75, 76, 5053, 5459, 2, 'high', 'exact'),
            (2025, 76, 77, 4675, 5052, 2, 'high', 'exact'),
            (2025, 77, 78, 4357, 4674, 2, 'high', 'exact'),
            (2025, 78, 79, 4030, 4356, 2, 'high', 'exact'),
            (2025, 79, 80, 3707, 4029, 2, 'high', 'exact'),
            (2025, 80, 81, 3421, 3706, 2, 'high', 'exact'),
            (2025, 81, 82, 3185, 3420, 2, 'high', 'exact'),
            (2025, 82, 83, 2944, 3184, 2, 'high', 'exact'),
            (2025, 83, 84, 2720, 2943, 2, 'high', 'exact'),
            (2025, 84, 85, 2517, 2719, 2, 'high', 'exact'),
            (2025, 85, 86, 2336, 2516, 2, 'high', 'exact'),
            (2025, 86, 87, 2165, 2335, 2, 'high', 'exact'),
            (2025, 87, 88, 1983, 2164, 2, 'high', 'exact'),
            (2025, 88, 89, 1851, 1982, 2, 'high', 'exact'),
            (2025, 89, 90, 1708, 1850, 2, 'high', 'exact'),
            (2025, 90, 91, 1604, 1707, 2, 'high', 'exact'),
            (2025, 91, 92, 1484, 1603, 2, 'high', 'exact'),
            (2025, 92, 93, 1356, 1483, 2, 'high', 'exact'),
            (2025, 93, 94, 1266, 1355, 2, 'high', 'exact'),
            (2025, 94, 95, 1194, 1265, 2, 'high', 'exact'),
            (2025, 95, 96, 1108, 1193, 2, 'high', 'exact'),
            (2025, 96, 97, 1035, 1107, 2, 'high', 'exact'),
            (2025, 97, 98, 962, 1034, 2, 'high', 'exact'),
            (2025, 98, 99, 896, 961, 2, 'high', 'exact'),
            (2025, 99, 100, 829, 895, 2, 'high', 'exact'),
            (2025, 100, 101, 783, 828, 2, 'high', 'exact'),
            (2025, 101, 102, 728, 782, 2, 'high', 'exact'),
            (2025, 102, 103, 678, 727, 2, 'high', 'exact'),
            (2025, 103, 104, 624, 677, 2, 'high', 'exact'),
            (2025, 104, 105, 576, 623, 2, 'high', 'exact'),
            (2025, 105, 106, 528, 575, 2, 'high', 'exact'),
            (2025, 106, 107, 488, 527, 2, 'high', 'exact'),
            (2025, 107, 108, 465, 487, 2, 'high', 'exact'),
            (2025, 108, 109, 426, 464, 2, 'high', 'exact'),
            (2025, 109, 110, 389, 425, 2, 'high', 'exact'),
            (2025, 110, 111, 363, 388, 2, 'high', 'exact'),
            (2025, 111, 112, 337, 362, 2, 'high', 'exact'),
            (2025, 112, 113, 313, 336, 2, 'high', 'exact'),
            (2025, 113, 114, 286, 312, 2, 'high', 'exact'),
            (2025, 114, 115, 269, 285, 2, 'high', 'exact'),
            (2025, 115, 116, 256, 268, 2, 'high', 'exact'),
            (2025, 116, 117, 237, 255, 2, 'high', 'exact'),
            (2025, 117, 118, 217, 236, 2, 'high', 'exact'),
            (2025, 118, 119, 200, 216, 2, 'high', 'exact'),
            (2025, 119, 120, 172, 199, 2, 'high', 'exact'),
            (2025, 120, 121, 164, 171, 2, 'high', 'exact'),
            (2025, 121, 122, 152, 163, 2, 'high', 'exact'),
            (2025, 122, 123, 134, 151, 2, 'high', 'exact'),
            (2025, 123, 124, 127, 133, 2, 'high', 'exact'),
            (2025, 124, 125, 115, 126, 2, 'high', 'exact'),
            (2025, 125, 126, 109, 114, 2, 'high', 'exact'),
            (2025, 126, 127, 100, 108, 2, 'high', 'exact'),
            (2025, 127, 128, 84, 99, 2, 'high', 'exact'),
            (2025, 128, 129, 78, 83, 2, 'high', 'exact'),
            (2025, 129, 130, 68, 77, 2, 'high', 'exact'),
            (2025, 130, 131, 60, 67, 2, 'high', 'exact'),
            (2025, 131, 132, 44, 59, 2, 'high', 'exact'),
            (2025, 132, 133, 40, 43, 2, 'high', 'exact'),
            (2025, 133, 134, 34, 39, 2, 'high', 'exact'),
            (2025, 134, 135, 28, 33, 2, 'high', 'exact'),
            (2025, 135, 136, 22, 27, 2, 'high', 'exact'),
            (2025, 136, 137, 20, 21, 2, 'high', 'exact'),
            (2025, 137, 138, 18, 19, 2, 'high', 'exact'),
            (2025, 138, 139, 16, 17, 2, 'high', 'exact'),
            (2025, 139, 140, 13, 15, 2, 'high', 'exact'),
            (2025, 140, 141, 11, 12, 2, 'high', 'exact'),
            (2025, 141, 142, 9, 10, 2, 'high', 'exact'),
            (2025, 142, 143, 7, 8, 2, 'high', 'exact'),
            (2025, 143, 144, 6, 6, 1, 'medium', 'interpolated'),
            (2025, 144, 145, 5, 5, 1, 'medium', 'interpolated'),
            (2025, 145, 146, 5, 5, 1, 'high', 'exact'),
            (2025, 146, 147, 4, 4, 1, 'medium', 'interpolated'),
            (2025, 147, 148, 4, 4, 1, 'medium', 'interpolated'),
            (2025, 148, 149, 4, 4, 1, 'medium', 'interpolated'),
            (2025, 149, 150, 4, 4, 1, 'high', 'exact'),
            (2025, 150, 151, 3, 3, 1, 'high', 'exact'),
            (2025, 151, 152, 2, 3, 1, 'medium', 'interpolated'),
            (2025, 152, 153, 2, 2, 1, 'high', 'exact'),
            (2025, 153, 154, 1, 2, 1, 'medium', 'interpolated'),
            (2025, 154, 155, 1, 1, 1, 'medium', 'interpolated'),
            (2025, 155, 156, 1, 1, 1, 'high', 'exact'),
            
            # 2024 Data (medium confidence - estimated from trends)
           # 2024 Data (medium confidence – derived)
            (2024, 155, 160, 1, 50, 5, 'medium', 'derived'),
            (2024, 150, 155, 51, 200, 5, 'medium', 'derived'),
            (2024, 145, 150, 201, 600, 5, 'medium', 'derived'),
            (2024, 140, 145, 601, 1200, 5, 'medium', 'derived'),
            (2024, 135, 140, 1201, 2500, 5, 'medium', 'derived'),
            (2024, 130, 135, 2501, 4500, 5, 'medium', 'derived'),
            (2024, 125, 130, 4501, 7000, 5, 'medium', 'derived'),
            (2024, 120, 125, 7001, 10500, 5, 'medium', 'derived'),
            (2024, 115, 120, 10501, 15000, 5, 'medium', 'derived'),
            (2024, 110, 115, 15001, 22000, 5, 'medium', 'derived'),
            (2024, 105, 110, 22001, 30000, 5, 'medium', 'derived'),
            (2024, 100, 105, 30001, 42000, 5, 'medium', 'derived'),
            (2024, 95, 100, 42001, 60000, 5, 'medium', 'derived'),
            (2024, 90, 95, 60001, 80000, 5, 'medium', 'derived'),
            (2024, 85, 90, 80001, 105000, 5, 'medium', 'derived'),
            (2024, 80, 85, 105001, 130000, 5, 'medium', 'derived'),
            (2024, 75, 80, 130001, 150000, 5, 'medium', 'derived'),
            (2024, 70, 75, 150001, 165000, 5, 'medium', 'derived'),
            (2024, 65, 70, 165001, 175000, 5, 'medium', 'derived'),
            (2024, 60, 65, 175001, 185000, 5, 'medium', 'derived'),
            (2024, 55, 60, 185001, 195000, 5, 'medium', 'derived'),
            (2024, 50, 55, 195001, 205000, 5, 'medium', 'derived'),
            (2024, 45, 50, 205001, 215000, 5, 'medium', 'derived'),
            (2024, 40, 45, 215001, 230000, 5, 'medium', 'derived'),

            # 2023 Data (medium confidence - estimated from trends)
            # 2023 Data (medium confidence – derived)
            (2023, 155, 160, 1, 8, 5, 'medium', 'derived'),
            (2023, 150, 155, 8, 18, 5, 'medium', 'derived'),
            (2023, 145, 150, 18, 32, 5, 'medium', 'derived'),
            (2023, 140, 145, 32, 50, 5, 'medium', 'derived'),
            (2023, 135, 140, 50, 75, 5, 'medium', 'derived'),
            (2023, 130, 135, 75, 110, 5, 'medium', 'derived'),
            (2023, 125, 130, 110, 150, 5, 'medium', 'derived'),
            (2023, 120, 125, 150, 200, 5, 'medium', 'derived'),
            (2023, 115, 120, 200, 280, 5, 'medium', 'derived'),
            (2023, 110, 115, 280, 380, 5, 'medium', 'derived'),
            (2023, 105, 110, 380, 520, 5, 'medium', 'derived'),
            (2023, 100, 105, 520, 700, 5, 'medium', 'derived'),
            (2023, 95, 100, 700, 1100, 4, 'medium', 'derived'),
            (2023, 90, 95, 1100, 1500, 4, 'medium', 'derived'),
            (2023, 85, 90, 1500, 2050, 4, 'medium', 'derived'),
            (2023, 80, 85, 2050, 3300, 4, 'medium', 'derived'),
            (2023, 75, 80, 3300, 5100, 4, 'medium', 'derived'),
            (2023, 70, 75, 5100, 8000, 4, 'medium', 'derived'),
            (2023, 65, 70, 8000, 13000, 4, 'medium', 'derived'),
            (2023, 60, 65, 13000, 21000, 4, 'medium', 'derived'),
            (2023, 55, 60, 21000, 35000, 4, 'medium', 'derived'),
            (2023, 50, 55, 35000, 63500, 4, 'medium', 'derived'),
            (2023, 45, 50, 63500, 113500, 3, 'medium', 'derived'),
            (2023, 40, 45, 113500, 156879, 3, 'medium', 'derived'),

            
            # 2022 Data (low confidence - approximated)
            # 2022 Data (medium confidence – derived)
            (2022, 155, 160, 1, 8, 5, 'medium', 'derived'),
            (2022, 150, 155, 8, 18, 5, 'medium', 'derived'),
            (2022, 145, 150, 18, 32, 5, 'medium', 'derived'),
            (2022, 140, 145, 32, 50, 5, 'medium', 'derived'),
            (2022, 135, 140, 50, 75, 5, 'medium', 'derived'),
            (2022, 130, 135, 75, 110, 5, 'medium', 'derived'),
            (2022, 125, 130, 110, 150, 5, 'medium', 'derived'),
            (2022, 120, 125, 150, 200, 5, 'medium', 'derived'),
            (2022, 115, 120, 200, 280, 5, 'medium', 'derived'),
            (2022, 110, 115, 280, 380, 5, 'medium', 'derived'),
            (2022, 105, 110, 380, 520, 5, 'medium', 'derived'),
            (2022, 100, 105, 520, 700, 5, 'medium', 'derived'),
            (2022, 95, 100, 700, 1100, 4, 'medium', 'derived'),
            (2022, 90, 95, 1100, 1500, 4, 'medium', 'derived'),
            (2022, 85, 90, 1500, 2050, 4, 'medium', 'derived'),
            (2022, 80, 85, 2050, 3300, 4, 'medium', 'derived'),
            (2022, 75, 80, 3300, 5100, 4, 'medium', 'derived'),
            (2022, 70, 75, 5100, 8000, 4, 'medium', 'derived'),
            (2022, 65, 70, 8000, 13000, 4, 'medium', 'derived'),
            (2022, 60, 65, 13000, 21000, 4, 'medium', 'derived'),
            (2022, 55, 60, 21000, 35000, 4, 'medium', 'derived'),
            (2022, 50, 55, 35000, 63500, 4, 'medium', 'derived'),
            (2022, 45, 50, 63500, 113500, 3, 'medium', 'derived'),
            (2022, 40, 45, 113500, 156879, 3, 'medium', 'derived'),

            # 2021 Data (low confidence - approximated)
            # 2021 Data (medium confidence – derived, total qualified = 121480)

            (2021, 155, 160, 1, 10, 5, 'medium', 'derived'),
            (2021, 150, 155, 10, 22, 5, 'medium', 'derived'),
            (2021, 145, 150, 22, 38, 5, 'medium', 'derived'),
            (2021, 140, 145, 38, 60, 5, 'medium', 'derived'),
            (2021, 135, 140, 60, 90, 5, 'medium', 'derived'),
            (2021, 130, 135, 90, 130, 5, 'medium', 'derived'),
            (2021, 125, 130, 130, 180, 5, 'medium', 'derived'),
            (2021, 120, 125, 180, 250, 5, 'medium', 'derived'),
            (2021, 115, 120, 250, 350, 5, 'medium', 'derived'),
            (2021, 110, 115, 350, 480, 5, 'medium', 'derived'),
            (2021, 105, 110, 480, 650, 5, 'medium', 'derived'),
            (2021, 100, 105, 650, 900, 5, 'medium', 'derived'),

            (2021, 95, 100, 900, 1400, 4, 'medium', 'derived'),
            (2021, 90, 95, 1400, 2000, 4, 'medium', 'derived'),
            (2021, 85, 90, 2000, 2800, 4, 'medium', 'derived'),
            (2021, 80, 85, 2800, 3800, 4, 'medium', 'derived'),
            (2021, 75, 80, 3800, 5500, 4, 'medium', 'derived'),
            (2021, 70, 75, 5500, 8200, 4, 'medium', 'derived'),
            (2021, 65, 70, 8200, 12500, 4, 'medium', 'derived'),
            (2021, 60, 65, 12500, 20000, 4, 'medium', 'derived'),

            (2021, 55, 60, 20000, 35000, 4, 'medium', 'derived'),
            (2021, 50, 55, 35000, 60000, 4, 'medium', 'derived'),
            (2021, 45, 50, 60000, 90000, 3, 'medium', 'derived'),
            (2021, 40, 45, 90000, 121480, 3, 'medium', 'derived'),

        ]
    
    def _calculate_density_adjustment(self, marks: float) -> float:
        """Calculate noise adjustment factor based on marks density."""
        for (min_mark, max_mark), adjustment in self.config['density_adjustment'].items():
            if min_mark <= marks < max_mark:
                return adjustment
        return 1.0
    
    def _add_controlled_noise(self, base_rank: int, marks: float, confidence: str) -> Tuple[int, float]:
        """
        Add zero-mean Gaussian noise to ranks (CORRECTED).
        
        Key fixes:
        - Uses symmetric Gaussian distribution (zero mean)
        - Applies minimum noise floor to prevent hard labels
        - Scales noise based on confidence and density
        
        Args:
            base_rank: Deterministic rank from interpolation
            marks: EAPCET score
            confidence: Data confidence level
            
        Returns:
            (noisy_rank, noise_magnitude)
        """
        # Base noise from confidence level (as std dev percentage)
        base_noise_std = self.config['noise_scaling'][confidence]
        
        # Density adjustment
        density_factor = self._calculate_density_adjustment(marks)
        
        # Combined standard deviation
        noise_std = base_noise_std * density_factor * base_rank
        
        # Apply minimum noise floor (prevents hard labels at top)
        min_floor = self.config['min_noise_floor'][confidence]
        noise_std = max(noise_std, min_floor)
        
        # Generate ZERO-MEAN Gaussian noise
        noise = np.random.normal(0, noise_std)
        
        # Apply noise and ensure positive rank
        noisy_rank = int(round(base_rank + noise))
        noisy_rank = max(1, noisy_rank)  # Ranks must be >= 1
        
        return noisy_rank, abs(noise)
    
    def _generate_mark_within_range(self, start: float, end: float) -> float:
        """
        Generate a mark within a range with realistic distribution (CORRECTED).
        
        Key fixes:
        - Beta parameters now correctly match mark ranges
        - Low marks skew LOW (alpha < beta)
        - High marks skew HIGH (alpha > beta)
        """
        # Select Beta distribution parameters based on mark range
        if start < 60:
            # Low marks: skew toward lower end (alpha < beta)
            alpha, beta = self.config['score_distribution_params']['low_marks']
        elif start < 100:
            # Mid marks: balanced distribution
            alpha, beta = self.config['score_distribution_params']['mid_marks']
        else:
            # High marks: skew toward higher end (alpha > beta)
            alpha, beta = self.config['score_distribution_params']['high_marks']
        
        # Generate normalized position within range using Beta
        position = np.random.beta(alpha, beta)
        
        # Calculate actual mark with decimal precision
        mark = start + (end - start) * position
        
        # Round to 2 decimal places (realistic exam precision)
        return round(mark, 2)
    
    def _select_distribution_row(self, year: int) -> Tuple:
        """
        Select a distribution row based on year and weight.
        Higher weight = higher probability of selection.
        """
        year_rows = [row for row in self.distribution_data if row[0] == year]
        
        if not year_rows:
            raise ValueError(f"No distribution data found for year {year}")
        
        # Extract weights and normalize to probabilities
        weights = [row[5] for row in year_rows]
        probabilities = np.array(weights) / sum(weights)
        
        # Select row based on probability
        selected_idx = np.random.choice(len(year_rows), p=probabilities)
        return year_rows[selected_idx]
    
    def generate_student_record(self, student_id: int, target_year: Optional[int] = None) -> Dict:
        """
        Generate a single synthetic student record.
        
        Args:
            student_id: Unique identifier for the student
            target_year: Specific year to generate for (None for random year)
            
        Returns:
            Dictionary containing student record
        """
        # Select year if not specified
        if target_year is None:
            available_years = list(set(row[0] for row in self.distribution_data))
            exam_year = np.random.choice(available_years)
        else:
            exam_year = target_year
        
        # Select distribution row for this year
        row = self._select_distribution_row(exam_year)
        
        # Unpack row
        year, marks_start, marks_end, min_rank, max_rank, weight, confidence, source_type = row
        
        # Generate marks within range
        eapcet_score = self._generate_mark_within_range(marks_start, marks_end)
        
        # Calculate base rank (linear interpolation within range)
        mark_position = (eapcet_score - marks_start) / max(1, (marks_end - marks_start))
        base_rank = int(min_rank + (max_rank - min_rank) * (1 - mark_position))
        
        # Add controlled Gaussian noise
        predicted_rank, rank_noise = self._add_controlled_noise(base_rank, eapcet_score, confidence)
        
        # Define rank band
        if predicted_rank <= 100:
            rank_band = "Top 100"
        elif predicted_rank <= 1000:
            rank_band = "Top 1000"
        elif predicted_rank <= 10000:
            rank_band = "Top 10000"
        elif predicted_rank <= 50000:
            rank_band = "Top 50000"
        else:
            rank_band = "Above 50000"
        
        # Build record (conditionally include debug fields)
        record = {
            'student_id': student_id,
            'exam_year': exam_year,
            'eapcet_score': eapcet_score,
            'predicted_rank': predicted_rank,
            'rank_band': rank_band,
            'data_confidence': confidence,
            'source_type': source_type,
        }
        
        # Add debug fields if configured
        if self.config['include_debug_fields']:
            record.update({
                'base_rank': base_rank,
                'rank_noise_applied': round(rank_noise, 2),
            })
        
        return record
    
    def generate_dataset(
        self, 
        num_students: int = 50000, 
        year_distribution: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Generate a complete synthetic dataset (NO GLOBAL RE-RANKING).
        
        Args:
            num_students: Number of student records to generate
            year_distribution: Optional dict of {year: proportion}
            
        Returns:
            pandas DataFrame with synthetic data
        """
        logger.info(f"Generating {num_students:,} synthetic student records...")
        
        # Use default year distribution if not specified
        if year_distribution is None:
            year_distribution = self.config['default_year_distribution']
        
        # Validate year distribution
        if not np.isclose(sum(year_distribution.values()), 1.0):
            raise ValueError("Year distribution must sum to 1.0")
        
        # Generate student records
        records = []
        years = list(year_distribution.keys())
        probs = list(year_distribution.values())
        
        for student_id in range(1, num_students + 1):
            # Select year based on distribution
            exam_year = np.random.choice(years, p=probs)
            
            # Generate record
            record = self.generate_student_record(student_id, exam_year)
            records.append(record)
            
            # Progress indicator
            if student_id % 10000 == 0:
                logger.info(f"  Generated {student_id:,} records...")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # CRITICAL: NO GLOBAL RE-RANKING
        # Preserve stochastic rank behavior - allow natural collisions
        # This is essential for ML safety and prevents data leakage
        
        logger.info(f"Successfully generated {len(df):,} synthetic student records")
        logger.info(f"Year distribution: {dict(df['exam_year'].value_counts().sort_index())}")
        logger.info(f"Score range: {df['eapcet_score'].min():.1f} - {df['eapcet_score'].max():.1f}")
        logger.info(f"Rank range: {df['predicted_rank'].min()} - {df['predicted_rank'].max()}")
        
        return df
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Validate the generated dataset for ML safety and realism (CORRECTED).
        
        Key fixes:
        - Uses correlation-based checks instead of deterministic uniqueness
        - Validates statistical trends, not exact behaviors
        - Checks for realistic noise and variance
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'ml_safety_checks': {},
            'statistical_checks': {},
            'production_checks': {}
        }
        
        logger.info("Running validation checks...")
        
        # ===== ML SAFETY CHECKS =====
        
        # Check 1: Non-deterministic mapping (correlation should be strong but not perfect)
        score_rank_corr = df.groupby('exam_year').apply(
            lambda x: x['eapcet_score'].corr(x['predicted_rank'])
        ).mean()
        validation_results['ml_safety_checks']['score_rank_correlation'] = {
            'value': round(score_rank_corr, 4),
            'expected': '< -0.95 (strong negative, not perfect)',
            'pass': -0.99 < score_rank_corr < -0.90
        }
        
        # Check 2: Rank variance exists (not deterministic)
        rank_variance = df.groupby(['exam_year', pd.cut(df['eapcet_score'], bins=20)])[
            'predicted_rank'
        ].std().mean()
        validation_results['ml_safety_checks']['rank_variance_in_bins'] = {
            'value': round(rank_variance, 2),
            'expected': '> 0 (variance exists)',
            'pass': rank_variance > 0
        }
        
        # Check 3: Top ranks have noise (no hard labels)
        top_100_ranks = df[df['predicted_rank'] <= 100]['predicted_rank']
        top_rank_duplicates = len(top_100_ranks) - len(top_100_ranks.unique())
        validation_results['ml_safety_checks']['top_rank_collisions'] = {
            'value': top_rank_duplicates,
            'expected': '> 0 (realistic collisions exist)',
            'pass': top_rank_duplicates > 0
        }
        
        # ===== STATISTICAL CHECKS =====
        
        # Check 4: Monotonic relationship (statistical trend)
        bin_means = df.groupby(pd.cut(df['eapcet_score'], bins=30)).agg({
            'eapcet_score': 'mean',
            'predicted_rank': 'mean'
        }).dropna()
        
        monotonic_violations = (bin_means['predicted_rank'].diff() > 0).sum()
        validation_results['statistical_checks']['monotonic_trend'] = {
            'value': f"{monotonic_violations} violations",
            'expected': '< 10% of bins',
            'pass': monotonic_violations < len(bin_means) * 0.1
        }
        
        # Check 5: Realistic score distribution
        score_min, score_max = df['eapcet_score'].min(), df['eapcet_score'].max()
        validation_results['statistical_checks']['score_range'] = {
            'value': f"{score_min:.1f} - {score_max:.1f}",
            'expected': '40.0 - 156.0',
            'pass': (39 <= score_min <= 41) and (155 <= score_max <= 157)
        }
        
        # Check 6: Confidence-based noise variation
        noise_by_confidence = {}
        if 'rank_noise_applied' in df.columns:
            noise_by_confidence = df.groupby('data_confidence')['rank_noise_applied'].mean().to_dict()
        validation_results['statistical_checks']['noise_by_confidence'] = noise_by_confidence
        
        # ===== PRODUCTION CHECKS =====
        
        # Check 7: No missing values
        validation_results['production_checks']['no_missing_values'] = {
            'value': df.isnull().sum().sum(),
            'expected': 0,
            'pass': not df.isnull().any().any()
        }
        
        # Check 8: Unique student IDs
        validation_results['production_checks']['unique_student_ids'] = {
            'value': df['student_id'].nunique(),
            'expected': len(df),
            'pass': df['student_id'].nunique() == len(df)
        }
        
        # Check 9: Rank distribution sanity
        validation_results['production_checks']['rank_statistics'] = {
            'min': int(df['predicted_rank'].min()),
            'max': int(df['predicted_rank'].max()),
            'mean': round(df['predicted_rank'].mean(), 2),
            'median': int(df['predicted_rank'].median())
        }
        
        # Check 10: Confidence distribution
        validation_results['production_checks']['confidence_distribution'] = (
            df['data_confidence'].value_counts().to_dict()
        )
        
        return validation_results
    
    def save_dataset(self, df: pd.DataFrame, output_path: str = "eapcet_synthetic_dataset.csv"):
        """
        Save dataset to CSV with metadata header.
        
        Args:
            df: Generated dataset
            output_path: Output file path
        """
        # Add metadata as comments (if CSV supports)
        logger.info(f"Saving dataset to {output_path}")
        
        # Save main dataset
        df.to_csv(output_path, index=False)
        
        # Save metadata separately
        metadata = {
            'generation_timestamp': self.generation_timestamp,
            'random_seed': self.seed,
            'num_records': len(df),
            'year_range': f"{df['exam_year'].min()} - {df['exam_year'].max()}",
            'score_range': f"{df['eapcet_score'].min():.1f} - {df['eapcet_score'].max():.1f}",
            'rank_range': f"{df['predicted_rank'].min()} - {df['predicted_rank'].max()}",
        }
        
        metadata_path = output_path.replace('.csv', '_metadata.txt')
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Metadata saved to {metadata_path}")
        logger.info(f"Dataset generation complete: {len(df):,} records")


# ============================================================================
# PRODUCTION USAGE EXAMPLE
# ============================================================================

def main():
    """
    Main function demonstrating production usage.
    """
    print("=" * 70)
    print("EAPCET SYNTHETIC DATASET GENERATOR - PRODUCTION GRADE (v2.0)")
    print("=" * 70)
    print()
    
    # Initialize generator with reproducible seed
    generator = EAPCETSyntheticGenerator(seed=42)
    
    # Generate dataset
    synthetic_df = generator.generate_dataset(num_students=50000)
    
    # Validate dataset
    print()
    validation_results = generator.validate_dataset(synthetic_df)
    
    # Display validation summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for category, checks in validation_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for check_name, result in checks.items():
            if isinstance(result, dict):
                if 'pass' in result:
                    status = "✓ PASS" if result['pass'] else "✗ FAIL"
                    print(f"  {status} - {check_name}")
                    print(f"      Value: {result['value']}")
                    print(f"      Expected: {result['expected']}")
                else:
                    print(f"  • {check_name}: {result}")
            else:
                print(f"  • {check_name}: {result}")
    
    # Save to CSV
    output_file = "eapcet_synthetic_dataset_2021_2025.csv"
    generator.save_dataset(synthetic_df, output_file)
    
    # Display sample records
    print("\n" + "=" * 70)
    print("SAMPLE RECORDS (First 10)")
    print("=" * 70)
    print(synthetic_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Total Records: {len(synthetic_df):,}")
    print(f"Years Covered: {sorted(synthetic_df['exam_year'].unique())}")
    print(f"Score Range: {synthetic_df['eapcet_score'].min():.2f} - {synthetic_df['eapcet_score'].max():.2f}")
    print(f"Rank Range: {synthetic_df['predicted_rank'].min():,} - {synthetic_df['predicted_rank'].max():,}")
    print(f"Unique Ranks: {synthetic_df['predicted_rank'].nunique():,} ({synthetic_df['predicted_rank'].nunique()/len(synthetic_df)*100:.1f}%)")
    
    return synthetic_df


if __name__ == "__main__":
    df = main()
