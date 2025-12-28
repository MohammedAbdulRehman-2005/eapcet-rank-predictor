import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EAPCETRankPredictor:
    """
    Production-grade EAPCET rank prediction model using quantile regression.
    
    Features:
    - Quantile regression (10%, 50%, 90%) for confidence intervals
    - Explainable gradient boosting models
    - Realistic rank scaling with score
    - Year-aware predictions
    """
    
    def __init__(self, total_candidates: int = 151_779):
        """
        Initialize the rank predictor.
        
        Args:
            total_candidates: Total qualified candidates (default: TS EAPCET 2025 = 151,779)
        """
        self.total_candidates = total_candidates
        self.models = {}  # {quantile: model}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = ['eapcet_score', 'exam_year', 'attempted', 'correct']
        
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature matrix from raw data.
        
        Note: For synthetic data training, attempted and correct are derived
        from score. For real predictions, users provide these directly.
        """
        features = df.copy()
        
        # Ensure required columns exist
        if 'attempted' not in features.columns:
            # Estimate attempted questions (simulation: ~80-90% of total)
            features['attempted'] = np.random.randint(120, 160, size=len(features))
        
        if 'correct' not in features.columns:
            # Estimate correct answers from score (each correct = ~1 mark)
            features['correct'] = (features['eapcet_score'] * 0.65).astype(int)
        
        return features[self.feature_columns]
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Train quantile regression models on synthetic dataset.
        
        Args:
            df: DataFrame with columns [eapcet_score, exam_year, predicted_rank]
            test_size: Proportion of data for validation
        """
        logger.info("Training EAPCET rank prediction models...")
        
        # Create features
        X = self._create_features(df)
        y = df['predicted_rank'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Fit scaler
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models for different quantiles
        quantiles = [0.10, 0.50, 0.90]  # 10th, 50th (median), 90th percentile
        
        for q in quantiles:
            logger.info(f"Training quantile {q} model...")
            
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            train_mae = np.mean(np.abs(train_pred - y_train))
            test_mae = np.mean(np.abs(test_pred - y_test))
            
            logger.info(f"  Quantile {q}: Train MAE={train_mae:.0f}, Test MAE={test_mae:.0f}")
            
            self.models[q] = model
        
        self.is_trained = True
        logger.info("Model training complete!")
        
        return {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'quantiles': quantiles
        }
    
    def predict_rank(
        self, 
        score: float, 
        exam_year: int, 
        attempted: int, 
        correct: int
    ) -> Dict:
        """
        Predict rank with confidence intervals.
        
        Args:
            score: Total EAPCET score (0-160)
            exam_year: Exam year
            attempted: Number of questions attempted
            correct: Number of correct answers
            
        Returns:
            Dictionary with rank predictions and derived metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")
        
        # Validate inputs
        score = max(0, min(160, score))
        attempted = max(0, min(160, attempted))
        correct = max(0, min(attempted, correct))
        
        # Create feature vector
        X = pd.DataFrame([{
            'eapcet_score': score,
            'exam_year': exam_year,
            'attempted': attempted,
            'correct': correct
        }])
        
        X_scaled = self.scaler.transform(X)
        
        # Predict ranks for all quantiles
        predictions = {}
        for q, model in self.models.items():
            rank = model.predict(X_scaled)[0]
            predictions[f'rank_q{int(q*100)}'] = max(1, int(rank))
        
        # Use median (50th percentile) as primary prediction
        ai_rank = predictions['rank_q50']
        
        # Calculate derived metrics
        percentile = self._calculate_percentile(ai_rank)
        accuracy = self._calculate_accuracy(correct, attempted)
        performance_label, helper_text = self._get_performance_label(percentile)
        
        # Rank confidence interval
        rank_lower = predictions['rank_q10']
        rank_upper = predictions['rank_q90']
        
        return {
            # Primary metrics
            'ai_rank': ai_rank,
            'percentile': percentile,
            'accuracy': accuracy,
            'performance_label': performance_label,
            'helper_text': helper_text,
            
            # Confidence intervals
            'rank_lower_bound': rank_lower,
            'rank_upper_bound': rank_upper,
            
            # Input echo
            'score': score,
            'attempted': attempted,
            'correct': correct,
            'exam_year': exam_year,
            
            # Additional stats
            'score_percentage': (score / 160) * 100,
            'correct_percentage': (correct / 160) * 100 if attempted > 0 else 0
        }
    
    def _calculate_percentile(self, rank: int) -> float:
        """
        Calculate percentile from rank.
        
        Formula: percentile = 100 * (1 - rank/total_candidates)
        """
        percentile = 100 * (1 - (rank / self.total_candidates))
        return max(0, min(100, percentile))
    
    def _calculate_accuracy(self, correct: int, attempted: int) -> float:
        """
        Calculate accuracy percentage.
        
        Formula: accuracy = (correct / attempted) * 100
        """
        if attempted == 0:
            return 0.0
        return (correct / attempted) * 100
    
    def _get_performance_label(self, percentile: float) -> Tuple[str, str]:
        """
        Get performance label and helper text based on percentile.
        
        Returns:
            (label, helper_text) tuple
        """
        if percentile < 30:
            return "Below Average", "💪 Room for improvement!"
        elif percentile < 60:
            return "Average", "📈 Keep practicing to improve!"
        elif percentile < 85:
            return "Good", "👍 Great work, keep it up!"
        else:
            return "Excellent", "🌟 Outstanding performance!"
    
    def save_model(self, filepath: str = "eapcet_rank_model.pkl"):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'total_candidates': self.total_candidates,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "eapcet_rank_model.pkl"):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.total_candidates = model_data['total_candidates']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


# ============================================================================
# TRAINING SCRIPT
# ============================================================================

def train_model_from_synthetic_data(csv_path: str = "eapcet_synthetic_dataset_2021_2025.csv"):
    """
    Train the rank prediction model using synthetic dataset.
    
    Args:
        csv_path: Path to synthetic dataset CSV
    """
    # Load synthetic dataset
    logger.info(f"Loading synthetic dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Score range: {df['eapcet_score'].min():.1f} - {df['eapcet_score'].max():.1f}")
    logger.info(f"Rank range: {df['predicted_rank'].min()} - {df['predicted_rank'].max()}")
    
    # Initialize and train model with official TS EAPCET 2025 figures
    predictor = EAPCETRankPredictor(total_candidates=151_779)
    training_stats = predictor.train(df, test_size=0.2)
    
    logger.info(f"Training statistics: {training_stats}")
    
    # Save model
    predictor.save_model("eapcet_rank_model.pkl")
    
    # Test predictions
    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("=" * 70)
    
    test_cases = [
        (150, 2025, 160, 150),  # High scorer
        (100, 2025, 150, 100),  # Medium scorer
        (50, 2025, 140, 50),    # Low scorer
        (4, 2025, 4, 0),        # Very low scorer (like screenshot)
    ]
    
    for score, year, attempted, correct in test_cases:
        result = predictor.predict_rank(score, year, attempted, correct)
        print(f"\nScore: {score}/300, Attempted: {attempted}, Correct: {correct}")
        print(f"  AI Rank: {result['ai_rank']:,}")
        print(f"  Percentile: {result['percentile']:.2f}%")
        print(f"  Accuracy: {result['accuracy']:.1f}%")
        print(f"  Performance: {result['performance_label']} - {result['helper_text']}")
        print(f"  Confidence Interval: [{result['rank_lower_bound']:,} - {result['rank_upper_bound']:,}]")
    
    return predictor


if __name__ == "__main__":
    # Train model
    predictor = train_model_from_synthetic_data()
    print("\nModel training complete! Use 'eapcet_rank_model.pkl' in Streamlit app.")