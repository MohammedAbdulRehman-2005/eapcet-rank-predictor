"""
EAPCET Analytics System - Comprehensive Test Suite
Run this to validate the entire system before deployment.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_dependencies():
    """Test 1: Check all required dependencies are installed."""
    print_header("TEST 1: Checking Dependencies")
    
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'joblib', 'streamlit'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed")
    return True


def test_dataset_exists():
    """Test 2: Check if synthetic dataset exists."""
    print_header("TEST 2: Checking Dataset")
    
    dataset_path = "eapcet_synthetic_dataset_2021_2025.csv"
    
    if not Path(dataset_path).exists():
        print(f"  ❌ Dataset not found: {dataset_path}")
        print("  Run: python eapcet_generator_fixed.py")
        return False
    
    # Load and validate dataset
    try:
        df = pd.read_csv(dataset_path)
        
        # Check required columns
        required_cols = ['student_id', 'exam_year', 'eapcet_score', 'predicted_rank']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ❌ Missing columns: {missing_cols}")
            return False
        
        print(f"  ✅ Dataset loaded: {len(df):,} records")
        print(f"  ✅ Score range: {df['eapcet_score'].min():.1f} - {df['eapcet_score'].max():.1f}")
        print(f"  ✅ Rank range: {df['predicted_rank'].min():,} - {df['predicted_rank'].max():,}")
        print(f"  ✅ Years: {sorted(df['exam_year'].unique())}")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Error loading dataset: {str(e)}")
        return False


def test_model_exists():
    """Test 3: Check if trained model exists."""
    print_header("TEST 3: Checking ML Model")
    
    model_path = "eapcet_rank_model.pkl"
    
    if not Path(model_path).exists():
        print(f"  ❌ Model not found: {model_path}")
        print("  Run: python eapcet_rank_model.py")
        return False
    
    # Try loading model
    try:
        from eapcet_rank_model import EAPCETRankPredictor
        
        predictor = EAPCETRankPredictor()
        predictor.load_model(model_path)
        
        print(f"  ✅ Model loaded successfully")
        print(f"  ✅ Total candidates: {predictor.total_candidates:,}")
        print(f"  ✅ Quantiles trained: {list(predictor.models.keys())}")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Error loading model: {str(e)}")
        return False


def test_predictions():
    """Test 4: Test model predictions with sample inputs."""
    print_header("TEST 4: Testing Predictions")
    
    try:
        from eapcet_rank_model import EAPCETRankPredictor
        
        predictor = EAPCETRankPredictor()
        predictor.load_model("eapcet_rank_model.pkl")
        
        # Test cases
        test_cases = [
            ("High Scorer", 150, 2025, 160, 150),
            ("Medium Scorer", 100, 2025, 150, 100),
            ("Low Scorer", 50, 2025, 140, 50),
            ("Very Low (Screenshot)", 4, 2025, 4, 0),
        ]
        
        all_passed = True
        
        for name, score, year, attempted, correct in test_cases:
            try:
                result = predictor.predict_rank(score, year, attempted, correct)
                
                # Validate output structure
                required_keys = ['ai_rank', 'percentile', 'accuracy', 
                               'performance_label', 'helper_text']
                
                missing_keys = [k for k in required_keys if k not in result]
                if missing_keys:
                    print(f"  ❌ {name}: Missing keys {missing_keys}")
                    all_passed = False
                    continue
                
                # Validate ranges
                if not (1 <= result['ai_rank'] <= 2_000_000):
                    print(f"  ❌ {name}: Invalid rank {result['ai_rank']}")
                    all_passed = False
                    continue
                
                if not (0 <= result['percentile'] <= 100):
                    print(f"  ❌ {name}: Invalid percentile {result['percentile']}")
                    all_passed = False
                    continue
                
                if not (0 <= result['accuracy'] <= 100):
                    print(f"  ❌ {name}: Invalid accuracy {result['accuracy']}")
                    all_passed = False
                    continue
                
                print(f"  ✅ {name}: Rank={result['ai_rank']:,}, "
                      f"Percentile={result['percentile']:.2f}%, "
                      f"Accuracy={result['accuracy']:.1f}%")
            
            except Exception as e:
                print(f"  ❌ {name}: Prediction failed - {str(e)}")
                all_passed = False
        
        if all_passed:
            print("\n✅ All prediction tests passed")
            return True
        else:
            print("\n❌ Some prediction tests failed")
            return False
    
    except Exception as e:
        print(f"  ❌ Prediction testing failed: {str(e)}")
        return False


def test_performance_labels():
    """Test 5: Validate performance label logic."""
    print_header("TEST 5: Testing Performance Labels")
    
    try:
        from eapcet_rank_model import EAPCETRankPredictor
        
        predictor = EAPCETRankPredictor()
        
        test_percentiles = [
            (10, "Below Average"),
            (40, "Average"),
            (70, "Good"),
            (90, "Excellent"),
        ]
        
        all_passed = True
        
        for percentile, expected_label in test_percentiles:
            label, helper = predictor._get_performance_label(percentile)
            
            if label == expected_label:
                print(f"  ✅ {percentile}% → {label}")
            else:
                print(f"  ❌ {percentile}% → Expected '{expected_label}', got '{label}'")
                all_passed = False
        
        if all_passed:
            print("\n✅ Performance label logic correct")
            return True
        else:
            print("\n❌ Performance label logic has issues")
            return False
    
    except Exception as e:
        print(f"  ❌ Performance label testing failed: {str(e)}")
        return False


def test_edge_cases():
    """Test 6: Test edge cases and boundary conditions."""
    print_header("TEST 6: Testing Edge Cases")
    
    try:
        from eapcet_rank_model import EAPCETRankPredictor
        
        predictor = EAPCETRankPredictor()
        predictor.load_model("eapcet_rank_model.pkl")
        
        edge_cases = [
            ("Zero score", 0, 2025, 0, 0),
            ("Perfect score", 160, 2025, 160, 160),
            ("Attempted but zero correct", 160, 2025, 160, 0),
            ("Old year", 150, 2021, 160, 150),
        ]
        
        all_passed = True
        
        for name, score, year, attempted, correct in edge_cases:
            try:
                result = predictor.predict_rank(score, year, attempted, correct)
                
                # Basic sanity checks
                if result['ai_rank'] < 1:
                    print(f"  ❌ {name}: Rank < 1")
                    all_passed = False
                elif result['percentile'] < 0 or result['percentile'] > 100:
                    print(f"  ❌ {name}: Percentile out of range")
                    all_passed = False
                else:
                    print(f"  ✅ {name}: Handled correctly")
            
            except Exception as e:
                print(f"  ❌ {name}: Failed - {str(e)}")
                all_passed = False
        
        if all_passed:
            print("\n✅ All edge cases handled")
            return True
        else:
            print("\n❌ Some edge cases failed")
            return False
    
    except Exception as e:
        print(f"  ❌ Edge case testing failed: {str(e)}")
        return False


def test_streamlit_imports():
    """Test 7: Check if Streamlit app can be imported."""
    print_header("TEST 7: Testing Streamlit App")
    
    if not Path("eapcet_streamlit_app.py").exists():
        print("  ❌ Streamlit app file not found")
        return False
    
    try:
        # Try importing key components
        import streamlit
        print("  ✅ Streamlit imported")
        
        # Check if app file has syntax errors
        with open("eapcet_streamlit_app.py", 'r', encoding="utf-8") as f:
            code = f.read()
            compile(code, "eapcet_streamlit_app.py", "exec")
        
        print("  ✅ App file syntax valid")
        print("  ℹ️  Run 'streamlit run eapcet_streamlit_app.py' to test UI")
        
        return True
    
    except SyntaxError as e:
        print(f"  ❌ Syntax error in app: {str(e)}")
        return False
    except Exception as e:
        print(f"  ❌ App testing failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "🧪" * 35)
    print("  EAPCET ANALYTICS SYSTEM - TEST SUITE")
    print("🧪" * 35)
    
    tests = [
        test_dependencies,
        test_dataset_exists,
        test_model_exists,
        test_predictions,
        test_performance_labels,
        test_edge_cases,
        test_streamlit_imports,
    ]
    
    results = []
    for test_func in tests:
        try:
            passed = test_func()
            results.append(passed)
        except Exception as e:
            print(f"\n❌ Test crashed: {str(e)}")
            results.append(False)
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n  Tests Passed: {passed} / {total}")
    print(f"  Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        print("  System is ready for deployment.")
        print("\n  Next steps:")
        print("    1. Run: streamlit run eapcet_streamlit_app.py")
        print("    2. Open browser at http://localhost:8501")
        return True
    else:
        print("\n  ⚠️  SOME TESTS FAILED")
        print("  Review errors above and fix issues.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)