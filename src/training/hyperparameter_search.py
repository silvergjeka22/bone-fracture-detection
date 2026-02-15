# Save as hyperparameter_search.py or run in a cell
"""
Hyperparameter optimization using Optuna.

This module contains functions for finding optimal hyperparameters using Optuna.
"""

import optuna


def optimize_hyperparameters(model_class, train_dataset, device, 
                            n_trials=20, k_folds=3, num_epochs=10):
    """
    Find optimal hyperparameters using Optuna with k-fold cross-validation.
    
    Args:
        model_class: Model class to optimize
        train_dataset: Training dataset
        device: Device to train on (cpu/cuda)
        n_trials: Number of Optuna trials (hyperparameter combinations to test)
        k_folds: Number of folds for cross-validation during search
        num_epochs: Number of epochs per fold during search
        
    Returns:
        study: Optuna study object with all results
        best_params: Dictionary with best hyperparameters
    """
    
    def objective(trial):
        """
        Objective function for Optuna to optimize.
        
        HOW OPTUNA WORKS:
        -----------------
        1. Optuna calls this function multiple times (n_trials times)
        2. Each time, it suggests different hyperparameters
        3. We train the model with those hyperparameters
        4. We return the validation accuracy
        5. Optuna learns which hyperparameters work best
        6. It gets smarter with each trial!
        """
        
        # ============================================================
        # STEP 1: OPTUNA SUGGESTS HYPERPARAMETERS
        # ============================================================
        
        # Learning rate: between 0.00001 and 0.01 (log scale)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # Batch size: one of these values
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Dropout rate: between 0.2 and 0.7
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
        
        # Print what we're testing this trial
        print(f"\n{'='*70}")
        print(f"🔬 TRIAL {trial.number + 1}/{n_trials}")
        print(f"{'='*70}")
        print(f"Testing hyperparameters:")
        print(f"  📊 Learning Rate: {learning_rate:.6f}")
        print(f"  📦 Batch Size:    {batch_size}")
        print(f"  💧 Dropout Rate:  {dropout_rate:.3f}")
        print(f"{'='*70}\n")
        
        # ============================================================
        # STEP 2: TRAIN MODEL WITH THESE HYPERPARAMETERS
        # ============================================================
        
        try:
            # Use train_kfold_cv with the suggested hyperparameters
            fold_results, fold_histories, mean_acc, std_acc = train_kfold_cv(
                model_class=model_class,
                train_dataset=train_dataset,
                k_folds=k_folds,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                device=device
            )
            
            # ============================================================
            # STEP 3: PRINT RESULTS FOR THIS TRIAL
            # ============================================================
            
            print(f"\n{'='*70}")
            print(f"📈 TRIAL {trial.number + 1} RESULTS")
            print(f"{'='*70}")
            print(f"Mean Validation Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            
            # Show results for each fold
            print(f"\n📊 Fold-by-fold results:")
            for fold_result in fold_results:
                print(f"  Fold {fold_result['fold']}: {fold_result['best_val_acc']:.4f}")
            
            # Check if this is the best trial so far
            if trial.number > 0:
                current_best = trial.study.best_value
                if mean_acc > current_best:
                    improvement = mean_acc - current_best
                    print(f"\n🎉 NEW BEST ACCURACY!")
                    print(f"   Previous best: {current_best:.4f}")
                    print(f"   New best:      {mean_acc:.4f}")
                    print(f"   Improvement:   +{improvement:.4f}")
                else:
                    print(f"\n   Current best remains: {current_best:.4f}")
            else:
                print(f"\n🎯 First trial complete!")
            
            print(f"{'='*70}\n")
            
            # ============================================================
            # STEP 4: RETURN THE METRIC TO OPTIMIZE
            # ============================================================
            # Optuna will try to MAXIMIZE this value
            return mean_acc
            
        except Exception as e:
            print(f"❌ Trial failed with error: {e}")
            return 0.0  # Return bad score if training fails
    
    # ================================================================
    # CREATE OPTUNA STUDY
    # ================================================================
    # direction='maximize' means we want the HIGHEST accuracy
    study = optuna.create_study(direction='maximize')
    
    print(f"\n{'#'*70}")
    print(f"🚀 STARTING HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print(f"{'#'*70}")
    print(f"Model:              {model_class.__name__}")
    print(f"Number of trials:   {n_trials}")
    print(f"K-folds per trial:  {k_folds}")
    print(f"Epochs per fold:    {num_epochs}")
    print(f"{'#'*70}\n")
    
    # ================================================================
    # RUN OPTIMIZATION
    # ================================================================
    # Optuna will call objective() n_trials times
    # Each time with different (smarter) hyperparameters!
    study.optimize(objective, n_trials=n_trials)
    
    # ================================================================
    # PRINT FINAL RESULTS
    # ================================================================
    print(f"\n{'#'*70}")
    print(f"✅ HYPERPARAMETER OPTIMIZATION COMPLETE!")
    print(f"{'#'*70}")
    print(f"\n🏆 BEST HYPERPARAMETERS FOUND:")
    print(f"  📊 Learning Rate: {study.best_params['learning_rate']:.6f}")
    print(f"  📦 Batch Size:    {study.best_params['batch_size']}")
    print(f"  💧 Dropout Rate:  {study.best_params['dropout_rate']:.3f}")
    print(f"\n📈 BEST VALIDATION ACCURACY: {study.best_value:.4f}")
    print(f"\n{'#'*70}\n")
    
    return study, study.best_params
