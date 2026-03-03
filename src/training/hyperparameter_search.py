import optuna
from .cross_validation import train_kfold_cv


def optimize_hyperparameters(model_class, train_dataset, device, 
                            n_trials=20, k_folds=3, num_epochs=10):
    
    def objective(trial):

        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Dropout rate: between 0.2 and 0.7
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
        
        # Print what we're testing this trial
        print(f"TRIAL {trial.number + 1}/{n_trials}")
        print(f"Testing hyperparameters:")
        print(f"Learning Rate: {learning_rate:.6f}")
        print(f"Batch Size:    {batch_size}")
        print(f"Dropout Rate:  {dropout_rate:.3f}")
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
            
            print(f"TRIAL {trial.number + 1} RESULTS")
            print(f"Mean Validation Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            
            # Show results for each fold
            print(f"\nFold-by-fold results:")
            for fold_result in fold_results:
                print(f"  Fold {fold_result['fold']}: {fold_result['best_val_acc']:.4f}")
            
            # Check if this is the best trial so far
            if trial.number > 0:
                current_best = trial.study.best_value
                if mean_acc > current_best:
                    improvement = mean_acc - current_best
                    print(f"\nNEW BEST ACCURACY!")
                    print(f"   Previous best: {current_best:.4f}")
                    print(f"   New best:      {mean_acc:.4f}")
                    print(f"   Improvement:   +{improvement:.4f}")
                else:
                    print(f"\n   Current best remains: {current_best:.4f}")
            else:
                print(f"\nFirst trial complete!")
            
            print(f"{'='*70}\n")
            
            return mean_acc
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return 0.0 
    
    study = optuna.create_study(direction='maximize')
    
    print(f"STARTING HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print(f"Model:              {model_class.__name__}")
    print(f"Number of trials:   {n_trials}")
    print(f"K-folds per trial:  {k_folds}")
    print(f"Epochs per fold:    {num_epochs}")
    
    study.optimize(objective, n_trials=n_trials)
    
    print(f"HYPERPARAMETER OPTIMIZATION COMPLETE!")
    print(f"\nBEST HYPERPARAMETERS FOUND:")
    print(f"Learning Rate: {study.best_params['learning_rate']:.6f}")
    print(f"Batch Size:    {study.best_params['batch_size']}")
    print(f"Dropout Rate:  {study.best_params['dropout_rate']:.3f}")
    print(f"\nBEST VALIDATION ACCURACY: {study.best_value:.4f}")
    
    return study, study.best_params
