import optuna
from .cross_validation import train_kfold_cv


def optimize_hyperparameters(model_class, train_dataset, device,
                             n_trials=20, k_folds=3, num_epochs=10,
                             base_model_kwargs=None):
    """
    Optuna hyperparameter optimisation using K-Fold CV as the objective.

    Args:
        model_class: BoneFractureCNN or BoneFractureScatNet
        train_dataset: PyTorch Dataset
        device: 'cpu' / 'cuda' / 'mps'
        n_trials: number of Optuna trials
        k_folds: folds used in each trial (keep small, e.g. 3)
        num_epochs: epochs per fold per trial (keep small, e.g. 10)
        base_model_kwargs: fixed model kwargs (e.g. {'num_classes': 2}).
                           dropout_rate is added per trial.

    Returns:
        (study, best_params) where best_params is a dict.
    """
    if base_model_kwargs is None:
        base_model_kwargs = {'num_classes': 2}

    def objective(trial):
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        bs = trial.suggest_categorical('batch_size', [16, 32, 64])
        dr = trial.suggest_float('dropout_rate', 0.2, 0.7)

        model_kwargs = {**base_model_kwargs, 'dropout_rate': dr}

        print(f"\nTrial {trial.number+1}/{n_trials}  lr={lr:.2e}  bs={bs}  dr={dr:.3f}")

        try:
            _, _, mean_acc, std_acc, mean_f1, _ = train_kfold_cv(
                model_class=model_class,
                train_dataset=train_dataset,
                model_kwargs=model_kwargs,
                k_folds=k_folds,
                num_epochs=num_epochs,
                batch_size=bs,
                learning_rate=lr,
                device=device,
            )
            print(f"  → mean_acc={mean_acc:.4f}±{std_acc:.4f}  mean_f1={mean_f1:.4f}")
            return mean_acc
        except Exception as e:
            print(f"  Trial failed: {e}")
            return 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("\n=== HYPERPARAMETER OPTIMISATION COMPLETE ===")
    print(f"Best trial: acc={study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study, study.best_params
