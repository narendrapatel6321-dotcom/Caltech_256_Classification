"""
ResumableTrainer — A plug-and-play training utility for Google Colab
Handles checkpointing, state persistence, and seamless resume across sessions.

Usage:
    trainer = ResumableTrainer(
        project_name="Cifar_10",
        experiment_name="model_1",
        model_fn=create_model,         # function that returns a compiled model
        checkpoint_root=find_checkpoint_root("Colab_Experiments")  # works on any account
    )
    trainer.fit(train_data, val_data, epochs=100, batch_size=64)
"""

import json
import glob
from pathlib import Path
from datetime import datetime
import hashlib
import gc
import re
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, CSVLogger, Callback
)


# ─────────────────────────────────────────────
# Shared Drive path helper
# ─────────────────────────────────────────────

def find_checkpoint_root(folder_name: str) -> str:
    """
    Auto-detects the correct path to a shared checkpoint folder,
    regardless of which Google account is running the notebook.

    - Owner account (A): folder lives at MyDrive/<folder_name>
    - Shared accounts (B, C, D): folder lives at MyDrive/Shared with me/<folder_name>

    Usage:
        trainer = ResumableTrainer(
            ...
            checkpoint_root=find_checkpoint_root("Colab_Experiments")
        )

    Args:
        folder_name (str): Name of the shared checkpoint folder on Google Drive.

    Returns:
        str: Absolute path to the folder, whichever account is running.

    Raises:
        FileNotFoundError: If the folder cannot be found in either location.
            Make sure the folder is shared with this account and Drive is mounted.
    """
    base = Path("/content/drive/MyDrive")

    # Owner account — folder lives directly in MyDrive
    direct = base / folder_name
    if direct.exists():
        return str(direct)

    # Shared accounts — folder lives under "Shared with me"
    shared = base / "Shared with me" / folder_name
    if shared.exists():
        return str(shared)

    raise FileNotFoundError(
        f"Could not find '{folder_name}' in:\n"
        f"  {direct}\n"
        f"  {shared}\n"
        f"Make sure the folder is shared with this Google account and Drive is mounted."
    )


# ─────────────────────────────────────────────
# State persistence callback
# ─────────────────────────────────────────────

class TrainingStateCallback(Callback):
    """
    Saves full training state after every epoch.
    Tracks: current epoch, best val metric, early stopping counter, completion flag.
    """

    def __init__(self, state_path, monitor='val_accuracy', mode='max', early_stopping_cb=None):
        super().__init__()
        self.state_path = Path(state_path)
        self.monitor = monitor
        self.mode = mode
        self.state = {}
        self.early_stopping_cb = early_stopping_cb

    def set_state(self, state: dict):
        """Load existing state (called before training begins)."""
        self.state = state.copy()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val = logs.get(self.monitor, None)

        # Update best val metric
        if current_val is not None:
            best = self.state.get('best_val_metric', None)
            if best is None:
                self.state['best_val_metric'] = float(current_val)
                self.state['patience_counter'] = 0
                self.state['best_epoch'] = epoch + 1
            else:
                if self.mode == 'max' and current_val > best:
                    self.state['best_val_metric'] = float(current_val)
                    self.state['patience_counter'] = 0
                    self.state['best_epoch'] = epoch + 1
                elif self.mode == 'min' and current_val < best:
                    self.state['best_val_metric'] = float(current_val)
                    self.state['patience_counter'] = 0
                    self.state['best_epoch'] = epoch + 1
                else:
                    if self.early_stopping_cb is not None:
                        self.state['patience_counter'] = int(self.early_stopping_cb.wait)
                    else:
                        self.state['patience_counter'] = self.state.get('patience_counter', 0) + 1

        self.state['last_epoch'] = epoch + 1  # 1-indexed (next epoch to run)
        self.state['last_updated'] = datetime.now().isoformat()
        self.state['training_complete'] = False
        if self.model is not None and self.model.optimizer is not None:
            try:
                lr = self.model.optimizer.learning_rate
                self.state['learning_rate'] = float(
                    tf.keras.backend.get_value(lr)
                )
            except AttributeError:
                pass  
        self._atomic_save()

    def on_train_end(self, logs=None):
        self.state['training_complete'] = True
        self.state['last_updated'] = datetime.now().isoformat()
        if self.early_stopping_cb is not None and self.early_stopping_cb.stopped_epoch > 0:
            self.state['stop_reason'] = 'early_stopping'
        else:
            self.state['stop_reason'] = 'completed'
        self._atomic_save()
        print(f"\n Training state saved → {self.state_path}")

    def _atomic_save(self):
        """Write to a temp file first, then rename — prevents corruption on crash."""
        tmp = self.state_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.state, f, indent=2)
        tmp.replace(self.state_path)


# ─────────────────────────────────────────────
# Stateful EarlyStopping
# ─────────────────────────────────────────────

class StatefulEarlyStopping(EarlyStopping):
    """
    EarlyStopping that restores its internal counter and best value
    from a saved state — so patience carries over across sessions.
    """

    def __init__(self, saved_best=None, saved_patience_counter=0, best_model_path=None, **kwargs):
        super().__init__(**kwargs)
        self._saved_best = saved_best
        self._saved_patience_counter = saved_patience_counter
        self.best_model_path = best_model_path

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if self._saved_best is not None:
            self.best = self._saved_best
            self.wait = self._saved_patience_counter
            if self.best_model_path is not None and Path(self.best_model_path).exists():
                checkpoint_weights = self.model.get_weights()
                best_model = tf.keras.models.load_model(str(self.best_model_path))
                self.best_weights = best_model.get_weights()
                del best_model
                gc.collect()
                self.model.set_weights(checkpoint_weights)
            else:
                self.best_weights = self.model.get_weights()
            print(f" EarlyStopping restored — best={self.best:.4f}, patience_counter={self.wait}")


# ─────────────────────────────────────────────
# Safe CSV Logger
# ─────────────────────────────────────────────

class SafeCSVLogger(CSVLogger):
    """
    CSVLogger that prevents duplicate header rows when resuming across sessions.

    The problem: on resume, the parent CSVLogger (in older TF/Keras versions) writes
    a fresh header even with append=True, leaving stray header rows mid-file that
    break pandas parsing.

    The fix: at the START of each session (on_train_begin), before the parent opens
    the file, we strip any duplicate headers from the existing log. This means even
    if the previous session crashed before on_train_end fired, the cleanup still
    happens correctly at the top of the next session.
    """

    @staticmethod
    def _strip_duplicate_headers(log_path: Path):
        """Remove any duplicate header rows from the CSV log file."""
        if not log_path.exists():
            return
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                return
            header = lines[0]
            cleaned = [header] + [l for l in lines[1:] if l.strip() != header.strip()]
            if len(cleaned) != len(lines):
                with open(log_path, 'w') as f:
                    f.writelines(cleaned)
        except OSError:
            pass  # non-critical — log cleanup failed silently, training continues

    def on_train_begin(self, logs=None):
        # Clean up any duplicate headers left by a previous crashed session,
        # before the parent opens the file for appending this session
        self._strip_duplicate_headers(Path(self.filename))
        super().on_train_begin(logs)


# ─────────────────────────────────────────────
# Core ResumableTrainer
# ─────────────────────────────────────────────

class ResumableTrainer:
    """
    A plug-and-play resumable training utility for Google Colab.

    Features:
    - Auto-detects and resumes from latest checkpoint
    - Persists full training state (epoch, best metric, patience counter)
    - Stateful EarlyStopping that carries over across sessions
    - CSV logging with append support
    - Works across multiple Colab accounts via shared Google Drive

    Args:
        project_name (str):     Top-level folder name (e.g., "Cifar_10")
        experiment_name (str):  Sub-folder / model name (e.g., "model_1")
        model_fn (callable):    Function that returns a freshly compiled Keras model
        checkpoint_root (str):  Root path on Google Drive
        monitor (str):          Metric to monitor (default: 'val_accuracy')
        mode (str):             'max' or 'min' depending on monitor metric
        patience (int):         EarlyStopping patience (default: 7)
        save_freq (str/int):    How often to save epoch checkpoints (default: 'epoch')
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        model_fn: callable,
        checkpoint_root: str = "/content/drive/MyDrive/Colab_Experiments",
        monitor: str = "val_accuracy",
        mode: str = "max",
        patience: int = 7,
        save_freq="epoch",
    ):
        self.experiment_name = experiment_name
        self.model_fn = model_fn
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.save_freq = save_freq

        # Directory setup
        self.ckpt_dir = Path(checkpoint_root) / project_name / experiment_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.best_model_path = self.ckpt_dir / f"{experiment_name}_best.keras"
        self.epoch_template    = str(self.ckpt_dir / f"{experiment_name}_epoch_{{epoch:04d}}.keras")
        self.state_path        = self.ckpt_dir / "training_state.json"
        self.csv_log_path      = self.ckpt_dir / "training_log.csv"

        # Will be populated on fit()
        self.model = None
        self.initial_epoch = 0
        self.state = {}

        print(f" Checkpoint directory: {self.ckpt_dir}")

    # ── Internal helpers ──────────────────────────────────────
    def _get_architecture_hash(self, model) -> str:
        """Hash the model architecture to detect changes between sessions."""
        if model is None:
            raise ValueError("_get_architecture_hash() requires an explicit model argument.")
        config = model.to_json()
        return hashlib.md5(config.encode()).hexdigest()
        
    def _load_state(self) -> dict:
        """Load training state from JSON if it exists."""
        # Clean up any abandoned .tmp file from a previous crash
        tmp = self.state_path.with_suffix('.tmp')
        if tmp.exists():
            try:
                with open(tmp) as f:
                    tmp_state = json.load(f)
                if not self.state_path.exists():
                    tmp.replace(self.state_path)
                    print(" Recovered state from leftover .tmp file")
                else:
                    with open(self.state_path) as f:
                        json_epoch = json.load(f).get('last_epoch', -1)
                    tmp_epoch = tmp_state.get('last_epoch', -1)
                    if tmp_epoch > json_epoch:
                        tmp.replace(self.state_path)
                        print(f" .tmp had newer state (epoch {tmp_epoch} vs {json_epoch}) — recovered")
                    else:
                        tmp.unlink()
                        print(" Cleaned up leftover .tmp state file")
            except (json.JSONDecodeError, OSError):
                tmp.unlink()
                print(" Leftover .tmp was corrupted — discarded")

        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    state = json.load(f)
                print(f" State loaded — last epoch: {state.get('last_epoch', 0)}, "
                      f"best {self.monitor}: {state.get('best_val_metric', 'N/A')}, "
                      f"patience counter: {state.get('patience_counter', 0)}")
                return state
            except (json.JSONDecodeError, OSError):
                print("️  State file corrupted — starting from last checkpoint only")
                return {}
        return {}

    def _save_state(self):
        """Atomically save current training state to JSON."""
        tmp = self.state_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.state, f, indent=2)
        tmp.replace(self.state_path)

    def _get_latest_checkpoint(self):
        """
        Find the latest valid (non-corrupted) epoch checkpoint file.
        Falls back to best_model_path if all epoch checkpoints are corrupted.
        """
        pattern = str(self.ckpt_dir / f"{self.experiment_name}_epoch_*.keras")
        files = glob.glob(pattern)

        if files:
            def epoch_num(f):
                match = re.search(r'_epoch_(\d+)\.keras$', f)
                return int(match.group(1)) if match else -1

            # Try from newest to oldest — skip suspiciously small (corrupted) files
            for f in sorted(files, key=epoch_num, reverse=True):
                if Path(f).stat().st_size > 1024:  # must be > 1KB to be valid
                    return f, epoch_num(f)
                else:
                    print(f" Checkpoint {Path(f).name} appears corrupted (too small) — trying previous...")

            print(" All epoch checkpoints corrupted — checking for best model fallback...")

        if self.best_model_path.exists() and self.best_model_path.stat().st_size > 1024:
            last_epoch = self.state.get('best_epoch', self.state.get('last_epoch', 0))
            print(f" Falling back to best model checkpoint (epoch ~{last_epoch})")
            return str(self.best_model_path), last_epoch

        print(" No valid checkpoints found — starting from scratch")
        return None, 0
    
    def _build_callbacks(self) -> list:
        """Build all callbacks with restored state."""
        callbacks = []

        # 1. Best model checkpoint
        callbacks.append(ModelCheckpoint(
            filepath=str(self.best_model_path),
            monitor=self.monitor,
            save_best_only=True,
            mode=self.mode,
            verbose=1
        ))

        # 2. Per-epoch checkpoint
        callbacks.append(ModelCheckpoint(
            filepath=self.epoch_template,
            save_freq=self.save_freq,
            save_best_only=False,
            verbose=0
        ))
        
        # 3. Stateful EarlyStopping (restores patience counter)
        early_stopping = StatefulEarlyStopping(
            monitor=self.monitor,
            patience=self.patience,
            mode=self.mode,
            restore_best_weights=True,
            verbose=1,
            saved_best=self.state.get('best_val_metric', None),
            saved_patience_counter=self.state.get('patience_counter', 0),
            best_model_path=self.best_model_path
        )
        callbacks.append(early_stopping)

        # 4. Safe CSV Logger 
        callbacks.append(SafeCSVLogger(
            filename=str(self.csv_log_path),
            append=True
        ))

        # 5. Training state saver — receives reference to EarlyStopping
        state_cb = TrainingStateCallback(
            state_path=self.state_path,
            monitor=self.monitor,
            mode=self.mode,
            early_stopping_cb=early_stopping  
        )
        state_cb.set_state(self.state)
        callbacks.append(state_cb)
        
        return callbacks

    def _check_already_complete(self, epochs: int) -> bool:
        if self.state.get('training_complete', False):
            last_epoch = self.state.get('last_epoch', 0)
            stop_reason = self.state.get('stop_reason', 'completed')

            if stop_reason == 'early_stopping':
                print(f" Training was stopped early at epoch {last_epoch} by EarlyStopping. Not resuming.")
                print(" If you want to continue anyway, call fit() with reset_patience=True.")
                return True

            if epochs > last_epoch:
                print(f" Previous run completed at epoch {last_epoch}, but epochs={epochs} — resuming for {epochs - last_epoch} more epochs.")
                self.state['training_complete'] = False
                self._save_state()
                return False

            print(" Training already complete! Nothing to resume.")
            return True
        return False

    # ── Public API ────────────────────────────────────────────

    def fit(self, train_data, val_data, epochs: int, reset_patience: bool = False, **fit_kwargs):
        """
        Start or resume training.

        Args:
            train_data:  Training dataset or (x_train, y_train)
            val_data:    Validation dataset or (x_val, y_val)
            epochs (int): Total number of epochs (same value every session)
            **fit_kwargs: Any additional args passed to model.fit()

        Returns:
            Keras History object
        """

        # 1. Load state
        self.state = self._load_state()

        # 2. Check if already done
        if self._check_already_complete(epochs):
            return None
        if reset_patience:
            self.state['patience_counter'] = 0
            self.state['best_val_metric'] = None
            self.state['best_epoch'] = None
            self.state['stop_reason'] = 'completed'
            print("Patience counter reset. Training will start fresh evaluation.")
            self._save_state()
        
        # 3. Find latest checkpoint
        latest_ckpt, resume_epoch = self._get_latest_checkpoint()

        # 4. Load or build model
        
        if latest_ckpt:
            print(f"\n Resuming from epoch {resume_epoch} → {latest_ckpt}")
            self.model = tf.keras.models.load_model(latest_ckpt)
            temp_model = self.model_fn()
            try :
                saved_hash = self.state.get('architecture_hash')
                if saved_hash:
                    current_hash = self._get_architecture_hash(model = temp_model)
                    if saved_hash != current_hash:
                        raise RuntimeError(
                        "Model architecture has changed since the last checkpoint!\n"
                        "If intentional, delete the checkpoint directory and retrain from scratch." )
   
                # Hard errors
                if type(temp_model.optimizer).__name__ != self.state.get('optimizer'):
                    raise RuntimeError(
                        f"Optimizer changed since last checkpoint: "
                        f"{self.state.get('optimizer')} -> {type(temp_model.optimizer).__name__}\n"
                        "If intentional, delete the checkpoint directory and retrain from scratch."
                        )
                if temp_model.loss != self.state.get('loss'):
                    raise RuntimeError(
                        f"Loss function changed since last checkpoint: "
                        f"{self.state.get('loss')} -> {temp_model.loss}\n"
                        "If intentional, delete the checkpoint directory and retrain from scratch."
                        )
                new_metrics = temp_model.metrics_names
                if new_metrics != self.state.get('metrics'):
                    raise RuntimeError(
                        f"Metrics changed since last checkpoint: "
                        f"{self.state.get('metrics')} -> {new_metrics}\n"
                        "If intentional, delete the checkpoint directory and retrain from scratch."
                        )
                if self.monitor != self.state.get('monitor'):
                    raise RuntimeError(
                        f"Monitor metric changed since last checkpoint: "
                        f"{self.state.get('monitor')} -> {self.monitor}\n"
                        "If intentional, delete the checkpoint directory and retrain from scratch."
                        )
                if self.mode != self.state.get('mode'):
                    raise RuntimeError(
                        f"Mode changed since last checkpoint: "
                        f"{self.state.get('mode')} -> {self.mode}\n"
                        "If intentional, delete the checkpoint directory and retrain from scratch."
                        )
                
                # Warn + auto-apply
                new_lr = float(temp_model.optimizer.learning_rate)
                old_lr = float(self.model.optimizer.learning_rate)
                if new_lr != old_lr:
                    print(f"Learning rate changed: {old_lr} -> {new_lr}. Applying new LR.")
                    self.model.optimizer.learning_rate.assign(new_lr)
                    self.state['learning_rate'] = new_lr 

                if self.patience != self.state.get('patience'):
                    print(f"Patience changed: {self.state.get('patience')} -> {self.patience}. Applying.")
                    self.state['patience'] = self.patience

                if self.save_freq != self.state.get('save_freq'):
                    print(f"Save frequency changed: {self.state.get('save_freq')} -> {self.save_freq}. Applying.")
                    self.state['save_freq'] = self.save_freq
                
                self.initial_epoch = resume_epoch
            finally : 
                del temp_model
                gc.collect()
        else:
            print("\n No checkpoint found — starting from scratch")
            self.model = self.model_fn()
            self.state['architecture_hash'] = self._get_architecture_hash(model=self.model)
            self.state['optimizer'] = type(self.model.optimizer).__name__
            self.state['loss'] = self.model.loss
            self.state['metrics'] = self.model.metrics_names
            self.state['monitor'] = self.monitor
            self.state['mode'] = self.mode
            self.state['patience'] = self.patience
            self.state['save_freq'] = self.save_freq
            self.state['learning_rate'] = float(self.model.optimizer.learning_rate)
            self.initial_epoch = 0
            self._save_state()

        # 5. Build callbacks
        callbacks = self._build_callbacks()

        # 6. Guard against wrong epochs value
        if self.initial_epoch >= epochs:
            print(f"  initial_epoch ({self.initial_epoch}) >= epochs ({epochs}). Nothing to train.")
            print("  Did you pass the same total epochs value as the original session?")
            return None
            
        # 7. Train
        for _key in ('epochs', 'initial_epoch', 'validation_data', 'callbacks'):
            fit_kwargs.pop(_key, None)

        fit_args = dict(
            validation_data=val_data,
            epochs=epochs,
            initial_epoch=self.initial_epoch,
            callbacks=callbacks,
            **fit_kwargs
        )
        
        print(f"\n  Training from epoch {self.initial_epoch} → {epochs}\n")
        if isinstance(train_data, tf.data.Dataset):
            history = self.model.fit(train_data, **fit_args)
        elif isinstance(train_data, tuple):
            x, y = train_data
            history = self.model.fit(x, y, **fit_args)
        else:
            history = self.model.fit(train_data, **fit_args)

        return history

    def load_best_model(self):
        """Load and return the best saved model."""
        if self.best_model_path.exists():
            print(f" Loading best model from {self.best_model_path}")
            return tf.keras.models.load_model(str(self.best_model_path))
        else:
            raise FileNotFoundError(f"No best model found at {self.best_model_path}")

    def get_training_summary(self) -> dict:
        """Print and return the current training state."""
        state = self._load_state()
        print("\n── Training Summary ──────────────────────")
        for k, v in state.items():
            print(f"  {k}: {v}")
        print("──────────────────────────────────────────\n")
        return state


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────

"""
# ── In your Colab notebook ──────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')

from resumable_trainer import ResumableTrainer

def create_model():
    model = tf.keras.Sequential([...])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

trainer = ResumableTrainer(
    project_name="Cifar_10",
    experiment_name="model_1",
    model_fn=create_model,
    checkpoint_root=find_checkpoint_root("Colab_Experiments"),  # works on any account
    monitor="val_accuracy",
    mode="max",
    patience=7
)

history = trainer.fit(
    train_dataset,
    val_dataset,
    epochs=100,
    batch_size=64
)

# Load best model anytime
best_model = trainer.load_best_model()

# Check training progress
trainer.get_training_summary()
"""
