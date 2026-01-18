import keras
import wandb
import tensorflow as tf

from wandb.sdk.lib import telemetry
from typing import Optional, Dict, Union

class WandbSingleUpdatesLogger(keras.callbacks.Callback):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.current_step_count = 0
        self.current_batch = 0

        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before WandbMetricsLogger()"
            )

        with telemetry.context(run=wandb.run) as tel:
            tel.feature.keras_metrics_logger = True

        # define custom x-axis for step-wise logging.
        wandb.define_metric("step/step")
        # set all step-wise metrics to be logged against step.
        wandb.define_metric("step/*", step_metric="step/step")

    def _get_lr(self) -> Union[float, None]:
        if isinstance(
            self.model.optimizer.learning_rate,
            (tf.Variable, tf.Tensor),
        ) or (
            hasattr(self.model.optimizer.learning_rate, "shape")
            and self.model.optimizer.learning_rate.shape == ()
        ):
            return float(self.model.optimizer.learning_rate.numpy().item())
        try:
            return float(
                self.model.optimizer.learning_rate(step=self.current_batch).numpy().item()
            )
        except Exception as e:
            wandb.termerror(f"Unable to log learning rate: {e}", repeat=False)
            return None

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, object]] = None) -> None:
        assert "step_delta" in logs, f"{self.__class__.__name__} needs `step_delta` found in logs."
        
        step_delta = logs["step_delta"]

        self.current_batch += 1
        self.current_step_count += step_delta
        logs = {f"step/{k}": v for k, v in logs.items()} if logs else {}
        logs["step/step"] = self.current_step_count

        lr = self._get_lr()
        if lr is not None:
            logs["step/learning_rate"] = lr

        wandb.log(logs)
