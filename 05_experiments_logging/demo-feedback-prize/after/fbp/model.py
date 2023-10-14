from typing import Any

import torch
import transformers
import lightning.pytorch as pl
import omegaconf


class MyModel(pl.LightningModule):
    def __init__(self, conf: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.backbone = transformers.AutoModel.from_pretrained(
            conf["model"]["name"],
            return_dict=True,
            output_hidden_states=True,
        )
        self.drop = torch.nn.Dropout(p=conf["model"]["dropout"])
        self.fc = torch.nn.LazyLinear(out_features=len(conf["labels"]))
        self.loss_fn = torch.nn.SmoothL1Loss()

    def forward(self, input_ids, attention_mask):
        embeddings = self.backbone(input_ids, attention_mask)["last_hidden_state"][:, 0]
        embeddings = self.drop(embeddings)
        logits = self.fc(embeddings)
        return logits

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        """Here you compute and return the training loss and some additional metrics
        for e.g. the progress bar or logger.

        Args:
            batch: The output of your :class:`~torch.utils.data.DataLoader`.
                A tensor, tuple or list.
            batch_idx (``int``): Integer displaying index of this batch

        Return:
            Any of.

            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the
                key ``'loss'``
            - ``None`` - Training will skip to the next batch. This is only for automatic
                optimization. This is not supported for multi-GPU, TPU, IPU, or
                    DeepSpeed.

        In this step you'd normally do the forward pass and calculate the loss for a
        batch. You can also do fancier things like multiple forward passes or something
        model specific.

        Example::

            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss

        To use multiple optimizers, you can switch to 'manual optimization' and control
        their stepping:

        .. code-block:: python

            def __init__(self):
                super().__init__()
                self.automatic_optimization = False


            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx):
                opt1, opt2 = self.optimizers()

                # do training_step with encoder
                ...
                opt1.step()
                # do training_step with decoder
                ...
                opt2.step()

        Note:
            When ``accumulate_grad_batches`` > 1, the loss returned here will be
            automatically normalized by ``accumulate_grad_batches`` internally.
        """
        input_ids, attention_mask, labels = batch
        y_preds = self(input_ids, attention_mask).squeeze()
        labels = labels.squeeze()
        loss = self.loss_fn(y_preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like
        accuracy.

        Args:
            batch: The output of your :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple val dataloaders used)

        Return:
            - Any object or value
            - ``None`` - Validation will skip to the next batch

        .. code-block:: python

            # if you have one val dataloader:
            def validation_step(self, batch, batch_idx):
                ...


            # if you have multiple val dataloaders:
            def validation_step(self, batch, batch_idx, dataloader_idx=0):
                ...

        Examples::

            # CASE 1: A single validation dataset
            def validation_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # log the outputs!
                self.log_dict({'val_loss': loss, 'val_acc': val_acc})

        If you pass in multiple val dataloaders, :meth:`validation_step` will have an
        additional argument. We recommend setting the default value of 0 so that you can
        quickly switch between single and multiple dataloaders.

        .. code-block:: python

            # CASE 2: multiple validation dataloaders
            def validation_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...

        Note:
            If you don't need to validate you don't need to implement this method.

        Note:
            When the :meth:`validation_step` is called, the model has been put in eval
            mode and PyTorch gradients have been disabled. At the end of validation,
            the model goes back to training mode and gradients are enabled.
        """
        input_ids, attention_mask, labels = batch
        y_preds = self(input_ids, attention_mask).squeeze()
        labels = labels.squeeze()
        loss = self.loss_fn(y_preds, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return {"val_loss": loss}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest
        such as accuracy.

        Args:
            batch: The output of your :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_id: The index of the dataloader that produced this batch.
                (only if multiple test dataloaders used).

        Return:
           Any of.

            - Any object or value
            - ``None`` - Testing will skip to the next batch

        .. code-block:: python

            # if you have one test dataloader:
            def test_step(self, batch, batch_idx):
                ...


            # if you have multiple test dataloaders:
            def test_step(self, batch, batch_idx, dataloader_idx=0):
                ...

        Examples::

            # CASE 1: A single test dataset
            def test_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # log the outputs!
                self.log_dict({'test_loss': loss, 'test_acc': test_acc})

        If you pass in multiple test dataloaders, :meth:`test_step` will have an
        additional argument. We recommend setting the default value of 0 so that you can
        quickly switch between single and multiple dataloaders.

        .. code-block:: python

            # CASE 2: multiple test dataloaders
            def test_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...

        Note:
            If you don't need to test you don't need to implement this method.

        Note:
            When the :meth:`test_step` is called, the model has been put in eval mode and
            PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
            to training mode and gradients are enabled.
        """
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Step function called during
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`. By default, it
        calls :meth:`~pytorch_lightning.core.module.LightningModule.forward`.
        Override to add any processing logic.

        The :meth:`~pytorch_lightning.core.module.LightningModule.predict_step` is used
        to scale inference on multi-devices.

        To prevent an OOM error, it is possible to use
        :class:`~pytorch_lightning.callbacks.BasePredictionWriter`
        callback to write the predictions to disk or database after each batch or on
        epoch end.

        The :class:`~pytorch_lightning.callbacks.BasePredictionWriter` should be used
        while using a spawn based accelerator.
        This happens for ``Trainer(strategy="ddp_spawn")``
        or training on 8 TPU cores with ``Trainer(accelerator="tpu", devices=8)``
        as predictions won't be returned.

        Example ::

            class MyModel(LightningModule):

                def predict_step(self, batch, batch_idx, dataloader_idx=0):
                    return self(batch)

            dm = ...
            model = MyModel()
            trainer = Trainer(accelerator="gpu", devices=2)
            predictions = trainer.predict(model, dm)


        Args:
            batch: Current batch.
            batch_idx: Index of current batch.
            dataloader_idx: Index of the current dataloader.

        Return:
            Predicted output
        """
        pass

    def configure_optimizers(self) -> Any:
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or similar you
        might have multiple. Optimization with multiple optimizers only works in the
        manual optimization mode.

        Return:
            Any of these 6 options.

            - **Single optimizer**.
            - **List or Tuple** of optimizers.
            - **Two lists** - The first list has multiple optimizers, and the second has
                multiple LR schedulers (or multiple ``lr_scheduler_config``).
            - **Dictionary**, with an ``"optimizer"`` key, and (optionally) a
                ``"lr_scheduler"`` key whose value is a single LR scheduler or
                ``lr_scheduler_config``.
            - **None** - Fit will run without any optimizer.

        The ``lr_scheduler_config`` is a dictionary which contains the scheduler and its
        associated configuration. The default configuration is shown below.

        .. code-block:: python

            lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a
                # warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }

        When there are schedulers in which the ``.step()`` method is conditioned on a
        value, such as the :class:`torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler,
        Lightning requires that the ``lr_scheduler_config`` contains the keyword
        ``"monitor"`` set to the metric name that the scheduler should be conditioned on.

        .. testcode::

            # The ReduceLROnPlateau scheduler requires a monitor
            def configure_optimizers(self):
                optimizer = Adam(...)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(optimizer, ...),
                        "monitor": "metric_to_track",
                        "frequency": "indicates how often the metric is updated"
                        # If "monitor" references validation metrics, then "frequency"
                        # should be set to a
                        # multiple of "trainer.check_val_every_n_epoch".
                    },
                }


            # In the case of two optimizers, only one using the ReduceLROnPlateau
            # scheduler
            def configure_optimizers(self):
                optimizer1 = Adam(...)
                optimizer2 = SGD(...)
                scheduler1 = ReduceLROnPlateau(optimizer1, ...)
                scheduler2 = LambdaLR(optimizer2, ...)
                return (
                    {
                        "optimizer": optimizer1,
                        "lr_scheduler": {
                            "scheduler": scheduler1,
                            "monitor": "metric_to_track",
                        },
                    },
                    {"optimizer": optimizer2, "lr_scheduler": scheduler2},
                )

        Metrics can be made available to monitor by simply logging it using
        ``self.log('metric_to_track', metric_val)`` in your
        :class:`~pytorch_lightning.core.module.LightningModule`.

        Note:
            Some things to know:

            - Lightning calls ``.backward()`` and ``.step()`` automatically in case of
                automatic optimization.
            - If a learning rate scheduler is specified in ``configure_optimizers()``
                with key``"interval"`` (default "epoch") in the scheduler configuration,
                Lightning will call the scheduler's ``.step()`` method automatically in
                case of automatic optimization.
            - If you use 16-bit precision (``precision=16``), Lightning will
                automatically handle the optimizer.
            - If you use :class:`torch.optim.LBFGS`, Lightning handles the closure
                function automatically for you.
            - If you use multiple optimizers, you will have to switch to
                'manual optimization' mode and step them yourself.
            - If you need to control how often the optimizer steps, override the
                :meth:`optimizer_step` hook.
        """
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.conf["train"]["weight_decay"],
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.conf["train"]["learning_rate"],
        )
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.conf["train"]["num_warmup_steps"],
            num_training_steps=self.conf["train"]["num_training_steps"],
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
