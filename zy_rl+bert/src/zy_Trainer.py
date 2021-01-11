from transformers import Trainer
from torch.utils.data.dataloader import DataLoader
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
import torch
import numpy as np
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    active_mask: np.ndarray


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: np.ndarray
    label_ids: np.ndarray


class MyTrainer(Trainer):
    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        active_mask: torch.Tensor = None
        model.eval()

        # if is_torch_tpu_available():
        #     dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            past = None

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            if self.args.past_index >= 0:
                inputs["mems"] = past

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    # 取出active mask用于表示哪些词被删除
                    step_eval_loss, logits, seg_active_mask = outputs[:3]  # active mask (bs,seqlen ) 1 is retain
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]
                if self.args.past_index >= 0:
                    past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)
                if active_mask is None:
                    active_mask = seg_active_mask.detach()
                else:
                    active_mask = torch.cat((active_mask, seg_active_mask.detach()), dim=0)

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
            if active_mask is not None:
                active_mask = self.distributed_concat(active_mask, num_total_examples=self.num_examples(dataloader))
        # elif is_torch_tpu_available():
        #     # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
        #     if preds is not None:
        #         preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
        #     if label_ids is not None:
        #         label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()
        if active_mask is not None:
            active_mask = active_mask.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics, active_mask=active_mask)
