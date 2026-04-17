import argparse
import json
import os
from contextlib import nullcontext

import torch
from accelerate import Accelerator, DistributedType
from peft import LoraConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification_dataset_harmony import ClassificationDatasetHarmony
from modeling_harmony import HarmonyTwoStageClassifier


def _gather_param_for_save(param):
    if not hasattr(param, "ds_id"):
        return param.detach().cpu().clone()

    try:
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    except ImportError as exc:
        raise RuntimeError(
            "DeepSpeed ZeRO-3 parameter gathering requested, but deepspeed is not installed."
        ) from exc

    if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
        with zero.GatheredParameters([param], modifier_rank=0):
            return param.detach().cpu().clone()
    return param.detach().cpu().clone()


def _build_local_state_dict(module, trainable_only=False):
    state_dict = {}
    for name, parameter in module.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        state_dict[name] = _gather_param_for_save(parameter)
    for name, buffer in module.named_buffers():
        state_dict[name] = buffer.detach().cpu().clone()
    return state_dict


def _gather_embedding_row(param, row_index):
    if not hasattr(param, "ds_id"):
        return param.detach()[row_index].cpu().clone()

    try:
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    except ImportError as exc:
        raise RuntimeError(
            "DeepSpeed ZeRO-3 parameter gathering requested, but deepspeed is not installed."
        ) from exc

    if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
        with zero.GatheredParameters([param], modifier_rank=0):
            return param.detach()[row_index].cpu().clone()
    return param.detach()[row_index].cpu().clone()


def save_checkpoint(accelerator, model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    adapter_dir = os.path.join(save_dir, "adapter")
    head_path = os.path.join(save_dir, "classifier_head.pth")
    tokenizer_dir = os.path.join(save_dir, "tokenizer")
    assess_token_config_path = os.path.join(save_dir, "assess_token_config.json")
    assess_token_embedding_path = os.path.join(save_dir, "assess_token_embedding.pth")

    unwrapped_model = accelerator.unwrap_model(model)
    classifier_head_state = _build_local_state_dict(unwrapped_model.classifier_head)

    if accelerator.is_main_process:
        unwrapped_model.model.save_pretrained(
            adapter_dir,
            selected_adapters=["classifier"],
            safe_serialization=False,
            save_embedding_layers=False,
        )
        torch.save(classifier_head_state, head_path)
        unwrapped_model.tokenizer.save_pretrained(tokenizer_dir)

        input_embedding_row = unwrapped_model.assess_token_embedding.detach().cpu().clone()
        output_embedding_layer = unwrapped_model.model.get_output_embeddings()
        output_embedding_row = None
        if output_embedding_layer is not None and hasattr(output_embedding_layer, "weight"):
            output_embedding_row = _gather_embedding_row(
                output_embedding_layer.weight,
                unwrapped_model.special_token_id,
            )

        with open(assess_token_config_path, "w", encoding="utf-8") as file_object:
            json.dump(
                {
                    "assess_token": unwrapped_model.assess_token,
                    "special_token_id": int(unwrapped_model.special_token_id),
                    "num_assess_tokens": int(unwrapped_model.num_assess_tokens),
                    "tokenizer_dir": "tokenizer",
                    "embedding_path": "assess_token_embedding.pth",
                },
                file_object,
                ensure_ascii=False,
                indent=2,
            )
        torch.save(
            {
                "assess_token": unwrapped_model.assess_token,
                "special_token_id": int(unwrapped_model.special_token_id),
                "input_embedding": input_embedding_row,
                "output_embedding": output_embedding_row,
            },
            assess_token_embedding_path,
        )
    accelerator.wait_for_everyone()


def broadcast_bool_flag(accelerator, value):
    flag = torch.tensor(
        [1 if value else 0],
        device=accelerator.device,
        dtype=torch.int32,
    )
    if accelerator.num_processes > 1 and torch.distributed.is_initialized():
        torch.distributed.broadcast(flag, src=0)
    return bool(flag.item())


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "y", "1"}:
        return True
    if lowered in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_csv_arg(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def detect_dtype(dtype_str):
    dtype_str = dtype_str.lower().strip()
    if dtype_str == "auto":
        return None
    if dtype_str in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_str in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported --dtype: {dtype_str}. Use auto|bf16|fp16.")


def get_device_map_for_distributed(accelerator):
    if not torch.cuda.is_available():
        return None
    if accelerator.distributed_type in {DistributedType.DEEPSPEED, DistributedType.FSDP}:
        return None
    if "LOCAL_RANK" not in os.environ:
        return None
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return {"": f"cuda:{local_rank}"}


def model_uses_mxfp4(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as file_object:
            config = json.load(file_object)
    except (OSError, json.JSONDecodeError):
        return False

    quantization_config = config.get("quantization_config") or {}
    quant_method = quantization_config.get("quant_method")
    return isinstance(quant_method, str) and quant_method.strip().lower() == "mxfp4"


def get_model_init_context(accelerator, model_path=None):
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        return nullcontext()

    plugin = accelerator.state.deepspeed_plugin
    if plugin is None or not plugin.is_zero3_init_enabled():
        return nullcontext()
    return plugin.zero3_init_context_manager(enable=True)


def evaluate(accelerator, model, val_dataloader, val_dataset, global_step, epoch):
    model.eval()
    all_mse_scores = []
    all_preds_list = []
    all_labels_list = []

    progress_bar = tqdm(
        val_dataloader,
        desc=f"Epoch {epoch} [Validate]",
        disable=not accelerator.is_main_process,
    )
    with torch.no_grad():
        for batch in progress_bar:
            outputs = model(
                prefix_ids=batch["prefix_ids"],
                prefix_attention_mask=batch["prefix_attention_mask"],
                labels=batch["labels"],
            )
            logits = outputs["logits"]
            soft_labels = batch["labels"]

            predicted_probs = torch.softmax(logits, dim=-1)
            mse = torch.nn.functional.mse_loss(
                predicted_probs,
                soft_labels,
                reduction="none",
            ).mean(dim=-1)
            predicted_hard_labels = torch.argmax(predicted_probs, dim=-1)
            true_hard_labels = torch.argmax(soft_labels, dim=-1)

            all_mse_scores.append(accelerator.gather_for_metrics(mse).cpu())
            all_preds_list.append(
                accelerator.gather_for_metrics(predicted_hard_labels).cpu()
            )
            all_labels_list.append(
                accelerator.gather_for_metrics(true_hard_labels).cpu()
            )

    if not accelerator.is_main_process:
        return None

    actual_val_size = len(val_dataset)
    all_mse_scores_cat = torch.cat(all_mse_scores)
    final_mse = all_mse_scores_cat[:actual_val_size].mean().item()

    all_preds = torch.cat(all_preds_list)[:actual_val_size]
    all_labels = torch.cat(all_labels_list)[:actual_val_size]
    final_accuracy = (all_preds == all_labels).float().mean().item()

    accelerator.print(
        f"Epoch {epoch} Validation - MSE: {final_mse:.4f}, Accuracy: {final_accuracy:.4f}"
    )
    accelerator.log(
        {
            "validation_mse": final_mse,
            "validation_accuracy": final_accuracy,
            "epoch": epoch,
        },
        step=global_step,
    )
    return final_mse, final_accuracy


def build_dataset(args, tokenizer, data_path):
    return ClassificationDatasetHarmony(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_hard_labels=args.use_hard_labels,
    )


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
    )

    if accelerator.is_main_process:
        print(f"Accelerator device: {accelerator.device}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Distributed type: {accelerator.distributed_type}")
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name}},
        )

    model_dtype = detect_dtype(args.dtype)
    device_map = get_device_map_for_distributed(accelerator)
    if accelerator.is_main_process and accelerator.distributed_type == DistributedType.DEEPSPEED:
        print("DeepSpeed ZeRO-3 init is enabled for model loading.")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    with get_model_init_context(accelerator, model_path=args.model_path):
        model = HarmonyTwoStageClassifier(
            model_path=args.model_path,
            lora_config=lora_config,
            num_labels=2,
            num_assess_tokens=args.num_assess_tokens,
            assess_token=args.assess_token,
            torch_dtype=model_dtype,
            local_files_only=args.local_files_only,
            device_map=device_map,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    tokenizer = model.tokenizer

    train_dataset = build_dataset(args, tokenizer, args.data_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    val_dataset = None
    val_dataloader = None
    if args.val_data_path:
        val_dataset = build_dataset(args, tokenizer, args.val_data_path)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
        )

    optimizer = AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    if val_dataloader is not None:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
        )

    best_mse = float("inf")
    global_step = 0
    accelerator.print("Training started...")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs} [Train]",
            disable=not accelerator.is_main_process,
        )

        for batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(
                    prefix_ids=batch["prefix_ids"],
                    prefix_attention_mask=batch["prefix_attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"]
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    accelerator.log(
                        {
                            "train_loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

                total_train_loss += loss.item()
                if accelerator.is_main_process:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / max(len(train_dataloader), 1)
        accelerator.print(
            f"Epoch {epoch + 1} Training finished. Average Loss: {avg_train_loss:.4f}"
        )

        if val_dataloader is not None:
            metrics = evaluate(
                accelerator,
                model,
                val_dataloader,
                val_dataset,
                global_step,
                epoch + 1,
            )
            should_save_best = False
            if accelerator.is_main_process and metrics is not None:
                final_mse, _ = metrics
                if final_mse < best_mse:
                    best_mse = final_mse
                    should_save_best = True
                    accelerator.print(
                        "New best validation MSE found. Saving to 'best_checkpoint'..."
                    )

            should_save_best = broadcast_bool_flag(accelerator, should_save_best)
            if should_save_best:
                save_checkpoint(
                    accelerator,
                    model,
                    os.path.join(args.output_dir, "best_checkpoint"),
                )

            # Intentionally keep only the best checkpoint when validation is enabled.

    if val_dataloader is None:
        accelerator.print("Training finished. Saving final model...")
        accelerator.wait_for_everyone()
        save_checkpoint(accelerator, model, args.output_dir)

    accelerator.end_training()
    accelerator.print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output_harmony_model")

    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument(
        "--target_modules",
        type=parse_csv_arg,
        default=[
            "q_proj",
            "v_proj",
        ],
    )
    parser.add_argument("--num_assess_tokens", type=int, default=4)
    parser.add_argument("--assess_token", type=str, default="[ASSESS]")
    parser.add_argument("--use_hard_labels", type=str2bool, default=False)

    parser.add_argument("--wandb_project", type=str, default="harmony_two_stage_classifier")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    if args.wandb_run_name is None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        num_gpus = len([device for device in visible_devices if device.strip()])
        global_batch_size = args.batch_size * args.gradient_accumulation_steps * max(num_gpus, 1)
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.wandb_run_name = (
            f"harmony-{model_name}-lr{args.learning_rate}-bs{global_batch_size}"
        )

    main(args)
