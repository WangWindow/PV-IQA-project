from __future__ import annotations

from pathlib import Path

import onnx
import torch
from onnx.external_data_helper import convert_model_from_external_data
from onnx.numpy_helper import to_array
from torch import nn

from pv_iqa.config import AppConfig
from pv_iqa.models.iqa import LightweightIQARegressor
from pv_iqa.utils.common import save_json

IMAGENET_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGENET_NORMALIZE_STD = [0.229, 0.224, 0.225]
ONNX_INPUT_NAME = "image"
ONNX_OUTPUT_NAME = "score"
RUST_EXPORT_PROFILE = "candle-upstream-static-v2"


class IQAScoreExportWrapper(nn.Module):
    def __init__(self, model: LightweightIQARegressor) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)["score"]


def _fallback_model_kwargs(config: AppConfig) -> dict[str, str | bool | float | int]:
    return {
        "backbone_name": config.iqa.backbone,
        "pretrained": False,
        "image_size": config.data.image_size,
        "use_waveformer_layer": config.iqa.use_waveformer_layer,
        "waveformer_mlp_ratio": config.iqa.waveformer_mlp_ratio,
    }


def resolve_onnx_artifact_paths(checkpoint_path: str | Path) -> tuple[Path, Path]:
    resolved_checkpoint = Path(checkpoint_path)
    onnx_path = resolved_checkpoint.with_suffix(".onnx")
    metadata_path = resolved_checkpoint.with_suffix(".onnx.json")
    return onnx_path, metadata_path


def load_iqa_model_for_export(
    config: AppConfig,
    checkpoint_path: str | Path,
) -> tuple[LightweightIQARegressor, dict[str, str | bool | float | int]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = checkpoint.get("model_kwargs", _fallback_model_kwargs(config))
    model = LightweightIQARegressor(
        **{
            **model_kwargs,
            "pretrained": False,
        }
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, model_kwargs


def _make_scalar_initializer(name: str, value: float) -> onnx.TensorProto:
    return onnx.helper.make_tensor(
        name=name,
        data_type=onnx.TensorProto.FLOAT,
        dims=[],
        vals=[value],
    )


def _decompose_hard_sigmoid(model_proto: onnx.ModelProto) -> onnx.ModelProto:
    graph = model_proto.graph
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []

    for index, node in enumerate(graph.node):
        if node.op_type != "HardSigmoid":
            new_nodes.append(node)
            continue

        attrs = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        alpha = float(attrs.get("alpha", 1.0 / 6.0))
        beta = float(attrs.get("beta", 0.5))
        prefix = f"hard_sigmoid_{index}"
        alpha_name = f"{prefix}_alpha"
        beta_name = f"{prefix}_beta"
        zero_name = f"{prefix}_zero"
        one_name = f"{prefix}_one"
        mul_output = f"{prefix}_mul"
        add_output = f"{prefix}_add"

        new_initializers.extend(
            [
                _make_scalar_initializer(alpha_name, alpha),
                _make_scalar_initializer(beta_name, beta),
                _make_scalar_initializer(zero_name, 0.0),
                _make_scalar_initializer(one_name, 1.0),
            ]
        )
        new_nodes.extend(
            [
                onnx.helper.make_node(
                    "Mul",
                    inputs=[node.input[0], alpha_name],
                    outputs=[mul_output],
                    name=f"{prefix}_mul",
                ),
                onnx.helper.make_node(
                    "Add",
                    inputs=[mul_output, beta_name],
                    outputs=[add_output],
                    name=f"{prefix}_add",
                ),
                onnx.helper.make_node(
                    "Clip",
                    inputs=[add_output, zero_name, one_name],
                    outputs=list(node.output),
                    name=f"{prefix}_clip",
                ),
            ]
        )

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)
    return model_proto


def _collect_value_ranks(model_proto: onnx.ModelProto) -> dict[str, int]:
    return {
        name: len(shape)
        for name, shape in _collect_value_shapes(model_proto).items()
    }


def _collect_value_shapes(model_proto: onnx.ModelProto) -> dict[str, list[int | None]]:
    inferred = onnx.shape_inference.infer_shapes(model_proto)
    shapes: dict[str, list[int | None]] = {}

    def _update_shape(value_info: onnx.ValueInfoProto) -> None:
        tensor_type = value_info.type.tensor_type
        if tensor_type.HasField("shape"):
            shapes[value_info.name] = [
                dim.dim_value if dim.HasField("dim_value") else None
                for dim in tensor_type.shape.dim
            ]

    for value_info in inferred.graph.input:
        _update_shape(value_info)
    for value_info in inferred.graph.value_info:
        _update_shape(value_info)
    for value_info in inferred.graph.output:
        _update_shape(value_info)
    return shapes


def _rewrite_opset17_compat(model_proto: onnx.ModelProto) -> onnx.ModelProto:
    reduction_ops = {"ReduceMean", "ReduceL2", "ReduceSum"}
    axes_values = {
        initializer.name: [int(item) for item in to_array(initializer).reshape(-1).tolist()]
        for initializer in model_proto.graph.initializer
    }
    value_ranks = _collect_value_ranks(model_proto)
    new_nodes: list[onnx.NodeProto] = []
    initializer_updates: dict[str, list[int]] = {}
    initializers_to_drop: set[str] = set()

    for node in model_proto.graph.node:
        if node.op_type == "Split":
            attrs = {
                attribute.name: onnx.helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            if "num_outputs" not in attrs:
                new_nodes.append(node)
                continue
            new_nodes.append(
                onnx.helper.make_node(
                    "Split",
                    inputs=[node.input[0]],
                    outputs=list(node.output),
                    name=node.name,
                    axis=int(attrs.get("axis", 0)),
                )
            )
            continue

        if node.op_type not in reduction_ops or len(node.input) < 2:
            new_nodes.append(node)
            continue

        axes_name = node.input[1]
        axes = axes_values.get(axes_name)
        if axes is None:
            new_nodes.append(node)
            continue

        rank = value_ranks.get(node.input[0])
        if rank is not None:
            axes = [axis if axis >= 0 else rank + axis for axis in axes]

        attrs = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if node.op_type in {"ReduceMean", "ReduceL2"}:
            initializers_to_drop.add(axes_name)
            new_nodes.append(
                onnx.helper.make_node(
                    node.op_type,
                    inputs=[node.input[0]],
                    outputs=list(node.output),
                    name=node.name,
                    axes=axes,
                    keepdims=int(attrs.get("keepdims", 1)),
                )
            )
        else:
            if axes != axes_values[axes_name]:
                initializer_updates[axes_name] = axes
            new_nodes.append(node)

    del model_proto.graph.node[:]
    model_proto.graph.node.extend(new_nodes)
    used_initializer_names = {input_name for node in new_nodes for input_name in node.input}
    if initializer_updates or initializers_to_drop:
        new_initializers: list[onnx.TensorProto] = []
        for initializer in model_proto.graph.initializer:
            if initializer.name in initializers_to_drop and initializer.name not in used_initializer_names:
                continue
            updated_axes = initializer_updates.get(initializer.name)
            if updated_axes is None:
                new_initializers.append(initializer)
                continue
            new_initializers.append(
                onnx.helper.make_tensor(
                    name=initializer.name,
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(updated_axes)],
                    vals=updated_axes,
                )
            )
        del model_proto.graph.initializer[:]
        model_proto.graph.initializer.extend(new_initializers)
    return model_proto


def export_iqa_onnx(
    config: AppConfig,
    *,
    checkpoint_path: str | Path | None = None,
    opset_version: int = 17,
    dynamic_batch: bool = False,
) -> Path:
    resolved_checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else config.experiment_dir / "iqa" / "best.pt"
    )
    model, model_kwargs = load_iqa_model_for_export(config, resolved_checkpoint)
    wrapper = IQAScoreExportWrapper(model)
    wrapper.eval()
    onnx_path, metadata_path = resolve_onnx_artifact_paths(resolved_checkpoint)

    dummy_input = torch.randn(
        1,
        3,
        config.data.image_size,
        config.data.image_size,
        dtype=torch.float32,
    )
    export_opset_version = max(opset_version, 18)
    with torch.no_grad():
        export_kwargs = {
            "input_names": [ONNX_INPUT_NAME],
            "output_names": [ONNX_OUTPUT_NAME],
            "opset_version": export_opset_version,
            "do_constant_folding": True,
            "external_data": False,
        }
        if dynamic_batch:
            export_kwargs["dynamic_axes"] = {
                ONNX_INPUT_NAME: {0: "batch_size"},
                ONNX_OUTPUT_NAME: {0: "batch_size"},
            }
        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path,
            **export_kwargs,
        )

    model_proto = onnx.load(onnx_path)
    convert_model_from_external_data(model_proto)
    model_proto = _decompose_hard_sigmoid(model_proto)
    model_proto = _rewrite_opset17_compat(model_proto)
    if model_proto.opset_import[0].version != opset_version:
        model_proto.opset_import[0].version = opset_version
    onnx.save_model(model_proto, onnx_path, save_as_external_data=False)
    external_data_path = onnx_path.with_name(f"{onnx_path.name}.data")
    if external_data_path.exists():
        external_data_path.unlink()
    model_proto = onnx.load(onnx_path, load_external_data=False)

    metadata = {
        "checkpoint_path": str(resolved_checkpoint),
        "model_path": str(onnx_path),
        "input_name": ONNX_INPUT_NAME,
        "output_name": ONNX_OUTPUT_NAME,
        "image_size": config.data.image_size,
        "grayscale_to_rgb": config.data.grayscale_to_rgb,
        "normalize_mean": IMAGENET_NORMALIZE_MEAN,
        "normalize_std": IMAGENET_NORMALIZE_STD,
        "opset_version": model_proto.opset_import[0].version,
        "dynamic_batch": dynamic_batch,
        "export_profile": RUST_EXPORT_PROFILE,
        "model_kwargs": model_kwargs,
    }
    save_json(metadata_path, metadata)
    return onnx_path


__all__ = [
    "ONNX_INPUT_NAME",
    "ONNX_OUTPUT_NAME",
    "RUST_EXPORT_PROFILE",
    "export_iqa_onnx",
    "load_iqa_model_for_export",
    "resolve_onnx_artifact_paths",
]
