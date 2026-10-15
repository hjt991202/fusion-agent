"""Gradio demonstration with public examples and an optional local Agent mode."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from fusion_agent.models.runtime import ModelOptions
from fusion_agent.schemas import FusionPlan, selected_methods
from fusion_agent.workflow import run_workflow


def public_demo() -> bool:
    return os.getenv("FUSION_PUBLIC_DEMO") == "1"


def public_output_root() -> Path:
    """Bound transient public runs; remove only old, app-created run directories."""
    root = Path(tempfile.gettempdir()) / "fusion-agent-public"
    root.mkdir(exist_ok=True)
    for path in root.glob("run-*"):
        if (
            path.is_dir()
            and not path.is_symlink()
            and path.resolve().parent == root.resolve()
            and time.time() - path.stat().st_mtime > 3600
        ):
            shutil.rmtree(path)
    if len(list(root.glob("run-*"))) >= 100:
        raise ValueError("Demo is busy. Please retry after older results expire.")
    return root


def example_pairs() -> list[list[str]]:
    root = Path(
        os.getenv("FUSION_EXAMPLES_DIR", str(Path(__file__).resolve().parents[3] / "examples"))
    )
    pairs = [
        [
            str(root / "tno" / "visible" / f"{name}.png"),
            str(root / "tno" / "infrared" / f"{name}.png"),
        ]
        for name in ["lake", "helicopter", "house"]
    ]
    pairs.append([str(root / "visible.png"), str(root / "infrared.png")])
    manifest = root / "jiandu" / "manifest.json"
    if manifest.is_file():
        for pair in json.loads(manifest.read_text(encoding="utf-8"))["pairs"]:
            pairs.append(
                [
                    str(root / "jiandu" / pair["visible"]["file"]),
                    str(root / "jiandu" / pair["infrared"]["file"]),
                ]
            )
    return [pair for pair in pairs if all(Path(path).is_file() for path in pair)]


def run_demo(
    visible,
    infrared,
    task,
    method,
    resize,
    use_agent,
    output="outputs/ui",
    visible_y=None,
    valid_mask=None,
    material_mask=None,
    material_mode="visible",
):
    import gradio as gr

    if not visible or not infrared:
        raise gr.Error("Upload both a visible and an infrared image.")
    try:
        plan = FusionPlan(methods=selected_methods(method), resize=resize)
        options = ModelOptions(
            visible_y=visible_y,
            valid_mask=valid_mask,
            material_mask=material_mask,
            material_mode=material_mode,
        )
        if public_demo():
            if use_agent:
                raise ValueError("Agent mode is disabled on the public demo; run it locally.")
            output = public_output_root()
        if use_agent:
            from fusion_agent.agents.coordinator import run_agent_workflow

            report = run_agent_workflow(
                visible, infrared, output, task, plan, model_options=options
            )
        else:
            report = run_workflow(visible, infrared, output, task, plan, model_options=options)
    except (ValueError, OSError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from None
    gallery = [(report.files[name], name) for name in report.metrics]
    gallery.extend((path, name) for name, path in report.files.items() if name.endswith("_color"))
    rows = [[name, *metric.model_dump().values()] for name, metric in report.metrics.items()]
    trace = [record.model_dump() for record in report.records]
    markdown = Path(report.files["report"]).read_text(encoding="utf-8")
    return gallery, rows, trace, markdown, report.files["report"]


def build_app():
    import gradio as gr

    with gr.Blocks(title="FusionAgent", analytics_enabled=False, delete_cache=(600, 3600)) as app:
        gr.Markdown(
            "# FusionAgent\nInfrared + visible fusion · Baselines + 简牍研究方法\n"
            "weighted / wavelet 为基础方法；icig / letters / prcv 需安装 research 依赖和本地权重。"
            "research 比较三种简牍方法，all 比较全部五种。"
        )
        if public_demo():
            gr.Markdown(
                "公开演示 · 无需 API 密钥。请选择下面的公开样例，或上传非敏感图像。"
                "图像会发送到此服务器处理；结果为临时文件，请及时下载。"
            )
        with gr.Row():
            visible = gr.Image(label="Visible image", type="filepath", image_mode=None)
            infrared = gr.Image(label="Infrared image", type="filepath", image_mode=None)
        task = gr.Textbox(
            label="Task / 任务", value="Compare selected methods and preserve details"
        )
        with gr.Row():
            method = gr.Dropdown(
                ["both", "weighted", "wavelet", "icig", "letters", "prcv", "research", "all"],
                value="both",
                label="Methods",
            )
            resize = gr.Checkbox(label="Allow infrared resize (not registration)")
            agent = gr.Checkbox(
                label="Agent (sends task + metadata; API key required)",
                value=False,
                visible=not public_demo(),
            )
        with gr.Accordion("研究方法输入 / Research input options", open=False):
            prepared_y = gr.File(
                label="ICIG: prepared visible luminance (optional)", type="filepath"
            )
            valid = gr.File(label="Letters: observed valid mask (optional)", type="filepath")
            material = gr.File(label="Letters: material silhouette (optional)", type="filepath")
            material_mode = gr.Dropdown(
                ["visible", "valid"],
                value="visible",
                label="Letters material support",
            )
            gr.Markdown(
                "不提供有效区域图时，letters 视整幅图为有效。visible 从彩色可见光估计材料区域；"
                "valid 使用有效区域作为材料范围。上传的 mask 必须与可见光尺寸一致。"
            )
        pairs = example_pairs()
        if pairs:
            sample_rows = []
            for v, i in pairs:
                visible_path = Path(v)
                prepared_path = visible_path.parent.parent / "visible_y" / visible_path.name
                y = str(prepared_path) if prepared_path.is_file() else None
                sample_rows.append([v, i, y, None, None, "visible"])
            gr.Examples(
                examples=sample_rows,
                inputs=[visible, infrared, prepared_y, valid, material, material_mode],
                cache_examples=False,
                label="点击加载样例 / Select a sample pair",
            )
            gr.Markdown(
                "公开样例：TNO / Alexander Toet，"
                "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。"
                "[来源及署名](https://doi.org/10.6084/m9.figshare.1008029.v2)。"
                "另含一组合成图和作者授权展示的 15 对简牍图（不适用 TNO 的 CC BY 许可）。"
                "选择样例后点击 Run fusion。"
            )
        visible.upload(lambda: (None, None, None), outputs=[prepared_y, valid, material])
        infrared.upload(lambda: (None, None), outputs=[valid, material])
        button = gr.Button("Run fusion", variant="primary")
        gallery = gr.Gallery(label="Fusion results", columns=2)
        table = gr.Dataframe(
            headers=["Method", "Entropy", "Std", "Avg gradient", "Spatial freq"], interactive=False
        )
        trace = gr.JSON(label="Executed tools")
        report = gr.Markdown()
        download = gr.File(label="Download Markdown report")
        button.click(
            lambda v, i, t, m, r, a, y, valid, material, mode: run_demo(
                v,
                i,
                t,
                m,
                r,
                a,
                visible_y=y,
                valid_mask=valid,
                material_mask=material,
                material_mode=mode,
            ),
            [
                visible,
                infrared,
                task,
                method,
                resize,
                agent,
                prepared_y,
                valid,
                material,
                material_mode,
            ],
            [gallery, table, trace, report, download],
            api_name="fuse",
            concurrency_limit=1,
        )
    return app.queue(max_size=8)


def main():
    build_app().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"), share=False, max_file_size="32mb"
    )


if __name__ == "__main__":
    main()
