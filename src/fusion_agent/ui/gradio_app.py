"""Gradio demonstration with public examples and an optional local Agent mode."""

import os
import shutil
import tempfile
import time
from pathlib import Path

from fusion_agent.schemas import FusionPlan
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
    return [pair for pair in pairs if all(Path(path).is_file() for path in pair)]


def run_demo(visible, infrared, task, method, resize, use_agent, output="outputs/ui"):
    import gradio as gr

    if not visible or not infrared:
        raise gr.Error("Upload both a visible and an infrared image.")
    plan = FusionPlan(
        methods=["weighted", "wavelet"] if method == "both" else [method], resize=resize
    )
    try:
        if public_demo():
            if use_agent:
                raise ValueError("Agent mode is disabled on the public demo; run it locally.")
            output = public_output_root()
        if use_agent:
            from fusion_agent.agents.coordinator import run_agent_workflow

            report = run_agent_workflow(visible, infrared, output, task, plan)
        else:
            report = run_workflow(visible, infrared, output, task, plan)
    except (ValueError, OSError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from None
    gallery = [(report.files[name], name) for name in report.metrics]
    rows = [[name, *metric.model_dump().values()] for name, metric in report.metrics.items()]
    trace = [record.model_dump() for record in report.records]
    markdown = Path(report.files["report"]).read_text(encoding="utf-8")
    return gallery, rows, trace, markdown, report.files["report"]


def build_app():
    import gradio as gr

    with gr.Blocks(title="FusionAgent", analytics_enabled=False, delete_cache=(600, 3600)) as app:
        gr.Markdown("# FusionAgent\nInfrared + visible fusion · CPU baselines · Measured evidence")
        if public_demo():
            gr.Markdown(
                "公开演示 · 无需 API 密钥。请选择下面的公开样例，或上传非敏感图像。"
                "图像会发送到此服务器处理；结果为临时文件，请及时下载。"
            )
        with gr.Row():
            visible = gr.Image(label="Visible image", type="filepath", image_mode=None)
            infrared = gr.Image(label="Infrared image", type="filepath", image_mode=None)
        task = gr.Textbox(label="Task / 任务", value="Compare both methods and preserve details")
        with gr.Row():
            method = gr.Dropdown(["both", "weighted", "wavelet"], value="both", label="Methods")
            resize = gr.Checkbox(label="Allow infrared resize (not registration)")
            agent = gr.Checkbox(
                label="Agent (sends task + metadata; API key required)",
                value=False,
                visible=not public_demo(),
            )
        pairs = example_pairs()
        if pairs:
            gr.Examples(
                examples=pairs,
                inputs=[visible, infrared],
                cache_examples=False,
                label="点击加载样例 / Select a sample pair",
            )
            gr.Markdown(
                "公开样例：TNO / Alexander Toet，"
                "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。"
                "[来源及署名](https://doi.org/10.6084/m9.figshare.1008029.v2)。"
                "另含一组合成图。选择样例后点击 Run fusion。"
            )
        button = gr.Button("Run fusion", variant="primary")
        gallery = gr.Gallery(label="Fusion results", columns=2)
        table = gr.Dataframe(
            headers=["Method", "Entropy", "Std", "Avg gradient", "Spatial freq"], interactive=False
        )
        trace = gr.JSON(label="Executed tools")
        report = gr.Markdown()
        download = gr.File(label="Download Markdown report")
        button.click(
            run_demo,
            [visible, infrared, task, method, resize, agent],
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
