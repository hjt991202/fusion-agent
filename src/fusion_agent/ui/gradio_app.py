"""Local-only Gradio UI, with reusable callback for offline tests."""

from pathlib import Path

from fusion_agent.schemas import FusionPlan
from fusion_agent.workflow import run_workflow


def run_demo(visible, infrared, task, method, resize, use_agent, output="outputs/ui"):
    import gradio as gr

    if not visible or not infrared:
        raise gr.Error("Upload both a visible and an infrared image.")
    plan = FusionPlan(
        methods=["weighted", "wavelet"] if method == "both" else [method], resize=resize
    )
    try:
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

    with gr.Blocks(title="FusionAgent", analytics_enabled=False) as app:
        gr.Markdown("# FusionAgent\nInfrared + visible fusion · CPU baselines · Measured evidence")
        with gr.Row():
            visible = gr.Image(label="Visible image", type="filepath", image_mode=None)
            infrared = gr.Image(label="Infrared image", type="filepath", image_mode=None)
        task = gr.Textbox(label="Task / 任务", value="Compare both methods and preserve details")
        with gr.Row():
            method = gr.Dropdown(["both", "weighted", "wavelet"], value="both", label="Methods")
            resize = gr.Checkbox(label="Allow infrared resize (not registration)")
            agent = gr.Checkbox(label="Agent (sends task + metadata; API key required)")
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
        )
    return app


def main():
    build_app().launch(server_name="127.0.0.1", share=False, max_file_size="32mb")


if __name__ == "__main__":
    main()
