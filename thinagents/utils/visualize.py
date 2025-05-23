import contextlib
import os
from typing import Optional
from graphviz import Digraph  # type: ignore
from thinagents.core.agent import Agent
from thinagents.core.tool import ThinAgentsTool


def visualize_agent_flow(
    agent: Agent,
    filename: Optional[str] = None,
    display_in_notebook: Optional[bool] = True,
) -> None:
    """
    Visualize the flow of agents, sub-agents, and tools using Graphviz.

    Args:
        agent (Agent): The root agent to visualize.
        filename (str, optional): If provided, saves the rendered graph to this file.
                                  The output type is inferred from the file extension (e.g., 'agent_flow.png' saves as PNG).
                                  If no extension is provided, defaults to SVG (e.g., 'agent_flow' saves as 'agent_flow.svg').
                                  If also in a notebook and `display_in_notebook` is True, the graph will be displayed
                                  using the inferred or default type, in addition to being saved.
        display_in_notebook (Optional[bool], default=True):
            Controls inline display in Jupyter notebooks.
            - If True and running in a notebook, attempts to display the graph inline.
              The display type will match the type inferred from `filename` (or SVG if no `filename`).
            - If False, inline display is suppressed.

    Note:
        Requires the 'graphviz' Python package and Graphviz system binaries (e.g. 'brew install graphviz').
        Install with: pip install graphviz
    """

    determined_output_type = "svg"
    if filename:
        _name, ext = os.path.splitext(filename)
        if ext:
            determined_output_type = ext[1:].lower()

    dot = Digraph(comment="Agent Flow", format=determined_output_type)
    dot.attr(nodesep='0.5', ranksep='0.7')
    visited = set()

    def add_agent_node(current_agent, parent_id=None):
        agent_id = f"agent_{id(current_agent)}"
        if agent_id in visited:
            return
        visited.add(agent_id)
        label = f"{current_agent.name}\n[{current_agent.model}]"
        dot.node(agent_id, label, shape="box", style="filled", fillcolor="#e0f7fa")
        if parent_id:
            dot.edge(parent_id, agent_id)

        for tool in getattr(current_agent, '_provided_tools', []):
            if isinstance(tool, ThinAgentsTool):
                tool_name = tool.__name__
                tool_id = f"tool_{tool_name}_{id(tool)}"
                dot.node(tool_id, tool_name, shape="ellipse", style="filled", fillcolor="#fff9c4")
                dot.edge(agent_id, tool_id)
            elif callable(tool):
                tool_name = getattr(tool, "__name__", str(tool))
                tool_id = f"tool_{tool_name}_{id(tool)}"
                dot.node(tool_id, tool_name, shape="ellipse", style="filled", fillcolor="#fff9c4")
                dot.edge(agent_id, tool_id)

        for sub_agent in getattr(current_agent, 'sub_agents', []):
            add_agent_node(sub_agent, parent_id=agent_id)

    add_agent_node(agent)

    legend = Digraph(name='cluster_legend')
    legend.attr(
        label='Legend',
        fontsize='10',
        fontcolor='#555555',
        labelloc='b',
        margin='8',
        penwidth='0',
        rank='sink',
        rankdir='LR'
    )
    # Legend items
    legend.node('legend_agent', 'Agent', shape='box', style='filled', fillcolor='#e0f7fa', fontsize='9', width='0.6', height='0.3')
    legend.node('legend_tool', 'Tool', shape='ellipse', style='filled', fillcolor='#fff9c4', fontsize='9', width='0.6', height='0.3')
    legend.node('legend_subagent', 'Sub-agent', shape='box', style='filled', fillcolor='#e0f7fa', fontsize='9', width='0.6', height='0.3')
    legend.edge('legend_agent', 'legend_tool', label='has tool', style='dotted', fontsize='8')
    legend.edge('legend_agent', 'legend_subagent', label='has sub-agent', style='dashed', fontsize='8')
    dot.subgraph(legend)

    is_in_notebook_and_should_display = False
    if display_in_notebook:
        with contextlib.suppress(Exception):
            from IPython.core.getipython import get_ipython
            shell = get_ipython()
            if shell and shell.__class__.__name__ == "ZMQInteractiveShell":
                is_in_notebook_and_should_display = True

    if filename:
        dot.render(filename=filename, cleanup=True)

    if is_in_notebook_and_should_display:
        try:
            from IPython.display import SVG, Image, display
            data = dot.pipe(format=determined_output_type)
            if determined_output_type == "svg":
                display(SVG(data))
            elif determined_output_type == "png":
                display(Image(data))
            else:
                print(f"Auto-display in notebook for type '{determined_output_type}' not directly supported. Try 'svg' or 'png'. File may have been saved if a filename was provided.")
        except ImportError:
            print("IPython is not available. Cannot display in notebook.")
        except Exception as e:
            print(f"Error displaying graph in notebook: {e}") 