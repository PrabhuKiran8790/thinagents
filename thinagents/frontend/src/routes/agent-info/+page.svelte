<script lang="ts">
	import dagre from '@dagrejs/dagre';
	import {
		SvelteFlow,
		Background,
		Controls,
		MarkerType,
		Position,
		type Node,
		type Edge,
		type ColorMode
	} from '@xyflow/svelte';

	import '@xyflow/svelte/dist/style.css';
	import { onMount } from 'svelte';
	import AgentNode from './AgentNode.svelte';
	import ToolNode from './ToolNode.svelte';
	import { agentInfoStore } from '$lib/stores.svelte';

	interface Tool {
		type: string;
		function: {
			name: string;
			description: string;
			parameters: any;
		};
	}

	interface AgentInfo {
		name: string;
		model: string;
		tools: Tool[];
		sub_agents: AgentInfo[];
		memory: string;
	}

	let nodes = $state.raw<Node[]>([]);
	let edges = $state.raw<Edge[]>([]);
	let colorMode: ColorMode = $state('dark');
	let loading = $state(true);
	let error = $state<string | null>(null);

	let toolCounter = 0;
	let agentCounter = 0;

	const AGENT_NODE_WIDTH = 320;
	const AGENT_NODE_HEIGHT = 200;
	const TOOL_NODE_WIDTH = 280;
	const TOOL_NODE_HEIGHT = 180;

	function getRequiredParams(parameters: any): string[] {
		return parameters?.required || [];
	}

	function getParamCount(parameters: any): number {
		return Object.keys(parameters?.properties || {}).length;
	}

	const nodeTypes = {
		agent: AgentNode,
		tool: ToolNode
	};

	// Function to create nodes and edges from agent hierarchy
	function createNodesAndEdges(
		agentInfo: AgentInfo,
		agentId: string,
		level: number = 0,
		parentId: string | null = null
	): { nodes: Node[]; edges: Edge[] } {
		const nodes: Node[] = [];
		const edges: Edge[] = [];

		// Create agent node
		const agentNode: Node = {
			id: agentId,
			position: { x: 0, y: 0 }, // Position will be set by dagre
			data: {
				label: agentInfo.name,
				model: agentInfo.model,
				memory: agentInfo.memory,
				toolCount: agentInfo.tools.length,
				subAgentCount: agentInfo.sub_agents.length,
				level
			},
			type: 'agent'
		};
		nodes.push(agentNode);

		// Add edge from parent agent to this agent
		if (parentId) {
			edges.push({
				id: `e-${parentId}-${agentId}`,
				source: parentId,
				target: agentId,
				animated: true,
				type: 'smoothstep',
				style: 'stroke-width: 2px; stroke: #8b5cf6',
				markerEnd: {
					type: MarkerType.ArrowClosed,
					width: 20,
					height: 20,
					color: '#8b5cf6'
				}
			});
		}

		// Create tool nodes
		agentInfo.tools.forEach((tool) => {
			const toolId = `tool-${agentId}-${toolCounter++}`;
			const paramCount = getParamCount(tool.function.parameters);
			const requiredParams = getRequiredParams(tool.function.parameters);

			const toolNode: Node = {
				id: toolId,
				position: { x: 0, y: 0 }, // Position will be set by dagre
				data: {
					label: tool.function.name,
					description: tool.function.description,
					paramCount,
					requiredParams,
					parameters: tool.function.parameters
				},
				type: 'tool'
			};
			nodes.push(toolNode);

			// Create edge from agent to tool
			edges.push({
				id: `e-${agentId}-${toolId}`,
				source: agentId,
				target: toolId,
				animated: true,
				type: 'smoothstep',
				style: 'stroke-width: 1.5px; stroke: #6b7280',
				markerEnd: {
					type: MarkerType.ArrowClosed,
					width: 20,
					height: 20,
					color: '#FF4000'
				}
			});
		});

		// Recursively process sub-agents
		agentInfo.sub_agents.forEach((subAgent) => {
			const subAgentId = `agent-${agentCounter++}`;
			const result = createNodesAndEdges(subAgent, subAgentId, level + 1, agentId);

			nodes.push(...result.nodes);
			edges.push(...result.edges);
		});

		return { nodes, edges };
	}

	// Apply dagre layout to nodes and edges
	function applyLayout(nodes: Node[], edges: Edge[], direction = 'TB') {
		const dagreGraph = new dagre.graphlib.Graph();
		dagreGraph.setDefaultEdgeLabel(() => ({}));
		dagreGraph.setGraph({
			rankdir: direction,
			nodesep: 100,
			ranksep: 200
		});

		// Add nodes to dagre graph with appropriate sizes
		nodes.forEach((node) => {
			const width = node.type === 'agent' ? AGENT_NODE_WIDTH : TOOL_NODE_WIDTH;
			const height = node.type === 'agent' ? AGENT_NODE_HEIGHT : TOOL_NODE_HEIGHT;
			dagreGraph.setNode(node.id, { width, height });
		});

		// Add edges to dagre graph
		edges.forEach((edge) => {
			dagreGraph.setEdge(edge.source, edge.target);
		});

		// Apply layout
		dagre.layout(dagreGraph);

		// Update node positions based on dagre output
		const layoutedNodes = nodes.map((node) => {
			const nodeWithPosition = dagreGraph.node(node.id);
			const width = node.type === 'agent' ? AGENT_NODE_WIDTH : TOOL_NODE_WIDTH;
			const height = node.type === 'agent' ? AGENT_NODE_HEIGHT : TOOL_NODE_HEIGHT;

			node.targetPosition = Position.Top;
			node.sourcePosition = Position.Bottom;

			return {
				...node,
				position: {
					x: nodeWithPosition.x - width / 2,
					y: nodeWithPosition.y - height / 2
				}
			};
		});

		return { nodes: layoutedNodes, edges };
	}

	onMount(async () => {
		try {
			const response = await agentInfoStore.get_agent_info();

			let agentInfo = response as AgentInfo;
			// Reset counters
			toolCounter = 0;
			agentCounter = 0;

			// Create nodes and edges from agent hierarchy
			const { nodes: rawNodes, edges: rawEdges } = createNodesAndEdges(
				agentInfo,
				'root-agent',
				0,
				null
			);

			// Apply dagre layout
			const { nodes: layoutedNodes, edges: layoutedEdges } = applyLayout(rawNodes, rawEdges);

			nodes = layoutedNodes;
			edges = layoutedEdges;
		} catch (err) {
			error = err instanceof Error ? err.message : 'An error occurred';
		}
	});
</script>

<SvelteFlow bind:nodes bind:edges {nodeTypes} {colorMode} fitView>
	<Background />
	<Controls />
</SvelteFlow>
