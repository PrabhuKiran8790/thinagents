<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';

	let { data, isConnectable }: NodeProps = $props();

	// Visual differentiation based on agent level
	const level: number = (data.level as number) ?? 0;

	// Different border colors for different levels
	const borderColors = [
		'border-primary/30 border-2', // Root agent
		'border-purple-500 border-2', // Sub-agent level 1
		'border-pink-500 border-2', // Sub-agent level 2
		'border-green-500 border-2' // Sub-agent level 3
	];


	// Different background colors
	const bgColors = [
		'bg-card', // Root agent
		'bg-purple-950/30', // Sub-agent level 1
		'bg-pink-950/30', // Sub-agent level 2
		'bg-green-950/30' // Sub-agent level 3
	];

	const borderClass = borderColors[level] || borderColors[borderColors.length - 1];
	const bgColor = bgColors[level] || bgColors[bgColors.length - 1];

	// Label for agent type
	const agentLabel = level === 0 ? 'Root Agent' : `Sub-Agent (L${level})`;
</script>

<div class="text-card-foreground {bgColor} h-full w-full rounded-lg {borderClass} shadow-sm">
	<div class="flex flex-col">
		<div class="flex items-center gap-3 border-b p-6 pb-4">
			<div class="flex flex-col gap-1">
				<div class="text-lg font-semibold leading-none tracking-tight">{data.label}</div>
				<div class="text-muted-foreground text-xs">{agentLabel}</div>
			</div>
		</div>
		<div class="p-6 pt-4">
			<div class="mb-3 flex items-center gap-2 text-sm">
				<span class="text-muted-foreground font-medium">Model:</span>
				<span class="bg-muted rounded px-2 py-1 font-mono text-xs">{data.model}</span>
			</div>
			<div class="mt-3 flex gap-4">
				<div class="bg-muted/50 flex flex-1 flex-col items-center rounded-lg px-3 py-3">
					<span class="text-2xl font-bold leading-none">{data.toolCount}</span>
					<span class="text-muted-foreground mt-1 text-[11px] uppercase tracking-wider">Tools</span>
				</div>
				<div class="bg-muted/50 flex flex-1 flex-col items-center rounded-lg px-3 py-3">
					<span class="text-2xl font-bold leading-none">{data.subAgentCount}</span>
					<span class="text-muted-foreground mt-1 text-[11px] uppercase tracking-wider"
						>Sub-agents</span
					>
				</div>
			</div>
		</div>
	</div>
</div>

<Handle type="source" position={Position.Bottom} {isConnectable} />
<Handle type="target" position={Position.Top} {isConnectable} />
