<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';

	let { data, isConnectable }: NodeProps = $props();

	const requiredParams = (data.requiredParams || []) as string[];
	const requiredParamsText = requiredParams.length > 0 ? requiredParams.join(', ') : 'None';
	const description = (data.description as string) || '';
	const hasDescription = description.trim().length > 0;
</script>

<div
	class="bg-card text-card-foreground h-full w-[280px] rounded-lg border border-orange-500/50 shadow-sm"
>
	<div class="flex flex-col">
		<div class="flex items-center gap-3 border-b p-4 pb-3">
			<div class="text-xl">🔧</div>
			<div class="flex-1">
				<div class="text-base font-semibold leading-none tracking-tight">{data.label}</div>
			</div>
		</div>
		<div class="p-4 pt-3">
			{#if hasDescription}
				<div class="text-muted-foreground mb-3 border-b pb-3 text-xs leading-relaxed">
					{description}
				</div>
			{/if}
			<div class="flex flex-col gap-2">
				<div class="flex items-center justify-between text-sm">
					<span class="text-muted-foreground font-medium">Parameters:</span>
					<span
						class="bg-primary text-primary-foreground rounded-full px-2.5 py-0.5 text-xs font-bold"
						>{data.paramCount}</span
					>
				</div>
				{#if requiredParams.length > 0}
					<div class="flex flex-col gap-1 text-sm">
						<span class="text-muted-foreground font-medium">Required:</span>
						<div class="flex flex-wrap gap-1">
							{#each requiredParams as param}
								<span class="bg-muted rounded px-2 py-0.5 font-mono text-xs">{param}</span>
							{/each}
						</div>
					</div>
				{:else}
					<div class="flex items-center justify-between text-sm">
						<span class="text-muted-foreground font-medium">Required:</span>
						<span class="bg-muted rounded px-2 py-0.5 font-mono text-xs">None</span>
					</div>
				{/if}
			</div>
		</div>
	</div>
</div>

<Handle type="target" position={Position.Top} {isConnectable} />
