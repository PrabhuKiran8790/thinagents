<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';
	import AppSidebar from '$lib/components/app-sidebar.svelte';
	import { Separator } from '$lib/components/ui/separator/index.js';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import { agentInfoStore } from '$lib/stores.svelte';
	import { Database, TriangleAlert } from '@lucide/svelte';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<Sidebar.Provider>
	<AppSidebar />
	<Sidebar.Inset class="flex h-screen flex-col">
		<header
			class="group-has-data-[collapsible=icon]/sidebar-wrapper:h-12 flex h-12 shrink-0 items-center gap-2 transition-[width,height] ease-linear"
		>
			<div class="flex items-center gap-2 px-4">
				<Sidebar.Trigger class="-ml-1" />
			</div>
			<div>
				{#await agentInfoStore.get_agent_info()}
					<div>Loading...</div>
				{:then agentInfo}
					{#if agentInfo.memory}
						<div class="flex items-center justify-center gap-2 text-sm">
							<div class="size-1 animate-ping rounded-full bg-green-500"></div>
							<span>{agentInfo.memory}</span>
							<Database class="size-3" />
						</div>
					{:else}
						<div class="flex items-center justify-center gap-2 text-sm">
							<div class="size-1 animate-ping rounded-full bg-orange-500"></div>
							<span>No memory</span>
							<TriangleAlert class="size-3 text-orange-500" />
						</div>
					{/if}
				{/await}
			</div>
		</header>
		<main class="flex-1 overflow-hidden">
			{@render children?.()}
		</main>
	</Sidebar.Inset>
</Sidebar.Provider>
