<script lang="ts">
	import {
		PromptInput,
		PromptInputAction,
		PromptInputActions,
		PromptInputTextarea
	} from '$lib/components/prompt-kit/prompt-input';
	import { Button } from '$lib/components/ui/button';
	import { cn } from '$lib/utils';
	import { ArrowUp, Square } from '@lucide/svelte';
	import { ToolCallResult } from '$lib/components/ai-elements/tool-result';
	import { Response } from '$lib/components/ai-elements/response';
	import { AgentCall } from '$lib/components/ai-elements/agent-call';
	import { ThinkingLoader } from '$lib/components/ai-elements/thinking-loader';

	import { onMount } from 'svelte';

	type ToolCallData = {
		id: string;
		name: string;
		args: Record<string, unknown>;
		status: 'pending' | 'success' | 'error';
		result?: unknown;
		error?: string;
		agentName?: string;
		isSubagent?: boolean;
	};

	type AgentCallData = {
		agentName: string;
		isSubagent: boolean;
		toolCalls: ToolCallData[];
		nestedAgents: AgentCallData[];
		textResponse?: string;
	};

	type MessageItem =
		| { type: 'text'; content: string }
		| { type: 'tool_call'; data: ToolCallData }
		| { type: 'agent_call'; data: AgentCallData };

	type MessageType = {
		id: string;
		role: 'user' | 'assistant';
		content?: string;
		items?: MessageItem[];
	};

	let input = $state('');
	let isLoading = $state(false);
	let messages = $state<MessageType[]>([]);
	let messagesEndRef = $state<HTMLDivElement | null>(null);
	let currentAssistantMessage: MessageType | null = null;
	let abortController: AbortController | null = null;
	let currentAgentBatch: AgentCallData | null = null;
	let currentAgentBatchIndex: number = -1;
	let parentAgentStack: AgentCallData[] = [];

	function scrollToBottom() {
		if (messagesEndRef) {
			messagesEndRef.scrollIntoView({ behavior: 'smooth' });
		}
	}

	$effect(() => {
		if (messages.length > 0) {
			scrollToBottom();
		}
	});

	function resetAgentHierarchy() {
		currentAgentBatch = null;
		currentAgentBatchIndex = -1;
		parentAgentStack = [];
	}

	async function handleSubmit() {
		if (!input.trim() || isLoading) return;

		const userMessage: MessageType = {
			id: crypto.randomUUID(),
			role: 'user',
			content: input.trim()
		};

		messages = [...messages, userMessage];
		const userInput = input.trim();
		input = '';
		isLoading = true;

		resetAgentHierarchy();

		const assistantMessageId = crypto.randomUUID();
		currentAssistantMessage = {
			id: assistantMessageId,
			role: 'assistant',
			items: []
		};
		messages = [...messages, currentAssistantMessage];

		try {
			abortController = new AbortController();

			const response = await fetch('/api/agent/run', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ input: userInput }),
				signal: abortController.signal
			});

			if (!response.ok) {
				throw new Error('Failed to get response');
			}

			const reader = response.body?.getReader();
			const decoder = new TextDecoder('utf-8');

			if (!reader) {
				throw new Error('No response body');
			}

			let buffer = '';

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n');

				buffer = lines.pop() || '';

				for (const line of lines) {
					if (line.trim() === '') continue;

					if (line.startsWith('data: ')) {
						try {
							const data = JSON.parse(line.slice(6));

							if (data.done) {
								isLoading = false;
								currentAssistantMessage = null;
								break;
							}

							if (data.error) {
								console.error('Error:', data.error);
								if (currentAssistantMessage) {
									const lastMessageIndex = messages.length - 1;
									const updatedMessage = { ...messages[lastMessageIndex] };
									if (!updatedMessage.items) {
										updatedMessage.items = [];
									}
									updatedMessage.items = [
										...updatedMessage.items,
										{ type: 'text', content: `Error: ${data.error}` }
									];
									messages = [...messages.slice(0, -1), updatedMessage];
								}
								isLoading = false;
								currentAssistantMessage = null;
								break;
							}

							if (currentAssistantMessage) {
								const lastMessageIndex = messages.length - 1;
								const updatedMessage = { ...messages[lastMessageIndex] };

								if (!updatedMessage.items) {
									updatedMessage.items = [];
								}

								if (data.type === 'text') {
									currentAgentBatch = null;
									currentAgentBatchIndex = -1;
									parentAgentStack = [];

									const lastItem = updatedMessage.items[updatedMessage.items.length - 1];
									if (lastItem && lastItem.type === 'text') {
										lastItem.content += data.content;
									} else {
										updatedMessage.items.push({ type: 'text', content: data.content });
									}

									currentAssistantMessage.items = [...updatedMessage.items];
									messages = [...messages.slice(0, -1), updatedMessage];
								} else if (data.type === 'tool_call') {
									const agentName = data.agent_name || 'Agent';
									const isSubagent = data.is_subagent || false;
									const isSubagentCall = data.tool_name.startsWith('subagent_');

									if (isSubagentCall) {
										if (!currentAgentBatch || currentAgentBatch.agentName !== agentName) {
											currentAgentBatch = {
												agentName,
												isSubagent,
												toolCalls: [],
												nestedAgents: [],
												textResponse: ''
											};
											updatedMessage.items.push({
												type: 'agent_call',
												data: currentAgentBatch
											});
											currentAgentBatchIndex = updatedMessage.items.length - 1;
										}

										parentAgentStack.push(currentAgentBatch);

										updatedMessage.items = [...updatedMessage.items];
										currentAssistantMessage.items = updatedMessage.items;
										messages = [...messages.slice(0, -1), updatedMessage];
									} else {
										const parentAgent =
											parentAgentStack.length > 0
												? parentAgentStack[parentAgentStack.length - 1]
												: null;

										if (parentAgent && isSubagent) {
											let subagent = parentAgent.nestedAgents.find(
												(na) => na.agentName === agentName
											);

											if (!subagent) {
												subagent = {
													agentName,
													isSubagent: true,
													toolCalls: [],
													nestedAgents: [],
													textResponse: ''
												};

												// Create a NEW parent agent object with the new nested agent
												const updatedParent: AgentCallData = {
													...parentAgent,
													toolCalls: [...parentAgent.toolCalls],
													nestedAgents: [...parentAgent.nestedAgents, subagent]
												};

												// Find and replace the parent in the items array BY NAME (not by reference)
												const parentIndex = updatedMessage.items.findIndex(
													(item) =>
														item.type === 'agent_call' &&
														item.data.agentName === parentAgent.agentName
												);

												if (parentIndex !== -1) {
													updatedMessage.items[parentIndex] = {
														type: 'agent_call',
														data: updatedParent
													};
													updatedMessage.items = [...updatedMessage.items];

													// Update the parent stack to point to the new parent object
													parentAgentStack[parentAgentStack.length - 1] = updatedParent;

													currentAssistantMessage.items = updatedMessage.items;
													messages = [...messages.slice(0, -1), updatedMessage];
												}
											}

											currentAgentBatch = subagent;
										} else {
											if (!currentAgentBatch || currentAgentBatch.agentName !== agentName) {
												currentAgentBatch = {
													agentName,
													isSubagent,
													toolCalls: [],
													nestedAgents: [],
													textResponse: ''
												};
												updatedMessage.items.push({
													type: 'agent_call',
													data: currentAgentBatch
												});
												currentAgentBatchIndex = updatedMessage.items.length - 1;
											}
										}

										const newToolCall = {
											id: data.tool_call_id,
											name: data.tool_name,
											args: data.tool_call_args || {},
											status: 'pending' as const,
											agentName,
											isSubagent
										};
										currentAgentBatch.toolCalls.push(newToolCall);

										updatedMessage.items = [...updatedMessage.items];
										currentAssistantMessage.items = updatedMessage.items;
										messages = [...messages.slice(0, -1), updatedMessage];
									}
								} else if (data.type === 'tool_result') {
									const isSubagentResult = data.tool_name && data.tool_name.startsWith('subagent_');

									if (isSubagentResult) {
										// Set textResponse on the nested agent
										if (currentAgentBatch && currentAgentBatch.agentName === data.agent_name) {
											currentAgentBatch.textResponse = data.content;

											// Deep clone to force reactivity
											function cloneAgentWithNestedUpdate(agent: AgentCallData): AgentCallData {
												return {
													...agent,
													toolCalls: [...agent.toolCalls],
													nestedAgents: agent.nestedAgents.map((nested) =>
														nested.agentName === data.agent_name
															? {
																	...nested,
																	textResponse: data.content,
																	toolCalls: [...nested.toolCalls]
																}
															: cloneAgentWithNestedUpdate(nested)
													)
												};
											}

											// Update the items array with cloned agents
											updatedMessage.items = updatedMessage.items.map((item) => {
												if (item.type === 'agent_call') {
													return {
														...item,
														data: cloneAgentWithNestedUpdate(item.data)
													};
												}
												return item;
											});
										}

										// Update parent stack and current batch references to use the new cloned objects
										if (parentAgentStack.length > 0) {
											parentAgentStack.pop();

											// Update parent stack to point to cloned objects
											if (parentAgentStack.length > 0) {
												const parentAgentName =
													parentAgentStack[parentAgentStack.length - 1].agentName;
												const updatedParent = updatedMessage.items.find(
													(item) =>
														item.type === 'agent_call' && item.data.agentName === parentAgentName
												);
												if (updatedParent && updatedParent.type === 'agent_call') {
													currentAgentBatch = updatedParent.data;
													parentAgentStack[parentAgentStack.length - 1] = currentAgentBatch;
												}
											} else {
												currentAgentBatch = null;
											}
										} else {
											currentAgentBatch = null;
										}

										// Force Svelte reactivity
										currentAssistantMessage.items = updatedMessage.items;
										messages = [...messages.slice(0, -1), updatedMessage];
									} else {
										function findAndUpdateTool(agent: AgentCallData): boolean {
											const toolIndex = agent.toolCalls.findIndex(
												(tc) => tc.id === data.tool_call_id
											);

											if (toolIndex !== -1) {
												agent.toolCalls[toolIndex].status =
													data.tool_status === 'error' ? 'error' : 'success';
												agent.toolCalls[toolIndex].result = data.content;
												agent.toolCalls[toolIndex].error =
													data.tool_status === 'error' ? data.content : undefined;
												return true;
											}

											for (const nested of agent.nestedAgents) {
												if (findAndUpdateTool(nested)) {
													return true;
												}
											}

											return false;
										}

										for (const item of updatedMessage.items) {
											if (item.type === 'agent_call') {
												if (findAndUpdateTool(item.data)) {
													break;
												}
											}
										}
									}

									updatedMessage.items = [...updatedMessage.items];
									currentAssistantMessage.items = updatedMessage.items;
									messages = [...messages.slice(0, -1), updatedMessage];
								}
							}
						} catch (e) {
							console.error('Failed to parse SSE data:', line, e);
						}
					}
				}
			}
		} catch (error) {
			console.error('Error streaming response:', error);
			if (currentAssistantMessage && messages.length > 0) {
				const lastMessageIndex = messages.length - 1;
				const updatedMessage = { ...messages[lastMessageIndex] };
				if (!updatedMessage.items) {
					updatedMessage.items = [];
				}
				updatedMessage.items = [
					...updatedMessage.items,
					{ type: 'text', content: 'Sorry, an error occurred while processing your request.' }
				];
				messages = [...messages.slice(0, -1), updatedMessage];
			}
			isLoading = false;
			currentAssistantMessage = null;
		}
	}

	function handleStopGeneration() {
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
		isLoading = false;
		currentAssistantMessage = null;
	}

	function handleValueChange(value: string) {
		input = value;
	}

	onMount(() => {
		return () => {
			if (abortController) {
				abortController.abort();
			}
		};
	});
</script>

<div class="flex h-full">
	{#if messages.length === 0}
		<div
			class="mx-auto -mt-20 flex h-full w-full max-w-4xl select-none flex-col items-center justify-center gap-6"
		>
			<div>
				<span class="font-mono text-3xl font-bold">Talk to your Agent</span>
			</div>
			<PromptInput
				value={input}
				onValueChange={handleValueChange}
				{isLoading}
				onSubmit={handleSubmit}
				class="max-w-(--breakpoint-4xl) bg-sidebar w-full border-0"
			>
				<PromptInputTextarea placeholder="Ask anything" />
				<PromptInputActions class="justify-end pt-2">
					<PromptInputAction>
						{#snippet tooltip()}
							{isLoading ? 'Stop generation' : 'Send message'}
						{/snippet}
						<Button
							variant="default"
							size="icon"
							class="h-8 w-8 rounded-full"
							onclick={handleSubmit}
						>
							{#if isLoading}
								<Square class="size-5 fill-current" />
							{:else}
								<ArrowUp class={cn('size-5', input ? 'rotate-90' : '', 'duration-700')} />
							{/if}
						</Button>
					</PromptInputAction>
				</PromptInputActions>
			</PromptInput>
		</div>
	{:else}
		<div class="mx-auto flex h-full w-full max-w-4xl flex-col">
			<div class="flex-1 overflow-y-auto px-4 py-6">
				{#each messages as message (message.id)}
					<div
						data-message-id={message.id}
						class={cn(
							'text-primary/90 group mb-4 flex w-full gap-3',
							message.role === 'user' ? 'justify-end' : 'justify-start'
						)}
					>
						{#if message.role === 'user'}
							<div class="bg-sidebar max-w-[80%] rounded-2xl px-4 py-3">
								<div class="whitespace-pre-wrap text-sm">{message.content}</div>
							</div>
						{:else}
							<div class="flex w-full max-w-[85%] flex-col gap-3">
								{#if isLoading && (!message.items || message.items.length === 0) && !message.content}
									<ThinkingLoader />
								{:else if message.items && message.items.length > 0}
									{#each message.items as item, idx (idx)}
										{#if item.type === 'text'}
											<Response content={item.content} class="text-sm" />
										{:else if item.type === 'agent_call'}
											<AgentCall
												agentName={item.data.agentName}
												isSubagent={item.data.isSubagent}
												toolCalls={item.data.toolCalls}
												nestedAgents={item.data.nestedAgents}
												textResponse={item.data.textResponse}
												class="w-full"
											/>
										{:else if item.type === 'tool_call'}
											<ToolCallResult
												toolName={item.data.name}
												status={item.data.status}
												arguments={item.data.args}
												result={item.data.result}
												error={item.data.error}
												class="w-full"
											/>
										{/if}
									{/each}
								{:else if message.content}
									<Response content={message.content} class="text-sm" />
								{/if}
							</div>
						{/if}
					</div>
				{/each}
				<div bind:this={messagesEndRef}></div>
			</div>
			<div class="px-4 py-4">
				<PromptInput
					value={input}
					onValueChange={handleValueChange}
					{isLoading}
					onSubmit={handleSubmit}
					class="max-w-(--breakpoint-4xl) bg-sidebar w-full border-0"
				>
					<PromptInputTextarea placeholder="Ask anything" />
					<PromptInputActions class="justify-end pt-2">
						<PromptInputAction>
							{#snippet tooltip()}
								{isLoading ? 'Stop generation' : 'Send message'}
							{/snippet}
							<Button
								variant="default"
								size="icon"
								class="h-8 w-8 rounded-full"
								onclick={isLoading ? handleStopGeneration : handleSubmit}
							>
								{#if isLoading}
									<Square class="size-5 fill-current" />
								{:else}
									<ArrowUp class={cn('size-5', input ? 'rotate-90' : '', 'duration-700')} />
								{/if}
							</Button>
						</PromptInputAction>
					</PromptInputActions>
				</PromptInput>
			</div>
		</div>
	{/if}
</div>
