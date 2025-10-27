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

	import { onMount } from 'svelte';

	type ToolCallData = {
		id: string;
		name: string;
		args: Record<string, unknown>;
		status: 'pending' | 'success' | 'error';
		result?: unknown;
		error?: string;
	};

	type MessageItem = { type: 'text'; content: string } | { type: 'tool_call'; data: ToolCallData };

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

	function scrollToBottom() {
		if (messagesEndRef) {
			messagesEndRef.scrollIntoView({ behavior: 'smooth' });
		}
	}

	$effect(() => {
		console.log('Messages changed:', messages.length, messages);
		if (messages.length > 0) {
			scrollToBottom();
		}
	});

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

		const assistantMessageId = crypto.randomUUID();
		currentAssistantMessage = {
			id: assistantMessageId,
			role: 'assistant',
			items: []
		};
		messages = [...messages, currentAssistantMessage];
		console.log('Initial messages:', messages.length);

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
									console.log('Received text:', data.content);
									const lastItem = updatedMessage.items[updatedMessage.items.length - 1];
									if (lastItem && lastItem.type === 'text') {
										lastItem.content += data.content;
										updatedMessage.items = [...updatedMessage.items];
									} else {
										updatedMessage.items = [
											...updatedMessage.items,
											{ type: 'text', content: data.content }
										];
									}
									currentAssistantMessage.items = updatedMessage.items;
									messages = [...messages.slice(0, -1), updatedMessage];
									console.log('Messages updated, count:', messages.length);
								} else if (data.type === 'tool_call') {
									console.log('Tool call data received:', data);
									const existingItemIndex = updatedMessage.items.findIndex(
										(item) => item.type === 'tool_call' && item.data.id === data.tool_call_id
									);
									if (existingItemIndex === -1) {
										console.log('Creating tool call with args:', data.tool_call_args);
										updatedMessage.items = [
											...updatedMessage.items,
											{
												type: 'tool_call',
												data: {
													id: data.tool_call_id,
													name: data.tool_name,
													args: data.tool_call_args || {},
													status: 'pending' as const
												}
											}
										];
										currentAssistantMessage.items = updatedMessage.items;
										messages = [...messages.slice(0, -1), updatedMessage];
									}
								} else if (data.type === 'tool_result') {
									const itemIndex = updatedMessage.items.findIndex(
										(item) => item.type === 'tool_call' && item.data.id === data.tool_call_id
									);
									if (itemIndex !== -1) {
										updatedMessage.items = updatedMessage.items.map((item, idx) =>
											idx === itemIndex && item.type === 'tool_call'
												? {
														...item,
														data: {
															...item.data,
															status:
																data.tool_status === 'error'
																	? ('error' as const)
																	: ('success' as const),
															result: data.content,
															error: data.tool_status === 'error' ? data.content : undefined
														}
													}
												: item
										);
										currentAssistantMessage.items = updatedMessage.items;
										messages = [...messages.slice(0, -1), updatedMessage];
									}
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
								{#if message.items && message.items.length > 0}
									{#each message.items as item, idx (idx)}
										{#if item.type === 'text'}
											<Response content={item.content} class="text-primary/90 text-sm" />
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
									<Response content={message.content} class="text-primary/90 text-sm" />
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
