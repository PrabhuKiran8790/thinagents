<script lang="ts">
	import {
		PromptInput,
		PromptInputAction,
		PromptInputActions,
		PromptInputTextarea
	} from '$lib/components/prompt-kit/prompt-input';
	import { Message, MessageAvatar, MessageContent } from '$lib/components/ai-elements/message';
	import { Response } from '$lib/components/ai-elements/response/index';
	import { ToolResult } from '$lib/components/ai-elements/tool-result/index';
	import { Button } from '$lib/components/ui/button';
	import { ArrowUp, Square } from '@lucide/svelte';

	type MessageChunk =
		| { type: 'text'; content: string }
		| { type: 'tool_call'; tool_name: string; tool_call_args: string; content: string }
		| { type: 'tool_result'; tool_name: string; content: string };

	interface ChatMessage {
		id: string;
		chunks: MessageChunk[];
		timestamp: string;
		role: 'user' | 'assistant';
	}

	let input = $state('');
	let isLoading = $state(false);
	let messages = $state<ChatMessage[]>([]);
	let streamingText = $state('');

	async function handleSubmit() {
		if (!input.trim()) return;

		isLoading = true;

		const userMessage: ChatMessage = {
			id: crypto.randomUUID(),
			chunks: [{ type: 'text', content: input }],
			timestamp: new Date().toISOString(),
			role: 'user'
		};

		messages = [...messages, userMessage];
		const currentInput = input;
		input = '';

		const assistantMessageId = crypto.randomUUID();
		const assistantMessage: ChatMessage = {
			id: assistantMessageId,
			chunks: [],
			timestamp: new Date().toISOString(),
			role: 'assistant'
		};

		messages = [...messages, assistantMessage];

		try {
			const response = await fetch('/api/agent/run', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ input: currentInput })
			});

			if (!response.ok) {
				throw new Error('Failed to get response');
			}

			const reader = response.body?.getReader();
			const decoder = new TextDecoder();

			if (!reader) {
				throw new Error('No reader available');
			}

			let buffer = '';

			while (true) {
				const { done, value } = await reader.read();

				if (done) {
					break;
				}

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n');
				buffer = lines.pop() || '';

				for (const line of lines) {
					if (line.startsWith('data: ')) {
						const data = JSON.parse(line.slice(6));

						if (data.error) {
							throw new Error(data.error);
						}

						const idx = messages.findIndex((m) => m.id === assistantMessageId);
						if (idx !== -1) {
							if (data.type === 'text') {
								if (data.content) {
									streamingText += data.content;
								}
							} else if (data.type === 'tool_call' || data.type === 'tool_result') {
								if (streamingText) {
									messages[idx].chunks.push({ type: 'text', content: streamingText });
									streamingText = '';
								}
								messages[idx].chunks.push(data as MessageChunk);
								messages = [...messages];
							}
						}

						if (data.done) {
							if (streamingText) {
								const idx = messages.findIndex((m) => m.id === assistantMessageId);
								if (idx !== -1) {
									messages[idx].chunks.push({ type: 'text', content: streamingText });
									streamingText = '';
									messages = [...messages];
								}
							}
							break;
						}
					}
				}
			}
		} catch (error) {
			console.error('Error:', error);
			const idx = messages.findIndex((m) => m.id === assistantMessageId);
			if (idx !== -1) {
				messages[idx].chunks = [
					{ type: 'text', content: 'Sorry, there was an error processing your request.' }
				];
				messages = [...messages];
			}
		} finally {
			streamingText = '';
			isLoading = false;
		}
	}

	function handleValueChange(value: string) {
		input = value;
	}

	function isToolError(result: string): boolean {
		try {
			const parsed = JSON.parse(result);
			return parsed && (parsed.error || parsed.message?.includes('failed'));
		} catch {
			return result.toLowerCase().includes('error') || result.toLowerCase().includes('failed');
		}
	}

	function processChunks(chunks: MessageChunk[]) {
		const items: Array<
			| { type: 'tool'; toolName: string; result: string; status: 'success' | 'error' }
			| { type: 'text'; content: string }
		> = [];
		let textBuffer = '';

		for (let i = 0; i < chunks.length; i++) {
			const chunk = chunks[i];

			if (chunk.type === 'text') {
				textBuffer += chunk.content;
			} else if (chunk.type === 'tool_call') {
				if (textBuffer) {
					items.push({ type: 'text', content: textBuffer });
					textBuffer = '';
				}

				const nextChunk = chunks[i + 1];
				if (
					nextChunk &&
					nextChunk.type === 'tool_result' &&
					nextChunk.tool_name === chunk.tool_name
				) {
					const hasError = isToolError(nextChunk.content);
					items.push({
						type: 'tool',
						toolName: chunk.tool_name,
						result: nextChunk.content,
						status: hasError ? 'error' : 'success'
					});
					i++;
				}
			}
		}

		if (textBuffer) {
			items.push({ type: 'text', content: textBuffer });
		}

		return items;
	}
</script>

<div class="flex h-full flex-col">
	<div class="flex-1 overflow-y-auto p-4">
		<div class="mx-auto max-w-4xl">
			<div class="flex flex-col gap-8">
				{#each messages as message (message.id)}
					<Message from={message.role}>
						<MessageContent variant="flat">
							{#each processChunks(message.chunks) as item, i (i)}
								{#if item.type === 'text'}
									<Response content={item.content} />
								{:else if item.type === 'tool'}
									<ToolResult toolName={item.toolName} result={item.result} status={item.status} />
								{/if}
							{/each}
							{#if message === messages[messages.length - 1] && streamingText}
								<Response content={streamingText} />
							{/if}
						</MessageContent>
						{#if message.role === 'assistant'}
							<MessageAvatar name="AI" icon="bot" />
						{:else}
							<MessageAvatar name="You" icon="user" />
						{/if}
					</Message>
				{/each}
			</div>
		</div>
	</div>

	<div class="bg-background shrink-0 border-t p-4">
		<div class="mx-auto max-w-4xl">
			<PromptInput
				value={input}
				onValueChange={handleValueChange}
				{isLoading}
				onSubmit={handleSubmit}
				class="w-full"
			>
				<PromptInputTextarea placeholder="Ask me anything..." />
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
								<ArrowUp class="size-5" />
							{/if}
						</Button>
					</PromptInputAction>
				</PromptInputActions>
			</PromptInput>
		</div>
	</div>
</div>
