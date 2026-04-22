<script lang="ts">
	import {
		WAITLIST_ENDPOINT,
		WAITLIST_FAILURE_MESSAGE,
		WAITLIST_HONEYPOT_FIELD,
		type WaitlistFormStatus,
		type WaitlistResponse
	} from '$lib/waitlist';

	export let source: string;
	export let maxWidth = '460px';
	export let marginBottom = '0';

	let email = '';
	let honeypot = '';
	let status: WaitlistFormStatus = 'idle';
	let message = '';

	$: sourceId = source.toLowerCase().replace(/[^a-z0-9]+/g, '-');
	$: emailId = `waitlist-email-${sourceId}`;
	$: honeypotId = `waitlist-honeypot-${sourceId}`;

	async function handleSubmit(event: SubmitEvent) {
		const form = event.currentTarget;

		if (!(form instanceof HTMLFormElement)) {
			return;
		}

		status = 'submitting';
		message = '';

		const formData = new FormData(form);
		formData.set('source', source);

		try {
			const result = await fetch(WAITLIST_ENDPOINT, {
				method: 'POST',
				body: formData,
				headers: {
					accept: 'application/json'
				}
			});

			const data = (await result.json().catch(() => null)) as WaitlistResponse | null;
			const nextMessage =
				data?.message ?? (result.ok ? "Thanks — we'll be in touch." : WAITLIST_FAILURE_MESSAGE);

			if (result.ok && data?.ok) {
				status = 'success';
				message = nextMessage;
				email = '';
				honeypot = '';
				form.reset();
				return;
			}

			status = 'error';
			message = nextMessage;
		} catch {
			status = 'error';
			message = WAITLIST_FAILURE_MESSAGE;
		}
	}
</script>

<div
	class="waitlist-shell"
	style={`--waitlist-max-width: ${maxWidth}; --waitlist-margin-bottom: ${marginBottom};`}
>
	{#if status === 'success'}
		<p class="status ok" role="status" aria-live="polite">{message}</p>
	{:else}
		<form class="waitlist" on:submit|preventDefault={handleSubmit}>
			<label class="sr-only" for={emailId}>Email</label>
			<input
				id={emailId}
				type="email"
				name="email"
				bind:value={email}
				placeholder="you@example.com"
				autocomplete="email"
				required
				disabled={status === 'submitting'}
				aria-invalid={status === 'error'}
			/>

			<div class="honeypot" aria-hidden="true">
				<label for={honeypotId}>Company</label>
				<input
					id={honeypotId}
					type="text"
					name={WAITLIST_HONEYPOT_FIELD}
					bind:value={honeypot}
					tabindex="-1"
					autocomplete="off"
				/>
			</div>

			<input type="hidden" name="source" value={source} />

			<button type="submit" disabled={status === 'submitting'}>
				{status === 'submitting' ? 'Joining…' : 'Join waitlist'}
			</button>
		</form>
	{/if}

	{#if status === 'error'}
		<p class="status error" role="alert">{message}</p>
	{/if}

	<p class="hint">
		We'll only use this email for launch updates. <a href="/privacy">Privacy policy</a>.
	</p>
</div>

<style>
	.waitlist-shell {
		width: 100%;
		max-width: var(--waitlist-max-width);
		margin-bottom: var(--waitlist-margin-bottom);
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.waitlist {
		display: flex;
		gap: 8px;
		width: 100%;
		flex-wrap: wrap;
	}

	input {
		flex: 1 1 220px;
		padding: 12px 14px;
		border-radius: 8px;
		border: 1px solid var(--border);
		background: var(--bg-surface);
		color: var(--text);
		font-size: 1rem;
		outline: none;
	}

	input:focus {
		border-color: var(--accent);
	}

	input:disabled,
	button:disabled {
		opacity: 0.7;
		cursor: wait;
	}

	button {
		padding: 12px 20px;
		border-radius: 8px;
		border: none;
		background: var(--accent);
		color: white;
		font-weight: 600;
		cursor: pointer;
		font-size: 0.95rem;
	}

	.status {
		font-weight: 600;
	}

	.ok {
		color: var(--win);
	}

	.error {
		color: var(--loss);
	}

	.hint {
		color: var(--text-muted);
		font-size: 0.85rem;
		line-height: 1.5;
	}

	.honeypot,
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}
</style>
