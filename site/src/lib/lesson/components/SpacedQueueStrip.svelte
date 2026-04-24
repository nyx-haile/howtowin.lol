<script lang="ts">
	import type { SpacedQueueItem } from '$lib/lesson/fixtures/spaced_queue';

	export let items: SpacedQueueItem[];
	export let scaffoldLevel: 0 | 1 | 2 = 0;

	function dayLabel(offset: number): string {
		if (offset === 0) return 'today';
		if (offset === 1) return 'tomorrow';
		return `day +${offset}`;
	}
</script>

<section class="strip" data-scaffold={scaffoldLevel}>
	<header>
		<h3>On deck — spaced review</h3>
		<p class="sub">
			Interleaved across themes. Expanding intervals (Kang 2016). <em>stub</em>
		</p>
	</header>
	<ol class="queue">
		{#each items as item (item.id)}
			<li class="item">
				<span class="day">{dayLabel(item.dayOffset)}</span>
				<span class="theme" style="--theme: {item.themeColorVar}">{item.theme}</span>
				<span class="summary">{item.summary}</span>
			</li>
		{/each}
	</ol>
</section>

<style>
	.strip {
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding: 14px 16px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 6px;
		opacity: 0.75;
	}
	header h3 {
		font-size: 0.95rem;
	}
	header .sub {
		color: var(--text-muted);
		font-size: 0.8rem;
		margin-top: 2px;
	}
	em {
		font-style: normal;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		font-size: 0.7rem;
	}
	.queue {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.item {
		display: grid;
		grid-template-columns: 80px 90px 1fr;
		gap: 10px;
		align-items: baseline;
		padding: 6px 0;
		border-top: 1px dashed var(--border);
		font-size: 0.88rem;
	}
	.item:first-child {
		border-top: none;
	}
	.day {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		color: var(--text-muted);
		font-size: 0.8rem;
	}
	.theme {
		color: var(--theme, var(--text-muted));
		text-transform: uppercase;
		font-size: 0.72rem;
		letter-spacing: 0.06em;
	}
	.summary {
		color: var(--text);
	}
</style>
