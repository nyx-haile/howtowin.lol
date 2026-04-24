<script lang="ts">
	import { fixtureSummaries } from '$lib/lesson/fixtures';
</script>

<section class="intro">
	<h1>Pedagogy flow preview</h1>
	<p>
		Each fixture renders a complete mock lesson — pretest, reveal, intervention — with Layers 4–6
		(spaced reinforcement, retrieval check, engagement scaffolding) visible but inert. No real
		match data, no model inference.
	</p>
	<p class="note">
		Streak advances only when you commit an intent. It will not advance from opening the page or
		viewing a reveal.
	</p>
</section>

<ul class="fixtures">
	{#each fixtureSummaries as fx (fx.id)}
		<li class="fixture">
			<a href="/review/preview/{fx.id}/">
				<div class="head">
					<span class="polarity polarity-{fx.polarity}">{fx.polarity}</span>
					<span class="id">{fx.id}</span>
				</div>
				<p class="summary">{fx.summary}</p>
			</a>
		</li>
	{/each}
</ul>

<style>
	.intro {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	h1 {
		font-size: 1.6rem;
	}
	.intro p {
		color: var(--text-muted);
		line-height: 1.6;
	}
	.note {
		font-size: 0.9rem;
		color: var(--text-muted);
		border-left: 2px solid var(--accent);
		padding-left: 12px;
	}
	.fixtures {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.fixture a {
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 14px 16px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 6px;
		color: var(--text);
	}
	.fixture a:hover {
		border-color: var(--accent);
		background: var(--bg-hover);
		color: var(--text);
	}
	.head {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.polarity {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		padding: 2px 8px;
		border-radius: 4px;
		font-weight: 600;
	}
	.polarity-mistake {
		background: color-mix(in srgb, var(--loss) 22%, transparent);
		color: var(--loss);
	}
	.polarity-strength {
		background: color-mix(in srgb, var(--win) 22%, transparent);
		color: var(--win);
	}
	.polarity-neutral {
		background: color-mix(in srgb, var(--text-muted) 22%, transparent);
		color: var(--text-muted);
	}
	.id {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.85rem;
		color: var(--text-muted);
	}
	.summary {
		color: var(--text-muted);
		font-size: 0.95rem;
		line-height: 1.5;
	}
</style>
