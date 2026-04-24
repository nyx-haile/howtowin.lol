<script lang="ts">
	import { toLessonViewModel } from '$lib/lesson/adapter';
	import LessonStage from '$lib/lesson/components/LessonStage.svelte';
	import StreakBadge from '$lib/lesson/components/StreakBadge.svelte';
	import ProgressiveDisclosureGate from '$lib/lesson/components/ProgressiveDisclosureGate.svelte';
	import SpacedQueueStrip from '$lib/lesson/components/SpacedQueueStrip.svelte';
	import InterleavedBadge from '$lib/lesson/components/InterleavedBadge.svelte';
	import RetrievalDrillCard from '$lib/lesson/components/RetrievalDrillCard.svelte';
	import WeeklyLeagueCard from '$lib/lesson/components/WeeklyLeagueCard.svelte';
	import SupportivePushPreview from '$lib/lesson/components/SupportivePushPreview.svelte';
	import { spacedQueueItems } from '$lib/lesson/fixtures/spaced_queue';
	import type { PageData } from './$types';

	export let data: PageData;
	$: viewModel = toLessonViewModel(data.fixture.lesson, data.fixture.extras);

	const todaysThemes = ['vision', 'tempo'];
</script>

<div class="fixture-head">
	<a class="back" href="/review/preview/">← all fixtures</a>
	<StreakBadge />
</div>

<LessonStage {viewModel} fixtureId={data.fixture.id} />

<ProgressiveDisclosureGate>
	<section class="more-layers">
		<div class="layer-head">
			<h2>After the lesson</h2>
			<InterleavedBadge themes={todaysThemes} />
		</div>
		<SpacedQueueStrip items={spacedQueueItems} />
		<RetrievalDrillCard />
		<WeeklyLeagueCard />
		<SupportivePushPreview />
	</section>
</ProgressiveDisclosureGate>

<style>
	.fixture-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	.back {
		font-size: 0.9rem;
		color: var(--text-muted);
	}
	.back:hover {
		color: var(--accent);
	}
	.more-layers {
		display: flex;
		flex-direction: column;
		gap: 16px;
		margin-top: 8px;
		padding-top: 24px;
		border-top: 1px solid var(--border);
	}
	.layer-head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 12px;
	}
	h2 {
		font-size: 1.1rem;
		color: var(--text-muted);
		font-weight: 500;
	}
</style>
