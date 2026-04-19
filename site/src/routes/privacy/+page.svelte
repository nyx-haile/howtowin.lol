<svelte:head>
	<title>Privacy Policy — howtowin.lol</title>
	<meta name="description" content="Privacy Policy for howtowin.lol." />
</svelte:head>

<article class="legal-page">
	<h1>Privacy Policy</h1>
	<p class="meta">Last updated: 2026-04-18</p>

	<section>
		<h2>1. Scope</h2>
		<p>
			This policy describes what data howtowin.lol collects, how we use it, and who we
			share it with. It applies to all visitors and users of the Service.
		</p>
	</section>

	<section>
		<h2>2. Data we collect</h2>
		<ul>
			<li>
				<strong>Riot account identifiers:</strong> when you search a summoner, we request
				their public <code>puuid</code>, Riot ID, and rank from the Riot Games API. We do
				not ask for your password or any private account credential.
			</li>
			<li>
				<strong>Match data:</strong> we retrieve publicly available match history and
				per-game timeline events to generate lessons.
			</li>
			<li>
				<strong>Email (optional):</strong> if you join the waitlist, we store the email
				address you submit. We never sell or share it.
			</li>
			<li>
				<strong>Basic request logs:</strong> IP address, user agent, and timestamps for
				abuse prevention and rate limiting. Retained for up to 30 days.
			</li>
		</ul>
	</section>

	<section>
		<h2>3. How we use data</h2>
		<ul>
			<li>To generate per-game lessons and progression metrics for the user who searched.</li>
			<li>
				To train and improve our machine-learning models. Training data is aggregated and
				anonymized; we do not associate model outputs with a specific Riot ID beyond the
				user who requested that lesson.
			</li>
			<li>To operate, secure, and debug the Service.</li>
		</ul>
	</section>

	<section>
		<h2>4. How we share data</h2>
		<p>
			We do not sell personal data. We share data only with infrastructure providers
			strictly necessary to operate the Service (hosting, error tracking). We may disclose
			data if required by law.
		</p>
	</section>

	<section>
		<h2>5. Riot Games API</h2>
		<p>
			Data obtained from the Riot Games API is used in accordance with the
			<a
				href="https://developer.riotgames.com/policies/general"
				target="_blank"
				rel="noopener noreferrer">Riot Developer Policies</a
			>. We honor deletion requests propagated via the Riot API and will remove cached data
			for any account on request.
		</p>
	</section>

	<section>
		<h2>6. Cookies</h2>
		<p>
			We use only essential cookies required for the Service to function. We do not use
			third-party advertising or cross-site tracking cookies.
		</p>
	</section>

	<section>
		<h2>7. Your rights</h2>
		<p>
			You can request deletion of any data associated with your Riot ID or email by
			contacting <a href="mailto:xyn@mit.edu">xyn@mit.edu</a>. If you're covered by GDPR
			or CCPA, you additionally have rights to access, rectify, and port your data — the
			same contact applies.
		</p>
	</section>

	<section>
		<h2>8. Children</h2>
		<p>
			The Service is not directed at children under 13. We do not knowingly collect data
			from children under 13. If you believe we have, please contact us and we will delete
			it.
		</p>
	</section>

	<section>
		<h2>9. Changes</h2>
		<p>
			We may update this policy. Material changes will be announced on this page with an
			updated revision date.
		</p>
	</section>

	<section>
		<h2>10. Contact</h2>
		<p>
			Privacy questions or deletion requests: <a href="mailto:xyn@mit.edu">xyn@mit.edu</a>.
		</p>
	</section>
</article>

<style>
	.legal-page { max-width: 720px; display: flex; flex-direction: column; gap: 24px; }
	h1 { font-size: 1.8rem; margin-bottom: 4px; }
	.meta { color: var(--text-muted); font-size: 0.85rem; margin-top: -16px; }
	h2 { font-size: 1.05rem; margin-bottom: 8px; }
	p, li { color: var(--text-muted); line-height: 1.65; font-size: 0.95rem; }
	ul { padding-left: 20px; display: flex; flex-direction: column; gap: 6px; }
	code {
		background: var(--bg-hover); padding: 1px 6px; border-radius: 3px;
		font-size: 0.85rem; color: var(--text);
	}
	strong { color: var(--text); }
</style>
